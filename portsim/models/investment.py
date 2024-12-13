import pandas as pd
import seaborn as sns
from typing import Any
from copy import deepcopy
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, Extra

from portsim.models.rate import Rate
from portsim.enums.investment import InvestmentType, InvestmentFrequency

pd.set_option('display.float_format', lambda x: '%.2f' % x)


class Investment(BaseModel):
    name: str = Field(description="Name of the Investment")
    investment_type: InvestmentType = Field(
        description="Type of investment. Accepts enum `InvestmentType` as input"
    )
    principal_amount: float = Field(
        gt=0, frozen=True,
        description="Principal Amount Invested"
    )
    time: int = Field(
        gt=0, frozen=True,
        description="Time Period (in years)"
    )
    rate: Rate = Field(
        frozen=True,
        description="Period by period rates of return. Defined by the `Rate` class"
    )
    investment_freq: InvestmentFrequency = Field(
        description="Frequency of the investment. Accepts enum `InvestmentFrequency` as input"
    )
    compounding_freq: int = Field(
        default=1, frozen=True, gt=1,
        description="Number of times the amount is compounded per time period"
    )
    metadata: dict = Field(default={})

    _amount: list[float] = PrivateAttr(default=[])
    _interest: list[float] = PrivateAttr(default=[])
    _acc_amount: float = PrivateAttr(default=0)
    _amount_invested: float = PrivateAttr(default=0)
    _total_interest: float = PrivateAttr(default=0)
    _period_earnings: list[float] = PrivateAttr(default=[])
    _forecast_df: pd.DataFrame = PrivateAttr(default=pd.DataFrame())
    _forecasted: bool = PrivateAttr(default=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra=Extra.forbid
    )

    @property
    def amount(self):
        return self._amount

    @property
    def interest(self):
        return self._interest

    @property
    def acc_amount(self):
        return self._acc_amount

    @property
    def amount_invested(self):
        return self._amount_invested

    @property
    def total_interest(self):
        return self._total_interest

    @property
    def period_earnings(self):
        return self._period_earnings

    @property
    def forecast_df(self):
        return self._forecast_df

    def plot(self, save_path=None):
        if not self._forecasted:
            self.forecast()

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(7)
        fig.set_figwidth(14)

        ax = axs[0, 0]
        ax = sns.scatterplot(ax=ax, x="n_years", y="amount", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="amount", marker='o', data=self.forecast_df)
        ylabels = [f"{round(x)}K" for x in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Number of Years")
        ax.set_ylabel(" Total Amount")

        ax = axs[0, 1]
        ax = sns.scatterplot(ax=ax, x="n_years", y="interest", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="interest", marker='o', data=self.forecast_df)
        ylabels = [f"{round(x)}K" for x in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Number of Years")
        ax.set_ylabel("Accumulated Interest")

        ax = axs[1, 0]
        ax = sns.scatterplot(ax=ax, x="n_years", y="period_earnings", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="period_earnings", marker='o', data=self.forecast_df)
        ylabels = [f"{round(x)}K" for x in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Number of Years")
        ax.set_ylabel("Annual Earnings")

        ax = axs[1, 1]
        ax = sns.scatterplot(ax=ax, x="n_years", y="interest_rate", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="interest_rate", marker='o', data=self.forecast_df)
        ylabels = [f"{round(x * 100, 2)}%" for x in ax.get_yticks()]
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Number of Years")
        ax.set_ylabel("Annual Interest Rate")

        if save_path is not None:
            fig.savefig(save_path)

    def _compute_earnings(self) -> None:
        self._forecast_df = pd.DataFrame(
            {
                "amount": self.amount,
                "interest": self.interest
            }
        ).round(1)
        for idx, data in self.forecast_df.iterrows():
            i = int(idx) - 1 if idx != 0 else 0
            try:
                period_earnings_ = data.get("amount") - self.forecast_df.iloc[i].get("amount")
            except IndexError:
                break

            self.period_earnings.append(period_earnings_)

        self.forecast_df["period_earnings"] = self.period_earnings
        self.forecast_df.insert(0, "n_years", self.forecast_df.index)
        interest_rates = list(self.rate)
        if len(interest_rates) != len(self.forecast_df):
            interest_rates.insert(0, 0)
        self.forecast_df.insert(3, "interest_rate", pd.Series(interest_rates))

    def forecast_lumpsum(self) -> dict[str, Any]:
        if not self._forecasted:
            self._acc_amount = deepcopy(self.principal_amount)
            if self.investment_type == InvestmentType.LUMPSUM:

                self._amount.append(self.acc_amount)
                self._interest.append(0)
                for r in self.rate:
                    self._amount_invested += self.principal_amount
                    rate_per_period = r / self.rate.investment_freq(self.investment_freq)
                    total_amount = self.acc_amount * (1 + (rate_per_period / self.compounding_freq)) ** self.compounding_freq
                    interest = total_amount - self.principal_amount
                    self._amount.append(total_amount)
                    self._interest.append(interest)
                    self._acc_amount = total_amount

                self._total_interest = self._interest[-1]
                self._compute_earnings()
                self._forecasted = True

        return dict(self)

    def forecast_sip(self) -> dict[str, Any]:
        if not self._forecasted:
            n_months = (self.time + 1) * 12
            self._acc_amount = deepcopy(self.principal_amount)
            total_amount_invested = 0

            self._amount.append(self.principal_amount)
            self._interest.append(0)
            for t, r in zip(range(1, n_months + 1), self.rate):
                self._amount_invested += self.principal_amount
                rate_per_period = r / self.rate.investment_freq(self.investment_freq)
                total_amount_invested += self.principal_amount
                amount_ = self.acc_amount if t == 1 else self.acc_amount + self.principal_amount
                amount = amount_ * (((1 + rate_per_period) - 1) * (1 + rate_per_period)) / rate_per_period
                self._acc_amount = amount
                interest = self.acc_amount - total_amount_invested
                self._amount.append(amount)
                self._interest.append(interest)

            self._total_interest = self._interest[-1]
            self._compute_earnings()
            self._forecasted = True

        return dict(self)

    def forecast(self):
        self.rate.generate(self.time, self.investment_freq)
        if self.investment_type == InvestmentType.LUMPSUM:
            return self.forecast_lumpsum()
        elif self.investment_type == InvestmentType.SIP:
            return self.forecast_sip()

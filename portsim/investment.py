import pandas as pd
import seaborn as sns
from enum import Enum
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Optional, Any, Literal
from pydantic import BaseModel, ConfigDict

from portsim.rate import Rate

pd.set_option('display.float_format', lambda x: '%.2f' % x)


class InvestmentType(Enum):
    LUMPSUM = "lumpsum"
    SIP = "sip"


class Investment(BaseModel):
    name: str
    investment_type: InvestmentType
    principal_amount: float
    time: int
    rate: Rate
    investment_freq: Literal["monthly", "quarterly", "yearly"]
    compounding_freq: int = 1
    increase_every: Optional[int] = None
    increase_by: Optional[float] = None

    amount_yoy: list[float] = []
    interest_yoy: list[float] = []
    acc_amount: float = 0
    total_interest: float = 0
    annual_earnings: list[float] = []
    forecast_df: pd.DataFrame = pd.DataFrame()
    metadata: dict = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _forecasted: bool = False

    def plot(self, save_path=None):
        if not self._forecasted:
            self.forecast()

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(7)
        fig.set_figwidth(14)

        ax = axs[0, 0]
        ax = sns.scatterplot(ax=ax, x="n_years", y="amount_yoy", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="amount_yoy", marker='o', data=self.forecast_df)
        ylabels = [f"{round(x)}K" for x in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Number of Years")
        ax.set_ylabel(" Total Amount")

        ax = axs[0, 1]
        ax = sns.scatterplot(ax=ax, x="n_years", y="interest_yoy", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="interest_yoy", marker='o', data=self.forecast_df)
        ylabels = [f"{round(x)}K" for x in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Number of Years")
        ax.set_ylabel("Accumulated Interest")

        ax = axs[1, 0]
        ax = sns.scatterplot(ax=ax, x="n_years", y="annual_earnings", marker='o', data=self.forecast_df)
        ax = sns.lineplot(ax=ax, x="n_years", y="annual_earnings", marker='o', data=self.forecast_df)
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
        forecast_data = {k: dict(self).get(k) for k in ["amount_yoy", "interest_yoy"]}
        self.forecast_df = pd.DataFrame(forecast_data).round(1)
        for idx, data in self.forecast_df.iterrows():
            i = int(idx) - 1 if idx != 0 else 0
            try:
                annual_earnings_ = data.get("amount_yoy") - self.forecast_df.iloc[i].get("amount_yoy")
            except IndexError:
                break

            self.annual_earnings.append(annual_earnings_)

        self.forecast_df["annual_earnings"] = self.annual_earnings
        self.forecast_df.insert(0, "n_years", self.forecast_df.index)
        interest_rates = list(self.rate)
        if len(interest_rates) != len(self.forecast_df):
            interest_rates.insert(0, 0)
        self.forecast_df.insert(3, "interest_rate", pd.Series(interest_rates))

    def forecast_lumpsum(self) -> dict[str, Any]:
        if not self._forecasted:
            self.acc_amount = deepcopy(self.principal_amount)
            if self.investment_type == InvestmentType.LUMPSUM:

                self.amount_yoy.append(self.acc_amount)
                self.interest_yoy.append(0)
                for r in self.rate:
                    total_amount = self.acc_amount * (1 + (r / self.compounding_freq)) ** self.compounding_freq
                    interest = total_amount - self.principal_amount
                    self.amount_yoy.append(total_amount)
                    self.interest_yoy.append(interest)
                    self.acc_amount = total_amount

                self.total_interest = self.interest_yoy[-1]
                self._compute_earnings()
                self._forecasted = True

        return dict(self)  # TODO: return self only ?

    def forecast_sip(self) -> dict[str, Any]:
        if not self._forecasted:
            n_months = self.time * 12
            acc_amount = deepcopy(self.principal_amount)
            total_amount_invested = 0

            for t, r in zip(range(1, n_months + 1), self.rate):
                rate_per_period = r / self.rate.investment_freq(self.investment_freq)
                total_amount_invested += self.principal_amount
                amount_ = acc_amount if t == 1 else acc_amount + self.principal_amount
                amount = amount_ * (((1 + rate_per_period) - 1) * (1 + rate_per_period)) / rate_per_period
                acc_amount = amount
                interest = acc_amount - total_amount_invested
                self.amount_yoy.append(amount)
                self.interest_yoy.append(interest)

            self.total_interest = self.interest_yoy[-1]
            self._compute_earnings()
            self._forecasted = True

        return dict(self)

    def forecast(self):
        self.rate.generate(self.time)
        if self.investment_type == InvestmentType.LUMPSUM:
            return self.forecast_lumpsum()
        elif self.investment_type == InvestmentType.SIP:
            return self.forecast_sip()

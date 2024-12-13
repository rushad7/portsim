import numpy as np
from scipy.stats import truncnorm
from typing import Optional, Callable
from pydantic import BaseModel, Field, PrivateAttr


from portsim.enums.rate import RateType
from portsim.enums.investment import InvestmentFrequency


class RateParameters(BaseModel):
    # TODO: Write val method for overlapping types
    rate_type: RateType = Field(
        description="Defines the function the rate will follow for the given time period"
    )
    constant: float = Field(
        default=None, description="Constant rate of the provided value"
    )
    mean: float = Field(
        default=None,
        description="""
        Mean of the Gaussian distribution describing the rate.
        Only applicable when rate_type is either one of 
        RateType.GAUSSIAN or RateType.GAUSSIAN_CENTERED
        """
    )
    std_dev: float = Field(
        default=None, description="""
        Standard Deviation of the gaussian distribution
        describing the rate. Only applicable when rate_type
        is either one of RateType.GAUSSIAN or RateType.GAUSSIAN_CENTERED
        """
    )
    minimum: float = Field(
        default=None, le=0, ge=-1,
        description="Minimum modeled loss"
    )
    maximum: float = Field(
        default=None, le=1, ge=0,
        description="Maximum modeled profit"
    )
    pos_prob: float = Field(
        default=None,
        description="""
        Probability of gaussian noise positively biasing the constant value.
        Only applicable when rate_type == RateType.GAUSSIAN_CENTERED
        """
    )
    _every: InvestmentFrequency = PrivateAttr(default=None)

    @property
    def every(self):
        return self._every

    @every.setter
    def every(self, value):
        self._every = value


class Rate:

    def __init__(self, rate_config: RateParameters) -> None:
        self.rate_params = rate_config
        self.rate_yoy: Optional[list[float]] = None
        self.time_period: Optional[int] = None
        self._index = 0

    def generate(self, time_period: int, investment_frq: InvestmentFrequency):
        self.time_period = time_period
        self.rate_params.every = investment_frq
        RateReturnType = Callable[[RateParameters], list[float] | None]
        rate_map: dict[str, RateReturnType] = {
            "constant": self.constant,
            "random": self.random,
            "gaussian": self.gaussian,
            "gaussian_centered": self.gaussian_centered
        }
        rate_map.get(self.rate_params.rate_type.value)(self.rate_params)

    @staticmethod
    def investment_freq(investment_freq: InvestmentFrequency):
        inv_freq_map = {
            "monthly": 12,
            "quarterly": 4,
            "yearly": 1
        }
        return inv_freq_map.get(investment_freq.value)

    def constant(self, params: RateParameters):
        inv_freq = self.investment_freq(params.every)
        adjusted_time_period = self.time_period * inv_freq
        self.rate_yoy = [params.constant] * adjusted_time_period
        return self.rate_yoy

    def random(self, params: RateParameters):
        self.rate_yoy = np.random.rand(self.time_period).tolist()
        return self.rate_yoy

    def gaussian(self, params: RateParameters):
        mean, std_dev = params.mean, params.std_dev
        min_val, max_val = params.minimum, params.maximum

        a = (min_val - mean) / std_dev
        b = (max_val - mean) / std_dev
        self.rate_yoy = truncnorm(a, b, loc=mean, scale=std_dev).rvs(self.time_period).tolist()
        return self.rate_yoy

    def gaussian_centered(self, params: RateParameters):
        rate_const = self.constant(params)
        rate_gaussian = self.gaussian(params)
        rate = []
        for rc, rg in zip(rate_const, rate_gaussian):
            operation = np.random.choice(["-", "+"], p=[1 - params.pos_prob, params.pos_prob])
            rate_ = rc + rg if operation == "+" else rc - rg
            rate.append(rate_)

        self.rate_yoy = rate

    def __len__(self) -> int:
        return len(self.rate_yoy)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.rate_yoy):
            raise StopIteration
        result = self.rate_yoy[self._index]
        self._index += 1
        return result

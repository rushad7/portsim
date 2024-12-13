from enum import Enum


class RateType(Enum):
    CONSTANT = "constant"
    RANDOM = "random"
    GAUSSIAN = "gaussian"
    GAUSSIAN_CENTERED = "gaussian_centered"
    CUSTOM = "custom"

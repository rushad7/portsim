from enum import Enum


class InvestmentType(Enum):
    LUMPSUM = "lumpsum"
    SIP = "sip"


class InvestmentFrequency(Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

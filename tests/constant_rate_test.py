import unittest
from portsim.enums import InvestmentFrequency, InvestmentType, RateType
from portsim.models import Investment, Rate, RateParameters


class ConstantRateTests(unittest.TestCase):
    def test_sip(self):
        print(f"*"*30, "TEST FOR SIP INVESTMENT", f"*"*30, sep="\n")
        rate_config = RateParameters(rate_type=RateType.CONSTANT, constant=0.10)
        rate = Rate(rate_config)
        inv = Investment(
            name="Project_X",
            investment_type=InvestmentType.SIP,
            principal_amount=10_000,
            time=3, rate=rate,
            investment_freq=InvestmentFrequency.MONTHLY,
        )

        inv.forecast()
        print("Accumulated Amount: ", inv.acc_amount.__round__(2))
        print("Amount Invested: ", inv.amount_invested.__round__(2))
        print("Total Interest Earned: ", inv.total_interest.__round__(2))
        print(f"Time Period: {inv.time} Years")
        print("*"*30, sep="\n")

    def test_lumpsum(self):
        print(f"*" * 30, "TEST FOR LUMPSUM INVESTMENT", f"*" * 30, sep="\n")
        rate_config = RateParameters(rate_type=RateType.CONSTANT, constant=0.10)
        rate = Rate(rate_config)
        inv = Investment(
            name="Project_X",
            investment_type=InvestmentType.LUMPSUM,
            principal_amount=10_000,
            time=3, rate=rate,
            investment_freq=InvestmentFrequency.YEARLY
        )

        inv.forecast()
        print("Accumulated Amount: ", inv.acc_amount.__round__(2))
        print("Amount Invested: ", inv.amount_invested.__round__(2))
        print("Total Interest Earned: ", inv.total_interest.__round__(2))
        print(f"Time Period: {inv.time} Years")
        print("*"*30, sep="\n")


if __name__ == '__main__':
    unittest.main()

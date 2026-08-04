"""
Building a European option price surface.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import EuropeanOption
from KQuant.pricing import AnalyticEngine
from KQuant.calibration import PriceSurface


# Market data
market = MarketData(spot=100)

r = 0.02
sigma = 0.20

# Model
model = BlackAndScholes(r=r, sigma=sigma)

# Pricing engine
engine = AnalyticEngine()

# Create option grid
options = [
    EuropeanOption(K=90, T=0.5, option_type="call"),
    EuropeanOption(K=100, T=0.5, option_type="call"),
    EuropeanOption(K=110, T=0.5, option_type="call"),

    EuropeanOption(K=90, T=1.0, option_type="call"),
    EuropeanOption(K=100, T=1.0, option_type="call"),
    EuropeanOption(K=110, T=1.0, option_type="call"),

    EuropeanOption(K=90, T=2.0, option_type="call"),
    EuropeanOption(K=100, T=2.0, option_type="call"),
    EuropeanOption(K=110, T=2.0, option_type="call"),
]

# Compute prices
prices = []

for option in options:

    price = engine.price(market, model, option)
    prices.append(price)


# Build price surface
surface = PriceSurface(options,prices)

print("European Call Price Surface")
print("---------------------------")

for option, price in zip(surface.options, surface.prices):
    print(f"K={option.K}, "f"T={option.T} year : "f"{price:.4f}")
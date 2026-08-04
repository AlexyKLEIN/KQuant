"""
Building a volatility surface from European option prices.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import EuropeanOption
from KQuant.pricing import AnalyticEngine
from KQuant.calibration import ImpliedVolatility, VolatilitySurface


# Market data
market = MarketData(spot=100)

r = 0.02

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

# Market implied volatilities
market_vols = [
    # T = 0.5
    0.22, 0.20, 0.22,
    # T = 1.0
    0.21, 0.19, 0.21,
    # T = 2.0
    0.20, 0.18, 0.20
]

# Generate market prices
prices = []

for option, sigma in zip(options, market_vols):

    model = BlackAndScholes(r=r, sigma=sigma)
    price = engine.price(market, model, option)
    prices.append(price)


# Recover implied volatilities
iv_solver = ImpliedVolatility()

implied_vols = []

for price, option in zip(prices, options):

    iv = iv_solver.implied_volatility(price, market, option, r)
    implied_vols.append(iv)


# Build volatility surface
surface = VolatilitySurface(
    options,
    implied_vols
)


print("Volatility Surface")
print("------------------")

for option, vol in zip(surface.options, surface.vols):
    print(f"K={option.K}, "f"T={option.T} year : "f"{vol:.2%}")
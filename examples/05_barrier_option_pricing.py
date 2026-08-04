"""
Example: Pricing a down-and-in put option using Monte Carlo simulation with the Black-Scholes model.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import BarrierOption
from KQuant.pricing import MonteCarloEngine

# Market data
market = MarketData(spot=100)

# Model
model = BlackAndScholes(r=0.02, sigma=0.20)

# Product
option = BarrierOption(K=100, T=1.0, option_type="put", barrier_type="down-and-in", barrier=70)

# Pricing
engine = MonteCarloEngine(n_paths = 10000, n_steps = 252)
price = engine.price(market, model, option)

print("Down-and-in Put Option")
print("----------------------")
print(f"Spot       : {market.spot}")
print(f"Strike     : {option.K}")
print(f"Maturity   : {option.T} year")
print(f"Volatility : {model.sigma:.0%}")
print(f"Rate       : {model.r:.0%}")
print(f"Barrier    : {option.barrier}")
print()
print("Monte Carlo parameters")
print(f"Paths      : {engine.n_paths}")
print(f"Steps      : {engine.n_steps}")
print()
print(f"Price      : {price:.4f}")


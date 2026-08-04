"""
Price a European call option using the Black-Scholes model.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import EuropeanOption
from KQuant.pricing import AnalyticEngine

# Market data
market = MarketData(spot=100)

# Model
model = BlackAndScholes(r=0.02, sigma=0.20)

# Product
option = EuropeanOption(K=100, T=1.0, option_type="call")

# Pricing
engine = AnalyticEngine()
price = engine.price(market, model, option)

print("European Call Option")
print("----------------------")
print(f"Spot       : {market.spot}")
print(f"Strike     : {option.K}")
print(f"Maturity   : {option.T} year")
print(f"Volatility : {model.sigma:.0%}")
print(f"Rate       : {model.r:.0%}")
print()
print(f"Price      : {price:.4f}")
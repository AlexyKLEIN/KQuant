"""
Computing the implied volatility of a European call option.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import EuropeanOption
from KQuant.pricing import AnalyticEngine
from KQuant.calibration import ImpliedVolatility


# Market data
market = MarketData(spot=100)

# Option
option = EuropeanOption(K=100, T=1.0, option_type="call")

# Market assumptions
r = 0.02
true_vol = 0.20

# Generate option price
model = BlackAndScholes(r=r, sigma=true_vol)

engine = AnalyticEngine()

market_price = engine.price(market,model,option)

# Implied volatility calculation
iv_solver = ImpliedVolatility()
implied_vol = iv_solver.implied_volatility(market_price, market, option,r)


print("European Call Implied Volatility")
print("--------------------------------")
print(f"Spot          : {market.spot}")
print(f"Strike        : {option.K}")
print(f"Maturity      : {option.T} year")
print(f"Rate          : {r:.2%}")
print(f"Market price  : {market_price:.4f}")
print()
print(f"True volatility    : {true_vol:.2%}")
print(f"Implied volatility : {implied_vol:.2%}")
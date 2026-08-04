"""
Example: Compute Greeks of a European call option using the Black-Scholes model.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import EuropeanOption
from KQuant.greek import AnalyticGreeks

# Market data
market = MarketData(spot=100)

# Model
model = BlackAndScholes(r=0.02, sigma=0.20)

# Product
option = EuropeanOption(K=100, T=1.0, option_type="call")

# Greek
greek = AnalyticGreeks()

# Delta
delta = greek.delta(market, model, option)

# Gamma
gamma = greek.gamma(market, model, option)

# Vega
vega = greek.vega(market, model, option)

# Theta
theta = greek.theta(market, model, option)

# Rho
rho = greek.rho(market, model, option)

print("European Call Option")
print("----------------------")
print(f"Spot       : {market.spot}")
print(f"Strike     : {option.K}")
print(f"Maturity   : {option.T} year")
print(f"Volatility : {model.sigma:.0%}")
print(f"Rate       : {model.r:.0%}")
print()
print(f"Delta      : {delta:.4f}")
print(f"Gamma      : {gamma:.4f}")
print(f"Vega       : {vega:.4f}")
print(f"Theta      : {theta:.4f}")
print(f"Rho        : {rho:.4f}")
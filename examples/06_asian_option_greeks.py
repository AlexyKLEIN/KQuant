"""
Example: Compute Greeks of an Asian call option using the Black-Scholes model.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import AsianOption
from KQuant.greek import FiniteDifferenceGreeks
from KQuant.pricing import MonteCarloEngine

# Market data
market = MarketData(spot=100)

# Model
model = BlackAndScholes(r=0.02, sigma=0.20)

# Product
option = AsianOption(K=100, T=1.0, option_type="call")

# Engine
engine = MonteCarloEngine(n_paths=10000, n_steps=252)

# Greek
greek = FiniteDifferenceGreeks()

# Delta
delta = greek.delta(market, model, engine, option)

# Gamma
gamma = greek.gamma(market, model, engine, option)

# Vega
vega = greek.vega(market, model, engine, option)

# Theta
theta = greek.theta(market, model, engine, option)

# Rho
rho = greek.rho(market, model, engine, option)

print("Asian Call Option")
print("----------------------")
print(f"Spot       : {market.spot}")
print(f"Strike     : {option.K}")
print(f"Maturity   : {option.T} year")
print(f"Volatility : {model.sigma:.0%}")
print(f"Rate       : {model.r:.0%}")
print()
print("Monte Carlo parameters")
print(f"Paths      : {engine.n_paths}")
print(f"Steps      : {engine.n_steps}")
print()
print(f"Delta      : {delta:.4f}")
print(f"Gamma      : {gamma:.4f}")
print(f"Vega       : {vega:.4f}")
print(f"Theta      : {theta:.4f}")
print(f"Rho        : {rho:.4f}")
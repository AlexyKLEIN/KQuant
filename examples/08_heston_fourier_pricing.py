"""
Price a European call option using the Heston model with Fourier inversion.
"""

from KQuant.market import MarketData
from KQuant.models import Heston
from KQuant.products import EuropeanOption
from KQuant.pricing import HestonFourierEngine

# Market data
market = MarketData(spot=100)

# Model
model = Heston(r=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

# Product
option = EuropeanOption(K=100, T=1.0, option_type="call")

# Pricing
engine = HestonFourierEngine()
price = engine.price(market, model, option)

print("Heston Fourier European Call")
print("----------------------------")
print(f"Spot       : {market.spot}")
print(f"Strike     : {option.K}")
print(f"Maturity   : {option.T} year")
print()
print("Heston parameters")
print(f"v0         : {model.v0:.4f}")
print(f"kappa      : {model.kappa:.4f}")
print(f"theta      : {model.theta:.4f}")
print(f"xi         : {model.xi:.4f}")
print(f"rho        : {model.rho:.4f}")
print(f"Rate       : {model.r:.2%}")
print()
print(f"Price      : {price:.4f}")
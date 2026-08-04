"""
Heston model calibration using a synthetic option price surface.

Option prices are generated using a Black-Scholes model and are then used
to calibrate the Heston stochastic volatility model.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes
from KQuant.products import EuropeanOption
from KQuant.pricing import AnalyticEngine, HestonFourierEngine
from KQuant.calibration import PriceSurface, HestonCalibrator


# Market data
market = MarketData(spot=100)

# Generate synthetic option prices using Black-Scholes
bs_model = BlackAndScholes(r=0.02, sigma=0.20)
bs_engine = AnalyticEngine()

options = [
    EuropeanOption(K=80, T=0.5, option_type="call"),
    EuropeanOption(K=90, T=0.5, option_type="call"),
    EuropeanOption(K=100, T=0.5, option_type="call"),
    EuropeanOption(K=110, T=0.5, option_type="call"),
    EuropeanOption(K=120, T=0.5, option_type="call"),
]

prices = []

for option in options:
    price = bs_engine.price(market, bs_model, option)
    prices.append(price)

surface = PriceSurface(options, prices)


# Heston Fourier pricing engine
engine = HestonFourierEngine()


# Heston calibrator
calibrator = HestonCalibrator(engine=engine, method="price", max_iter=50)


initial_guess = [
    0.04,   # v0
    2.0,    # kappa
    0.04,   # theta
    0.30,   # xi
    -0.70,  # rho
]


bounds = [
    (0.001, 0.5),   # v0
    (0.1, 10.0),    # kappa
    (0.001, 0.5),   # theta
    (0.01, 2.0),    # xi
    (-0.99, 0.99),  # rho
]


# Calibrate Heston parameters
params = calibrator.calibrate_scipy(market=market, surface=surface, r=0.02, initial_guess=initial_guess, bounds=bounds)


print("Calibrated Heston parameters:")
print(f"v0    = {params[0]:.4f}")
print(f"kappa = {params[1]:.4f}")
print(f"theta = {params[2]:.4f}")
print(f"xi    = {params[3]:.4f}")
print(f"rho   = {params[4]:.4f}")
"""
Example of Heston model calibration using a synthetic implied volatility
surface generated from Black-Scholes prices.
"""

from KQuant.market import MarketData
from KQuant.models import BlackAndScholes, Heston
from KQuant.products import EuropeanOption
from KQuant.pricing import AnalyticEngine, HestonFourierEngine
from KQuant.calibration import ImpliedVolatility, VolatilitySurface, HestonCalibrator

# Market data
market = MarketData(spot=100)

# Generate synthetic market data using Black-Scholes
bs_model = BlackAndScholes(r=0.02, sigma=0.20)
bs_engine = AnalyticEngine()

options = [
    EuropeanOption(K=80, T=0.25, option_type="call"),
    EuropeanOption(K=90, T=0.25, option_type="call"),
    EuropeanOption(K=100, T=0.25, option_type="call"),

    EuropeanOption(K=80, T=0.5, option_type="call"),
    EuropeanOption(K=100, T=0.5, option_type="call"),
    EuropeanOption(K=120, T=0.5, option_type="call"),

    EuropeanOption(K=90, T=1.0, option_type="call"),
    EuropeanOption(K=100, T=1.0, option_type="call"),
    EuropeanOption(K=110, T=1.0, option_type="call"),

    EuropeanOption(K=90, T=2.0, option_type="call"),
    EuropeanOption(K=100, T=2.0, option_type="call"),
    EuropeanOption(K=110, T=2.0, option_type="call"),
]

vols = []
for option in options:
    price = bs_engine.price(market, bs_model, option)
    iv = ImpliedVolatility.implied_volatility(price, market, option, r=bs_model.r)
    vols.append(iv)

surface = VolatilitySurface(options, vols)

# Heston pricing engine
engine = HestonFourierEngine()

# Calibrator
calibrator = HestonCalibrator(engine=engine, method="implied_volatility", max_iter=50)

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
    (0.001, 1.0),    # xi
    (-0.99, 0.99),  # rho
]

# Calibrate the Heston model using SciPy
params, error= calibrator.calibrate_scipy(market=market, surface=surface, r=0.02, initial_guess=initial_guess, bounds=bounds)

print("Calibration error:", error)

print("Calibrated Heston parameters:")
print(f"v0    = {params[0]:.4f}")
print(f"kappa = {params[1]:.4f}")
print(f"theta = {params[2]:.4f}")
print(f"xi    = {params[3]:.4f}")
print(f"rho   = {params[4]:.4f}")
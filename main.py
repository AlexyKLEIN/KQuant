from KQuant.models.heston import Heston
from KQuant.models.black_and_scholes import BlackAndScholes
from KQuant.products.asian_option import AsianOption
from KQuant.products.european_option import EuropeanOption
from KQuant.market.market_data import MarketData
from KQuant.pricing.monte_carlo import MonteCarloEngine
from KQuant.pricing.analytic import AnalyticEngine
from KQuant.greek.finite_difference import FiniteDifferenceGreeks
from KQuant.pricing.heston_fourier import HestonFourierEngine
from KQuant.calibration.implied_volatility import ImpliedVolatility
from KQuant.calibration.heston_calibrator import HestonCalibrator
from KQuant.calibration.volatility_surface import VolatilitySurface
from KQuant.calibration.price_surface import PriceSurface
from KQuant.products.barrier_option import BarrierOption



import numpy as np
# Market data
market = MarketData(
    spot=100
)

# Model
model = BlackAndScholes(
    r=0.02,
    sigma=0.2
)

# Barrier option
option = BarrierOption(
    K=100,
    T=1.0,
    option_type="call",
    barrier_type="down-and-out",
    barrier=90
)
# Barrier option
option_2 = BarrierOption(
    K=100,
    T=1.0,
    option_type="call",
    barrier_type="down-and-in",
    barrier=90
)

option_3 = EuropeanOption(
    K=100,
    T=1,
    option_type="call"
)

# Monte Carlo engine
engine = MonteCarloEngine(
    n_steps=252,
    n_paths=100000)

engine_2 = AnalyticEngine()

# Pricing
price = engine.price(
    market,
    model,
    option
)
price_2 = engine.price(
    market,
    model,
    option_2
)
price_3 = engine_2.price(
    market,
    model,
    option_3
)



print(f"Barrier option 1 price: {price:.6f}")
print(f"Barrier option 2 price: {price_2:.6f}")
print(f"Call option price: {price_3:.6f}")
print(f"DandI + DandO : {price+price_2}")





# Pour la doc
"""
"Since no QuantLib reference implementation exists for arithmetic Asian options under Heston dynamics, the Heston Asian Monte Carlo engine is validated indirectly. The Heston simulation is validated against QuantLib on European options, while the Asian payoff implementation is validated under Black-Scholes dynamics."
"""

### Test ###
# Mieux organiser conftest (si temps)
# Dire que les tolérance sont assez souple car pas beaucoup iteration dan sles tests pour rapidité (mettre plsu de rel) le gamma de la barriere a bcp de mal expliquer pk doc. En changeant barrier on peut avoir un truc plus stable pour le test
# rejouer tt les test pour voir tolérance homogénéiser ou en parler dans la doc 

### Bonus ###
# (ajout _reper_)
# Nouveaux produits barrier digital american option


### Semaine pro ###  
# finir exemple 13 remplir Git ignore et Read Me et ewemple plus poussé 'convergence monte carlo, comparaison vitesse, smile ....

#Dire dans la doc pas d example sur calibrate sans spicy pour des raison de vitesse
"""
Gamma estimation with Monte Carlo finite differences is less stable than other Greeks.
Gamma is computed using a second-order finite difference approximation:
    Γ = (V(S+h) - 2V(S) + V(S-h)) / h²
Unlike Delta, which relies on a first-order difference, Gamma amplifies Monte Carlo noise because it involves the difference of three simulated prices and a division by a small quantity h².
This issue is particularly significant for barrier options. The payoff is discontinuous because a small change in the underlying asset price can modify the barrier activation status of a path (knocked-in or knocked-out). Consequently, small spot perturbations can lead to large variations in the estimated Gamma.
Therefore, Gamma computed with Monte Carlo finite differences is expected to have a larger numerical error compared to other sensitivities such as Delta, Vega, or Rho. A larger tolerance is used when validating Gamma.
"""
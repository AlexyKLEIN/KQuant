"""
Validation tests for option Greeks.

The analytical/monte carlo/heston fourier engine and analytical/finite-difference Greek implementations are benchmarked
against QuantLib, which is used as an external reference library.

Tests cover the following Greeks:
- Delta
- Gamma
- Vega
- Theta
- Rho

for European and Asian call and put options under the Black-Scholes model and the Heston model.
"""

import pytest
from KQuant.greek import AnalyticGreeks, FiniteDifferenceGreeks
from KQuant.products import EuropeanOption, AsianOption

GREEKS = ["delta", "gamma", "vega", "theta", "rho"]
GREEKS_2 = ["delta", "gamma", "v0_sensitivity", "kappa_sensitivity", "xi_sensitivity", "theta", "rho"]

# Black and Scholes using analytic engine and analytics greeks VS Quantlib  (European Option : Call and Put)
@pytest.mark.parametrize("greek_name", GREEKS)
@pytest.mark.parametrize("option_type", ["put","call"])
def test_eu_option_analytic_greek_matches_quantlib(market, black_scholes_model, ql_greeks, option_type, greek_name):

    option = EuropeanOption(K=100, T=1.0, option_type=option_type)
    my_value = getattr(AnalyticGreeks, greek_name)(market, black_scholes_model, option)
    ql_value = ql_greeks(market, black_scholes_model, option, greek_name)
    tolerance = 1e-3 if greek_name == "gamma" else 1e-4

    assert my_value == pytest.approx(ql_value, rel=tolerance, abs=tolerance)



# Black and Scholes using analytic engine and finite differences greeks VS Quantlib  (European Option : Call and Put)
@pytest.mark.parametrize("greek_name", GREEKS)
@pytest.mark.parametrize("option_type", ["put", "call"])
def test_eu_option_analytical_engine_finite_difference_greek_matches_quantlib(market, black_scholes_model, analytic_engine, ql_greeks, option_type, greek_name):

    option = EuropeanOption(K=100, T=1.0, option_type=option_type)
    fd_value = getattr(FiniteDifferenceGreeks, greek_name)(market, black_scholes_model, analytic_engine, option)
    ql_value = ql_greeks(market, black_scholes_model, option, greek_name)
    tolerance = 1.e-2

    assert fd_value == pytest.approx(ql_value, rel=tolerance,abs=tolerance)



# Black and Scholes using mc engine and finite differences greeks VS Quantlib  (European Option : Call and Put)
@pytest.mark.parametrize("greek_name", GREEKS)
@pytest.mark.parametrize("option_type", ["put", "call"])
@pytest.mark.parametrize("model_fixture", ["black_scholes_model"])
def test_eu_option_bs_mc_engine_finite_difference_greek_matches_quantlib(request, market, model_fixture, mc_engine, ql_greeks, option_type, greek_name):

    model = request.getfixturevalue(model_fixture)
    option = EuropeanOption(K=100, T=1.0, option_type=option_type)
    fd_value = getattr(FiniteDifferenceGreeks, greek_name)(market, model, mc_engine, option)
    ql_value = ql_greeks(market, model, option, greek_name)
    tolerance = 1.e-2

    assert fd_value == pytest.approx(ql_value, rel=tolerance,abs=tolerance)


# Heston using mc engine and finite differences greeks VS Quantlib  (European Option : Call and Put)
@pytest.mark.parametrize("greek_name", GREEKS_2)
@pytest.mark.parametrize("option_type", ["put", "call"])
@pytest.mark.parametrize("model_fixture", ["heston_model"])
def test_eu_option_heston_mc_engine_finite_difference_greek_matches_quantlib(request, market, model_fixture, mc_engine, ql_greeks, option_type, greek_name):

    model = request.getfixturevalue(model_fixture)
    option = EuropeanOption(K=100, T=1.0, option_type=option_type)
    fd_value = getattr(FiniteDifferenceGreeks, greek_name)(market, model, mc_engine, option)
    ql_value = ql_greeks(market, model, option, greek_name)
    tolerance = 5.e-2

    assert fd_value == pytest.approx(ql_value, rel=tolerance,abs=tolerance)


# Heston using Heston Fourier engine and finite differences greeks VS Quantlib finite differences references (European Option : Call and Put)
@pytest.mark.parametrize("greek_name", GREEKS_2)
@pytest.mark.parametrize("option_type", ["put", "call"])
def test_eu_option_heston_engine_finite_difference_greek_matches_quantlib(market, heston_model, heston_engine, ql_greeks, option_type, greek_name):

    option = EuropeanOption(K=100, T=1.0, option_type=option_type)
    fd_value = getattr(FiniteDifferenceGreeks, greek_name)(market, heston_model, heston_engine, option)
    ql_value = ql_greeks(market, heston_model, option, greek_name)
    tolerance = 1.e-2

    assert fd_value == pytest.approx(ql_value, rel=tolerance,abs=tolerance)


# Black and Scholes using Monte Carlo engine and finite differences greeks VS Quantlib  (Asian Option : Call and Put)
@pytest.mark.parametrize("greek_name", GREEKS)
@pytest.mark.parametrize("option_type", ["put", "call"])
@pytest.mark.parametrize("model_fixture", ["black_scholes_model"])
def test_asian_option_mc_engine_finite_difference_greek_matches_quantlib(request, market, model_fixture, mc_engine, ql_greeks, option_type, greek_name):
    
    model = request.getfixturevalue(model_fixture)
    option = AsianOption(K=100, T=1.0, option_type=option_type)
    fd_value = getattr(FiniteDifferenceGreeks, greek_name)(market, model, mc_engine, option)
    ql_value = ql_greeks(market, model, option, greek_name)
    tolerance = 5.e-2

    assert fd_value == pytest.approx(ql_value, rel=tolerance,abs=tolerance)


# Barrier option Greeks consistency:
# Down-In + Down-Out = European
# Model : Black-Scholes
# Greeks : Finite Differences Monte Carlo
@pytest.mark.parametrize("greek_name", GREEKS)
@pytest.mark.parametrize("option_type",["call", "put"])
def test_down_in_plus_down_out_greek_matches_european(request, market_high, black_scholes_model, mc_engine, greek_name, option_type):

    european_option = request.getfixturevalue(f"{option_type}_option")
    down_in_option = request.getfixturevalue(f"down_in_{option_type}_option")
    down_out_option = request.getfixturevalue(f"down_out_{option_type}_option")

    european_greek = getattr(FiniteDifferenceGreeks, greek_name)(market_high,black_scholes_model,mc_engine,european_option)
    down_in_greek = getattr(FiniteDifferenceGreeks, greek_name)(market_high, black_scholes_model, mc_engine, down_in_option)
    down_out_greek = getattr(FiniteDifferenceGreeks, greek_name)(market_high, black_scholes_model, mc_engine, down_out_option)

    if greek_name == "gamma":
        tolerance = 0.25
    else:
        tolerance = 0.15

    assert down_in_greek + down_out_greek == pytest.approx(european_greek, abs=tolerance, rel = tolerance)


# Barrier option Greeks consistency:
# Up-and-In +Up-and-Out = European
# Model : Black-Scholes
# Greeks : Finite Differences Monte Carlo
@pytest.mark.parametrize("greek_name", GREEKS)
@pytest.mark.parametrize("option_type",["call", "put"])
def test_up_in_plus_up_out_greek_matches_european(request, market_low, black_scholes_model, mc_engine, greek_name, option_type):

    european_option = request.getfixturevalue(f"{option_type}_option")
    up_in_option = request.getfixturevalue(f"up_in_{option_type}_option")
    up_out_option = request.getfixturevalue(f"up_out_{option_type}_option")

    european_greek = getattr(FiniteDifferenceGreeks, greek_name)(market_low,black_scholes_model,mc_engine,european_option)
    up_in_greek = getattr(FiniteDifferenceGreeks, greek_name)(market_low, black_scholes_model, mc_engine, up_in_option)
    up_out_greek = getattr(FiniteDifferenceGreeks, greek_name)(market_low, black_scholes_model, mc_engine, up_out_option)

    if greek_name == "gamma":
        tolerance = 0.25
    else:
        tolerance = 0.15

    assert up_in_greek + up_out_greek == pytest.approx(european_greek, abs=tolerance, rel = tolerance)
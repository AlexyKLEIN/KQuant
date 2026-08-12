"""
Validation tests for option pricing engines.

The analytical and Monte Carlo pricing engines are benchmarked against
QuantLib, which is used as an external reference implementation.

Tests cover:
- European call and put options
- Asian call and put options
- Barrier call and put option
- Analytical Black-Scholes pricing
- Monte Carlo simulation-based pricing
- Semi-analytical Heston Fourier pricing
- Heston and Black-Scholes model
"""

import pytest

# European Option : Call and Put
# Model : Black and Scholes 
# Engine : Analytic
# VS Quantlib
def test_analytic_call_price_matches_quantlib(market, black_scholes_model, call_option, analytic_engine, ql_pricer) :

    my_price = analytic_engine.price(market, black_scholes_model, call_option)
    ql_price = ql_pricer(market, black_scholes_model, call_option)

    assert my_price == pytest.approx(ql_price, abs=1e-10)

def test_analytic_put_price_matches_quantlib(market, black_scholes_model, put_option, analytic_engine, ql_pricer) :

    my_price = analytic_engine.price(market, black_scholes_model, put_option)
    ql_price = ql_pricer(market, black_scholes_model, put_option)

    assert my_price == pytest.approx(ql_price, abs=1e-10)


# European Option : Call and Put
# Models : Black and Scholes and Heston
# Engine : Monte Carlo
# VS Quantlib
@pytest.mark.parametrize("model_fixture",["black_scholes_model","heston_model"])
def test_mc_put_price_matches_quantlib(request, market, model_fixture, put_option, mc_engine, ql_pricer) :

    model = request.getfixturevalue(model_fixture)
    my_price, stderr = mc_engine.price_with_error(market, model, put_option)
    ql_price = ql_pricer(market, model, put_option)

    assert my_price == pytest.approx(ql_price, abs = 3 * stderr)

@pytest.mark.parametrize("model_fixture",["black_scholes_model","heston_model"])
def test_mc_call_price_matches_quantlib(request, market, model_fixture, call_option, mc_engine, ql_pricer) :

    model = request.getfixturevalue(model_fixture)
    my_price, stderr = mc_engine.price_with_error(market, model, call_option)
    ql_price = ql_pricer(market, model, call_option)

    assert my_price == pytest.approx(ql_price, abs=3*stderr)



# European Option : Call and Put
# Model : Heston  
# Engine : Heston Fourier
# VS Quantlib
def test_heston_fourier_put_price_matches_quantlib(market, heston_model, put_option, heston_engine, ql_pricer) :

    my_price = heston_engine.price(market, heston_model, put_option)
    ql_price = ql_pricer(market, heston_model, put_option)

    assert my_price == pytest.approx(ql_price, abs = 1e-5, rel = 1e-5)

def test_heston_fourier_call_price_matches_quantlib(market, heston_model, call_option, heston_engine, ql_pricer) :

    my_price = heston_engine.price(market, heston_model, call_option)
    ql_price = ql_pricer(market, heston_model, call_option)

    assert my_price == pytest.approx(ql_price, abs = 1e-5, rel = 1e-5)



# Asian Option : Call and Put
# Model : Black and Scholes 
# Engine : Monte Carlo  
# VS Quantlib
@pytest.mark.parametrize("model_fixture",["black_scholes_model"])
def test_mc_asian_put_price_matches_quantlib(request, market, model_fixture, asian_put_option, mc_engine, ql_pricer) :

    model = request.getfixturevalue(model_fixture)
    my_price, stderr = mc_engine.price_with_error(market, model, asian_put_option)
    ql_price = ql_pricer(market, model, asian_put_option)

    assert my_price == pytest.approx(ql_price, abs=3*stderr)

@pytest.mark.parametrize("model_fixture",["black_scholes_model"])
def test_mc_asian_call_price_matches_quantlib(request, market, model_fixture, asian_call_option, mc_engine, ql_pricer) :

    model = request.getfixturevalue(model_fixture)
    my_price, stderr = mc_engine.price_with_error(market, model, asian_call_option)
    ql_price = ql_pricer(market, model, asian_call_option)

    assert my_price == pytest.approx(ql_price, abs=3*stderr)


# down-and-in + down-and-out call option
# Model : Black and Scholes 
# Engine : Monte Carlo  
# VS European call
def test_call_down_in_plus_down_out_matches_european(market, black_scholes_model, mc_engine, call_option, down_in_call_option, down_out_call_option):

    european_price = mc_engine.price(market, black_scholes_model, call_option)
    down_in_price = mc_engine.price(market, black_scholes_model, down_in_call_option)
    down_out_price = mc_engine.price(market, black_scholes_model, down_out_call_option)

    assert down_in_price + down_out_price == pytest.approx(european_price, rel=0.01)


# up-and-in + up-and-out call option
# Model : Black and Scholes 
# Engine : Monte Carlo  
# VS European call
def test_call_up_in_plus_up_out_matches_european(market, black_scholes_model, mc_engine, call_option, up_in_call_option, up_out_call_option):

    european_price = mc_engine.price(market, black_scholes_model, call_option)
    up_in_price = mc_engine.price(market, black_scholes_model, up_in_call_option)
    up_out_price = mc_engine.price(market, black_scholes_model, up_out_call_option)

    assert up_in_price + up_out_price == pytest.approx(european_price, rel=0.01)


# down-and-in + down-and-out put option
# Model : Black and Scholes 
# Engine : Monte Carlo  
# VS European put
def test_put_down_in_plus_down_out_matches_european(market, black_scholes_model, mc_engine, put_option, down_in_put_option, down_out_put_option):

    european_price = mc_engine.price(market, black_scholes_model, put_option)
    down_in_price = mc_engine.price(market, black_scholes_model, down_in_put_option)
    down_out_price = mc_engine.price(market, black_scholes_model, down_out_put_option)

    assert down_in_price + down_out_price == pytest.approx(european_price, rel=0.01)


# up-and-in + up-and-out put option
# Model : Black and Scholes 
# Engine : Monte Carlo  
# VS European put
def test_put_up_in_plus_up_out_matches_european(market, black_scholes_model, mc_engine, put_option, up_in_put_option, up_out_put_option):

    european_price = mc_engine.price(market, black_scholes_model, put_option)
    up_in_price = mc_engine.price(market, black_scholes_model, up_in_put_option)
    up_out_price = mc_engine.price(market, black_scholes_model, up_out_put_option)

    assert up_in_price + up_out_price == pytest.approx(european_price, rel=0.01)
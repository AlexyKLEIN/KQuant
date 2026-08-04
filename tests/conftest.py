"""
Shared pytest fixtures for the pricing library test suite.

Provides common market data, models, products, pricing engines,
and QuantLib-based reference implementations used for validation tests.
"""

import pytest
import QuantLib as ql
from KQuant.market import MarketData
from KQuant.models import BlackAndScholes, Heston
from KQuant.products import EuropeanOption, AsianOption, BarrierOption
from KQuant.pricing import AnalyticEngine, MonteCarloEngine, HestonFourierEngine

@pytest.fixture
def market():
    """Market data with an arbitrary but fixed spot price."""
    return MarketData(spot=100)

@pytest.fixture
def market_low():
    """Market data with an arbitrary but fixed spot price."""
    return MarketData(spot=50)

@pytest.fixture
def market_high():
    """Market data with an arbitrary but fixed spot price."""
    return MarketData(spot=150)

@pytest.fixture
def black_scholes_model():
    """Black-Scholes model with fixed rate and volatility."""
    return BlackAndScholes(r=0.02, sigma=0.2)


@pytest.fixture
def call_option():
    """At-the-money European call, 1 year maturity."""
    return EuropeanOption(K=100, T=1.0, option_type="call")


@pytest.fixture
def put_option():
    """At-the-money European put, 1 year maturity."""
    return EuropeanOption(K=100, T=1.0, option_type="put")

@pytest.fixture
def asian_put_option():
    """At-the-money asian put, 1 year maturity."""
    return AsianOption(K=100, T=1.0, option_type="put")

@pytest.fixture
def asian_call_option():
    """At-the-money asian call, 1 year maturity."""
    return AsianOption(K=100, T=1.0, option_type="call")

@pytest.fixture
def down_in_call_option():
    """At-the-money down-and-in call, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="call", barrier_type="down-and-in", barrier=90)

@pytest.fixture
def down_out_call_option():
    """At-the-money down-and-out call, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="call", barrier_type="down-and-out", barrier=90)

@pytest.fixture
def up_out_call_option():
    """At-the-money ip-and-out call, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="call", barrier_type="up-and-out", barrier=90)

@pytest.fixture
def up_in_call_option():
    """At-the-money up-and-in call, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="call", barrier_type="up-and-in", barrier=90)

@pytest.fixture
def down_in_put_option():
    """At-the-money down-and-in put, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="put", barrier_type="down-and-in", barrier=90)

@pytest.fixture
def down_out_put_option():
    """At-the-money down-and-out put, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="put", barrier_type="down-and-out", barrier=90)

@pytest.fixture
def up_out_put_option():
    """At-the-money ip-and-out put, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="put", barrier_type="up-and-out", barrier=90)

@pytest.fixture
def up_in_put_option():
    """At-the-money up-and-in put, 1 year maturity."""
    return BarrierOption(K=100, T=1.0, option_type="put", barrier_type="up-and-in", barrier=90)

@pytest.fixture
def analytic_engine():
    """Analytical engine"""
    return AnalyticEngine()


@pytest.fixture
def mc_engine():
    """Monte Carlo engine with a moderate number of paths."""
    return MonteCarloEngine(n_steps=100, n_paths=100000)


@pytest.fixture
def heston_engine():
    """Heston Fourier engine with a moderate number of paths."""
    return HestonFourierEngine()


@pytest.fixture
def heston_model():
    """
    Heston model with standard parameters.
    """
    return Heston(r=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


@pytest.fixture
def ql_pricer():

    def price(market, model, product):

        if isinstance(product, EuropeanOption) :
            return price_european_option(market, model, product)

        if isinstance(product, AsianOption) :
            return price_asian_option(market, model, product)

        else:
            raise NotImplementedError(f"QuantLib pricer not implemented for {type(model)} + {type(product)}")
    return price


def create_ql_process(market, model):

    today = ql.Date.todaysDate()
    spot = ql.QuoteHandle(ql.SimpleQuote(market.spot))
    rate = ql.YieldTermStructureHandle(ql.FlatForward(today,model.r, ql.Actual365Fixed()))

    if isinstance(model, BlackAndScholes):
        volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), model.sigma, ql.Actual365Fixed()))

        return ql.BlackScholesProcess(spot, rate, volatility)

    elif isinstance(model, Heston):
        dividend = ql.YieldTermStructureHandle(ql.FlatForward(today,0.0,ql.Actual365Fixed()))
        return ql.HestonProcess(rate, dividend, spot, model.v0, model.kappa, model.theta, model.xi, model.rho)

    else:
        raise NotImplementedError(f"QuantLib process not implemented for {type(model)}")


def price_european_option(market, model, product):

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    process = create_ql_process(market, model)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if product.option_type == "call" else ql.Option.Put, product.K)
    exercise = ql.EuropeanExercise(today + int(product.T * 365))
    option = ql.VanillaOption(payoff, exercise)

    if isinstance(model, BlackAndScholes):
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    elif isinstance(model, Heston):
        heston_model = ql.HestonModel(process)
        option.setPricingEngine(ql.AnalyticHestonEngine(heston_model))

    return option.NPV()


def price_asian_option(market, model, asian_option):

    if isinstance(model, Heston):
        raise NotImplementedError("QuantLib does not support arithmetic Asian option with HestonProcess")

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    process = create_ql_process(market, model)
    n_fixings = 252
    fixing_dates = ql.DateVector()
    maturity = today + int(asian_option.T * 365)

    for i in range(1, n_fixings + 1):
        fixing_time = asian_option.T * i / n_fixings
        fixing_dates.push_back(today + int(fixing_time * 365))

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if asian_option.option_type == "call" else ql.Option.Put, asian_option.K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.DiscreteAveragingAsianOption(ql.Average.Arithmetic, 0.0, 0, fixing_dates, payoff, exercise)
    option.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(process, "pseudorandom", requiredSamples=100000, seed=42))

    return option.NPV()

@pytest.fixture
def ql_greeks():

    def greeks(market, model, product, greek):

        if isinstance(product, EuropeanOption):
            return european_option_greek(market, model, product, greek)
        
        if isinstance(product, AsianOption):
            return asian_option_greek(market, model, product, greek)
        
        raise NotImplementedError(
            f"No QuantLib greeks for {type(product)}"
        )

    return greeks


def european_option_greek(market, model, product, greek_name):

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    process = create_ql_process(market, model)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if product.option_type == "call" else ql.Option.Put,product.K)
    exercise = ql.EuropeanExercise(today + int(product.T * 365))
    option = ql.VanillaOption(payoff, exercise)

    if isinstance(model, BlackAndScholes):
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        if greek_name == "delta":
            return option.delta()

        elif greek_name == "gamma":
            return option.gamma()

        elif greek_name == "vega":
            return option.vega()

        elif greek_name == "theta":
            return option.theta()

        elif greek_name == "rho":
            return option.rho()

    if isinstance(model, Heston):
        h = 1e-3

        if greek_name == "delta":
            ds = market.spot*h
            market_up = MarketData(market.spot + ds)
            market_down = MarketData(market.spot - ds)

            return (price_european_option(market_up, model, product)-price_european_option(market_down, model, product))/(2*ds)

        elif greek_name == "gamma":
            ds = market.spot*h
            market_up = MarketData(market.spot + ds)
            market_down = MarketData(market.spot - ds)

            return (price_european_option(market_up, model, product)-2*price_european_option(market, model, product)+price_european_option(market_down, model, product))/ds**2

        elif greek_name == "vega":
            if isinstance(model, BlackAndScholes):
                model_up = BlackAndScholes(model.r,model.sigma+h)
                model_down = BlackAndScholes(model.r,model.sigma-h)

            elif isinstance(model, Heston):
                model_up = model.bump_v0(h)
                model_down = model.bump_v0(-h)

            else:
                raise NotImplementedError()

            return (price_european_option(market, model_up, product)-price_european_option(market, model_down, product))/(2*h)

        elif greek_name == "rho":
            model_up = model.bump_risk_free_rate(h)
            model_down = model.bump_risk_free_rate(-h)

            return (price_european_option(market, model_up, product)-price_european_option(market, model_down, product))/(2*h)

        elif greek_name == "theta":
            dt = 5/365
            product_short = EuropeanOption(product.K, product.T-dt, product.option_type)
            product_long = EuropeanOption(product.K,product.T+dt,product.option_type)

            return (price_european_option(market, model, product_short)-price_european_option(market, model, product_long))/(2*dt)

        elif greek_name == "v0_sensitivity":
            model_up = model.bump_v0(h)
            model_down = model.bump_v0(-h)
        
            return (price_european_option(market, model_up, product)-price_european_option(market, model_down, product))/(2*h)

        elif greek_name == "kappa_sensitivity":
                    model_up = model.bump_kappa(h)
                    model_down = model.bump_kappa(-h)
                
                    return (price_european_option(market, model_up, product)-price_european_option(market, model_down, product))/(2*h)

        elif greek_name == "xi_sensitivity":
                    model_up = model.bump_xi(h)
                    model_down = model.bump_xi(-h)
                
                    return (price_european_option(market, model_up, product)-price_european_option(market, model_down, product))/(2*h)
        
    else:
        raise NotImplementedError(f"Greek {greek_name} not available")


def asian_option_greek(market, model, product, greek_name):
    h = 1e-3

    if isinstance(model, Heston):
        raise NotImplementedError("QuantLib does not support arithmetic Asian option with HestonProcess")
    
    if greek_name == "delta":
        ds = market.spot*h
        market_up = MarketData(market.spot + ds)
        market_down = MarketData(market.spot - ds)

        return (price_asian_option(market_up, model, product)-price_asian_option(market_down, model, product))/(2*ds)

    elif greek_name == "gamma":
        ds = market.spot*h
        market_up = MarketData(market.spot + ds)
        market_down = MarketData(market.spot - ds)

        return (price_asian_option(market_up, model, product)-2*price_asian_option(market, model, product)+price_asian_option(market_down, model, product))/ds**2

    elif greek_name == "vega":
        if isinstance(model, BlackAndScholes):
            model_up = BlackAndScholes(model.r,model.sigma+h)
            model_down = BlackAndScholes(model.r,model.sigma-h)

        elif isinstance(model, Heston):
            model_up = model.bump_v0(h)
            model_down = model.bump_v0(-h)

        else:
            raise NotImplementedError()

        return (price_asian_option(market, model_up, product)-price_asian_option(market, model_down, product))/(2*h)

    elif greek_name == "rho":
        model_up = model.bump_risk_free_rate(h)
        model_down = model.bump_risk_free_rate(-h)

        return (price_asian_option(market, model_up, product)-price_asian_option(market, model_down, product))/(2*h)

    elif greek_name == "theta":
        dt = 30/365
        product_short = AsianOption(product.K, product.T-dt, product.option_type)
        product_long = AsianOption(product.K,product.T+dt,product.option_type)

        return (price_asian_option(market, model, product_short)-price_asian_option(market, model, product_long))/(2*dt)

    else:
        raise NotImplementedError()
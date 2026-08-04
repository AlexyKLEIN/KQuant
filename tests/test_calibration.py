import pytest
from KQuant.products.european_option import EuropeanOption
from KQuant.calibration import ImpliedVolatility, PriceSurface, HestonCalibrator, VolatilitySurface
from KQuant.models.heston import Heston


@pytest.mark.parametrize("K",[80, 90, 100, 110, 120])
def test_implied_volatility_different_strikes(K, market, analytic_engine, black_scholes_model):
    """
    Validate that the implied volatility solver recovers the original
    Black-Scholes volatility for different strikes.
    """

    call_option = EuropeanOption(K, 1, "call")
    C = analytic_engine.price(market, black_scholes_model, call_option)
    implied_vol = ImpliedVolatility.implied_volatility(C, market, call_option, black_scholes_model.r)

    assert(implied_vol == pytest.approx(0.2, abs = 1e-5))


BOUNDS = [(0.01, 0.1),(0.1, 5),(0.01, 0.1),(0.1, 1),(-0.99, 0),]
def test_heston_calibration_price(market, heston_model, heston_engine):
    """
    Test Heston calibration using an price surface.

    The calibrated parameters are not directly compared with the true ones
    because Heston parameters are not always uniquely identifiable. Moreover,
    a reduced number of options and optimization iterations are used to limit
    computational cost. The test therefore validates that the calibrated model
    reproduces the original price.
    """
    
    options = [EuropeanOption(80, 1, "call"), EuropeanOption(90, 1, "call"), EuropeanOption(100, 1, "call"), EuropeanOption(110, 1, "call"), EuropeanOption(120, 1, "call")]
    prices = [heston_engine.price(market, heston_model, option) for option in options]
    surface = PriceSurface(options, prices)
    calibrator = HestonCalibrator(heston_engine, method="price", max_iter=20)
    params = calibrator.calibrate_scipy(heston_model.r, market, surface, [0.05, 1.0, 0.05, 0.2, -0.5], BOUNDS)[0]
    calibrated_model = Heston(r=heston_model.r, v0=params[0] ,kappa=params[1], theta=params[2], xi=params[3], rho=params[4])

    for option, price in zip(options, prices):
        assert heston_engine.price(market, calibrated_model, option) == pytest.approx(price, abs=1e-2, rel = 1e-3)


def test_heston_calibration_implied_volatility(market, heston_model, heston_engine):
    """
    Test Heston calibration using an implied volatility surface.

    The calibrated parameters are not directly compared with the true ones
    because Heston parameters are not always uniquely identifiable. Moreover,
    a reduced number of options and optimization iterations are used to limit
    computational cost. The test therefore validates that the calibrated model
    reproduces the original implied volatilities.
    """

    options = [EuropeanOption(80, 1, "call"), EuropeanOption(90, 1, "call"), EuropeanOption(100, 1, "call"),EuropeanOption(110, 1, "call"),EuropeanOption(120, 1, "call"),]
    vols = []

    for option in options:

        price = heston_engine.price(market, heston_model, option)
        iv = ImpliedVolatility.implied_volatility(price, market, option, heston_model.r)
        vols.append(iv)

    surface = VolatilitySurface(options, vols)
    calibrator = HestonCalibrator(heston_engine, method="implied_volatility", max_iter=20)
    params = calibrator.calibrate_scipy(heston_model.r, market, surface, [0.05, 1.0, 0.05, 0.2, -0.5], BOUNDS)[0]
    calibrated_model = Heston(r=heston_model.r, v0=params[0], kappa=params[1], theta=params[2], xi=params[3], rho=params[4])

    for option, market_iv in zip(options, vols):

        price = heston_engine.price(market, calibrated_model, option)
        calibrated_iv = ImpliedVolatility.implied_volatility(price, market, option, calibrated_model.r)

        assert calibrated_iv == pytest.approx(market_iv, abs=1e-2, rel = 1e-3)
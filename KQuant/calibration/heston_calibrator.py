import numpy as np
from ..models.heston import Heston
from .implied_volatility import ImpliedVolatility
from scipy.optimize import minimize

class HestonCalibrator():
    """
    Calibrate Heston model parameters using price or implied volatility errors.
    """

    def __init__(self, engine, method, max_iter = 50, tol = 1e-10):
        """
        Initialize the Heston calibrator.

        Parameters
        ----------
        engine : PricingEngine
            Pricing engine used for Heston valuation.
        method : str
            Calibration objective. Can be "price" or "implied_volatility".
        max_iter : int
            Maximum number of optimization iterations.
        tol : float
            Optimization tolerance controlling the convergence criterion.
        """
        if method not in ["price", "implied_volatility"]:

            raise ValueError("Calibration method must be 'price' or 'implied_volatility'")
        
        self.engine = engine
        self.method = method
        self.max_iter = max_iter
        self.tol = tol


    def objective_using_vol(self, params, r, market, surface):
        """
        Compute the calibration error between Heston and market implied volatilities.

        Parameters
        ----------
        params : list
            Heston parameters (v0, kappa, theta, xi, rho).
        market : MarketData
            Market information.
        surface : VolatilitySurface
            Market implied volatility surface.

        Returns
        -------
        float
            Sum of squared implied volatility errors.
        """

        v0, kappa, theta, xi, rho = params
        model = Heston(r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        error = 0

        for option, market_iv in zip(surface.options, surface.vols):

            price = self.engine.price(market, model, option)
            iv = ImpliedVolatility.implied_volatility(price, market, option, r)
            error += (iv - market_iv)**2

        return error


    def objective_using_price(self, params, r, market, surface): 
            """
            Compute the calibration error between Heston and market prices.
    
            Parameters
            ----------
            params : list
                Heston parameters (v0, kappa, theta, xi, rho).
            market : MarketData
                Market information.
            surface : PriceSurface
                Market price surface.
    
            Returns
            -------
            float
                Sum of squared price errors.
            """
    
            v0, kappa, theta, xi, rho = params
            model = Heston(r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
            error = 0
    
            for option, market_price in zip(surface.options, surface.prices):
    
                price = self.engine.price(market, model, option)
                error += (price - market_price)**2
    
            return error


    def calibrate(self,r, market, surface, bounds):
        """
        Calibrate Heston parameters using random search.

        Parameters
        ----------
        market : MarketData
            Market information.
        surface : VolatilitySurface
            Market implied volatility surface.
        bounds : list of tuple
            Parameter bounds.
        
        Returns
        -------
        tuple
            Calibrated Heston parameters and calibration error.
        """

        best_error = np.inf
        best_params = None

        for _ in range(self.max_iter):

            params = [np.random.uniform(low, high) for low, high in bounds]
            if(self.method == "implied_volatility"):
                error = self.objective_using_vol(params, r, market, surface)
            else :
                error = self.objective_using_price(params, r, market, surface)


            if error < best_error:
                best_error = error
                best_params = params

        return best_params, best_error


    def calibrate_scipy(self, r, market, surface, initial_guess, bounds):
        """
        Calibrate Heston parameters using scipy optimization.

        Parameters
        ----------
        r : float
            Risk-free interest rate.
        market : MarketData
            Market information.
        surface : VolatilitySurface or PriceSurface
            Market option data used for calibration.
        initial_guess : list
            Initial guess for Heston parameters
            (v0, kappa, theta, xi, rho).
        bounds : list of tuple
            Bounds for each Heston parameter.

        Returns
        -------
        tuple
        A tuple containing:
        - array
            Calibrated Heston parameters.
        - float
            Final value of the objective function. (equivalent to the error)
        """

        if(self.method == "price"):
            result = minimize(fun=self.objective_using_price, x0=initial_guess, args=(r, market, surface),
                               bounds=bounds, method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol": self.tol})

        else:
            result = minimize(fun=self.objective_using_vol, x0=initial_guess, args=(r, market, surface),
                                           bounds=bounds, method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol":self.tol})
            
        return result.x, result.fun
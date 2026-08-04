from ..models.black_and_scholes import BlackAndScholes
from ..pricing.analytic import AnalyticEngine
from ..greek.analytic import AnalyticGreeks

class ImpliedVolatility():
     
    @staticmethod
    def implied_volatility(price, market, product, r, initial_sigma=0.2, tol=1e-8, max_iter=100):
        """
        Compute the implied volatility of a European option using
        the Newton-Raphson method.
    
        Parameters
        ----------
        price : float
            Market price of the option.
        market : MarketData
            Market information.
        product : EuropeanOption
            European option contract.
        r : float
            Risk-free interest rate.
        initial_sigma : float, optional
            Initial volatility guess.
        tol : float, optional
            Convergence tolerance.
        max_iter : int, optional
            Maximum number of iterations.
    
        Returns
        -------
        float
            Implied volatility.
        """
        
        C = price
        sigma_n = initial_sigma
        engine = AnalyticEngine()
        i = 0
        f = tol + 1

        while i < max_iter and abs(f) > tol:
            f = engine.price(market, BlackAndScholes(r, sigma_n), product) - C
            f_prim = AnalyticGreeks.vega(market, BlackAndScholes(r, sigma_n), product)

            if abs(f_prim) < 1e-10:
                raise ValueError("Vega too small")
            
            sigma_n = sigma_n - f/f_prim
            i += 1

        return sigma_n

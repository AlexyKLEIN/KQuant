import math
import numpy as np
from .pricing_engine import PricingEngine
from ..products import BarrierOption

class MonteCarloEngine(PricingEngine) :
    """
    Monte Carlo pricing engine for financial products.
    """

    def __init__(self, n_steps=252, n_paths=5000):
        """
        Initialize the Monte Carlo pricing engine.

        Parameters
        ----------
        n_steps : int, optional
            Number of time steps used in each simulated path.
            Default is 252.
        n_paths : int, optional
            Number of simulated paths.
            Default is 5000.
        """
        self.n_steps = n_steps
        self.n_paths = n_paths


    def price(self, market, model, product, Z_1 = None, Z_2 = None) :
        """
        Compute the price of a financial product using Monte Carlo simulation.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.
        model : Model
            Stochastic model used to simulate asset price paths.
        product : Product
            Financial product to be priced.
        Z_1 : array-like, optional
            Pre-generated random variables used for the simulation.
        Z_2 : array-like, optional
                    Pre-generated random variables used for the simulation.

        Returns
        -------
        float
            Monte Carlo estimate of the product price.
        """
        if isinstance(product, BarrierOption):

            if product.barrier_type in ["down-and-in", "down-and-out"]:
                if product.barrier >= market.spot:
                    raise ValueError("Down barrier must be below spot price")

            if product.barrier_type in ["up-and-in", "up-and-out"]:
                if product.barrier <= market.spot:
                    raise ValueError("Up barrier must be above spot price")
    
        paths = model.simulate_paths(market.spot, product.T, self.n_steps, self.n_paths, Z_1, Z_2)
        payoffs = [product.payoff(path) for path in paths.T]

        return math.exp(-model.r*product.T)*sum(payoffs)/self.n_paths


    def price_with_error(self, market, model, product, Z_1=None, Z_2=None):
        """
        Compute the Monte Carlo price of a financial product and estimate
        the statistical error of the simulation.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.
        model : Model
            Stochastic model used to simulate asset price paths.
        product : Product
            Financial product to be priced.
        Z_1 : array-like, optional
            Pre-generated random variables used for the first Brownian motion.
        Z_2 : array-like, optional
            Pre-generated random variables used for the second Brownian motion.

        Returns
        -------
        tuple
            Monte Carlo price estimate and its standard error.
        """
        if isinstance(product, BarrierOption):

            if product.barrier_type in ["down-and-in", "down-and-out"]:
                if product.barrier >= market.spot:
                    raise ValueError("Down barrier must be below spot price")

            if product.barrier_type in ["up-and-in", "up-and-out"]:
                if product.barrier <= market.spot:
                    raise ValueError("Up barrier must be above spot price")
                
        paths = model.simulate_paths(market.spot,product.T,self.n_steps,self.n_paths,Z_1,Z_2)

        payoffs = [
            product.payoff(path)
            for path in paths.T
        ]

        price = math.exp(-model.r*product.T) * np.mean(payoffs)

        stderr = np.std(payoffs, ddof=1) / np.sqrt(self.n_paths)

        return price, stderr
    



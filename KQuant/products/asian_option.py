from .product import Product
import numpy as np

class AsianOption(Product):
    """Asian option contract (with arithmetic mean)."""


    def __init__(self, K, T, option_type):
        """
        Create an Asian option.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        K : float
            Strik price
        option_type : str
            Option type: "call" or "put".
        """

        super().__init__()

        if K <= 0:                                    
            raise ValueError("Strike must be positive")

        if T <= 0:
            raise ValueError("Maturity must be positive")

        if option_type not in ["call", "put"]:
            raise ValueError("Invalid option type")
        
        self.K = K
        self.T = T
        self.option_type = option_type
    

    def payoff(self, path) :
        """
        Return option payoff at maturity.
        
        Parameter
        ---------
        path : array-like
            Simulated underlying asset path, from t=0 to maturity.

        Returns
        -------
        float
            Option payoff at maturity.
        """

        average = np.mean(path[1:]) 

        if self.option_type == "call" :
            return max(0, average - self.K)

        return max(0, self.K - average)
    

    def bump_maturity(self, bump):
        """
        Create a new Asian option with a bumped maturity.

        Parameters
        ----------
        bump : float
            Maturity shift applied to the current time to maturity.

        Returns
        -------
        EuropeanOption
            New Asian option with adjusted maturity.
        """
        return AsianOption(self.K,self.T + bump,self.option_type)
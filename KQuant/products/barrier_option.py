from .product import Product
import numpy as np

class BarrierOption(Product):
    """Barrier option contract."""


    def __init__(self, K, T, barrier, option_type, barrier_type):
        """
        Create Barrier option.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        K : float
            Strik price
        barrier : float
            Value of the barrier
        option_type : str
            Option type: "call" or "put".
        barrier_type : str
            Barrier type : "down-and-in", "down-and-out", "up-and-in" or "up-and-out" 
        """

        super().__init__()

        if K <= 0:                                    
            raise ValueError("Strike must be positive")

        if T <= 0:
            raise ValueError("Maturity must be positive")

        if barrier <= 0:
            raise ValueError("Barrier must be positive")

        if option_type not in ["call", "put"]:
            raise ValueError("Invalid option type")

        if barrier_type not in ["down-and-in", "down-and-out", "up-and-in", "up-and-out"]:
            raise ValueError("Invalid barrier option type")
        
        self.K = K
        self.T = T
        self.barrier = barrier
        self.option_type = option_type
        self.barrier_type = barrier_type
    

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

        alive = True

        if(self.barrier_type == "down-and-in"):
            if(min(path) > self.barrier):
                alive = False
        elif(self.barrier_type == "down-and-out"):
            if(min(path) <= self.barrier):
                alive = False
        elif(self.barrier_type == "up-and-in"):
            if(max(path) < self.barrier):
                alive = False
        elif(self.barrier_type == "up-and-out"):
            if(max(path) >= self.barrier):
                alive = False

        if (alive == False) :
            return 0
        elif(self.option_type == "call"):
            return(max(path[-1]-self.K,0))
        else :
            return(max(self.K-path[-1],0))
    

    def bump_maturity(self, bump):
        """
        Create a new Barrier option with a bumped maturity.

        Parameters
        ----------
        bump : float
            Maturity shift applied to the current time to maturity.

        Returns
        -------
        EuropeanOption
            New Barrier option with adjusted maturity.
        """
        return BarrierOption(self.K,self.T + bump, self.barrier, self.option_type, self.barrier_type)
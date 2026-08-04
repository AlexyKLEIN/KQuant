from .product import Product

class EuropeanOption(Product) :
    """European vanilla option contract."""


    def __init__(self, K, T, option_type):
        """
        Create a European option.

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Time to maturity in years.
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


    def has_analytic_greeks(self):
        """
        Indicate whether the product has closed-form Greeks.

        Returns
        -------
        bool
            True if analytical Greeks are available, False otherwise.
        """
        return True
    
    def has_analytic_pricing(self):
        """
        Indicate whether the product has a closed-form pricing formula.

        Returns
        -------
        bool
            True if analytical pricing is available, False otherwise.
        """
        return True
    

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
        ST = path[-1]
        if ST <= 0:
            raise ValueError("Terminal spot must be positive")

        if self.option_type == "call" :
            return max(0, ST - self.K)

        return max(0, self.K - ST)
    
    def bump_maturity(self, bump):
        """
        Create a new European option with a bumped maturity.

        Parameters
        ----------
        bump : float
            Maturity shift applied to the current time to maturity.

        Returns
        -------
        EuropeanOption
            New European option with adjusted maturity.
        """
        return EuropeanOption(self.K,self.T + bump,self.option_type)

    





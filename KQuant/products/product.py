from abc import ABC, abstractmethod

class Product(ABC):
    """
    Abstract base class for financial products.
    """

    @abstractmethod
    def payoff(self, paths, times):
        """
        Compute the product payoff.

        Parameters
        ----------
        paths : array-like
            Simulated underlying asset path, from t=0 to maturity.

        Returns
        -------
        float
            Product payoff.
        """
        pass


    def has_analytic_pricing(self):
        """
        Indicate whether the product has a closed-form pricing formula.

        Returns
        -------
        bool
            True if analytical pricing is available, False otherwise.
        """
        return False


    def has_analytic_greeks(self):
        """
        Indicate whether the product has closed-form Greeks.

        Returns
        -------
        bool
            True if analytical Greeks are available, False otherwise.
        """
        return False


    @abstractmethod
    def bump_maturity(self, bump):
        """
        Return a new product with a bumped maturity.

        Parameters
        ----------
        bump : float
            Maturity shift applied to the product.

        Returns
        -------
        Product
            A new product instance with bumped maturity..
        """
        pass
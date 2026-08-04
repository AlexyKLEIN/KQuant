from abc import ABC, abstractmethod


class PricingEngine(ABC):
    """
    Abstract base class for pricing engines.
    """

    @abstractmethod
    def price(self, market, model, product, **kwargs):
        """
        Compute the price of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data.
        model : Model
            Pricing model.
        product : Product
            Financial product.

        Returns
        -------
        float
            Product price.
        """
        pass
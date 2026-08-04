from ..products import EuropeanOption

class PriceSurface:
    """
    Store market price associated with European options.

    Parameters
    ----------
    options : list
        List of European options.
    prices : list
        Corresponding market prices.
    """

    def __init__(self, options, prices):
        if len(options) != len(prices):
            raise ValueError(
                "Options and prices must have the same length"
            )
        
        if not all(isinstance(option, EuropeanOption) for option in options):
            raise TypeError(
                "Price surface only supports European options"
            )
        
        self.options = options
        self.prices = prices
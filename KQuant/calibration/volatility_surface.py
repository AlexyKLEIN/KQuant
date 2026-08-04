from ..products import EuropeanOption

class VolatilitySurface:
    """
    Store market implied volatilities associated with European options.

    Parameters
    ----------
    options : list
        List of European options.
    vols : list
        Corresponding implied volatilities.
    """

    def __init__(self, options, vols):
        if len(options) != len(vols):
            raise ValueError(
                "Options and volatilities must have the same length"
            )
        
        if not all(isinstance(option, EuropeanOption) for option in options):
            raise TypeError(
                "Volatility surface only supports European options"
            )
        
        self.options = options
        self.vols = vols
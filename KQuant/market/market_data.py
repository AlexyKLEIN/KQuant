class MarketData :
    """Market data required for financial product valuation."""

    def __init__(self, spot):
        """
        Parameters
        ----------
        spot : float
            Current price of the underlying asset.
        """

        if spot <= 0:
            raise ValueError("Spot price must be positive")
        
        self.spot = spot
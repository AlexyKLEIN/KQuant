import math
from .pricing_engine import PricingEngine
from ..models import BlackAndScholes
from ..tools.normal import Normal
from ..products import EuropeanOption

class AnalyticEngine(PricingEngine) :
    """
    Analytical pricing engine for financial products with closed-form solutions.
    """

    @staticmethod
    def european_call_price(market, model, option):
        """
        Compute the analytical price of a European call option.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.
        model : BlackScholes
            Pricing model used for valuation.
        option : EuropeanOption
            European call option contract.

        Returns
        -------
        float
            Theoretical call option price.
        """
        if isinstance(option, EuropeanOption) and isinstance(model, BlackAndScholes):
            C = market.spot*Normal.cdf(model.d1(market.spot, option.K, option.T),0,1)-option.K*math.exp(-model.r*option.T)*Normal.cdf(model.d2(market.spot, option.K, option.T),0,1)
            return(C)
        
        raise ValueError("This product or model doesn't have analytical price")

    
    @staticmethod
    def european_put_price(market, model, option):
        """
        Compute the analytical price of a European put option.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.
        model : BlackScholes
            Pricing model used for valuation.
        option : EuropeanOption
            European put option contract.

        Returns
        -------
        float
            Theoretical put option price.
        """
        if isinstance(option, EuropeanOption) and isinstance(model, BlackAndScholes):
            C = AnalyticEngine.european_call_price(market, model, option)
            P = C + option.K*math.exp(-model.r*option.T) - market.spot
            return(P)

        raise ValueError("This product or model doesn't have analytical price")

    
    def price(self, market, model, product, Z_1 = None, Z_2 = None) :
        """
        Compute the analytical price of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.
        model : Model
            Pricing model used for valuation.
        product : Product
            Financial product to be priced.
        Z_1 : array-like, optional
            Random variables used for simulation methods.
            This parameter is ignored for analytical pricing.
        Z_2 : array-like, optional
            Random variables used for simulation methods.
            This parameter is ignored for analytical pricing.

        Returns
        -------
        float
            Theoretical product price.
        """

        if product.has_analytic_pricing() :
            if isinstance(product, EuropeanOption) and isinstance(model, BlackAndScholes) :
                if product.option_type == "call" :
                    return AnalyticEngine.european_call_price(market, model, product)

                return AnalyticEngine.european_put_price(market, model, product)
            
        raise ValueError("This product or model doesn't have analytical price")

    

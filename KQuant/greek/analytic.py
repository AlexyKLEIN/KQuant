import math
from ..tools import Normal
from ..products.european_option import EuropeanOption
from ..models.black_and_scholes import BlackAndScholes

class AnalyticGreeks : 
    """
    Analytical computation of financial product sensitivities.
    """

    @staticmethod
    def delta(market, model, product) :
        """
        Compute the analytical delta of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : BlackAndScholes
            Pricing model used for the analytical computation.

        product : Product
            Financial product for which the delta is computed.

        Returns
        -------
        float
            Delta sensitivity of the product.

        """
         
        if not product.has_analytic_greeks() :
            raise ValueError("This product or model doesn't have analytical greeks")
        
        if isinstance(product, EuropeanOption) and isinstance(model, BlackAndScholes):
            d1 = model.d1(market.spot, product.K, product.T)

            if product.option_type == "call" :
            
                return Normal.cdf(d1,0,1)

            return Normal.cdf(d1,0,1) - 1
        
        
    

    @staticmethod
    def gamma(market, model, product) :
        """
        Compute the analytical gamma of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : BlackAndScholes
            Pricing model used for the analytical computation.

        product : Product
            Financial product for which the gamma is computed.

        Returns
        -------
        float
            Gamma sensitivity of the product.

        """
            
        if not product.has_analytic_greeks() :
            raise ValueError("This product or model doesn't have analytical greeks")
        
        if isinstance(product, EuropeanOption) and isinstance(model, BlackAndScholes):
            d1 = model.d1(market.spot, product.K, product.T)

            return Normal.pdf(d1,0,1)/(market.spot*model.sigma*math.sqrt(product.T))
    

    @staticmethod
    def vega(market, model, product) :
        """
        Compute the analytical vega of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : BlackAndScholes
            Pricing model used for the analytical computation.

        product : Product
            Financial product for which the vega is computed.

        Returns
        -------
        float
            Vega of the product.

        """

        if not product.has_analytic_greeks() and isinstance(model, BlackAndScholes):
            raise ValueError("This product or model doesn't have analytical greeks")
        
        if isinstance(product, EuropeanOption) :
            d1 = model.d1(market.spot, product.K, product.T)
            
            return Normal.pdf(d1, 0, 1)*market.spot*math.sqrt(product.T)
    

    @staticmethod
    def theta(market, model, product) :
        """
        Compute the analytical theta of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : BlackAndScholes
            Pricing model used for the analytical computation.

        product : Product
            Financial product for which the theta is computed.

        Returns
        -------
        float
            theta sensitivity of the product.

        """

        if not product.has_analytic_greeks() :
            raise ValueError("This product or model doesn't have analytical greeks")
        
        if isinstance(product, EuropeanOption) and isinstance(model, BlackAndScholes) :
            d1 = model.d1(market.spot, product.K, product.T)
            d2 = model.d2(market.spot, product.K, product.T)

            if product.option_type == "call" :
                return (-market.spot*Normal.pdf(d1, 0, 1)*model.sigma)/(2*math.sqrt(product.T)) - model.r*product.K*math.exp(-model.r*product.T)*Normal.cdf(d2, 0, 1)
        
            return (-market.spot*Normal.pdf(d1, 0, 1)*model.sigma)/(2*math.sqrt(product.T)) + model.r*product.K*math.exp(-model.r*product.T)*Normal.cdf(-d2, 0, 1)
        

    @staticmethod
    def rho(market, model, product) :
        """
        Compute the analytical rho of a financial product.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : BlackAndScholes
            Pricing model used for the analytical computation.

        product : Product
            Financial product for which the rho is computed.

        Returns
        -------
        float
            rho sensitivity of the product.

        """

        if not product.has_analytic_greeks() :
            raise ValueError("This product or model doesn't have analytical greeks")
        
        if isinstance(product, EuropeanOption) and isinstance(model, BlackAndScholes):
            d1 = model.d1(market.spot, product.K, product.T)
            d2 = model.d2(market.spot, product.K, product.T)

            if product.option_type == "call" :
                return product.K*product.T*math.exp(-model.r*product.T)*Normal.cdf(d2, 0, 1)
            
            return -product.K*product.T*math.exp(-model.r*product.T)*Normal.cdf(-d2, 0, 1)

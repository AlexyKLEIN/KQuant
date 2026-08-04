from ..pricing import MonteCarloEngine
from ..market import MarketData
import numpy as np

class FiniteDifferenceGreeks :
    """
    Compute greeks using central finite differences.
    """

    @staticmethod    
    def delta(market, model, engine, product, eps = 1e-4) :
        """
        Compute delta using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        eps : float, optional
            Relative bump applied to the spot price.

        Returns
        -------
        float
            Delta sensitivity of the product.
        """

        Z_1 = None
        Z_2 = None

        if isinstance(engine, MonteCarloEngine) :
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))


        bump = market.spot*eps
        market_up = MarketData(market.spot + bump)
        market_down = MarketData(market.spot - bump)
  
        price_down = engine.price(market_down, model, product, Z_1, Z_2)
        price_up = engine.price(market_up, model, product, Z_1, Z_2)

        return (price_up - price_down)/(2*bump)    


    @staticmethod    
    def gamma(market, model, engine, product, eps = 1e-3) :

        """
        Compute gamma using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        eps : float, optional
            Relative bump applied to the spot price.

        Returns
        -------
        float
            Gamma sensitivity of the product.
        """

        Z_1 = None
        Z_2 = None

        if isinstance(engine, MonteCarloEngine) :
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
        
        bump = market.spot*eps
        market_up = MarketData(market.spot + bump)
        market_down = MarketData(market.spot - bump)

        price = engine.price(market, model, product, Z_1, Z_2)         
        price_down = engine.price(market_down, model, product, Z_1, Z_2)
        price_up = engine.price(market_up, model, product, Z_1, Z_2)

        return (price_up + price_down - 2*price)/(bump**2)
    

    @staticmethod    
    def vega(market, model, engine, product, bump = 1.e-3) :
        """
        Compute vega using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        bump : float, optional
            Bump applied to the volatility.

        Returns
        -------
        float
            Vega sensitivity of the product.
        """

        if not hasattr(model, "bump_volatility"):
            raise ValueError("Vega is not available for this model.")

        Z_1 = None
        Z_2 = None
        
        if isinstance(engine, MonteCarloEngine) :
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
        
        vol_down = model.bump_volatility(-bump)
        vol_up = model.bump_volatility(+bump)
        
        price_down = engine.price(market, vol_down, product, Z_1, Z_2)
        price_up = engine.price(market, vol_up, product, Z_1, Z_2)

        return (price_up - price_down)/(2*bump)
    

    @staticmethod    
    def rho(market, model, engine, product, bump = 1e-3) :
        """
        Compute rho using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        bump : float, optional
            Bump applied to the risk free interest rate.

        Returns
        -------
        float
            Theta sensitivity of the product using market convention
        """

        if not hasattr(model, "bump_risk_free_rate"):
            raise ValueError("Rho is not available for this model.")
        
        Z_1 = None
        Z_2 = None
       
        if isinstance(engine, MonteCarloEngine) :
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
       

        rate_down = model.bump_risk_free_rate(-bump)
        rate_up = model.bump_risk_free_rate(+bump)
        
        price_down = engine.price(market, rate_down, product, Z_1, Z_2)
        price_up = engine.price(market, rate_up, product, Z_1, Z_2)

        return (price_up - price_down)/(2*bump)
    

    @staticmethod    
    def theta(market, model, engine, product, bump = 1e-3) :
        """
        Compute theta using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        bump : float, optional
            Bump applied to the maturity time.

        Returns
        -------
        float
            theta sensitivity of the product.
        """

        Z_1 = None
        Z_2 = None
        
        if isinstance(engine, MonteCarloEngine) :
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
        

        maturity_down = product.bump_maturity(-bump)
        maturity_up = product.bump_maturity(bump)
        
        price_down = engine.price(market, model, maturity_down, Z_1, Z_2)
        price_up = engine.price(market, model, maturity_up, Z_1, Z_2)

        return -(price_up - price_down)/(2*bump)


    @staticmethod
    def v0_sensitivity(market, model, engine, product, bump=1e-3):
        """
        Compute the sensitivity of the initial variance v0 using central finite differences.
        """

        if not hasattr(model, "bump_v0"):
            raise ValueError("V0 sensitivity is not available for this model.")

        Z_1 = None
        Z_2 = None

        if isinstance(engine, MonteCarloEngine):
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))

        model_down = model.bump_v0(-bump)
        model_up = model.bump_v0(+bump)

        price_down = engine.price(market, model_down, product, Z_1, Z_2)
        price_up = engine.price(market, model_up, product, Z_1, Z_2)

        return (price_up - price_down) / (2*bump)


    @staticmethod
    def kappa_sensitivity(market, model, engine, product, bump=1e-3):
        """
        Compute the sensitivity of the mean reversion speed kappa
        using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        bump : float, optional
            Bump applied to the kappa parameter.

        Returns
        -------
        float
            Kappa sensitivity of the product.
        """

        if not hasattr(model, "bump_kappa"):
            raise ValueError("Kappa sensitivity is not available for this model.")

        Z_1 = None
        Z_2 = None

        if isinstance(engine, MonteCarloEngine):
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))

        model_down = model.bump_kappa(-bump)
        model_up = model.bump_kappa(+bump)

        price_down = engine.price(market, model_down, product, Z_1, Z_2)
        price_up = engine.price(market, model_up, product, Z_1, Z_2)

        return (price_up - price_down) / (2*bump)


    @staticmethod
    def xi_sensitivity(market, model, engine, product, bump=1e-3):
        """
        Compute the sensitivity of the long-term variance parameter theta
        using central finite differences.

        Parameters
        ----------
        market : MarketData
            Market data containing the current spot price.

        model : Model
            Pricing model.

        engine : PricingEngine
            Pricing engine used to evaluate the product.

        product : Product
            Financial product.

        bump : float, optional
            Bump applied to the long-term variance parameter theta.

        Returns
        -------
        float
            Long-term variance sensitivity of the product.
        """

        if not hasattr(model, "bump_xi"):
            raise ValueError(
                "Long-term variance sensitivity is not available for this model."
            )

        Z_1 = None
        Z_2 = None

        if isinstance(engine, MonteCarloEngine):
            Z_1 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))
            Z_2 = np.random.normal(0, 1, (engine.n_steps, engine.n_paths))

        model_down = model.bump_xi(-bump)
        model_up = model.bump_xi(+bump)

        price_down = engine.price(market, model_down, product, Z_1, Z_2)
        price_up = engine.price(market, model_up, product, Z_1, Z_2)

        return (price_up - price_down) / (2*bump)
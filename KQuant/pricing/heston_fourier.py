import numpy as np
import math
from .pricing_engine import PricingEngine
from ..models.heston import Heston
from ..products import EuropeanOption


class HestonFourierEngine(PricingEngine):
    """
    Pricing engine based on the Heston stochastic volatility model using Fourier inversion.

    Supported:
        - Model: Heston
        - Product: EuropeanOption
    """

    def __init__(self, n_integration=2000):
        """
        Initialize the Fourier integration settings.

        Parameters
        ----------
        n_integration : int
            Number of integration points.
        """

        self.n_integration = n_integration

    
    def characteristic_function(self, u, market, model, product):
        """
        Compute the Heston characteristic function.

        Parameters
        ----------
        u : complex
            Fourier integration variable.
        market : MarketData
            Market information containing the spot price.
        model : Heston
            Heston model parameters.
        product : EuropeanOption
            Option contract information.

        Returns
        -------
        complex
            Value of the Heston characteristic function.
        """

        b = model.kappa - model.rho*model.xi*1j*u
        d = np.sqrt(b**2 + (model.xi)**2*(1j*u + u**2))
        g = (b-d)/(b+d)
        D = (b-d)*(1-np.exp(-d*product.T))/(model.xi**2*(1-g*np.exp(-d*product.T)))
        C = 1j*model.r*u*product.T+model.kappa*model.theta*((b-d)*product.T-2*np.log((1-g*np.exp(-d*product.T))/(1-g)))/model.xi**2
        return np.exp(C + D*model.v0 + 1j*u*np.log(market.spot))

    def _P1(self, market, model, product):
        """
        Compute probability P1 using Fourier inversion.
        P1 corresponds to the probability term weighting
        the underlying asset in the Heston pricing formula.

        Parameters
        ----------
        market : MarketData
            Market information.
        model : Heston
            Heston model parameters.
        product : EuropeanOption
            European option contract.

        Returns
        -------
        float
            Probability P1.
        """
        def integrand(u):
            phi_1 = self.characteristic_function(u-1j, market, model, product)/self.characteristic_function(-1j, market, model, product)
            value = (np.exp(-1j*u*np.log(product.K))*phi_1/(1j*u))
        
            return np.real(value)

        u_max = 150
        n = self.n_integration
        u_values = np.linspace(1e-8, u_max, n)
        values = np.array([integrand(u) for u in u_values])
        integral = np.trapezoid(values, u_values)

        return 0.5 + integral/np.pi


    def _P2(self, market, model, product):
        """
        Compute probability P2 using Fourier inversion.
        P2 corresponds to: P(S_T > K) in the Heston pricing formula.

        Parameters
        ----------
        market : MarketData
            Market information.
        model : Heston
            Heston model parameters.
        product : EuropeanOption
            European option contract.

        Returns
        -------
        float
            Probability P2.
        """
        def integrand(u):

            phi = self.characteristic_function(u, market, model, product) 
            value = (np.exp(-1j*u*np.log(product.K))*phi/(1j*u))

            return np.real(value)

        u_max = 150
        n = self.n_integration
        u_values = np.linspace(1e-8, u_max, n)
        values = np.array([integrand(u) for u in u_values])
        integral = np.trapezoid(values, u_values)

        return 0.5 + integral/np.pi


    def price(self, market, model, product, Z_1=None, Z_2=None):
        """
        Compute the European option price under the Heston model.

        Parameters
        ----------
        market : MarketData
            Market information.
        model : Heston
            Heston stochastic volatility model.
        product : EuropeanOption
            European option contract.
        Z_1 : array-like, optional
            Random variables used for simulation methods.
            This parameter is ignored here.
        Z_2 : array-like, optional
            Random variables used for simulation methods.
            This parameter is ignored here.

        Returns
        -------
        float
            Option price.
        """
        if not isinstance(model, Heston):
            raise NotImplementedError(
                "Heston Fourier only supports Heston model"
            )

        if not isinstance(product, EuropeanOption):
            raise NotImplementedError(
                "Heston Fourier only supports European options"
            )
    
        P1 = self._P1(market, model, product)
        P2 = self._P2(market, model, product)
        call = market.spot * P1 - product.K * np.exp(-model.r * product.T) * P2

        if product.option_type == "call":
            return call
        else:
            return call - market.spot + product.K * np.exp(-model.r * product.T)
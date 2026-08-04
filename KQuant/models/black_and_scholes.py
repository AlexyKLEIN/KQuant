import math
import numpy as np
from .model import Model

class BlackAndScholes(Model) :
    """
    Black-Scholes stochastic model.
    """

    def __init__(self, r, sigma) :
        """
        Initialize a Black-Scholes model.

        Parameters
        ----------
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility of the underlying asset.
        """
        if sigma <= 0 :
            raise ValueError("Invalid input sigma")
        
        self.r = r
        self.sigma = sigma


    def d1(self, S0, K, T) :
        """
        Compute the d1 term of the Black-Scholes formula.

        Parameters
        ----------
        S0 : float
            Current asset price.
        K : float
            Strike price.
        T : float
            Time to maturity in years.

        Returns
        -------
        float
            Value of d1.
        """
         
        d1 = (math.log(S0/K) + (self.r + (self.sigma**2)/2)*T)/(self.sigma*math.sqrt(T))

        return d1
    

    def d2(self, S0, K, T) :
        """
        Compute the d2 term of the Black-Scholes formula.

        Parameters
        ----------
        S0 : float
            Current asset price.
        K : float
            Strike price.
        T : float
            Time to maturity in years.

        Returns
        -------
        float
            Value of d2.
        """
        d2 = self.d1(S0, K, T)-self.sigma*math.sqrt(T)
        
        return d2
    

    def simulate_paths(self, S0, T, n_steps, n_paths, Z_1 = None, Z_2 = None) :
        """
        Simulate asset price paths using geometric Brownian motion.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of simulated paths.  
        Z_1 : array-like, optional
            Random variables used for simulation.
        Z_2 : array-like, optional
            Random variables used for simulation.

        Returns
        -------
        numpy.ndarray
            Simulated asset price paths with shape (n_steps + 1, n_paths).
        """

        if(Z_1 is None) : 
            Z_1 = np.random.normal(0, 1, (n_steps, n_paths))
        else:
            Z_1 = np.asarray(Z_1)

        if(Z_2 is None) : 
            Z_2 = np.random.normal(0, 1, (n_steps, n_paths))
        else:
            Z_2 = np.asarray(Z_2)

        if T <= 0 or n_steps <= 0 or n_paths <= 0 or Z_1.shape != (n_steps, n_paths) :
            raise ValueError("Invalid simulation parameters")

        dt = T/n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        
        paths[0] = S0

        for t in range(1, n_steps + 1) :

            paths[t] = paths[t-1]*np.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z_1[t-1])


        return paths
    
    def bump_volatility(self, bump):
        """
        Create a new Black-Scholes model with a bumped volatility.

        Parameters
        ----------
        bump : float
            Volatility shift applied to the current volatility.

        Returns
        -------
        BlackAndScholes
            New Black-Scholes model with adjusted volatility.
        """
        new_sigma = self.sigma + bump

        if new_sigma <= 0:
            raise ValueError("Invalid volatility bump")

        return BlackAndScholes(self.r, new_sigma)
    

    def bump_risk_free_rate(self, bump):
        """
        Create a new Black-Scholes model with a bumped risk free interest rate.

        Parameters
        ----------
        bump : float
            Risk free interest rate shift applied to the current rate.

        Returns
        -------
        BlackAndScholes
            New Black-Scholes model with adjusted rate.
        """

        new_r = self.r + bump

        if new_r <= 0:
            raise ValueError("Invalid risk free interest rate bump")

        return BlackAndScholes(new_r, self.sigma)
    

    


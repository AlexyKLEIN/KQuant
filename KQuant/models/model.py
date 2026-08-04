from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for stochastic pricing models.
    """

    @abstractmethod
    def simulate_paths(self, S0, T, n_steps, n_paths, Z_1=None, Z_2 = None):
        """
        Simulate asset price paths.

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
            Simulated asset price paths.
        """
        pass
    

    @abstractmethod
    def bump_risk_free_rate(self, bump):
        """
        Return a new model with bumped risk free interest rate.
        """
        pass
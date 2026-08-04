import math
import numpy as np
from .model import Model

class Heston(Model) :

    def __init__(self, r, v0, kappa, theta, xi, rho):
        super().__init__()
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0

    def simulate_paths(self, S0, T, n_steps, n_paths, Z_1 = None, Z_2 = None) :

        dt = T/n_steps

        if(Z_1 is None) : 
             Z_1 = np.random.normal(0, 1, (n_steps, n_paths))
        else:
            Z_1 = np.asarray(Z_1)

        if(Z_2 is None) : 
            Z_2 = np.random.normal(0, 1, (n_steps, n_paths))
        else:
            Z_2 = np.asarray(Z_2)

        DW_1 = math.sqrt(dt)*Z_1
        DW_2 = math.sqrt(dt)*(self.rho*Z_1 + math.sqrt(1-self.rho**2)*Z_2)

        paths_v = np.zeros((n_steps + 1, n_paths))
        paths = np.zeros((n_steps + 1, n_paths))

        paths_v[0] = self.v0
        paths[0] = S0

        for t in range(1, n_steps + 1) :
            v = np.maximum(paths_v[t-1], 0.0)
            paths_v[t] = (paths_v[t-1] + self.kappa*(self.theta - v)*dt + self.xi*np.sqrt(v)*DW_2[t-1])

            paths_v[t] = np.maximum(paths_v[t], 0.0)
            paths[t] = paths[t-1] * np.exp((self.r - 0.5*v)*dt+ np.sqrt(v)*DW_1[t-1])

        return paths
        

    def bump_risk_free_rate(self, bump):
        """
        Create a new Heston model with a bumped risk free interest rate.

        Parameters
        ----------
        bump : float
            Risk free interest rate shift applied to the current rate.

        Returns
        -------
        Heston
            New Heston model with adjusted rate.
        """

        new_r = self.r + bump

        if new_r <= 0:
            raise ValueError("Invalid risk free interest rate bump")

        return Heston(new_r, self.v0, self.kappa, self.theta, self.xi, self.rho)


    def bump_v0(self, bump):
        """
        Create a new Heston model with a bumped initial variance.

        Parameters
        ----------
        bump : float
            Shift applied to initial variance.

        Returns
        -------
        Heston
            New Heston model.
        """

        sigma = np.sqrt(self.v0)

        new_sigma = sigma + bump

        if new_sigma <= 0:
            raise ValueError("Invalid volatility bump")

        new_v0 = new_sigma**2

        return Heston(r=self.r, v0=new_v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho)


    def bump_kappa(self, bump):
        """
        Create a new Heston model with a bumped mean reversion speed.

        Parameters
        ----------
        bump : float
            Shift applied to kappa.

        Returns
        -------
        Heston
            New Heston model with adjusted kappa.
        """

        new_kappa = self.kappa + bump

        if new_kappa <= 0:
            raise ValueError("Invalid kappa bump")

        return Heston(r=self.r, v0=self.v0, kappa=new_kappa, theta=self.theta, xi=self.xi, rho=self.rho)


    def bump_xi(self, bump):
        """
        Create a new Heston model with a bumped vol-of-vol parameter.

        Parameters
        ----------
        bump : float
            Shift applied to xi.

        Returns
        -------
        Heston
            New Heston model with adjusted xi.
        """

        new_xi = self.xi + bump

        if new_xi <= 0:
            raise ValueError("Invalid xi bump")

        return Heston(r=self.r, v0=self.v0, kappa=self.kappa, theta=self.theta, xi=new_xi, rho=self.rho)


    def bump_theta(self, bump):
        """
        Create a new Heston model with a bumped long-term variance.

        Parameters
        ----------
        bump : float
            Shift applied to theta.

        Returns
        -------
        Heston
            New Heston model with adjusted theta.
        """

        new_theta = self.theta + bump

        if new_theta <= 0:
            raise ValueError("Invalid theta bump")

        return Heston(r=self.r, v0=self.v0, kappa=self.kappa, theta=new_theta, xi=self.xi, rho=self.rho) 
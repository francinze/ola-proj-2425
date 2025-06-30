from typing import Any, List
import numpy as np

from .setting import Setting


class Buyer:
    """
    Buyer class for the simulation of a market.
    """
    def __init__(self, name: str, setting: Setting, dist_params: np.ndarray[float, Any]):
        
        self.name = name
        self.setting = setting
        self.verbose = setting.verbose == 'all' or setting.verbose == 'buyer'
        self.valuations: np.ndarray[float, Any] = np.zeros(setting.n_products)
        self.T = setting.T  # Total number of rounds
        self.dist_params = dist_params  # Parameters for the valuation distribution

        if setting.distribution == "uniform":
            high = self.dist_params[0] if len(self.dist_params) > 0 else 1
            low = self.dist_params[1] if len(self.dist_params) > 1 else 0
            self.valuations = np.random.uniform(
                low=low, high=high, size=setting.n_products
            )
        elif setting.distribution == "bernoulli":
            n = self.dist_params[0][0] if len(self.dist_params) > 0 > 0 else 1
            p = self.dist_params[1] if len(self.dist_params) > 1 else 0.5
            self.valuations = np.random.binomial(
                n=n, p=p, size=setting.n_products
            )
        elif setting.distribution == "gaussian":
            loc = self.dist_params[0] if len(self.dist_params) > 0 else 0.5
            scale = self.dist_params[1] if len(self.dist_params) > 1 else 0.2
            self.valuations = np.random.normal(
                loc=loc, scale=scale, size=setting.n_products
            )
            self.valuations = np.clip(self.valuations, 0, 1)
        elif setting.distribution == "exponential":
            # mean=0.5, scale=0.5, clipped to [0,1]
            scale = self.dist_params[0] if len(self.dist_params) > 0 else 0.5
            self.valuations = np.clip(
                np.random.exponential(scale=scale, size=setting.n_products), 0, 1
            )
        elif setting.distribution == "beta":
            # Beta(2,5) is skewed toward 0, Beta(5,2) toward 1
            a = self.dist_params[0] if len(self.dist_params) > 0 else 2
            b = self.dist_params[1] if len(self.dist_params) > 1 else 5
            self.valuations = np.random.beta(a=a, b=b, size=setting.n_products)
        elif setting.distribution == "lognormal":
            # lognormal with mean ~0.5, clipped to [0,1]
            mean = self.dist_params[0] if len(self.dist_params) > 0 else -0.7
            sigma = self.dist_params[1] if len(self.dist_params) > 1 else 0.5
            self.valuations = np.clip(np.random.lognormal(
                mean=mean, sigma=sigma, size=setting.n_products
            ), 0, 1)
        elif setting.distribution == "test":
            # set valuations in a fixed range for testing purposes
            high = self.dist_params[:,0] if len(self.dist_params) > 0 else np.linspace(0.2, 1, setting.n_products)
            low = self.dist_params[:,1] if len(self.dist_params) > 1 else np.linspace(0, 0.8, setting.n_products)
            v = np.zeros(setting.n_products)
            for i in range(setting.n_products):
                v[i] = np.random.uniform(
                    low=low[i], high=high[i]
                )
            self.valuations = np.clip(v, 0, 1)
        elif setting.distribution == "constant":
            # All valuations are the same constant value
            v = self.dist_params if len(self.dist_params) > 0 else np.linspace(0.1, 1, setting.n_products)
            self.valuations = np.clip(v, 0, 1)
        else:
            raise ValueError(f"Unknown distribution: {setting.distribution}")

    def __str__(self):
        return f"Buyer(name={self.name}, valuations={self.valuations})"

    def __repr__(self):
        return f"Buyer number {self.name}"

    def yield_demand(self, prices: np.ndarray[float, Any]) -> List[float]:
        """
        Make purchases based on the given prices.
        :param prices: A list of prices for the products.
        :return: A list of purchased products.
        """
        purchased_products: List[float] = []
        if self.verbose:
            print(f"{self.name} valuations: {self.valuations}")
            print(f"{self.name} prices: {prices}")
        for i, price in enumerate(prices):
            if self.valuations[i] > price:
                purchased_products.append(price)
            else:
                purchased_products.append(0)
        if self.verbose:
            print(f"{self.name} purchased products: {purchased_products}")
        return purchased_products

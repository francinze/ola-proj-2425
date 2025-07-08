from typing import Any, List
import numpy as np

from .setting import Setting


class Buyer:
    """
    Buyer class for the simulation of a market.
    """
    def __init__(self, name: str, setting: Setting, dist_params: np.ndarray[float, Any] = None):
        
        self.name = name
        self.setting = setting
        self.verbose = setting.verbose == 'all' or setting.verbose == 'buyer'
        self.valuations = np.zeros((setting.n_buyers, setting.n_products))
        self.T = setting.T  # Total number of rounds
        self.dist_params = dist_params  # Parameters for the valuation distribution

        if setting.distribution == "uniform":
            high = self.dist_params[0] if self.dist_params != None and len(self.dist_params) > 0 else 1
            low = self.dist_params[1] if self.dist_params != None and len(self.dist_params) > 1 else 0
            self.valuations = np.random.uniform(
                low=low, high=high, size=setting.n_products
            )
        elif setting.distribution == "bernoulli":
            n = self.dist_params[0][0] if self.dist_params != None and len(self.dist_params) > 0 > 0 else 1
            p = self.dist_params[1] if self.dist_params != None and len(self.dist_params) > 1 else 0.5
            self.valuations = np.random.binomial(
                n=n, p=p, size=setting.n_products
            )
        elif setting.distribution == "gaussian":
            loc = self.dist_params[0] if self.dist_params != None and len(self.dist_params) > 0 else 0.5
            scale = self.dist_params[1] if self.dist_params != None and len(self.dist_params) > 1 else 0.2
            self.valuations = np.random.normal(
                loc=loc, scale=scale, size=setting.n_products
            )
            self.valuations = np.clip(self.valuations, 0, 1)
        elif setting.distribution == "exponential":
            # mean=0.5, scale=0.5, clipped to [0,1]
            # mean = self.dist_params[0] if self.dist_params != None and len(self.dist_params) > 0 else 0.5
            scale = self.dist_params[1] if self.dist_params != None and len(self.dist_params) > 1 else 0.5
            self.valuations = np.clip(
                np.random.exponential(scale=scale, size=setting.n_products), 0, 1
            )
        elif setting.distribution == "beta":
            # Beta(2,5) is skewed toward 0, Beta(5,2) toward 1
            a = self.dist_params[0] if self.dist_params != None and len(self.dist_params) > 0 else 2
            b = self.dist_params[1] if self.dist_params != None and len(self.dist_params) > 1 else 5
            self.valuations = np.random.beta(a=a, b=b, size=setting.n_products)
        elif setting.distribution == "lognormal":
            # lognormal with mean ~0.5, clipped to [0,1]
            mean = self.dist_params[0] if self.dist_params != None and len(self.dist_params) > 0 else -0.7
            sigma = self.dist_params[1] if self.dist_params != None and len(self.dist_params) > 1 else 0.5
            self.valuations = np.clip(np.random.lognormal(
                mean=mean, sigma=sigma, size=setting.n_products
            ), 0, 1)
        elif setting.distribution == "test":
            # set valuations in a fixed range for testing purposes
            high = self.dist_params[:,0] if self.dist_params != None and len(self.dist_params) > 0 else np.linspace(0.2, 1, setting.n_products)
            low = self.dist_params[:,1] if self.dist_params != None and len(self.dist_params) > 1 else np.linspace(0, 0.8, setting.n_products)
            v = np.zeros(setting.n_products)
            for i in range(setting.n_products):
                v[i] = np.random.uniform(
                    low=low[i], high=high[i]
                )
            self.valuations = np.clip(v, 0, 1)
        elif setting.distribution == "constant":
            # All valuations are the same constant value
            v = self.dist_params if self.dist_params != None and len(self.dist_params) > 0 else np.linspace(0.1, 1, setting.n_products)
            self.valuations = np.clip(v, 0, 1)
        else:
            raise ValueError(f"Unknown distribution: {setting.distribution}")

    def __str__(self):
        return f"Buyer(name={self.name}, valuations={self.valuations})"

    def __repr__(self):
        return f"Buyer number {self.name}"

    def yield_demand(self, prices: np.ndarray) -> np.ndarray:
        """
        Make purchases based on buyer valuations and prices.
        :param prices: A (num_products,) array of prices.
        :return: A (num_products,) array representing total demand per product.
        """
        prices = np.asarray(prices).reshape(1, -1)  # shape (1, n_products)
        prices = np.tile(prices, (self.setting.n_buyers, 1))  # shape (n_buyers, n_products)

        # Each buyer purchases product i if valuation > price
        willingness = self.valuations - prices
        demand = (willingness > 0).astype(int)  # shape (n_buyers, n_products)

        total_demand = np.sum(demand, axis=0)  # shape (n_products,)

        return total_demand

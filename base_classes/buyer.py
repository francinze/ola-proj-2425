from typing import Any, List
import numpy as np

from .setting import Setting


class Buyer:
    """
    Buyer class for the simulation of a market.
    """
    def __init__(self, name: str, setting: Setting):
        self.name = name
        self.setting = setting
        self.verbose = setting.verbose == 'all' or setting.verbose == 'buyer'
        self.valuations: np.ndarray[float, Any] = np.zeros(setting.n_products)
        if setting.distribution == "uniform":
            self.valuations = np.random.uniform(
                low=0, high=1, size=setting.n_products
            )
        elif setting.distribution == "bernoulli":
            self.valuations = np.random.binomial(
                n=1, p=0.5, size=setting.n_products
            )
        elif setting.distribution == "gaussian":
            self.valuations = np.random.normal(
                loc=0.5, scale=0.15, size=setting.n_products
            )
            self.valuations = np.clip(self.valuations, 0, 1)
        elif setting.distribution == "exponential":
            # mean=0.5, scale=0.5, clipped to [0,1]
            self.valuations = np.clip(
                np.random.exponential(scale=0.5, size=setting.n_products), 0, 1
            )
        elif setting.distribution == "beta":
            # Beta(2,5) is skewed toward 0, Beta(5,2) toward 1
            self.valuations = np.random.beta(a=2, b=5, size=setting.n_products)
        elif setting.distribution == "lognormal":
            # lognormal with mean ~0.5, clipped to [0,1]
            self.valuations = np.clip(np.random.lognormal(
                mean=-0.7, sigma=0.5, size=setting.n_products
            ), 0, 1)
        elif setting.distribution == "test":
            # set valuations in a fixed range for testing purposes
            limits = np.linspace(0.3, 0.8, setting.n_products)
            v= np.zeros(setting.n_products)
            for i in range(setting.n_products):
                v[i] = np.random.uniform(
                    low=limits[i] - 0.15, high=limits[i] + 0.15
                )
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

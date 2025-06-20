from typing import Any, List
import numpy as np


class Buyer:
    """
    Buyer class for the simulation of a market.
    """
    def __init__(
        self, name: str, n_products: int, distribution: str = "uniform"
    ):
        self.name = name
        self.valuations: np.ndarray[float, Any] = np.zeros(n_products)
        if distribution == "uniform":
            self.valuations = np.random.uniform(
                low=0, high=1, size=n_products
            )
        elif distribution == "bernoulli":
            self.valuations = np.random.binomial(
                n=1, p=0.5, size=n_products
            )
        elif distribution == "gaussian":
            self.valuations = np.random.normal(
                loc=0.5, scale=0.15, size=n_products
            )
            self.valuations = np.clip(self.valuations, 0, 1)
        elif distribution == "exponential":
            # mean=0.5, scale=0.5, clipped to [0,1]
            self.valuations = np.clip(
                np.random.exponential(scale=0.5, size=n_products), 0, 1
            )
        elif distribution == "beta":
            # Beta(2,5) is skewed toward 0, Beta(5,2) toward 1
            self.valuations = np.random.beta(a=2, b=5, size=n_products)
        elif distribution == "lognormal":
            # lognormal with mean ~0.5, clipped to [0,1]
            self.valuations = np.clip(
                np.random.lognormal(mean=-0.7, sigma=0.5, size=n_products),
                0,
                1
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def __str__(self):
        return f"Buyer(name={self.name}, valuations={self.valuations})"

    def __repr__(self):
        return f"Buyer number {self.name}"

    def make_purchases(self, prices: np.ndarray[float, Any]) -> List[float]:
        """
        Make purchases based on the given prices.
        :param prices: A list of prices for the products.
        :return: A list of purchased products.
        """
        purchased_products: List[float] = []
        for i, price in enumerate(prices):
            if self.valuations[i] > price:
                purchased_products.append(price)
            else:
                purchased_products.append(0)
        return purchased_products

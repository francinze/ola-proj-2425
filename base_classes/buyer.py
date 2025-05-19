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

    def __str__(self):
        return f"Buyer(name={self.name}, valuations={self.valuations})"

    def __repr__(self):
        return f"Buyer(name={self.name}, valuations={self.valuations})"

    def make_purchases(self, prices: np.ndarray[float, Any]) -> List[float]:
        """
        Make purchases based on the given prices.
        :param prices: A list of prices for the products.
        :return: A list of purchased products.
        """
        purchased_products: List[float] = []
        print(f"Buyer valuations: {self.valuations}")
        print(f"Prices: {prices}")
        for price in prices:
            product_index = prices.tolist().index(price)
            if self.valuations[product_index] > price:
                purchased_products.append(price)
        return purchased_products

"""
Seller class for the simulation of a market.
"""
from typing import Any

import numpy as np


class Seller:
    """
    Seller class for the simulation of a market.
    """
    def __init__(
        self,
        products: np.ndarray[float, Any],
        prices: np.ndarray[float, Any],
        B: int = np.random.randint(1, 1000),
        inventory_constraint: str = "lax"
    ):
        """
        Initialize the Seller class.
        :param products: A list of products.
        """
        self.products = products
        self.prices = prices
        self.B = B
        self.utility = 0
        self.inv_rule: str = inventory_constraint

    def choose_prices(self):
        """
        Choose prices for the products.
        :return: A list of prices.
        """
        # Assume a boolean mask indicating which products to sell: True = sell,
        # False = not sell
        # For demonstration, randomly decide which products to sell
        sell_mask = np.random.choice(
            [True, False], size=len(self.products)
        )
        new_prices = np.full(len(self.products), np.inf)
        new_prices[sell_mask] = np.random.uniform(
            low=0.1, high=1, size=np.sum(sell_mask)
        )
        self.prices = new_prices
        return self.prices

    def inventory_constraint(self, purchases: np.ndarray[float, Any]) -> None:
        """
        Check if the purchases exceed the production capacity.
        :param purchases: A list of purchases.
        """
        if self.inv_rule == "strict" and np.sum(purchases) > self.B:
            raise ValueError("Exceeds production capacity")
        exceeding_capacity = max(0, np.sum(purchases) - self.B)
        self.utility = np.sum(purchases) - exceeding_capacity

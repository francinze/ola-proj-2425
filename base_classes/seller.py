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
        price_grid: np.ndarray[float, Any],
        B: int = np.random.randint(1, 1000),
        inventory_constraint: str = "lax",
        verbose: bool = True
    ):
        """
        Initialize the Seller class.
        :param products: A list of products.
        """
        self.products = products
        if price_grid.ndim == 1:
            price_grid = price_grid.reshape((len(products), -1))
        self.price_grid = price_grid
        self.B = B  # Production capacity
        self.inv_rule: str = inventory_constraint
        self.verbose = verbose

        self.num_products = len(products)
        self.num_prices = price_grid.shape[1]

        # UCB1 stats for each product and price
        self.counts = np.zeros((self.num_products, self.num_prices))
        self.values = np.zeros((self.num_products, self.num_prices))
        self.total_steps = 0

        # History for tracking rewards and arm selections
        self.history_rewards = []
        self.history_chosen_prices = []

    def choose_prices(self):
        """
        For each product, choose a price using UCB1.
        Returns: array of chosen prices (one per product),
            array of chosen price indices (one per product)
        """
        self.total_steps += 1
        chosen_prices = []
        chosen_indices = []

        for i in range(self.num_products):
            if self.verbose:
                print(f"\n[DEBUG] Product {i}:")
            ucb_values = np.zeros(self.num_prices)
            for j in range(self.num_prices):
                if self.counts[i, j] == 0:
                    ucb_values[j] = np.inf
                    if self.verbose:
                        print(
                            f"Price index {j}: count=0, ucb=inf "
                            "(unexplored)"
                        )
                else:
                    avg_reward = self.values[i, j] / self.counts[i, j]
                    confidence = np.sqrt(
                        2 * np.log(self.total_steps) / self.counts[i, j]
                    )
                    ucb_values[j] = avg_reward + confidence
                    if self.verbose:
                        print(
                            f"Price index {j}: count={self.counts[i, j]}, "
                            f"avg_reward={avg_reward:.4f}, "
                            f"confidence={confidence:.4f}, "
                            f"ucb={ucb_values[j]:.4f}"
                        )
            max_ucb = np.max(ucb_values)
            candidates = np.where(ucb_values == max_ucb)[0]
            chosen_j = np.random.choice(candidates)
            if self.verbose:
                print(
                    f"Chosen price index {chosen_j} "
                    f"for product {i} (ucb={ucb_values[chosen_j]})"
                )
            chosen_prices.append(self.price_grid[i, chosen_j])
            chosen_indices.append(chosen_j)

        self.history_chosen_prices.append(chosen_indices)
        if self.verbose:
            print("Chosen prices for products:", chosen_prices)
        return np.array(chosen_prices), np.array(chosen_indices)

    def update_ucb(self, actions, rewards):
        """
        Update UCB statistics for all products in a single call.
        :param actions: ndarray of shape (num_products,)
            with price indices for each product.
        :param rewards: ndarray of shape (num_products,)
            with rewards for each product.
        """
        # Check inventory constraint before updating reward
        if not self.inventory_constraint(rewards):
            rewards = np.zeros_like(rewards)  # or apply a penalty

        for product_idx, (price_idx, reward) in enumerate(
            zip(actions, rewards)
        ):
            self.counts[product_idx, price_idx] += 1
            n = self.counts[product_idx, price_idx]
            value = self.values[product_idx, price_idx]
            self.values[
                product_idx, price_idx
            ] = ((n - 1) / n) * value + (1 / n) * reward

        self.history_rewards.append(np.sum(rewards))

    def inventory_constraint(self, purchases: np.ndarray[float, Any]) -> bool:
        """
        Check if the purchases exceed the production capacity.
        :param purchases: A list of purchases.
        """
        total_purchases = np.sum(purchases)
        exceeding_capacity = max(0, total_purchases - self.B)

        if self.inv_rule == "lax" and exceeding_capacity > 0:
            print(
                "Warning: Purchases exceed production capacity by "
                f"{exceeding_capacity}."
            )
            return True
        elif self.inv_rule == "strict" and exceeding_capacity > 0:
            print(
                "Error: Purchases exceed production capacity by "
                f"{exceeding_capacity}."
            )
            return False
        return True  # No constraint violation

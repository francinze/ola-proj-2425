"""
Seller class for the simulation of a market.
"""
from typing import Any

import numpy as np

from .setting import Setting


class Seller:
    """
    Seller class for the simulation of a market.
    """
    def __init__(self, setting: Setting):
        """
        Initialize the Seller class.
        :param products: A list of products.
        """
        self.products = np.arange(setting.n_products)  # Products
        self.price_grid = np.linspace(0.1, 1.0, int(1 / setting.epsilon))
        self.price_grid = np.tile(self.price_grid, (setting.n_products, 1))
        if self.price_grid.ndim == 1:
            self.price_grid = self.price_grid.reshape((len(self.products), -1))
        self.B = setting.B  # Production capacity
        self.inv_rule: str = setting.inventory_constraint
        self.setting = setting
        self.verbose = setting.verbose == 'all' or setting.verbose == 'seller'

        self.num_products = len(self.products)
        self.num_prices = self.price_grid.shape[1]

        # UCB1 stats for each product and price
        self.counts = np.zeros((self.num_products, self.num_prices))
        self.values = np.zeros((self.num_products, self.num_prices))
        self.total_steps = 0

        if self.verbose:
            print(f"Initialized Seller with {self.num_products} products "
                  f"and {self.num_prices} price options per product.")
            print(f"Production capacity (B): {self.B}")

        # History for tracking rewards and arm selections
        self.history_rewards = []
        self.history_chosen_prices = []

    def yield_prices(self, chosen_indices):
        """
        For each product, choose a price using UCB1.
        Returns: array of chosen prices (one per product),
            array of chosen price indices (one per product)
        """
        self.total_steps += 1
        chosen_prices = self.price_grid[
            np.arange(self.num_products), chosen_indices
        ]
        self.history_chosen_prices.append(chosen_indices)
        if self.verbose:
            print("Chosen prices for products:", chosen_prices)
        return np.array(chosen_prices)

    def update_ucb(self, actions, rewards):
        """
        Update UCB statistics for all products in a single call.
        :param actions: ndarray of shape (num_products,)
            with price indices for each product.
        :param rewards: ndarray of shape (num_products,)
            with rewards for each product.
        """
        self.total_steps += 1
        for i, price_idx in enumerate(actions):
            self.counts[i, price_idx] += 1
            # Incremental mean update
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]
            self.values[i, price_idx] += (rewards[i] - old_value) / n

        self.history_rewards.append(np.sum(rewards))

    def inventory_constraint(
        self, purchases: np.ndarray[float, Any]
    ) -> np.ndarray[float, Any]:
        """
        Check if the purchases exceed the production capacity.
        :param purchases: A list of purchases.
        """
        purchases = np.array(purchases, dtype=float)
        total_purchases = np.count_nonzero(purchases)
        exceeding_capacity = max(0, total_purchases - self.B)
        if self.inv_rule == "lax" and exceeding_capacity > 0:
            if self.verbose:
                print(
                    "Warning: Purchases exceed production capacity by "
                    f"{exceeding_capacity}."
                )
            # Set to 0 enough purchases (randomly)
            # until total does not exceed capacity
            if total_purchases > self.B:
                indices = np.where(purchases > 0)[0]
                np.random.shuffle(indices)
                running_total = total_purchases
                for idx in indices:
                    if running_total <= self.B:
                        break
                    running_total -= 1
                    purchases[idx] = 0
        elif self.inv_rule == "strict" and exceeding_capacity > 0:
            if self.verbose:
                print(
                    "Error: Purchases exceed production capacity by "
                    f"{exceeding_capacity}."
                )
            # Return all zeros, keeping the same length
            purchases = np.zeros_like(purchases)
        if self.verbose:
            print(
                f"Purchases after inventory constraint: {purchases}"
            )
        return purchases  # No constraint violation

    def pull_arm(self):
        """
        Select a price index for each product (like an agent choosing arms).
        Returns: array of chosen price indices (one per product)
        """
        try:
            chosen_indices = np.array([], dtype=int)
            for i in range(self.num_products):
                ucb = self.values[i] + np.sqrt(
                        2 * np.log(self.total_steps + 1) /
                        np.maximum(self.counts[i], 1)
                    )
                if self.verbose:
                    print(f"UCB values for product {i}: {ucb}")
                idx = np.argmax(ucb)
                chosen_indices = np.append(chosen_indices, idx)
            if self.verbose:
                print(f"Chosen price indices for products: {chosen_indices}")
            return chosen_indices
        except Exception as e:
            print(f"Error in pull_arm: {e}")

    def update(self, rewards, actions):
        """
        Update statistics after observing rewards.
        :param rewards: ndarray of rewards per product
        :param actions: ndarray of chosen price indices per product
        """
        # Optionally clip rewards to [0, 1] or another reasonable range
        rewards = np.clip(rewards, 0, 1)
        self.update_ucb(actions, rewards)

    def reset(self):
        """
        Reset the seller's statistics for a new trial.
        """
        self.counts = np.zeros((self.num_products, self.num_prices))
        self.values = np.ones((self.num_products, self.num_prices))
        self.total_steps = 0
        self.history_rewards = []
        self.history_chosen_prices = []

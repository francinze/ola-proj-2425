"""
Seller class for the simulation of a market.
"""
from typing import Any

import numpy as np

from .setting import Setting
from .logger import (log_seller, log_error, log_algorithm_choice,
                     log_ucb1_update, log_primal_dual_update,
                     log_arm_selection)


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
            log_seller("Price grid is 1D, reshaping to 2D for compatibility")
            self.price_grid = self.price_grid.reshape((len(self.products), -1))
        self.B = setting.B  # Production capacity
        self.inv_rule: str = setting.budget_constraint
        self.setting = setting
        self.algorithm = setting.algorithm  # Algorithm choice

        # Log algorithm choice
        log_algorithm_choice(self.algorithm)

        self.num_products = len(self.products)
        self.num_prices = self.price_grid.shape[1]

        # Necessary for UCB1 algorithm
        self.T = setting.T  # Total number of rounds
        if self.T is None:
            self.T = int(np.random.randint(99, 100))
        else:
            self.T = int(self.T)  # Ensure T is always an integer

        # Primal dual params
        self.cost = np.zeros((self.T, 1))
        self.cost_coeff = 0.5
        self.eta = 0.1  # Learning rate for primal-dual updates
        self.lambda_pd = np.zeros((self.T, 1))
        self.rho_pd = self.B/self.T  # Step size for primal-dual updates

        # UCB1 stats for each product and price
        self.counts = np.zeros((self.num_products, self.num_prices))
        self.values = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)
        self.total_steps = 0

        log_seller(f"Initialized Seller with {self.num_products} products "
                   f"and {self.num_prices} price options per product.")
        log_seller(f"Production capacity (B): {self.B}")

        # History for tracking rewards and arm selections
        self.history_rewards = []
        self.history_chosen_prices = []

    def yield_prices(self, chosen_indices):
        """
        For each product, choose a price using UCB1.
        Returns: array of chosen prices (one per product),
            array of chosen price indices (one per product)
        """
        chosen_prices = self.price_grid[
            np.arange(self.num_products), chosen_indices
        ]
        self.history_chosen_prices.append(chosen_indices)

        self.cost[self.total_steps] = np.sum(chosen_prices)*self.cost_coeff

        return np.array(chosen_prices)

    def update_ucb1(self, actions, rewards):
        """
        Update UCB1 statistics for all products.
        :param actions: ndarray of shape (num_products,)
            with price indices for each product.
        :param rewards: ndarray of shape (num_products,)
            with rewards for each product.
        """
        self.total_steps += 1

        # Calculate price-weighted rewards (price × purchase)
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        price_weighted_rewards = chosen_prices * rewards

        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)  # Ensure price_idx is an integer
            self.counts[i, price_idx] += 1
            # Incremental mean update
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]

            # UCB1 update formula - FIXED: use current time step, not horizon
            # Use price-weighted reward for update
            reward_i = price_weighted_rewards[i]
            self.values[i, price_idx] = (old_value * (n-1) / n +
                                         (reward_i - old_value) / n)
            self.ucbs[i, price_idx] = self.values[i, price_idx] + \
                np.sqrt(2 * np.log(self.total_steps) / n)

            log_ucb1_update(i, price_idx, self.counts[i, price_idx],
                            self.values[i, price_idx],
                            self.ucbs[i, price_idx])

        # Store price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))

    def update_primal_dual(self, actions, rewards):
        """
        Update UCB statistics for all products in a single call.
        :param actions: ndarray of shape (num_products,)
            with price indices for each product.
        :param rewards: ndarray of shape (num_products,)
            with rewards for each product.
        """
        # Calculate price-weighted rewards (price × purchase)
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        price_weighted_rewards = chosen_prices * rewards

        # Calculate cost for current step before updating lambda
        current_cost = np.sum(chosen_prices) * self.cost_coeff
        self.cost[self.total_steps] = current_cost

        # Update lambda for next step if not at final step
        if self.total_steps < self.T - 1:
            self.lambda_pd[self.total_steps + 1] = self.update_lambda(
                self.lambda_pd[self.total_steps], self.eta, current_cost)

        # FIXED: Don't modify budget B - it should remain constant
        # The budget constraint is handled in budget_constraint method

        log_primal_dual_update(self.lambda_pd[self.total_steps].item(),
                               self.cost[self.total_steps].item(), self.B)

        # Store price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))

        self.total_steps += 1

        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)  # Ensure price_idx is an integer
            self.counts[i, price_idx] += 1
            # Incremental mean update
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]
            reward_i = price_weighted_rewards[i]
            self.values[i, price_idx] = (old_value +
                                         (reward_i - old_value) / n)
            # FIXED: Use current time step for UCB calculation
            self.ucbs[i, price_idx] = self.values[i, price_idx] + \
                np.sqrt(2 * np.log(self.total_steps) / n)

            log_seller(f"Updated UCB for product {i}, "
                       f"price index {price_idx}: "
                       f"count={self.counts[i, price_idx]}, "
                       f"value={self.values[i, price_idx]}, "
                       f"UCB={self.ucbs[i, price_idx]}")

        # Note: history_rewards already appended above with price-weighted

    def update_lambda(self, lambda_prev, eta, current_cost):
        """
        Update the dual variable lambda.
        :param lambda_prev: Previous lambda value
        :param eta: Learning rate
        :param current_cost: Cost for current time step
        :return: Updated lambda value.
        """
        # FIXED: Proper primal-dual update using cost vs budget constraint
        lambda_raw = (
            lambda_prev + eta * (current_cost - self.rho_pd)
        )

        return np.minimum(np.maximum(lambda_raw, 0), self.T/self.B)

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
            log_seller(
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
            log_seller(
                "Error: Purchases exceed production capacity by "
                f"{exceeding_capacity}."
            )
            # Return all zeros, keeping the same length
            purchases = np.zeros_like(purchases)
        log_seller(
            f"Purchases after inventory constraint: {purchases}"
        )
        return purchases  # No constraint violation

    def budget_constraint(
        self, purchases: np.ndarray[float, Any]
    ) -> np.ndarray[float, Any]:
        """
        Check if the purchases exceed the budget.
        :param purchases: A list of purchases.
        """
        purchases = np.array(purchases, dtype=float)
        total_purchases = np.count_nonzero(purchases)
        # FIXED: Check actual purchases against budget, not max prices
        exceeding_capacity = max(0, total_purchases - self.B)
        if self.inv_rule == "lax" and exceeding_capacity > 0:
            log_seller(
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
            log_seller(
                "Error: Purchases exceed production capacity by "
                f"{exceeding_capacity}."
            )
            # Return all zeros, keeping the same length
            purchases = np.zeros_like(purchases)
        log_seller(
            f"Purchases after inventory constraint: {purchases}"
        )
        return purchases  # No constraint violation

    def pull_arm(self):
        """
        Select a price index for each product using the chosen algorithm.
        Returns: array of chosen price indices (one per product)
        """
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
            chosen_indices = np.array([], dtype=int)

            if self.algorithm == "ucb1":
                # UCB1 algorithm: select arm with highest UCB value
                for i in range(self.num_products):
                    idx = int(np.argmax(self.ucbs[i]))
                    chosen_indices = np.append(chosen_indices, idx)
            elif self.algorithm == "primal_dual":
                # Primal-dual: select based on UCB with lambda adjustment
                for i in range(self.num_products):
                    # Adjust UCB values by lambda for budget constraint
                    adjusted_ucbs = (self.ucbs[i] -
                                     self.lambda_pd[self.total_steps] *
                                     self.price_grid[i] * self.cost_coeff)
                    idx = int(np.argmax(adjusted_ucbs))
                    chosen_indices = np.append(chosen_indices, idx)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices
        except Exception as e:
            log_error(f"Error in pull_arm: {e}")
            # Return safe default
            return np.zeros(self.num_products, dtype=int)

    def update(self, purchased, actions):
        """
        Update statistics after observing rewards using the chosen algorithm.
        :param purchased: ndarray of purchases per product
        :param actions: ndarray of chosen price indices per product
        """
        # Clip purchases to reasonable range
        purchased = np.clip(purchased, 0, 1)

        if self.algorithm == "ucb1":
            # For UCB1, rewards are simply the purchases
            rewards = purchased
            self.update_ucb1(actions, rewards)
        elif self.algorithm == "primal_dual":
            # FIXED: For primal-dual, rewards are just purchases
            # Cost constraint is handled via lambda in arm selection
            rewards = purchased
            self.update_primal_dual(actions, rewards)
        else:
            log_error(f"Unknown algorithm in update: {self.algorithm}")

    def reset(self, setting):
        """
        Reset the seller's statistics for a new trial.
        """
        self.setting = setting
        self.algorithm = setting.algorithm  # Update algorithm if changed
        self.counts = np.zeros((self.num_products, self.num_prices))
        self.values = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)
        self.total_steps = 0
        self.history_rewards = []
        self.history_chosen_prices = []

        # Reset algorithm-specific parameters
        if self.algorithm == "primal_dual":
            self.cost = np.zeros((self.T, 1))
            self.lambda_pd = np.zeros((self.T, 1))
            self.B = setting.B  # Reset production capacity

        log_algorithm_choice(self.algorithm)

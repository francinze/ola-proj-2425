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
    def __init__(self, setting: Setting, budget_constraint: str = "lax"):
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
        self.inv_rule: str = budget_constraint
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

        # COMPREHENSIVE BUDGET TRACKING - Available for all seller classes
        self.initial_budget = self.B  # Store original budget
        self.remaining_budget = self.B  # Current remaining budget
        self.cost_history = []  # Track costs per round
        self.budget_depleted = False  # Flag for budget depletion
        self.budget_depletion_round = None  # Round when budget was depleted
        self.total_spent = 0.0  # Total amount spent so far

        log_seller(f"Budget tracking initialized: "
                   f"Initial budget = {self.initial_budget}")

    def calculate_price_weighted_rewards(self, actions, rewards):
        """
        Calculate price-weighted rewards (price × purchase).
        Common utility method used by specialized sellers.
        """
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        return chosen_prices * rewards

    def safe_pull_arm(self, pull_arm_func):
        """
        Common error handling wrapper for pull_arm methods.
        """
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
            return pull_arm_func()
        except Exception as e:
            log_error(f"Error in {self.algorithm} pull_arm: {e}")
            return np.zeros(self.num_products, dtype=int)

    def apply_constraints_and_calculate_rewards(self, purchased, actions,
                                                use_inventory_constraint=True):
        """
        Common method to apply constraints and calculate rewards.
        """
        purchased = np.clip(purchased, 0, 1)

        # Apply inventory constraint if enabled
        if use_inventory_constraint:
            purchased = self.budget_constraint(purchased)

        return purchased

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
        Update UCB1 statistics for all products with budget tracking.
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

        # Calculate costs for budget tracking
        costs = chosen_prices * self.cost_coeff
        total_cost = np.sum(costs)

        # UPDATE BUDGET TRACKING
        self.update_budget(total_cost)

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
        Update UCB statistics for all products in a single call with
        budget tracking.
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

        # UPDATE BUDGET TRACKING
        self.update_budget(current_cost)

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

        # Reset comprehensive budget tracking
        self.initial_budget = setting.B
        self.remaining_budget = setting.B
        self.cost_history = []
        self.budget_depleted = False
        self.budget_depletion_round = None
        self.total_spent = 0.0

        # Reset algorithm-specific parameters
        if self.algorithm == "primal_dual":
            self.cost = np.zeros((self.T, 1))
            self.lambda_pd = np.zeros((self.T, 1))
            self.B = setting.B  # Reset production capacity

        log_algorithm_choice(self.algorithm)

    def update_budget(self, cost):
        """
        Update budget tracking with the cost incurred in current round.
        This method should be called by all seller classes when costs are
        incurred.

        Args:
            cost (float): Cost incurred in current round
        """
        # Record the cost
        self.cost_history.append(cost)
        self.total_spent += cost

        # Update remaining budget
        self.remaining_budget = max(0, self.remaining_budget - cost)

        # Check for budget depletion
        if not self.budget_depleted and self.remaining_budget <= 0:
            self.budget_depleted = True
            # Current round number
            self.budget_depletion_round = len(self.cost_history)
            log_seller(f"Budget depleted at round "
                       f"{self.budget_depletion_round}")

    def get_budget_status(self):
        """
        Get detailed budget status information.

        Returns:
            dict: Budget status information including depletion status,
                  remaining budget, etc.
        """
        budget_util = ((self.total_spent / self.initial_budget) * 100
                       if self.initial_budget > 0 else 0)
        cost_hist = (np.array(self.cost_history) if self.cost_history
                     else np.array([]))

        return {
            'initial_budget': self.initial_budget,
            'remaining_budget': self.remaining_budget,
            'total_spent': self.total_spent,
            'budget_depleted': self.budget_depleted,
            'budget_depletion_round': self.budget_depletion_round,
            'budget_utilization': budget_util,
            'cost_history': cost_hist
        }

    def get_budget_summary_string(self):
        """
        Get a formatted string summary of budget status for display.

        Returns:
            str: Formatted budget status string
        """
        if self.budget_depleted:
            return f"Budget depleted at round: {self.budget_depletion_round}"
        else:
            return (f"Budget NOT depleted — "
                    f"Remaining: {self.remaining_budget:.0f}")

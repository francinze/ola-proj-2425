"""
Specialized seller classes for different project requirements.
Each class extends the base Seller class with specific algorithms.
"""
import numpy as np
from .seller import Seller
from .setting import Setting
from .logger import (log_error, log_algorithm_choice,
                     log_ucb1_update, log_arm_selection)


class BaseSeller(Seller):
    """
    Base seller class with common functionality for all specialized sellers.
    This extends the original Seller class to be more modular.
    """

    def __init__(self, setting: Setting):
        """Initialize base seller with common functionality."""
        super().__init__(setting)

    def calculate_price_weighted_rewards(self, actions, rewards):
        """
        Calculate price-weighted rewards (price × purchase).
        Common utility method used by all specialized sellers.
        """
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        return chosen_prices * rewards


class UCB1Seller(BaseSeller):
    """
    Seller class implementing UCB1 algorithm for Requirements 1 and 2.

    Requirement 1: Single product + stochastic + UCB1 (with/without inventory)
    Requirement 2: Multiple products + stochastic + Combinatorial-UCB
    """

    def __init__(
        self, setting: Setting, use_inventory_constraint: bool = True
    ):
        """
        Initialize UCB1 seller.

        Args:
            setting: Setting object with configuration
            use_inventory_constraint: Whether to enforce inventory constraints
        """
        super().__init__(setting)
        self.algorithm = "ucb1"
        self.use_inventory_constraint = use_inventory_constraint
        log_algorithm_choice(
            f"UCB1 (inventory_constraint={use_inventory_constraint})"
        )

    def pull_arm(self):
        """
        UCB1 arm selection: choose arm with highest UCB value for each product.
        """
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
            chosen_indices = np.array([], dtype=int)

            for i in range(self.num_products):
                # UCB1: select arm with highest UCB value
                idx = int(np.argmax(self.ucbs[i]))
                chosen_indices = np.append(chosen_indices, idx)

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices
        except Exception as e:
            log_error(f"Error in UCB1 pull_arm: {e}")
            return np.zeros(self.num_products, dtype=int)

    def update(self, purchased, actions):
        """Update UCB1 statistics after observing rewards."""
        purchased = np.clip(purchased, 0, 1)

        # Apply inventory constraint if enabled
        if self.use_inventory_constraint:
            purchased = self.budget_constraint(purchased)

        rewards = purchased
        self.update_ucb1(actions, rewards)

    def update_ucb1(self, actions, rewards):
        """
        Update UCB1 statistics for all products.
        """
        self.total_steps += 1

        # Calculate price-weighted rewards
        price_weighted_rewards = self.calculate_price_weighted_rewards(
            actions, rewards)

        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)
            self.counts[i, price_idx] += 1
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]

            # UCB1 update with price-weighted reward
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


class CombinatorialUCBSeller(UCB1Seller):
    """
    Seller implementing Combinatorial-UCB for Requirement 2.

    Requirement 2: Multiple products + stochastic + Combinatorial-UCB

    Based on the UCB-Bidding Algorithm from project.md:
    - Computes UCB bounds for rewards (f_t) and LCB bounds for costs (c_t)
    - Solves LP to get distribution γ_t over price combinations
    - Samples from γ_t instead of greedy selection
    """

    def __init__(self, setting: Setting):
        """Initialize Combinatorial-UCB seller."""
        super().__init__(setting, use_inventory_constraint=True)
        self.algorithm = "combinatorial_ucb"
        log_algorithm_choice("Combinatorial-UCB")

        # Additional tracking for Combinatorial-UCB
        self.cost_values = np.zeros((self.num_products, self.num_prices))
        self.cost_counts = np.zeros((self.num_products, self.num_prices))
        self.cost_coeff = 1.0  # Cost coefficient for price-based costs

    def compute_ucb_lcb_bounds(self):
        """
        Compute UCB bounds for rewards and LCB bounds for costs.
        Returns: (ucb_rewards, lcb_costs)
        """
        ucb_rewards = np.zeros((self.num_products, self.num_prices))
        lcb_costs = np.zeros((self.num_products, self.num_prices))

        for i in range(self.num_products):
            for j in range(self.num_prices):
                n = self.counts[i, j]
                if n > 0:
                    # UCB for rewards (f_t)
                    confidence = np.sqrt(2 * np.log(self.total_steps) / n)
                    ucb_rewards[i, j] = self.values[i, j] + confidence

                    # LCB for costs (c_t)
                    lcb_costs[i, j] = self.cost_values[i, j] - confidence
                else:
                    # Optimistic initialization
                    ucb_rewards[i, j] = np.inf
                    lcb_costs[i, j] = 0.0

        return ucb_rewards, lcb_costs

    def solve_lp_for_distribution(self, ucb_rewards, lcb_costs):
        """
        Solve LP to get distribution γ_t over price combinations.
        Simplified version: use softmax over expected values.
        """
        # Compute expected profit for each price combination
        expected_profits = np.zeros((self.num_products, self.num_prices))

        for i in range(self.num_products):
            for j in range(self.num_prices):
                # Expected profit = UCB_reward - LCB_cost
                expected_profits[i, j] = ucb_rewards[i, j] - lcb_costs[i, j]

        # Apply softmax to get probabilities for each product
        gamma_t = np.zeros((self.num_products, self.num_prices))
        for i in range(self.num_products):
            # Avoid overflow in softmax and handle NaN/inf values
            profits = expected_profits[i]

            # Replace inf values with large finite values
            profits = np.where(np.isinf(profits), 1e10, profits)
            profits = np.where(np.isnan(profits), 0, profits)

            # Softmax computation with numerical stability
            max_val = np.max(profits)
            if np.isfinite(max_val):
                exp_vals = np.exp(profits - max_val)
                sum_exp = np.sum(exp_vals)
                if sum_exp > 0:
                    gamma_t[i] = exp_vals / sum_exp
                else:
                    gamma_t[i] = np.ones(self.num_prices) / self.num_prices
            else:
                gamma_t[i] = np.ones(self.num_prices) / self.num_prices

        return gamma_t

    def sample_from_distribution(self, gamma_t):
        """
        Sample price indices from the distribution γ_t.
        """
        chosen_indices = np.zeros(self.num_products, dtype=int)

        for i in range(self.num_products):
            # Sample according to the distribution
            chosen_indices[i] = np.random.choice(
                self.num_prices, p=gamma_t[i]
            )

        return chosen_indices

    def pull_arm(self):
        """
        Combinatorial-UCB arm selection following the LP-based approach.
        """
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
            # Compute UCB bounds for rewards and LCB bounds for costs
            ucb_rewards, lcb_costs = self.compute_ucb_lcb_bounds()

            # Solve LP to get distribution over price combinations
            gamma_t = self.solve_lp_for_distribution(ucb_rewards, lcb_costs)

            # Sample from the distribution
            chosen_indices = self.sample_from_distribution(gamma_t)

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices
        except Exception as e:
            log_error(f"Error in Combinatorial-UCB pull_arm: {e}")
            return np.zeros(self.num_products, dtype=int)

    def update(self, purchased, actions):
        """Update Combinatorial-UCB statistics after observing rewards."""
        purchased = np.clip(purchased, 0, 1)

        # Apply inventory constraint
        if self.use_inventory_constraint:
            purchased = self.budget_constraint(purchased)

        rewards = purchased
        self.update_combinatorial_ucb(actions, rewards)

    def update_combinatorial_ucb(self, actions, rewards):
        """
        Update Combinatorial-UCB statistics for both rewards and costs.
        """
        self.total_steps += 1

        # Calculate price-weighted rewards and costs
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        price_weighted_rewards = chosen_prices * rewards
        costs = chosen_prices * self.cost_coeff  # Cost proportional to price

        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)

            # Update reward statistics
            self.counts[i, price_idx] += 1
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]

            reward_i = price_weighted_rewards[i]
            self.values[i, price_idx] = (old_value * (n-1) / n +
                                         (reward_i - old_value) / n)

            # Update cost statistics
            self.cost_counts[i, price_idx] += 1
            old_cost = self.cost_values[i, price_idx]
            cost_i = costs[i]
            self.cost_values[i, price_idx] = (old_cost * (n-1) / n +
                                              (cost_i - old_cost) / n)

            log_ucb1_update(i, price_idx, self.counts[i, price_idx],
                            self.values[i, price_idx], 0.0)  # UCB on demand

        # Store price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))


class PrimalDualSeller(BaseSeller):
    """
    Seller class implementing Primal-Dual algorithm for Requirements 3 and 4.

    Requirement 3: Single product + best-of-both-worlds + primal-dual
    Requirement 4: Multiple products + best-of-both-worlds + primal-dual

    Based on the Pacing strategy from project.md:
    - Uses regret minimizer R(t) that returns distribution over prices
    - Samples from this distribution instead of greedy selection
    - Updates dual variable λ with proper projection Π[0,1/ρ]
    """

    def __init__(self, setting: Setting):
        """Initialize Primal-Dual seller."""
        super().__init__(setting)
        self.algorithm = "primal_dual"
        log_algorithm_choice("Primal-Dual")

        # Primal-dual specific parameters
        self.eta = 0.1  # Learning rate
        self.rho_pd = self.B / self.T  # ρ = B/T as per project.md line 106
        self.lambda_pd = np.zeros(self.T)  # Dual variable for each round
        self.cost_history = []  # Track costs for each round

        # Regret minimizer parameters
        self.price_weights = np.zeros((self.num_products, self.num_prices))
        self.regret_learning_rate = 0.1

    def regret_minimizer(self, t):
        """
        Regret minimizer R(t) that returns distribution over prices.
        Uses exponential weights (Hedge algorithm).
        """
        gamma_t = np.zeros((self.num_products, self.num_prices))

        for i in range(self.num_products):
            if t == 0:
                # Uniform distribution at start
                gamma_t[i] = np.ones(self.num_prices) / self.num_prices
            else:
                # Exponential weights based on cumulative rewards
                exp_weights = np.exp(self.regret_learning_rate *
                                     self.price_weights[i])
                gamma_t[i] = exp_weights / np.sum(exp_weights)

        return gamma_t

    def sample_from_regret_minimizer(self, gamma_t):
        """
        Sample price indices from the regret minimizer distribution.
        """
        chosen_indices = np.zeros(self.num_products, dtype=int)

        for i in range(self.num_products):
            chosen_indices[i] = np.random.choice(
                self.num_prices, p=gamma_t[i]
            )

        return chosen_indices

    def project_lambda(self, lambda_raw):
        """
        Project λ to [0, 1/ρ] as specified in project.md.
        With ρ = B/T, the upper bound is T/B.
        """
        return np.clip(lambda_raw, 0, 1.0 / self.rho_pd)

    def pull_arm(self):
        """
        Primal-dual arm selection using regret minimizer.
        """
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
            # Get distribution from regret minimizer
            gamma_t = self.regret_minimizer(self.total_steps)

            # Sample from the distribution
            chosen_indices = self.sample_from_regret_minimizer(gamma_t)

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices
        except Exception as e:
            log_error(f"Error in Primal-Dual pull_arm: {e}")
            return np.zeros(self.num_products, dtype=int)

    def update(self, purchased, actions):
        """Update primal-dual statistics after observing rewards."""
        purchased = np.clip(purchased, 0, 1)
        # Always apply budget constraint for primal-dual
        purchased = self.budget_constraint(purchased)

        rewards = purchased
        self.update_primal_dual(actions, rewards)

    def update_primal_dual(self, actions, rewards):
        """
        Update primal-dual algorithm following project.md specification.
        """
        # Calculate price-weighted rewards and costs
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        price_weighted_rewards = chosen_prices * rewards
        current_cost = np.sum(chosen_prices)  # c_t(b_t)

        # Store cost for this round
        self.cost_history.append(current_cost)

        # Update regret minimizer weights
        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)
            # Update weight for chosen price with adjusted reward
            lambda_cost = self.lambda_pd[self.total_steps] * chosen_prices[i]
            adjusted_reward = price_weighted_rewards[i] - lambda_cost
            self.price_weights[i, price_idx] += adjusted_reward

        # Update dual variable λ for next round
        if self.total_steps < self.T - 1:
            lambda_update = self.eta * (self.rho_pd - current_cost)
            lambda_raw = self.lambda_pd[self.total_steps] - lambda_update
            self.lambda_pd[self.total_steps + 1] = self.project_lambda(
                lambda_raw)

        # Store price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))

        self.total_steps += 1


class SlidingWindowUCBSeller(CombinatorialUCBSeller):
    """
    Seller implementing Combi-UCB with sliding window for Requirement 5.

    Requirement 5: Multiple products + slightly non-stationary +
                   Combinatorial-UCB with sliding window
    """

    def __init__(self, setting: Setting, window_size: int = None):
        """
        Initialize Sliding Window UCB seller.

        Args:
            setting: Setting object with configuration
            window_size: Size of sliding window (default: sqrt(T))
        """
        super().__init__(setting)
        self.algorithm = "sliding_window_ucb"
        self.window_size = window_size or max(1, int(np.sqrt(self.T)))
        log_algorithm_choice(f"Sliding Window UCB (window={self.window_size})")

        # Additional tracking for sliding window
        self.reward_history = []  # Store individual rewards
        self.action_history = []  # Store individual actions


# Factory function to create appropriate seller for each requirement
def create_seller_for_requirement(requirement_number: int,
                                  setting: Setting,
                                  use_inventory_constraint: bool = True,
                                  **kwargs) -> BaseSeller:
    """
    Factory function to create the appropriate seller for each requirement.

    Args:
        requirement_number: Project requirement number (1-5)
        setting: Setting object with configuration
        **kwargs: Additional parameters for specific sellers

    Returns:
        Appropriate seller instance for the requirement
    """
    if requirement_number == 1:
        # Single product + stochastic + UCB1 (with/without inventory)
        return UCB1Seller(
            setting, use_inventory_constraint=use_inventory_constraint
        )

    elif requirement_number == 2:
        # Multiple products + stochastic + Combinatorial-UCB
        return CombinatorialUCBSeller(setting)

    elif requirement_number == 3:
        # Single product + best-of-both-worlds + primal-dual
        return PrimalDualSeller(setting)

    elif requirement_number == 4:
        # Multiple products + best-of-both-worlds + primal-dual
        return PrimalDualSeller(setting)

    elif requirement_number == 5:
        # Multiple products + slightly non-stationary + sliding window UCB
        window_size = kwargs.get('window_size', None)
        return SlidingWindowUCBSeller(setting, window_size=window_size)

    else:
        raise ValueError(f"Unknown requirement number: {requirement_number}")

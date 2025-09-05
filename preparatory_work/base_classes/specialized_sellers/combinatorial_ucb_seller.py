"""
Combinatorial UCB Seller for multi-product optimization.
"""
import numpy as np
from .ucb1_seller import UCB1Seller
from ..setting import Setting
from ..logger import (log_algorithm_choice, log_ucb1_update, log_arm_selection)


class CombinatorialUCBSeller(UCB1Seller):
    """
    FIXED Combinatorial-UCB implementation for Requirement 2.

    Requirement 2: Multiple products + stochastic + Combinatorial-UCB

    Based on the UCB-Bidding Algorithm from project.md:
    - Computes UCB bounds for rewards (f_t) and LCB bounds for costs (c_t)
    - Solves LP to get distribution γ_t over price combinations
    - Samples from γ_t instead of greedy selection

    Key fixes:
    - Proper cost calculation (costs = prices * cost_coeff, NOT rewards)
    - Simplified but effective exploration
    - Reasonable temperature scheduling
    - No overly complex budget constraints
    """

    def __init__(self, setting: Setting, cost_coeff: float = 0.01):
        """Initialize Optimized Combinatorial-UCB seller."""
        super().__init__(setting, use_inventory_constraint=True)
        self.algorithm = "combinatorial_ucb"
        log_algorithm_choice("Optimized Combinatorial-UCB")

        # Additional tracking for Combinatorial-UCB
        self.cost_values = np.zeros((self.num_products, self.num_prices))
        self.cost_counts = np.zeros((self.num_products, self.num_prices))
        # Use optimized cost coefficient
        self.cost_coeff = cost_coeff

        # Enhanced learning parameters - scale with problem size
        self.min_exploration_rounds = max(3, int(np.log(setting.T)))
        self.exploration_bonus_factor = 3.0  # Enhanced exploration bonus

        # Optimistic initialization for better convergence
        initial_value = 2.0  # Optimistic initial values
        self.values.fill(initial_value)
        self.cost_values.fill(0.005)  # Small initial cost estimate

        # Algorithm-specific parameters
        self.confidence_scaling = 1.5  # Enhanced confidence bounds
        self.temperature_scaling = 5.0  # Adaptive temperature

    def compute_ucb_lcb_bounds(self):
        """
        Enhanced UCB bounds with improved exploration.
        """
        ucb_rewards = np.zeros((self.num_products, self.num_prices))
        lcb_costs = np.zeros((self.num_products, self.num_prices))

        for i in range(self.num_products):
            for j in range(self.num_prices):
                n = self.counts[i, j]
                if n > 0:
                    # Enhanced confidence bounds for improved learning
                    log_factor = max(1, np.log(self.total_steps + 1))
                    confidence = (self.confidence_scaling *
                                  np.sqrt(2 * log_factor / n))

                    # Enhanced exploration bonus for better convergence
                    if n <= self.min_exploration_rounds:
                        exploration_bonus = (self.exploration_bonus_factor *
                                             np.sqrt(log_factor / (n + 1)))
                        confidence += exploration_bonus

                    ucb_rewards[i, j] = self.values[i, j] + confidence
                    lcb_costs[i, j] = max(0,
                                          self.cost_values[i, j] - confidence)
                else:
                    # Very large bonus for unexplored arms
                    ucb_rewards[i, j] = np.inf
                    lcb_costs[i, j] = 0.0

        return ucb_rewards, lcb_costs

    def solve_lp_for_distribution(self, ucb_rewards, lcb_costs):
        """Enhanced softmax with adaptive temperature scheduling."""
        expected_profits = ucb_rewards - lcb_costs
        gamma_t = np.zeros((self.num_products, self.num_prices))

        # Enhanced temperature scheduling for optimal learning
        # Adaptive decay with maintained exploration
        temperature = max(0.01, self.temperature_scaling /
                          np.log(self.total_steps + 5))

        for i in range(self.num_products):
            profits = expected_profits[i]

            # Handle infinite values properly
            finite_mask = np.isfinite(profits)
            if np.any(finite_mask):
                finite_profits = profits[finite_mask]
                max_finite = np.max(finite_profits)
                profits = np.where(np.isinf(profits), max_finite + 20, profits)
            profits = np.where(np.isnan(profits), 0, profits)

            # Enhanced exploration bonus for better convergence
            for j in range(self.num_prices):
                if self.counts[i, j] <= self.min_exploration_rounds:
                    exploration_factor = np.log(self.total_steps + 1)
                    # Enhanced exploration bonus
                    bonus = 5.0 * np.sqrt(exploration_factor /
                                          (self.counts[i, j] + 1))
                    profits[j] += bonus

            # Temperature-scaled softmax with adaptive convergence
            scaled_profits = profits / temperature
            max_val = np.max(scaled_profits)
            exp_vals = np.exp(scaled_profits - max_val)
            sum_exp = np.sum(exp_vals)

            if sum_exp > 0:
                gamma_t[i] = exp_vals / sum_exp
            else:
                # Uniform distribution as fallback
                gamma_t[i] = np.ones(self.num_prices) / self.num_prices

        return gamma_t

    def sample_from_distribution(self, gamma_t):
        """Sample price indices from the distribution γ_t."""
        chosen_indices = np.zeros(self.num_products, dtype=int)
        for i in range(self.num_products):
            chosen_indices[i] = np.random.choice(self.num_prices, p=gamma_t[i])
        return chosen_indices

    def pull_arm(self):
        """Combinatorial-UCB arm selection following the LP-based approach."""
        def combinatorial_selection():
            ucb_rewards, lcb_costs = self.compute_ucb_lcb_bounds()
            gamma_t = self.solve_lp_for_distribution(ucb_rewards, lcb_costs)
            chosen_indices = self.sample_from_distribution(gamma_t)
            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices
        return self.safe_pull_arm(combinatorial_selection)

    def update(self, purchased, actions):
        """Update Combinatorial-UCB statistics after observing rewards."""
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, self.use_inventory_constraint
        )
        self.update_combinatorial_ucb(actions, purchased)

    def update_combinatorial_ucb(self, actions, rewards):
        """
        FIXED update method with proper cost calculation and budget tracking.
        """
        self.total_steps += 1

        chosen_prices = self.price_grid[np.arange(self.num_products),
                                        actions.astype(int)]
        price_weighted_rewards = chosen_prices * rewards

        costs = chosen_prices * self.cost_coeff  # Simple proportional cost
        total_cost = np.sum(costs)  # Total cost for this round

        self.update_budget(total_cost)

        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)

            # Update reward statistics
            self.counts[i, price_idx] += 1
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]

            reward_i = price_weighted_rewards[i]
            self.values[i, price_idx] = old_value + (reward_i - old_value) / n

            # Update cost statistics
            old_cost = self.cost_values[i, price_idx]
            cost_i = costs[i]  # Use the proper cost, not reward-dependent!
            self.cost_values[i, price_idx] = old_cost + (cost_i - old_cost) / n

            # Log the update
            log_ucb1_update(i, price_idx, self.counts[i, price_idx],
                            self.values[i, price_idx], 0.0)

        # Store price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))

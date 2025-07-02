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
    """

    def __init__(self, setting: Setting):
        """Initialize Combinatorial-UCB seller."""
        super().__init__(setting, use_inventory_constraint=True)
        self.algorithm = "combinatorial_ucb"
        log_algorithm_choice("Combinatorial-UCB")


class PrimalDualSeller(BaseSeller):
    """
    Seller class implementing Primal-Dual algorithm for Requirements 3 and 4.

    Requirement 3: Single product + best-of-both-worlds + primal-dual
    Requirement 4: Multiple products + best-of-both-worlds + primal-dual
    """

    def __init__(self, setting: Setting):
        """Initialize Primal-Dual seller."""
        super().__init__(setting)
        self.algorithm = "primal_dual"
        log_algorithm_choice("Primal-Dual")

        # Primal-dual specific parameters
        self.eta = 0.1  # Learning rate
        self.rho_pd = self.B / self.T  # Step size

    def pull_arm(self):
        """
        Primal-dual arm selection: UCB values adjusted by dual variables.
        """
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
            chosen_indices = np.array([], dtype=int)

            for i in range(self.num_products):
                # Adjust UCB values by lambda for budget constraint
                adjusted_ucbs = (self.ucbs[i] -
                                 self.lambda_pd[self.total_steps] *
                                 self.price_grid[i] * self.cost_coeff)
                idx = int(np.argmax(adjusted_ucbs))
                chosen_indices = np.append(chosen_indices, idx)

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

        # For primal-dual, we still need to pass purchases to the algorithm,
        # but the reward tracking should be price-weighted
        self.update_primal_dual(actions, purchased)


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
        use_inventory = kwargs.get('use_inventory_constraint', True)
        return UCB1Seller(setting, use_inventory_constraint=use_inventory)

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

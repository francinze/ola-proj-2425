"""
Sliding Window UCB Seller for non-stationary environments.
"""
import numpy as np
from collections import deque
from .combinatorial_ucb_seller import CombinatorialUCBSeller
from ..setting import Setting
from ..logger import log_algorithm_choice, log_arm_selection


class SlidingWindowUCB1Seller(CombinatorialUCBSeller):
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

        # Sliding window storage using deque for efficient operations
        # stores tuples of (actions, rewards)
        self.history = deque(maxlen=self.window_size)

        # Reset counts and values for sliding window approach
        # Avoid divide-by-zero
        self.counts = np.ones((self.num_products, self.num_prices)) * 1e-3
        self.values = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)

    def pull_arm(self):
        """
        Sliding window UCB arm selection.
        Uses only the data from the sliding window to make decisions.
        """
        def sliding_window_selection():
            # If history is empty, return random actions for exploration
            if len(self.history) == 0:
                chosen_indices = np.random.randint(
                    0, self.num_prices, size=self.num_products
                )
                log_arm_selection(
                    self.algorithm, self.total_steps, chosen_indices
                )
                return chosen_indices

            # Recalculate statistics from sliding window
            self._recalculate_from_window()

            # Use UCB selection
            chosen_indices = np.zeros(self.num_products, dtype=int)
            for i in range(self.num_products):
                # Check for unvisited arms in the current window
                unvisited_arms = np.where(self.counts[i] <= 1e-3)[0]

                if len(unvisited_arms) > 0:
                    # Explore unvisited arms first
                    chosen_indices[i] = unvisited_arms[0]
                else:
                    # Use UCB selection
                    chosen_indices[i] = np.argmax(self.ucbs[i])

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices

        return self.safe_pull_arm(sliding_window_selection)

    def _recalculate_from_window(self):
        """
        Recalculate UCB statistics from the sliding window data.
        """
        # Reset counts and values
        self.counts = np.ones((self.num_products, self.num_prices)) * 1e-3
        self.values = np.zeros((self.num_products, self.num_prices))

        # Accumulate statistics from sliding window
        for actions, rewards in self.history:
            for i, (action, reward) in enumerate(zip(actions, rewards)):
                if i >= self.num_products:
                    continue
                action = int(action)
                if action >= self.num_prices:
                    continue

                self.counts[i, action] += 1
                self.values[i, action] += reward

        # Calculate means and UCB values
        means = np.divide(
            self.values, self.counts,
            out=np.zeros_like(self.values),
            where=self.counts > 0
        )
        total_counts = np.sum(self.counts)

        # Calculate UCB bounds
        with np.errstate(divide='ignore', invalid='ignore'):
            exploration_term = np.sqrt(
                2 * np.log(total_counts + 1) / self.counts
            )
            self.ucbs = means + exploration_term
            self.ucbs = np.nan_to_num(
                self.ucbs, nan=0.0, posinf=1e6, neginf=0.0
            )

        # Set infinite UCB for unvisited arms
        self.ucbs[self.counts <= 1e-3] = np.inf

        # Update average values
        self.values = np.divide(
            self.values, self.counts,
            out=np.zeros_like(self.values),
            where=self.counts > 0
        )

    def update(self, purchased, actions):
        """
        Update sliding window UCB statistics after observing rewards with
        budget tracking.
        """
        # Apply constraints and get processed rewards
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, self.use_inventory_constraint
        )

        # Calculate price-weighted rewards
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        price_weighted_rewards = chosen_prices * purchased

        # Calculate costs for budget tracking
        costs = chosen_prices * 0.1  # 10% of price as cost
        total_cost = np.sum(costs)

        # UPDATE BUDGET TRACKING - Available from base Seller class
        self.update_budget(total_cost)

        # Update sliding window
        self.history.append((actions.copy(), price_weighted_rewards.copy()))

        # Store in history for logging
        self.reward_history.append(price_weighted_rewards.copy())
        self.action_history.append(actions.copy())
        self.history_rewards.append(np.sum(price_weighted_rewards))

        # Recalculate statistics from current window
        self._recalculate_from_window()

        self.total_steps += 1

    def reset(self, setting):
        """
        Reset the sliding window UCB seller for a new trial.
        """
        super().reset(setting)

        # Reset sliding window specific parameters
        self.window_size = getattr(
            setting, 'sliding_window_size', max(1, int(np.sqrt(self.T)))
        )
        self.history = deque(maxlen=self.window_size)
        self.reward_history = []
        self.action_history = []

        # Reset UCB statistics
        self.counts = np.ones((self.num_products, self.num_prices)) * 1e-3
        self.values = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)

        log_algorithm_choice(
            f"Reset Sliding Window UCB (window={self.window_size})"
        )

    def get_diagnostics(self):
        """
        Get diagnostic information about the sliding window UCB algorithm.
        """
        diagnostics = {
            'window_size': self.window_size,
            'current_window_length': len(self.history),
            'total_actions_taken': len(self.action_history),
            'window_utilization': (
                len(self.history) / self.window_size
                if self.window_size > 0 else 0
            ),
            'average_reward_in_window': (
                np.mean([np.sum(rewards) for _, rewards in self.history])
                if self.history else 0
            ),
            'counts_matrix_shape': self.counts.shape,
            'values_matrix_shape': self.values.shape,
            'ucbs_matrix_shape': self.ucbs.shape
        }
        return diagnostics

"""
Specialized seller classes for different project requirements.
Each class extends the base Seller class with specific algorithms.
"""
import numpy as np
from .seller import Seller
from .setting import Setting
from .logger import (log_algorithm_choice,
                     log_ucb1_update, log_arm_selection)


class UCBBaseSeller(Seller):
    """
    Base class for UCB-based sellers with common functionality.
    """

    def __init__(self, setting: Setting,
                 use_inventory_constraint: bool = True):
        """Initialize UCB base seller."""
        super().__init__(setting)
        self.use_inventory_constraint = use_inventory_constraint

        # Common UCB tracking variables
        self.total_rewards = np.zeros((self.num_products, self.num_prices))
        self.avg_rewards = np.zeros((self.num_products, self.num_prices))

    def update_ucb_statistics(self, actions, price_weighted_rewards):
        """
        Common UCB statistics update method with budget tracking.
        """
        self.total_steps += 1

        # Calculate costs for budget tracking
        chosen_prices = self.price_grid[np.arange(self.num_products),
                                        actions.astype(int)]
        # Use a simple cost model: cost proportional to chosen prices
        costs = chosen_prices * 0.1  # 10% of price as cost
        total_cost = np.sum(costs)
        
        # UPDATE BUDGET TRACKING - Available from base Seller class
        self.update_budget(total_cost)

        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)

            # Update counts and rewards
            self.counts[i, price_idx] += 1
            self.total_rewards[i, price_idx] += price_weighted_rewards[i]

            # Incremental average update (numerically stable)
            self.avg_rewards[i, price_idx] = (
                self.total_rewards[i, price_idx] / self.counts[i, price_idx]
            )

            # Update UCB values for this arm
            if self.counts[i, price_idx] > 0:
                confidence_bound = np.sqrt(
                    2 * np.log(self.total_steps) / self.counts[i, price_idx]
                )
                self.ucbs[i, price_idx] = (
                    self.avg_rewards[i, price_idx] + confidence_bound
                )

            # Also update the values array for compatibility with base class
            self.values[i, price_idx] = self.avg_rewards[i, price_idx]

            log_ucb1_update(i, price_idx, self.counts[i, price_idx],
                            self.values[i, price_idx],
                            self.ucbs[i, price_idx])

        # Store total price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))


class UCB1Seller(UCBBaseSeller):
    """
    Seller class implementing UCB1 algorithm for Requirements 1 and 2.

    This implementation is based on the high-performing UCB1 algorithm from
    the demo notebook, adapted to work with the project's base classes.

    Requirement 1: Single product + stochastic + UCB1 (with/without inventory)
    Requirement 2: Multiple products + stochastic + Combinatorial-UCB
    """

    def __init__(
        self, setting: Setting, use_inventory_constraint: bool = True
    ):
        """
        Initialize UCB1 seller with optimized parameters.

        Args:
            setting: Setting object with configuration
            use_inventory_constraint: Whether to enforce inventory constraints
        """
        super().__init__(setting, use_inventory_constraint)
        self.algorithm = "ucb1"

        # UCB1-specific tracking variables
        self.last_chosen_arms = np.zeros(self.num_products, dtype=int)

        log_algorithm_choice(
            f"UCB1 (inventory_constraint={use_inventory_constraint})"
        )

    def pull_arm(self):
        """
        UCB1 arm selection with optimized exploration strategy.

        For each product:
        1. First, explore all arms once (initialization phase)
        2. Then use UCB1 formula: μ_i + √(2ln(t)/n_i)
        """
        def ucb1_selection():
            chosen_indices = np.zeros(self.num_products, dtype=int)

            for i in range(self.num_products):
                # Initialization phase: explore all arms at least once
                unexplored_arms = np.where(self.counts[i] == 0)[0]

                if len(unexplored_arms) > 0:
                    # Choose first unexplored arm
                    chosen_indices[i] = unexplored_arms[0]
                else:
                    # UCB1 phase: choose arm with highest UCB value
                    ucb_values = self.avg_rewards[i] + np.sqrt(
                        2 * np.log(self.total_steps + 1) / self.counts[i]
                    )
                    chosen_indices[i] = np.argmax(ucb_values)

                self.last_chosen_arms[i] = chosen_indices[i]

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices

        return self.safe_pull_arm(ucb1_selection)

    def update(self, purchased, actions):
        """
        Update UCB1 statistics after observing rewards.

        This method handles the reward calculation and constraint application
        before updating the UCB1 algorithm state.
        """
        # Apply constraints and get processed rewards
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, self.use_inventory_constraint
        )

        # Calculate price-weighted rewards (price × purchase indicator)
        price_weighted_rewards = self.calculate_price_weighted_rewards(
            actions, purchased)

        # Update UCB1 algorithm with the computed rewards
        self.update_ucb_statistics(actions, price_weighted_rewards)


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
        
        # FIXED: Proper cost calculation
        # Cost should be independent of rewards - use a reasonable model
        costs = chosen_prices * self.cost_coeff  # Simple proportional cost
        total_cost = np.sum(costs)  # Total cost for this round
        
        # UPDATE BUDGET TRACKING - Available from base Seller class
        self.update_budget(total_cost)
        
        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)
            
            # Update reward statistics
            self.counts[i, price_idx] += 1
            n = self.counts[i, price_idx]
            old_value = self.values[i, price_idx]
            
            reward_i = price_weighted_rewards[i]
            self.values[i, price_idx] = old_value + (reward_i - old_value) / n
            
            # Update cost statistics - FIXED VERSION
            old_cost = self.cost_values[i, price_idx]
            cost_i = costs[i]  # Use the proper cost, not reward-dependent!
            self.cost_values[i, price_idx] = old_cost + (cost_i - old_cost) / n

            # Log the update
            log_ucb1_update(i, price_idx, self.counts[i, price_idx],
                            self.values[i, price_idx], 0.0)
        
        # Store price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))


class PrimalDualSeller(Seller):
    """
    Primal–dual pacing agent with EXP3.P as the primal regret minimizer.

    Changes from previous version:
      - Dynamic pacing: rho_t = remaining_budget / (T - t)
      - Dual step (dual_lr) is tunable and uses rho_t in the gradient and
        projection
      - EXP3.P gain scaling uses current rho_t for stability

    API remains unchanged:
      choose_arm() -> int or None
      update(arm_index, reward)  # reward = price if sale else 0
    """

    def __init__(self, setting: Setting, eta=None, gamma=None, alpha=None,
                 dual_lr=None, rng=None):
        """Initialize PrimalDualSeller based on PrimalDualExp3PAgent."""
        super().__init__(setting)
        self.algorithm = "primal_dual"
        log_algorithm_choice("Primal-Dual EXP3.P")

        # Convert price grid to flat array for compatibility
        if self.price_grid.ndim > 1:
            self.prices = self.price_grid[0]
        else:
            self.prices = self.price_grid
        self.K = len(self.prices)
        
        # EXP3.P parameters (defaults are safe; tune in constructor if desired)
        if gamma is None:
            gamma = min(0.2, np.sqrt(self.K * np.log(max(2, self.K)) /
                                     ((np.e - 1) * max(1, self.T))))
        if eta is None:
            eta = min(0.2, np.sqrt(np.log(max(2, self.K)) /
                                   (self.K * max(1, self.T))))
        if alpha is None:
            alpha = gamma / self.K
        if dual_lr is None:
            dual_lr = 0.2  # more reactive than 1/sqrt(T); tune as needed

        self.eta = float(eta)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.dual_lr = float(dual_lr)

        # Internal state
        self.weights = np.ones(self.K, dtype=float)
        self.last_probs = np.full(self.K, 1.0 / self.K)
        self.last_arm = None
        self.lmbda = 0.0

        self.total_steps = 0
        self.remaining_budget = int(self.B)
        self.total_reward = 0.0

        self.rng = np.random.default_rng(rng)

    # ---------- helpers ----------
    def _probs(self):
        w = self.weights
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.ones_like(w)
            w_sum = w.sum()
        base = (1.0 - self.gamma) * (w / w_sum)
        mix = self.gamma / self.K
        p = base + mix
        p = np.clip(p, 1e-12, 1.0)
        p /= p.sum()
        return p

    def _rho_t_and_L(self):
        """
        Compute dynamic pacing target rho_t and its scaling L_t = 1/rho_t
        (capped).
        """
        # avoid div by zero at the very end
        rounds_left = max(1, self.T - self.total_steps)
        # allowed expected spend per remaining round
        rho_t = self.remaining_budget / rounds_left
        # Guard: if budget is 0 -> rho_t = 0; we still avoid division by zero
        # by capping L_t.
        if rho_t <= 0.0:
            # effectively infinite penalty scale when no budget remains
            L_t = 1e6
        else:
            L_t = 1.0 / rho_t
        return rho_t, L_t

    # ---------- public API ----------
    def pull_arm(self):
        """
        Select a price index using EXP3.P algorithm.
        Returns: array with single chosen price index for compatibility
        """
        def primal_dual_selection():
            # Stop if out of budget or rounds
            if self.remaining_budget <= 0 or self.total_steps >= self.T:
                return np.array([0])  # Return array for compatibility
            
            p = self._probs()
            a = int(self.rng.choice(self.K, p=p))
            self.last_probs = p
            self.last_arm = a
            
            log_arm_selection(self.algorithm, self.total_steps, [a])
            return np.array([a])  # Return as array for compatibility

        return self.safe_pull_arm(primal_dual_selection)

    def update(self, purchased, actions):
        """
        Update statistics after observing rewards.
        
        Args:
            purchased: Array of purchase outcomes per product
            actions: Array of chosen price indices per product
        """
        # Apply constraints and get processed rewards
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, use_inventory_constraint=True
        )
        
        # For single product case, get first element
        if len(actions) > 0:
            arm_index = int(actions[0])
            reward = self.prices[arm_index] if purchased[0] > 0 else 0.0
            self.update_primal_dual_exp3p(arm_index, reward)

    def update_primal_dual_exp3p(self, arm_index, reward):
        """
        reward: realized payoff (price if sale occurred else 0).
        Sale (cost=1) is inferred as (reward > 0).
        """
        if arm_index is None:
            return

        # Capture pacing BEFORE applying the cost of this round
        rho_t, L_t = self._rho_t_and_L()

        # Realized outcome
        sale = (reward > 0)
        cost = 1 if sale else 0
        self.total_reward += float(reward)

        # Penalized gain g_t = f - lambda * c, scaled to [0,1] using
        # current L_t
        g = float(reward) - self.lmbda * float(cost)
        g_tilde = (g + L_t) / (1.0 + L_t)
        g_tilde = float(np.clip(g_tilde, 0.0, 1.0))

        # EXP3.P update (bandit IW estimate with bias alpha)
        p_arm = float(self.last_probs[arm_index])
        est = (g_tilde + self.alpha) / max(p_arm, 1e-12)
        self.weights[arm_index] *= np.exp(self.eta * est)

        # Dual update with dynamic pacing (project onto [0, 1/rho_t])
        # Note: use rho_t computed BEFORE applying this round's cost.
        self.lmbda = self.lmbda - self.dual_lr * (rho_t - cost)
        lambda_max = L_t  # since L_t = 1/rho_t (or large cap if rho_t ~ 0)
        self.lmbda = float(np.clip(self.lmbda, 0.0, lambda_max))

        # Apply budget + time progression
        if sale and self.remaining_budget > 0:
            self.remaining_budget -= 1
            
        # Update budget tracking for base class compatibility
        # Small cost model
        price_cost = self.prices[arm_index] * 0.01 if sale else 0.0
        self.update_budget(price_cost)
        
        # Store rewards in history for compatibility
        self.history_rewards.append(reward)
        
        self.total_steps += 1

    # Optional inspectors
    def current_probs(self):
        return self._probs().copy()

    def get_diagnostics(self):
        """Get diagnostic information about the primal-dual algorithm."""
        rho_t, _ = self._rho_t_and_L()
        return {
            "t": self.total_steps,
            "remaining_budget": self.remaining_budget,
            "lambda": self.lmbda,
            "rho_t": rho_t,
            "probs": self._probs(),
            "weights": self.weights.copy(),
            "total_reward": self.total_reward,
        }

    def reset(self, setting):
        """Reset the primal-dual seller's statistics for a new trial."""
        super().reset(setting)
        
        # Reset primal-dual specific parameters
        if self.price_grid.ndim > 1:
            self.prices = self.price_grid[0]
        else:
            self.prices = self.price_grid
        self.K = len(self.prices)
        self.weights = np.ones(self.K, dtype=float)
        self.last_probs = np.full(self.K, 1.0 / self.K)
        self.last_arm = None
        self.lmbda = 0.0
        self.total_steps = 0
        self.remaining_budget = int(self.B)
        self.total_reward = 0.0
        
        log_algorithm_choice("Reset Primal-Dual EXP3.P")


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
        from collections import deque
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
        from collections import deque
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

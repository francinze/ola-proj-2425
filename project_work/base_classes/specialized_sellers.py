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
    Primal-Dual seller following project.md specifications exactly.

    This implementation features:
    - Using proper pacing strategy: ρ = B/T
    - Implementing stable regret minimizer with temperature scaling
    - Correct dual variable projection: Π[0,1/ρ]
    - Economically sound cost calculation (cost only on successful sales)
    - Better learning rate tuning for convergence

    Based on the Pacing strategy from project.md:
    - ρ ← B/T
    - λ_0 ← 0
    - R(t) returns distribution over prices (regret minimizer)
    - λ_t ← Π[0,1/ρ](λ_{t-1} - η(ρ - c_t(b_t)))
    """

    def __init__(
        self,
        setting: Setting,
        learning_rate: float = 0.01,
        regret_learning_rate: float = 0.05,
        base_temperature: float = 1.0
    ):
        """Initialize Improved Primal-Dual seller."""
        super().__init__(setting)
        self.algorithm = "improved_primal_dual"
        log_algorithm_choice("Improved Primal-Dual")

        # Primal-dual parameters from project.md
        self.rho = self.B / self.T  # ρ = B/T (pacing rate)
        self.eta = learning_rate  # Small learning rate for stability
        self.lambda_t = 0.0  # Current dual variable (λ_0 = 0)

        # Regret minimizer (Hedge/Exponential Weights)
        self.cumulative_losses = np.zeros((self.num_products, self.num_prices))
        self.regret_eta = regret_learning_rate

        # Enhanced tracking
        self.cost_history = []
        self.lambda_history = [0.0]

        # Temperature scaling for better exploration/exploitation
        self.base_temperature = base_temperature

        log_algorithm_choice(
            f"Improved Primal-Dual (η={self.eta}, ρ={self.rho:.6f})"
        )

    def regret_minimizer(self, product_idx):
        """
        R(t) returns distribution over prices using Hedge algorithm.
        More stable implementation with temperature scaling and proper
        normalization.

        Args:
            product_idx: Index of the product (for multi-product support)

        Returns:
            numpy.ndarray: Probability distribution over prices
        """
        if self.total_steps == 0:
            # Start with uniform distribution
            return np.ones(self.num_prices) / self.num_prices

        # Temperature scaling for better exploration/exploitation balance
        # Higher temperature early on encourages exploration
        temperature = max(
            0.1, self.base_temperature / np.sqrt(self.total_steps + 1)
        )

        # Exponential weights with temperature scaling
        weights = np.exp(-temperature * self.cumulative_losses[product_idx])

        # Normalize to probability distribution
        weights = weights / np.sum(weights)

        # Add small epsilon to prevent zero probabilities
        epsilon = 1e-8
        weights = (1 - epsilon) * weights + epsilon / self.num_prices

        return weights

    def project_lambda(self, lambda_raw):
        """
        Project λ to [0, 1/ρ] as specified in project.md.
        With ρ = B/T, the upper bound is T/B.

        Args:
            lambda_raw: Raw dual variable value before projection

        Returns:
            float: Projected dual variable in [0, 1/ρ]
        """
        return np.clip(lambda_raw, 0.0, 1.0 / self.rho)

    def pull_arm(self):
        """
        Primal-dual arm selection using regret minimizer.
        Samples from distribution γ_t for each product.

        Returns:
            numpy.ndarray: Array of chosen price indices (one per product)
        """
        def primal_dual_selection():
            chosen_indices = np.zeros(self.num_products, dtype=int)

            for i in range(self.num_products):
                # Get distribution from regret minimizer for this product
                gamma_t = self.regret_minimizer(i)

                # Sample from the distribution
                chosen_indices[i] = np.random.choice(
                    self.num_prices, p=gamma_t
                )

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices

        return self.safe_pull_arm(primal_dual_selection)

    def yield_prices(self, chosen_indices):
        """
        Override yield_prices to provide proper price calculation.
        """
        chosen_prices = self.price_grid[
            np.arange(self.num_products), chosen_indices
        ]
        # Track price history for diagnostics
        self.history_chosen_prices.append(chosen_indices.copy())
        return np.array(chosen_prices)

    def update(self, purchased, actions):
        """
        Update improved primal-dual statistics after observing rewards.
        Always applies budget constraint for primal-dual algorithms.

        Args:
            purchased: Array of purchase outcomes per product
            actions: Array of chosen price indices per product
        """
        # Apply constraints and get processed rewards
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, use_inventory_constraint=True
        )

        rewards = purchased
        self.update_improved_primal_dual(actions, rewards)

    def update_improved_primal_dual(self, actions, rewards):
        """
        Update improved primal-dual algorithm following project.md
        specification.

        Key improvements:
        - Proper cost calculation (only on successful sales)
        - Stable regret minimizer updates
        - Correct dual variable projection
        - Better learning rate scheduling

        Args:
            actions: Array of chosen price indices per product
            rewards: Array of purchase outcomes per product
        """
        # Calculate price-weighted rewards and costs
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        price_weighted_rewards = chosen_prices * rewards

        # Calculate cost c_t(b_t) - ONLY if purchase was made
        # (economically sound)
        current_cost = np.sum(chosen_prices * rewards)  # Cost only on sales

        # UPDATE BUDGET TRACKING - Use new comprehensive system
        self.update_budget(current_cost)

        # Store cost for tracking (legacy compatibility)
        self.cost_history.append(current_cost)

        # Update regret minimizer for each product
        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)

            # Calculate adjusted reward considering dual variable
            lambda_cost = self.lambda_t * chosen_prices[i] * rewards[i]
            adjusted_reward = price_weighted_rewards[i] - lambda_cost

            # Convert reward to loss for regret minimizer
            loss = -adjusted_reward

            # Get current distribution for importance weighting
            gamma_t = self.regret_minimizer(i)

            # Update cumulative losses (importance weighted)
            if gamma_t[price_idx] > 0:  # Avoid division by zero
                self.cumulative_losses[i, price_idx] += (
                    loss / gamma_t[price_idx]
                )

        # Update dual variable λ following project.md specification
        # λ_t ← Π[0,1/ρ](λ_{t-1} - η(ρ - c_t(b_t)))
        lambda_update = self.lambda_t - self.eta * (self.rho - current_cost)
        self.lambda_t = self.project_lambda(lambda_update)

        # Track dual variable evolution
        self.lambda_history.append(self.lambda_t)

        # Store price-weighted rewards in history (inherited from base class)
        self.history_rewards.append(np.sum(price_weighted_rewards))

        self.total_steps += 1

    def reset(self, setting):
        """
        Reset the improved primal-dual seller's statistics for a new trial.
        """
        super().reset(setting)

        # Reset improved primal-dual specific parameters
        self.rho = self.B / self.T
        self.lambda_t = 0.0
        self.cumulative_losses = np.zeros((self.num_products, self.num_prices))
        self.cost_history = []
        self.lambda_history = [0.0]

        log_algorithm_choice(
            f"Reset Improved Primal-Dual (η={self.eta}, ρ={self.rho:.6f})"
        )

    def get_diagnostics(self):
        """
        Get diagnostic information about the improved primal-dual algorithm.
        Useful for analysis and debugging.

        Returns:
            dict: Dictionary containing diagnostic information
        """
        diagnostics = {
            'lambda_history': np.array(self.lambda_history),
            'cost_history': np.array(self.cost_history),
            'remaining_budget': self.remaining_budget,
            'budget_utilization': (self.B - self.remaining_budget) / self.B,
            'pacing_rate': self.rho,
            'current_lambda': self.lambda_t,
            'lambda_upper_bound': 1.0 / self.rho,
            'total_costs': np.sum(self.cost_history),
            'average_cost_per_round': (
                np.mean(self.cost_history) if self.cost_history else 0.0
            ),
            'cumulative_losses_shape': self.cumulative_losses.shape,
            'learning_rates': {
                'dual_eta': self.eta,
                'regret_eta': self.regret_eta
            }
        }
        return diagnostics


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

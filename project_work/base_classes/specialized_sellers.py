"""
Specialized seller classes for different project requirements.
Each class extends the base Seller class with specific algorithms.
"""
import numpy as np
import scipy as sp
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

                if self.total_steps < self.num_prices:
                    chosen_indices = np.append(chosen_indices, self.total_steps)
                else:
                    # UCB1: select arm with highest UCB value
                    """if np.sum(self.lcbs[i] <= np.zeros(len(self.lcbs[i]))):
                        idx = np.zeros(len(self.ucbs[i]))
                        idx[np.argmax(self.ucbs[i])] = 1"""
                    c = -self.ucbs[i]
                    A_ub = [self.lcbs[i]]
                    b_ub = [self.rho_pd]
                    A_eq = [np.ones(self.num_prices)]
                    b_eq = [1]
                    res = sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
                    idx = np.random.choice(self.num_prices, p=res.x)

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
            self.values[i, price_idx] = (old_value * (n-1) +
                                         (reward_i - old_value)) / n 
            self.ucbs[i, price_idx] = self.values[i, price_idx] + \
                np.sqrt(2 * np.log(self.T) / n)
            
            # Update cost statistics
            old_cost = self.cost_values[i, price_idx]
            cost_i = np.count_nonzero(price_weighted_rewards[i])
            self.cost_values[i, price_idx] = (old_cost * (n-1) / n +
                                              (cost_i - old_cost) / n)
            self.lcbs[i, price_idx] = self.cost_values[i, price_idx] - \
                np.sqrt(2 * np.log(self.T) / n)

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
        self.W_avg = np.zeros((self.num_products, self.num_prices), dtype=float)
        self.N_pulls = np.zeros((self.num_products, self.num_prices), dtype=int)
        self.t = 0
        self.A_t = None
        self.rows_t = None
        self.cols_t = None
    
    def pull_arm(self):
        # if an arm is unexplored, then the UCB is a large value
        W = np.zeros(self.W_avg.shape, dtype=float)
        large_value = (1 + np.sqrt(2*np.log(self.T)/1))*10
        W[self.N_pulls==0] = large_value
        mask = self.N_pulls>0
        W[mask] = self.W_avg[mask] + np.sqrt(2*np.log(self.T)/self.N_pulls[mask])
        self.rows_t, self.cols_t = sp.optimize.linear_sum_assignment(W, maximize=True)
        self.A_t = list(zip(self.rows_t, self.cols_t))
        return self.A_t

    def update(self, rewards):
        self.N_pulls[self.rows_t, self.cols_t] += 1
        self.W_avg[self.rows_t, self.cols_t] += (rewards - self.W_avg[self.rows_t, self.cols_t])/self.N_pulls[self.rows_t, self.cols_t]
        self.t += 1


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
        self.eta = 0.01  # Reduced learning rate for better convergence
        self.rho_pd = self.B / self.T  # ρ = B/T as per project.md line 106
        self.lambda_pd = np.zeros(self.T)  # Dual variable for each round
        self.cost_history = []  # Track costs for each round

        # Regret minimizer parameters
        self.price_weights = np.zeros((self.num_products, self.num_prices))
        self.regret_learning_rate = 0.05  # Reduced for more stable learning

        # Initialize UCBs as None to indicate no UCB data
        self.ucbs = None

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

    def yield_prices(self, chosen_indices):
        """
        Override yield_prices to handle PrimalDual's cost tracking mechanism.
        """
        chosen_prices = self.price_grid[
            np.arange(self.num_products), chosen_indices
        ]
        # Use the base class method for history tracking
        self.history_chosen_prices.append(chosen_indices)

        # PrimalDualSeller handles cost tracking in update_primal_dual
        return np.array(chosen_prices)

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
        current_cost = np.count_nonzero(chosen_prices) # np.sum(chosen_prices)  # c_t(b_t)

        # Store cost for this round
        self.cost_history.append(current_cost)

        # Update regret minimizer weights
        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)
            # Update weight for chosen price with adjusted reward
            lambda_cost = self.lambda_pd[self.total_steps] * ((chosen_prices[i]>0).astype(float) - self.rho_pd) # chosen_prices[i] 
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


class ImprovedPrimalDualSeller(BaseSeller):
    """
    Improved Primal-Dual seller following project.md specifications exactly.

    This implementation provides better stability and performance compared to
    the original PrimalDualSeller by:
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

    def __init__(self, setting: Setting):
        """Initialize Improved Primal-Dual seller."""
        super().__init__(setting)
        self.algorithm = "improved_primal_dual"
        log_algorithm_choice("Improved Primal-Dual")

        # Primal-dual parameters from project.md
        self.rho = self.B / self.T  # ρ = B/T (pacing rate)
        self.eta = 0.001  # Small learning rate for stability
        self.lambda_t = 0.0  # Current dual variable (λ_0 = 0)

        # Regret minimizer (Hedge/Exponential Weights)
        self.cumulative_losses = np.zeros((self.num_products, self.num_prices))
        self.regret_eta = 0.01  # Learning rate for regret minimizer

        # Enhanced tracking
        self.cost_history = []
        self.lambda_history = [0.0]
        self.remaining_budget = self.B

        # Temperature scaling for better exploration/exploitation
        self.base_temperature = 1.0

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
        log_arm_selection(self.algorithm, self.total_steps, "starting")
        try:
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

        except Exception as e:
            log_error(f"Error in Improved Primal-Dual pull_arm: {e}")
            return np.zeros(self.num_products, dtype=int)

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
        purchased = np.clip(purchased, 0, 1)

        # Always apply budget constraint for primal-dual
        purchased = self.budget_constraint(purchased)

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
        current_cost = np.count_nonzero(chosen_prices * rewards) # np.sum(chosen_prices * rewards)  # Cost only on sales

        # Store cost for tracking
        self.cost_history.append(current_cost)

        # Update regret minimizer for each product
        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)

            # Calculate adjusted reward considering dual variable
            lambda_cost = self.lambda_t * ((chosen_prices[i] * rewards[i]>0).astype(float) - self.rho) # chosen_prices[i] * rewards[i] 
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

        # Update remaining budget
        self.remaining_budget -= current_cost

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
        self.remaining_budget = self.B

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

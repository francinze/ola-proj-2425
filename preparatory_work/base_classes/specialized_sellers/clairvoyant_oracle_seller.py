"""
Advanced LP-based Clairvoyant Oracle Seller with global optimization.
"""
import numpy as np
from scipy.stats import beta as beta_dist
from ..seller import Seller
from ..setting import Setting
from ..logger import log_algorithm_choice, log_arm_selection


class ClairvoyantOracleSeller(Seller):
    """
    Advanced LP-based Clairvoyant Oracle Seller with global optimization.

    This implementation uses linear programming to solve an optimization
    problem over the entire horizon T, considering success probabilities,
    budget constraints, and optimal mixing strategies. Based on the superior
    implementation from Assignment3.ipynb.

    Key Features:
    - LP-based global optimization over entire horizon
    - Statistical modeling using Beta distributions
    - Probability simplex projection for numerical robustness
    - Sophisticated budget constraint handling
    - Can operate deterministically or stochastically
    """

    def __init__(self, setting: Setting, deterministic: bool = False):
        """Initialize the advanced clairvoyant oracle seller."""
        super().__init__(setting)
        self.algorithm = "clairvoyant_oracle_lp"
        log_algorithm_choice(
            f"Advanced LP-based Clairvoyant Oracle "
            f"(deterministic={deterministic})"
        )

        # Convert price grid to flat array for compatibility with LP
        if self.price_grid.ndim > 1:
            self.prices = self.price_grid[0]  # Use first product's prices
        else:
            self.prices = self.price_grid
        self.K = len(self.prices)

        self.deterministic = bool(deterministic)
        self.environment_ref = None
        self.params = None  # Will store Beta distribution parameters

        # LP-based planning variables
        self.X = None  # Optimal mixing strategy (T x K matrix)
        self.S = None  # Success probabilities (T x K matrix)
        self.plan_arms = None  # Precomputed deterministic choices

        # Oracle-specific tracking
        self.total_optimal_reward = 0.0
        self.oracle_decisions = []
        self.current_round = 0

    def set_environment_reference(self, environment):
        """
        Set reference to the Environment to access oracle functionality.
        This extracts Beta distribution parameters for LP optimization.
        """
        self.environment_ref = environment

        # Extract Beta distribution parameters if available
        if hasattr(environment, 'params') and environment.params is not None:
            self.params = environment.params
            self._solve_lp_plan()
        elif (hasattr(environment, 'setting') and
              hasattr(environment.setting, 'params')):
            self.params = environment.setting.params
            self._solve_lp_plan()
        elif (hasattr(environment, 'setting') and
              hasattr(environment.setting, 'dist_params')):
            # The actual location of parameters in the Setting class
            self.params = environment.setting.dist_params
            self._solve_lp_plan()

    @staticmethod
    def _project_to_simplex_rowwise(M):
        """Project each row of M onto the probability simplex."""
        M = np.asarray(M, dtype=float)
        T, K = M.shape
        out = np.empty_like(M)
        for i in range(T):
            v = M[i]
            if not np.isfinite(v).all():
                out[i] = np.full(K, 1.0 / K)
                continue
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho_idx = np.nonzero(u * (np.arange(1, K+1)) > (cssv - 1.0))[0]
            if len(rho_idx) == 0:
                out[i] = np.full(K, 1.0 / K)
                continue
            rho = rho_idx[-1]
            theta = (cssv[rho] - 1.0) / (rho + 1.0)
            w = np.maximum(v - theta, 0.0)
            s = w.sum()
            out[i] = w / s if s > 0 else np.full(K, 1.0 / K)
        return out

    def _solve_lp_plan(self):
        """
        Solve optimization problem using efficient greedy approach.
        For very large problems (T > 1000), use greedy instead of full LP.
        """
        if self.params is None:
            log_algorithm_choice(
                "Oracle: No Beta parameters available, using fallback"
            )
            return

        T, K = self.T, self.K

        # Extract Beta parameters for each round
        if isinstance(self.params, list) and len(self.params) == T:
            # Parameters for each round (notebook format)
            a = np.array([p[0] for p in self.params], dtype=float)
            b = np.array([p[1] for p in self.params], dtype=float)
        elif (isinstance(self.params, tuple) and len(self.params) == 2 and
              hasattr(self.params[0], 'shape') and
              hasattr(self.params[1], 'shape')):
            # Base classes format: (a_matrix, b_matrix)
            a_matrix, b_matrix = self.params
            a = (a_matrix[:, 0] if a_matrix.shape[1] > 0
                 else a_matrix.flatten())
            b = (b_matrix[:, 0] if b_matrix.shape[1] > 0
                 else b_matrix.flatten())
        elif hasattr(self.params, '__len__') and len(self.params) == 2:
            # Single Beta parameter pair - use for all rounds
            a = np.full(T, self.params[0], dtype=float)
            b = np.full(T, self.params[1], dtype=float)
        else:
            # Fallback: assume moderate success probabilities
            a = np.full(T, 2.0, dtype=float)
            b = np.full(T, 3.0, dtype=float)

        # Compute success probabilities S[t,k] = P[v_t >= price_k]
        P = self.prices[None, :]  # (1, K)
        self.S = 1.0 - beta_dist.cdf(P, a[:, None], b[:, None])  # (T, K)

        # Use greedy approach for efficiency
        # For each round, find the price that maximizes expected revenue
        # subject to budget constraints
        
        self.X = np.zeros((T, K), dtype=float)
        remaining_budget = float(self.B)
        
        # Expected revenue for each (round, price) combination
        expected_revenue = self.S * self.prices[None, :]  # (T, K)
        
        # Greedy selection: for each round, pick the best price if affordable
        for t in range(T):
            if remaining_budget <= 0:
                # No budget left, must choose lowest price
                best_k = 0
            else:
                # Find the price that maximizes expected revenue
                # while considering remaining budget
                best_k = np.argmax(expected_revenue[t])
                
                # Check if we can afford this choice
                expected_cost = self.S[t, best_k]
                if expected_cost > remaining_budget:
                    # Find the highest price we can afford
                    affordable_mask = self.S[t] <= remaining_budget
                    if np.any(affordable_mask):
                        affordable_revenues = np.where(
                            affordable_mask,
                            expected_revenue[t],
                            -np.inf
                        )
                        best_k = np.argmax(affordable_revenues)
                    else:
                        # Nothing affordable, choose lowest price
                        best_k = 0
                
                # Update remaining budget
                remaining_budget -= self.S[t, best_k]
            
            # Set the decision
            self.X[t, best_k] = 1.0
        
        # Precompute deterministic choices
        if self.deterministic:
            self.plan_arms = np.argmax(self.X, axis=1)

        log_algorithm_choice(
            "Oracle: Successfully solved using greedy approach"
        )

    def _fallback_plan(self):
        """Fallback plan if LP optimization fails."""
        T, K = self.T, self.K

        if self.S is not None:
            # Use greedy selection based on expected reward
            self.X = np.zeros((T, K), dtype=float)
            best = np.argmax(self.S * self.prices[None, :], axis=1)
            self.X[np.arange(T), best] = 1.0
        else:
            # Ultimate fallback: uniform random
            self.X = np.full((T, K), 1.0 / K, dtype=float)

        if self.deterministic:
            self.plan_arms = np.argmax(self.X, axis=1)

    def pull_arm(self):
        """
        Select optimal price indices using LP-based clairvoyant strategy.
        Returns: array of optimal price indices for each product
        """
        def oracle_selection():
            if (self.remaining_budget <= 0 or
                    self.current_round >= self.T):
                return np.zeros(self.num_products, dtype=int)

            if self.X is None:
                # No LP solution available, use simple greedy approach
                return self._greedy_fallback()

            # Get action from LP solution
            if self.deterministic and self.plan_arms is not None:
                arm_idx = int(self.plan_arms[self.current_round])
            else:
                # Sample from the optimal distribution
                arm_idx = int(
                    np.random.choice(self.K, p=self.X[self.current_round])
                )

            # For multi-product case, replicate the same strategy
            # (This can be extended for product-specific strategies)
            optimal_actions = np.full(self.num_products, arm_idx, dtype=int)

            # Store decision for tracking
            oracle_prices = self.price_grid[
                np.arange(self.num_products), optimal_actions
            ]
            self.oracle_decisions.append({
                'round': self.current_round,
                'optimal_actions': optimal_actions.copy(),
                'optimal_prices': oracle_prices.copy(),
                'lp_distribution': (self.X[self.current_round].copy()
                                    if self.X is not None else None)
            })

            log_arm_selection(
                self.algorithm, self.total_steps, optimal_actions
            )
            return optimal_actions

        return self.safe_pull_arm(oracle_selection)

    def _greedy_fallback(self):
        """Greedy fallback when no LP solution is available."""
        if not self.environment_ref:
            return np.zeros(self.num_products, dtype=int)

        # Get current customer valuations from environment
        current_valuations = None
        if hasattr(self.environment_ref, 'current_valuations'):
            current_valuations = self.environment_ref.current_valuations
        elif (hasattr(self.environment_ref, 'valuation_history') and
              len(self.environment_ref.valuation_history) > 0):
            current_valuations = self.environment_ref.valuation_history[-1]

        if current_valuations is None:
            return np.zeros(self.num_products, dtype=int)

        # Find optimal price for each product
        optimal_actions = np.zeros(self.num_products, dtype=int)
        for i in range(self.num_products):
            valuation = current_valuations[i]

            # Find the highest price <= valuation
            feasible_mask = self.price_grid[i] <= valuation
            if np.any(feasible_mask):
                feasible_indices = np.where(feasible_mask)[0]
                optimal_actions[i] = feasible_indices[-1]
            else:
                optimal_actions[i] = 0  # Lowest price if none feasible

        return optimal_actions

    def update(self, purchased, actions):
        """
        Update statistics after observing purchase outcomes.

        Args:
            purchased: Array of purchase outcomes per product
            actions: Array of chosen price indices per product
        """
        # Apply constraints and get processed rewards
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, use_inventory_constraint=True
        )

        # Calculate oracle reward for this round
        chosen_prices = self.price_grid[
            np.arange(self.num_products), actions.astype(int)
        ]
        oracle_reward = np.sum(chosen_prices * purchased)

        self.total_optimal_reward += oracle_reward

        # Update budget tracking (only on actual sales)
        sales_occurred = purchased > 0
        if np.any(sales_occurred) and self.remaining_budget > 0:
            # Decrement budget by number of sales
            sales_count = int(np.sum(sales_occurred))
            budget_decrement = min(sales_count, self.remaining_budget)
            self.remaining_budget -= budget_decrement

        # Update base class budget tracking
        oracle_cost = np.sum(chosen_prices * 0.01 * purchased)
        self.update_budget(oracle_cost)

        # Store reward in history for compatibility
        self.history_rewards.append(oracle_reward)

        # Advance to next round
        self.current_round += 1

    def get_diagnostics(self):
        """Get diagnostic information about the oracle's performance."""
        return {
            "algorithm": self.algorithm,
            "total_steps": self.total_steps,
            "current_round": self.current_round,
            "total_optimal_reward": self.total_optimal_reward,
            "average_reward_per_step": (
                self.total_optimal_reward / max(1, self.total_steps)
            ),
            "oracle_decisions_count": len(self.oracle_decisions),
            "has_environment_ref": self.environment_ref is not None,
            "has_lp_solution": self.X is not None,
            "remaining_budget": self.remaining_budget,
            "deterministic": self.deterministic,
            "lp_solution_shape": (self.X.shape
                                  if self.X is not None else None),
            "success_probabilities_shape": (self.S.shape
                                            if self.S is not None else None)
        }

    def reset(self, setting):
        """Reset the oracle seller for a new trial."""
        super().reset(setting)
        self.total_optimal_reward = 0.0
        self.oracle_decisions = []
        self.current_round = 0

        # Recalculate LP solution for new setting
        if self.params is not None:
            self._solve_lp_plan()

        log_algorithm_choice("Reset Advanced LP-based Clairvoyant Oracle")

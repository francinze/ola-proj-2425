"""
Enhanced UCB1 Seller implementing advanced budgeted UCB algorithm.
"""
import numpy as np
from scipy.optimize import linprog
from ..seller import Seller
from ..setting import Setting
from ..logger import (log_algorithm_choice, log_ucb1_update, log_arm_selection)


class UCB1Seller(Seller):
    """
    Enhanced UCB1 Seller implementing advanced budgeted UCB algorithm.

    Based on the superior BudgetedUCBAgent design:
    - LP-based optimization with budget constraints
    - Dual tracking of rewards and success probabilities
    - Dynamic budget pacing
    - Robust numerical methods with simplex projection
    - Intelligent fallback strategies

    Requirement 1: Single product + stochastic + UCB1 (with/without inventory)
    Requirement 2: Multiple products + stochastic + Combinatorial-UCB
    """

    def __init__(
        self, setting: Setting, use_inventory_constraint: bool = True
    ):
        """
        Initialize enhanced UCB1 seller with advanced budget-aware features.

        Args:
            setting: Setting object with configuration
            use_inventory_constraint: Whether to enforce inventory constraints
        """
        super().__init__(setting, algorithm="ucb1")
        self.use_inventory_constraint = use_inventory_constraint

        # Enhanced tracking following BudgetedUCBAgent design
        self.last_chosen_arms = np.zeros(self.num_products, dtype=int)

        # Dual tracking: rewards AND success probabilities
        self.total_rewards = np.zeros((self.num_products, self.num_prices))
        self.successes = np.zeros((self.num_products, self.num_prices))
        self.avg_rewards = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)

        log_algorithm_choice(
            f"Enhanced UCB1 with LP optimization "
            f"(inventory_constraint={use_inventory_constraint})"
        )

    def _project_to_simplex(self, v):
        """
        Euclidean projection onto probability simplex: { x >= 0, sum x = 1 }.
        Robust against tiny negative entries from LP solvers.
        From BudgetedUCBAgent for numerical stability.
        """
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).all():
            return np.full_like(v, 1.0 / len(v))
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
        if len(rho_idx) == 0:
            return np.full_like(v, 1.0 / n)
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        w = np.maximum(v - theta, 0.0)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full_like(v, 1.0 / n)
        return w / s

    def pull_arm(self):
        """
        Enhanced UCB1 arm selection with LP-based budget optimization.

        For each product:
        1. Initialization phase: explore all arms once
        2. LP optimization phase: solve budget-constrained optimization
        3. Fallback: intelligent ratio-based selection if LP fails
        """
        def enhanced_ucb1_selection():
            chosen_indices = np.zeros(self.num_products, dtype=int)

            for i in range(self.num_products):
                # Initialization phase: explore all arms at least once
                unexplored_arms = np.where(self.counts[i] == 0)[0]

                if len(unexplored_arms) > 0:
                    # Choose first unexplored arm
                    chosen_indices[i] = unexplored_arms[0]
                else:
                    # Enhanced LP-based optimization phase
                    chosen_indices[i] = self._lp_based_arm_selection(i)

                self.last_chosen_arms[i] = chosen_indices[i]

            log_arm_selection(self.algorithm, self.total_steps, chosen_indices)
            return chosen_indices

        return self.safe_pull_arm(enhanced_ucb1_selection)

    def _lp_based_arm_selection(self, product_idx):
        """
        LP-based arm selection for a single product following
        BudgetedUCBAgent design.

        Args:
            product_idx: Index of the product to select arm for

        Returns:
            Selected arm index for the product
        """
        # Calculate confidence bounds
        confidence = np.sqrt(2.0 * np.log(self.T) /
                             np.maximum(self.counts[product_idx], 1e-10))

        # UCB for rewards and LCB for costs (success probabilities)
        mean_rewards = np.divide(
            self.total_rewards[product_idx],
            np.maximum(self.counts[product_idx], 1e-10)
        )
        mean_success_prob = np.divide(
            self.successes[product_idx],
            np.maximum(self.counts[product_idx], 1e-10)
        )

        ucb_reward = mean_rewards + confidence
        lcb_cost = np.maximum(mean_success_prob - confidence, 1e-6)

        # Dynamic budget pacing
        rounds_left = max(1, self.T - self.total_steps)
        rho = self.remaining_budget / rounds_left

        # Setup LP problem: maximize UCB rewards subject to LCB cost constraint
        num_arms = len(ucb_reward)
        c = -ucb_reward  # Negative because linprog minimizes
        A_ub = [lcb_cost]
        b_ub = [rho]
        A_eq = [np.ones(num_arms)]
        b_eq = [1.0]
        bounds = [(0.0, 1.0) for _ in range(num_arms)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')

            if res.success and res.x is not None:
                # Apply numerical hygiene with simplex projection
                gamma = self._project_to_simplex(res.x)

                # Final validation
                if (gamma >= 0).all() and np.isfinite(gamma).all() and \
                   abs(gamma.sum() - 1.0) <= 1e-6:
                    # Sample from the optimized distribution
                    return int(np.random.choice(num_arms, p=gamma))
        except Exception as e:
            log_arm_selection(f"LP failed for product {product_idx}: {e}",
                              self.total_steps, "fallback")

        # Intelligent fallback: reward-to-cost ratio
        ratios = ucb_reward / lcb_cost
        return int(np.argmax(ratios))

    def update(self, purchased, actions):
        """
        Enhanced update method tracking both rewards and success probabilities.

        This method follows the BudgetedUCBAgent design by maintaining
        separate tracking for:
        1. Total rewards (price-weighted)
        2. Success probabilities (for cost estimation)
        3. Dynamic budget tracking
        """
        # Apply constraints and get processed rewards
        purchased = self.apply_constraints_and_calculate_rewards(
            purchased, actions, self.use_inventory_constraint
        )

        # Calculate price-weighted rewards (price × purchase indicator)
        price_weighted_rewards = self.calculate_price_weighted_rewards(
            actions, purchased)

        self.total_steps += 1

        # Enhanced budget tracking following BudgetedUCBAgent pattern
        chosen_prices = self.price_grid[np.arange(self.num_products),
                                        actions.astype(int)]

        # Track actual sales for budget depletion (similar to BudgetedUCBAgent)
        total_sales = np.sum(purchased > 0)

        # Update remaining budget based on actual sales
        if total_sales > 0 and self.remaining_budget > 0:
            sales_cost = min(total_sales, self.remaining_budget)
            self.remaining_budget = max(0, self.remaining_budget - sales_cost)

        # Also track simple cost model for base class compatibility
        costs = chosen_prices * 0.01  # 1% of price as operational cost
        total_cost = np.sum(costs)
        self.update_budget(total_cost)

        # Enhanced dual tracking: rewards AND success probabilities
        for i, price_idx in enumerate(actions):
            price_idx = int(price_idx)

            # Update counts
            self.counts[i, price_idx] += 1

            # Update total rewards (price-weighted)
            self.total_rewards[i, price_idx] += price_weighted_rewards[i]

            # Update success tracking (key enhancement from BudgetedUCBAgent)
            if purchased[i] > 0:  # Track successful sales
                self.successes[i, price_idx] += 1.0

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

            # Update values array for base class compatibility
            self.values[i, price_idx] = self.avg_rewards[i, price_idx]

            log_ucb1_update(i, price_idx, self.counts[i, price_idx],
                            self.values[i, price_idx],
                            self.ucbs[i, price_idx])

        # Store total price-weighted rewards in history
        self.history_rewards.append(np.sum(price_weighted_rewards))

    def reset(self, setting):
        """
        Reset the enhanced UCB1 seller for a new trial.
        Includes resetting the dual tracking arrays.
        """
        super().reset(setting)

        # Reset enhanced dual tracking
        self.total_rewards = np.zeros((self.num_products, self.num_prices))
        self.successes = np.zeros((self.num_products, self.num_prices))
        self.avg_rewards = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)
        self.last_chosen_arms = np.zeros(self.num_products, dtype=int)

        log_algorithm_choice(
            f"Reset Enhanced UCB1 with LP optimization "
            f"(inventory_constraint={self.use_inventory_constraint})"
        )

"""
Primal-dual pacing agent with EXP3.P as the primal regret minimizer.
"""
import numpy as np
from ..seller import Seller
from ..setting import Setting
from ..logger import log_algorithm_choice, log_arm_selection


class PrimalDualSeller(Seller):
    """
    Primal-dual pacing agent with EXP3.P as the primal regret minimizer.

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
                 dual_lr=None, rng=None, eta_mult=1.0, forget_beta=0.0,
                 use_constant_pacing=True):
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

        # Pacing strategy selection
        self.use_constant_pacing = use_constant_pacing

        # Constant pacing parameters (notebook version)
        if self.use_constant_pacing:
            self.rho = self.B / float(self.T)
            self.lambda_cap = 1.0 / max(self.rho, 1e-12)

        # EXP3.P parameters (using notebook version defaults)
        if gamma is None:
            gamma = min(0.4, np.sqrt(self.K * np.log(max(2, self.K)) /
                                     max(1, self.T)))
        if eta is None:
            eta = gamma / (3.0 * max(1, self.K))
        if alpha is None:
            alpha = gamma / (3.0 * max(1, self.K))
        if dual_lr is None:
            dual_lr = eta  # per pacing pseudocode from notebook

        # Non-stationary knobs from notebook version
        # faster adaptation if eta_mult>1
        self.eta = float(eta) * float(eta_mult)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.dual_lr = float(dual_lr)
        # 0 = off; e.g., 0.02 = gentle forgetting
        self.forget_beta = float(forget_beta)

        # Internal state
        self.weights = np.ones(self.K, dtype=float)
        self.last_probs = np.full(self.K, 1.0 / self.K)
        self.last_arm = None
        self.lmbda = 0.0

        self.t = 0  # notebook version compatibility
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
        Compute pacing target rho_t and its scaling L_t = 1/rho_t (capped).
        Uses constant or dynamic pacing based on configuration.
        """
        if self.use_constant_pacing:
            # Notebook version: constant pacing
            rho_t = self.rho
            L_t = self.lambda_cap
        else:
            # Dynamic pacing (original PrimalDualSeller behavior)
            # avoid div by zero at the very end
            rounds_left = max(1, self.T - self.total_steps)
            # allowed expected spend per remaining round
            rho_t = self.remaining_budget / rounds_left
            # Guard: if budget is 0 -> rho_t = 0; avoid division by zero
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
        Update using notebook version's algorithm.
        reward: realized payoff (price if sale occurred else 0).
        Sale (cost=1) is inferred as (reward > 0).
        """
        if arm_index is None:
            return

        # Realized outcome & book-keeping
        sale = (reward > 0)
        cost = 1 if sale else 0
        self.total_reward += float(reward)

        # ---- Sale-centric reward for the primal (notebook version) ----
        if sale:
            bandit_reward = max(self.prices[arm_index] - self.lmbda, 0.0)
        else:
            bandit_reward = 0.0

        # ---- EXP3.P update (notebook version: all weights) ----
        p = self.last_probs
        xhat = np.zeros(self.K, dtype=float)
        p_arm = float(p[arm_index])
        xhat[arm_index] = bandit_reward / max(p_arm, 1e-12)

        bias = self.alpha / (np.maximum(p, 1e-12) *
                             np.sqrt(self.K * self.T))
        self.weights *= np.exp(self.eta * (xhat + bias))

        # Optional exponential forgetting (helps at regime switches)
        if self.forget_beta > 0.0:
            self.weights = self.weights ** (1.0 - self.forget_beta)

        # ---- Dual update (pacing) ----
        rho_t, lambda_max = self._rho_t_and_L()
        if self.use_constant_pacing:
            # Notebook version: constant pacing
            self.lmbda = float(np.clip(self.lmbda - self.dual_lr *
                                       (self.rho - cost), 0.0,
                                       self.lambda_cap))
        else:
            # Dynamic pacing (original behavior)
            self.lmbda = self.lmbda - self.dual_lr * (rho_t - cost)
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
        self.t += 1  # notebook version compatibility

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
            "rho": getattr(self, 'rho', rho_t),  # constant rho if available
            "probs": self._probs(),
            "weights": self.weights.copy(),
            "total_reward": self.total_reward,
            "use_constant_pacing": self.use_constant_pacing,
            "forget_beta": self.forget_beta,
        }

    def state(self):
        """
        Get current state (notebook version compatibility).
        """
        rho_t, _ = self._rho_t_and_L()
        state_dict = {
            "t": self.total_steps,
            "remaining_budget": self.remaining_budget,
            "lambda": self.lmbda,
            "rho_t": rho_t,
            "probs": self._probs(),
            "weights": self.weights.copy(),
            "total_reward": self.total_reward,
        }
        if self.use_constant_pacing:
            state_dict["rho"] = self.rho
        return state_dict

    def reset(self, setting):
        """Reset the primal-dual seller's statistics for a new trial."""
        super().reset(setting)

        # Reset primal-dual specific parameters
        if self.price_grid.ndim > 1:
            self.prices = self.price_grid[0]
        else:
            self.prices = self.price_grid
        self.K = len(self.prices)

        # Reset constant pacing parameters if using constant pacing
        if self.use_constant_pacing:
            self.rho = self.B / float(self.T)
            self.lambda_cap = 1.0 / max(self.rho, 1e-12)

        self.weights = np.ones(self.K, dtype=float)
        self.last_probs = np.full(self.K, 1.0 / self.K)
        self.last_arm = None
        self.lmbda = 0.0
        self.t = 0  # notebook version compatibility
        self.total_steps = 0
        self.remaining_budget = int(self.B)
        self.total_reward = 0.0

        log_algorithm_choice("Reset Primal-Dual EXP3.P")

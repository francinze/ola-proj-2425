from .seller import Seller
from .buyer import Buyer
from .setting import Setting
from .logger import (log_environment, log_simulation, log_error,
                     configure_logging, is_summary_mode)
import numpy as np


class Environment:
    """
    Environment class for the simulation of a market.
    This class is responsible for managing the interaction between the seller
    and buyer, including the pricing strategy of the seller and the purchasing
    behavior of the buyer.
    It also handles the simulation of multiple rounds of interactions.
    """
    def __init__(self, setting: Setting, seller: Seller = None):
        self.setting = setting
        self.t = 0
        self.distribution = setting.distribution
        self.seller = seller if seller is not None else Seller(setting)
        # Configure logging based on verbose setting
        configure_logging(setting.verbose)
        # Add valuation history tracking
        self.valuation_history = []
        # Collect log of results
        self.reset()

    def reset(self):
        """
        Reset the environment and seller for a new trial.
        """
        if self.seller is not None:
            self.seller.reset(self.setting)
        self.t = 0
        self.prices = np.zeros((self.setting.T, self.setting.n_products))
        self.purchases = np.zeros(
            (self.setting.T, self.setting.n_products),
            dtype=int
        )
        self.optimal_rewards = np.zeros(self.setting.T)
        if self.seller is not None:
            self.ucb_history = np.zeros(
                (self.setting.T, self.seller.num_products,
                 self.seller.num_prices)
            )
        self.regrets = np.zeros(self.setting.T)

    def round(self):
        """
        Play one round: seller chooses prices (or uses a_t if given),
        buyer responds, reward returned.
        """
        try:
            actions = self.seller.pull_arm()
            chosen_prices = self.seller.yield_prices(actions)
            # Note: history_chosen_prices is already updated in pull_arm()
            self.prices[self.t] = chosen_prices

            if (self.setting.non_stationary == 'slightly' or
                    self.setting.non_stationary == 'highly' or
                    self.setting.non_stationary == 'manual'):
                if len(self.setting.dist_params) == 2:
                    dist_params = (
                        self.setting.dist_params[0][self.t],
                        self.setting.dist_params[1][self.t, :]
                    )
                else:
                    dist_params = self.setting.dist_params[self.t, :]
            else:
                dist_params = self.setting.dist_params

            self.buyer = Buyer(
                name=f"Buyer at time {self.t}",
                setting=self.setting,
                dist_params=dist_params
            )

            # Store buyer's valuation for future reference
            self.valuation_history.append(self.buyer.valuations.copy())

            purchased = self.buyer.yield_demand(chosen_prices)

            # Update UCBs after this round (only if seller has UCBs)
            if (hasattr(self.seller, 'ucbs') and
                self.seller.ucbs is not None and
                hasattr(self.seller, 'algorithm') and
                self.seller.algorithm in ['ucb1', 'combinatorial_ucb',
                                          'sliding_window_ucb']):
                self.ucb_history[self.t] = self.seller.ucbs.copy()

            self.seller.update(purchased, actions)
            self.purchases[self.t] = purchased

            # --- Compute optimal reward for this round ---
            # Use adaptive optimal calculation for non-stationary environments
            if self.setting.non_stationary in ['slightly', 'highly']:
                optimal_reward = self.compute_optimal_reward_nonstationary()
            else:
                # Standard optimal calculation
                optimal_reward = self.compute_optimal_reward(
                    self.buyer.valuations)

            self.optimal_rewards[self.t] = optimal_reward
            self.regrets[self.t] = (
                optimal_reward - self.seller.history_rewards[-1]
            )
            # --------------------------------------------

            self.t += 1
        except Exception as e:
            log_error(f"Error in round {self.t}: {e}")
            # Ensure t increments even on error to avoid infinite loops
            self.t += 1

    def compute_optimal_reward_nonstationary(self):
        """
        Compute the true optimal reward for non-stationary environments.
        The optimal policy is clairvoyant - it knows the current valuation.
        """
        if len(self.valuation_history) == 0:
            return 0  # No valuations available

        # The optimal reward is the clairvoyant reward for current valuation
        # This is what a perfect algorithm would achieve if it knew it
        current_valuation = self.valuation_history[-1]  # Most recent valuation
        return self.compute_optimal_reward_with_valuation(current_valuation)

    def compute_optimal_reward_with_valuation(self, valuation):
        """
        Compute optimal reward for a given valuation.
        Similar to compute_optimal_reward but with specified valuation.
        """
        total = 0
        for i in range(self.setting.n_products):
            # Only consider prices <= valuation
            possible_prices = self.seller.price_grid[i][
                self.seller.price_grid[i] <= valuation[i]
            ]
            if len(possible_prices) > 0:
                best_price = np.max(possible_prices)
                total += best_price
        return total

    def compute_optimal_reward(self, valuations):
        """
        Compute the optimal (clairvoyant) reward for the current valuations.
        For each product, pick the price in the grid that maximizes reward,
        i.e., the highest price <= valuation.
        """
        total = 0
        for i in range(self.setting.n_products):
            # Only consider prices <= valuation
            possible_prices = self.seller.price_grid[i][
                self.seller.price_grid[i] <= valuations[i]
            ]
            if len(possible_prices) > 0:
                best_price = np.max(possible_prices)
                total += best_price
            # else: buyer would not buy at any price, reward is 0
        return total

    def play_all_rounds(self) -> None:
        '''Play all rounds of the simulation.'''
        for _ in range(self.setting.T):
            self.round()
        log_environment("Simulation finished.")
        # Use the actual number of completed rounds for indexing
        completed_rounds = min(self.t, self.setting.T)
        if completed_rounds > 0:
            final_index = completed_rounds - 1
            log_environment(f"Final prices: {self.prices[final_index]}")
            log_environment(f"Purchases: {self.purchases[final_index]}")

        # Generate detailed summary if summary mode is enabled
        if is_summary_mode():
            self._print_simulation_summary()

    def run_simulation(self, n_trials=20, distributions=[
        'uniform', 'gaussian', 'exponential', 'lognormal',
        'bernoulli', 'beta'
    ]):
        """
        Run the simulation for a given number of trials.
        Returns dictionaries containing regret and UCB data for analysis.
        """
        # Initialize dictionaries to collect data
        regrets_dict = {dist: [] for dist in distributions}
        ucb_dict = {dist: [] for dist in distributions}
        # Temporarily disable logging for batch simulation
        configure_logging(None)

        for trial in range(n_trials):
            self.reset()
            dist = distributions[trial % len(distributions)]
            self.setting.distribution = dist
            self.play_all_rounds()

            # --- Compute regret using optimal rewards ---
            rewards = np.array(self.seller.history_rewards)
            if len(rewards) == 0:
                continue  # Skip if no rewards recorded
            if rewards.ndim > 1:
                rewards = rewards.sum(axis=1)
            optimal_rewards = np.array(self.optimal_rewards)
            if (len(optimal_rewards) == 0 or
                    len(rewards) != len(optimal_rewards)):
                continue  # Skip if shapes don't match
            regret_per_round = optimal_rewards - rewards
            cumulative_regret = np.cumsum(regret_per_round)
            regrets_dict[dist].append(cumulative_regret)
            # Collect UCBs for product 0 for this trial
            ucb_product0 = []
            for t in range(self.setting.T):
                # After each round, seller.values[0] is UCBs for product 0
                if hasattr(self.seller, "values"):
                    ucb_product0.append(self.seller.ucbs[0].copy())
            ucb_dict[dist].append(np.array(ucb_product0))

            log_simulation(f"Trial {trial + 1} finished.")

        # Convert lists to arrays for return
        for dist in distributions:
            regrets_dict[dist] = np.array(regrets_dict[dist])
            ucb_dict[dist] = np.array(ucb_dict[dist])

        return regrets_dict, ucb_dict

    def _print_simulation_summary(self) -> None:
        """
        Print detailed simulation results when summary mode is enabled.
        """
        if (not hasattr(self.seller, 'history_rewards') or
                len(self.seller.history_rewards) == 0):
            print("No simulation data available for summary.")
            return

        rewards = np.array(self.seller.history_rewards)
        optimal_rewards = np.array(self.optimal_rewards)
        regrets = optimal_rewards - rewards
        cum_regret = np.cumsum(regrets)

        print("\n" + "="*70)
        print("🎯 SIMULATION SUMMARY")
        print("="*70)

        # Basic configuration
        print("📋 CONFIGURATION:")
        print(f"   Algorithm: {self.setting.algorithm}")
        print(f"   Products: {self.setting.n_products}")
        print(f"   Time steps: {self.setting.T}")
        print(f"   Price levels: {int(1/self.setting.epsilon)}")
        print(f"   Environment: {self.setting.non_stationary}")
        print(f"   Distribution: {self.setting.distribution}")
        # Fix the distribution params display
        if hasattr(self.setting, 'dist_params') and self.setting.dist_params:
            if (isinstance(self.setting.dist_params, tuple) and
                    len(self.setting.dist_params) == 2):
                mean, std = self.setting.dist_params
                print(f"   Distribution params: (mean={mean}, std={std})")
            else:
                param_type = type(self.setting.dist_params).__name__
                print(f"   Distribution params: {param_type}")
        print(f"   Budget constraint: {self.setting.B:.2f}")

        # Performance metrics
        print("\n📊 PERFORMANCE METRICS:")
        print(f"   Total rewards earned: {np.sum(rewards):.2f}")
        print(f"   Total optimal rewards: {np.sum(optimal_rewards):.2f}")
        print(f"   Final cumulative regret: {cum_regret[-1]:.2f}")
        efficiency = (np.sum(rewards) / np.sum(optimal_rewards)) * 100
        print(f"   Algorithm efficiency: {efficiency:.1f}%")
        print(f"   Average reward per round: {np.mean(rewards):.3f}")
        print(f"   Average regret per round: {np.mean(regrets):.3f}")

        # Time-based analysis
        print("\n📈 LEARNING PROGRESS:")
        quarter = self.setting.T // 4
        if quarter > 0:
            early_eff = (np.sum(rewards[:quarter]) /
                         np.sum(optimal_rewards[:quarter])) * 100
            late_eff = (np.sum(rewards[-quarter:]) /
                        np.sum(optimal_rewards[-quarter:])) * 100
            print(f"   Early performance (first 25%): {early_eff:.1f}%")
            print(f"   Late performance (last 25%): {late_eff:.1f}%")
            improvement = late_eff - early_eff
            print(f"   Learning improvement: {improvement:+.1f}%")

            # More detailed learning analysis
            print("\n🔍 DETAILED LEARNING ANALYSIS:")

            # Performance by segments
            segments = 5
            segment_size = self.setting.T // segments
            print(f"   Performance by {segments} segments:")
            for i in range(segments):
                start_idx = i * segment_size
                if i < segments - 1:
                    end_idx = (i + 1) * segment_size
                else:
                    end_idx = self.setting.T
                seg_rewards = rewards[start_idx:end_idx]
                seg_optimal = optimal_rewards[start_idx:end_idx]
                if np.sum(seg_optimal) > 0:
                    seg_eff = (np.sum(seg_rewards) / np.sum(seg_optimal)) * 100
                    print(f"     Segment {i+1} (steps {start_idx+1}-"
                          f"{end_idx}): {seg_eff:.1f}%")

            # Regret progression analysis
            print("\n   Regret progression:")
            regret_windows = [25, 50, 100]
            for window in regret_windows:
                if self.setting.T >= window:
                    early_regret = np.mean(regrets[:window])
                    late_regret = np.mean(regrets[-window:])
                    if early_regret > 0:
                        regret_change = ((late_regret - early_regret) /
                                         early_regret) * 100
                    else:
                        regret_change = 0
                    print(f"     Last {window} vs first {window} steps: "
                          f"{regret_change:+.1f}% regret change")

            # Learning velocity
            if len(cum_regret) >= 50:
                early_slope = np.polyfit(range(50), cum_regret[:50], 1)[0]
                late_slope = np.polyfit(range(50), cum_regret[-50:], 1)[0]
                print("   Learning velocity (regret slope):")
                print(f"     Early slope: {early_slope:.3f} regret/step")
                print(f"     Late slope: {late_slope:.3f} regret/step")
                if early_slope != 0:
                    slope_improvement = ((early_slope - late_slope) /
                                         abs(early_slope)) * 100
                else:
                    slope_improvement = 0
                print(f"     Slope improvement: {slope_improvement:+.1f}%")

        # Pricing strategy analysis
        if hasattr(self, 'prices') and len(self.prices) > 0:
            prices_used = self.prices[self.prices > 0]
            if len(prices_used) > 0:
                print("\n💰 PRICING STRATEGY:")
                print(f"   Average price used: {np.mean(prices_used):.3f}")
                min_price, max_price = np.min(prices_used), np.max(prices_used)
                print(f"   Price range: [{min_price:.3f}, {max_price:.3f}]")
                print(f"   Price volatility (std): {np.std(prices_used):.3f}")
        # Regret analysis
        print("\n📉 REGRET ANALYSIS:")
        regret_trend = (np.diff(cum_regret[-10:]) if len(cum_regret) >= 10
                        else np.diff(cum_regret))
        avg_recent_regret = (np.mean(regret_trend) if len(regret_trend) > 0
                             else 0)
        print(f"   Recent regret trend: {avg_recent_regret:.3f} per round")

        # Additional regret metrics
        print(f"   Maximum single-round regret: {np.max(regrets):.3f}")
        print(f"   Minimum single-round regret: {np.min(regrets):.3f}")
        print(f"   Regret standard deviation: {np.std(regrets):.3f}")

        # Convergence analysis
        if len(cum_regret) >= 100:
            last_100_slope = np.polyfit(range(100), cum_regret[-100:], 1)[0]
            print(f"   Last 100 rounds regret slope: {last_100_slope:.4f}")

            # Check for convergence patterns
            if last_100_slope < 0.01:
                convergence_status = "✅ Strong convergence"
            elif last_100_slope < 0.1:
                convergence_status = "🔄 Moderate convergence"
            elif last_100_slope < 0.5:
                convergence_status = "⚠️ Slow convergence"
            else:
                convergence_status = "❌ Poor convergence"

            print(f"   Convergence status: {convergence_status}")

        # Learning assessment
        print("\n🎓 LEARNING ASSESSMENT:")
        if avg_recent_regret < 0.01:
            print("   Status: ✅ Algorithm is converging well")
            print("   Recommendation: Current settings are effective")
        elif avg_recent_regret < 0.1:
            print("   Status: 🔄 Algorithm shows moderate learning")
            print("   Recommendation: Monitor performance, "
                  "consider parameter tuning")
        else:
            print("   Status: ⚠️  Algorithm may need adjustment")
            print("   Recommendation: Review algorithm parameters or approach")

        # Performance ranking
        if efficiency >= 80:
            perf_rank = "Excellent"
        elif efficiency >= 60:
            perf_rank = "Good"
        elif efficiency >= 40:
            perf_rank = "Fair"
        elif efficiency >= 20:
            perf_rank = "Poor"
        else:
            perf_rank = "Very Poor"

        print(f"   Overall performance: {perf_rank} "
              f"({efficiency:.1f}% efficiency)")

        print("="*70)

from .seller import Seller
from .buyer import Buyer
from .setting import Setting
from .logger import (log_environment, log_simulation, log_error,
                     configure_logging)
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

            # Update UCBs after this round (if seller has UCBs)
            if hasattr(self.seller, 'ucbs'):
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
        if self.t > 0:
            log_environment(f"Final prices: {self.prices[self.t - 1]}")
            log_environment(f"Purchases: {self.purchases[self.t - 1]}")

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

        # Return data for optional plotting by external functions
        return regrets_dict, ucb_dict

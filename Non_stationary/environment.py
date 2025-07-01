from .seller import Seller
from .buyer import Buyer
from .setting import Setting
import numpy as np
from plotting import (
    plot_all, plot_cumulative_regret_by_distribution,
    plot_ucb_product0_by_distribution
)


class Environment:
    """
    Environment class for the simulation of a market.
    This class is responsible for managing the interaction between the seller
    and buyer, including the pricing strategy of the seller and the purchasing
    behavior of the buyer.
    It also handles the simulation of multiple rounds of interactions.
    """
    def __init__(self, setting: Setting):
        self.setting = setting
        self.t = 0
        self.distribution = setting.distribution
        self.seller = Seller(setting)
        # Collect log of results
        self.reset()

    def reset(self):
        """
        Reset the environment and seller for a new trial.
        """
        self.seller.reset(self.setting)
        self.t = 0
        self.prices = np.zeros((self.setting.T, self.setting.n_products))
        self.purchases = np.zeros(
            (self.setting.T, self.setting.n_products),
            dtype=int
        )
        self.optimal_rewards = np.zeros(self.setting.T)
        self.ucb_history = np.zeros(
            (self.setting.T, self.seller.num_products, self.seller.num_prices)
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
            chosen_indices = actions
            self.seller.history_chosen_prices.append(chosen_indices)
            self.prices[self.t] = chosen_prices

            if self.setting.non_stationary == 'slightly' or self.setting.non_stationary == 'highly' or self.setting.non_stationary == 'manual':
                if len(self.setting.dist_params) == 2:
                    dist_params = (self.setting.dist_params[0][self.t], self.setting.dist_params[1][self.t, :])
                else:
                    dist_params = self.setting.dist_params[self.t, :]
            else:
                dist_params = self.setting.dist_params

            self.buyer = Buyer(
                name=f"Buyer at time {self.t}",
                setting=self.setting,
                dist_params=dist_params
            )
            demand = self.buyer.yield_demand(chosen_prices)
            purchased = self.seller.budget_constraint(demand)

            # Update UCBs after this round
            self.ucb_history[self.t] = self.seller.ucbs.copy()

            self.seller.update(purchased, actions)
            self.purchases[self.t] = purchased

            # --- Compute optimal reward for this round ---
            optimal_reward = self.compute_optimal_reward(self.buyer.valuations)
            self.optimal_rewards[self.t] = optimal_reward
            self.regrets[self.t] = (
                optimal_reward - self.seller.history_rewards[-1]
            )
            # --------------------------------------------

            self.t += 1
        except Exception as e:
            print(f"Error in round {self.t}: {e}")

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

    def play_all_rounds(self, plot=True) -> None:
        '''Play all rounds of the simulation.'''
        for _ in range(self.setting.T):
            self.round()
        if self.setting.verbose == 'all':
            print("Simulation finished.")
            print(f"Final prices: {self.prices[self.t - 1]}")
            print(f"Purchases: {self.purchases[self.t - 1]}")
        if plot:
            plot_all(
                self.seller, self.optimal_rewards,
                self.regrets, self.ucb_history
            )

    def run_simulation(self, n_trials=20, distributions=[
        'uniform', 'gaussian', 'exponential', 'lognormal',
        'bernoulli', 'beta'
    ]):
        """
        Run the simulation for a given number of trials
        and plot the cumulative regret.
        """
        # Initialize dictionaries to collect data
        regrets_dict = {dist: [] for dist in distributions}
        ucb_dict = {dist: [] for dist in distributions}
        self.setting.verbose = None

        for trial in range(n_trials):
            self.reset()
            dist = distributions[trial % len(distributions)]
            self.setting.distribution = dist
            self.play_all_rounds(plot=False)

            # --- Compute regret using optimal rewards ---
            rewards = np.array(self.seller.history_rewards)
            if rewards.ndim > 1:
                rewards = rewards.sum(axis=1)
            optimal_rewards = np.array(self.optimal_rewards)
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

            if self.setting.verbose == 'all':
                print(f"Trial {trial + 1} finished.")

        plot_all(self.seller, self.ucb_history)

        # Convert lists to arrays
        for dist in distributions:
            regrets_dict[dist] = np.array(regrets_dict[dist])
            ucb_dict[dist] = np.array(ucb_dict[dist])

        # For UCB plot, average over trials
        avg_ucb_dict = {}
        for dist in distributions:
            # Average over trials: shape (T, n_prices)
            avg_ucb_dict[dist] = ucb_dict[dist].mean(axis=0)

        plot_cumulative_regret_by_distribution(
            self.setting.T,
            regrets_dict,
            n_trials
        )
        plot_ucb_product0_by_distribution(avg_ucb_dict)
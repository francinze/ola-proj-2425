#from .seller import Seller
from .buyer import Buyer
from .setting import Setting
from matplotlib import pyplot as plt

#req_5
from .seller import Seller as BaseSeller
from .seller_sliding import SellerSliding
from .seller_prima_dual import PrimalDualSeller

 
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
        self.interval_len = 100  # or any suitable value like 50  #added for non-stationary
        self.manual_mu_table = None
        self.purchases = np.zeros((self.setting.T, self.setting.n_products), dtype=np.int32)


        if self.setting.non_stationary == 'slightly': 
            self._generate_mu_schedule()

        self.t = 0
        self.distribution = setting.distribution

        # REQ-5 Conditionally import the correct Seller class based on the algorithm
        """ if setting.algorithm == "ucb_sliding":
            from .seller_sliding import SellerSliding as Seller
        else:
            from .seller import Seller
        self.seller = Seller(setting)
        """
        self.seller = self._select_seller(setting)


        
        # Collect log of results
        self.reset()
        
    def _select_seller(self, setting):
        if setting.algorithm == "ucb_sliding":
            return SellerSliding(setting)
        elif setting.algorithm == "primal_dual":
            return PrimalDualSeller(setting)
        else:
            return BaseSeller(setting)

        
    def _generate_mu_schedule(self):
        """
        Generates a new dist_params schedule for slightly non-stationary setting.
        Changes mean every `interval_len` steps.
        """
        num_intervals = self.setting.T // self.interval_len + 1
        n = np.ones((self.setting.T, 1))  # Dummy for distributions like Bernoulli
        mu = np.zeros((self.setting.T, self.setting.n_products))

        for i in range(num_intervals):
            start = i * self.interval_len
            end = min((i + 1) * self.interval_len, self.setting.T)

            # Select new distribution each interval
            dist = self.setting.distribution
            if dist == 'uniform':
                new_mu = np.random.uniform(0.2, 0.8, size=(self.setting.n_products,))
            elif dist == 'gaussian':
                new_mu = np.clip(np.random.normal(loc=0.5, scale=0.15, size=self.setting.n_products), 0, 1)
            elif dist == 'beta':
                new_mu = np.random.beta(a=2, b=5, size=self.setting.n_products)
            elif dist == 'bernoulli':
                new_mu = np.random.binomial(1, p=0.5, size=self.setting.n_products)
            elif dist == 'lognormal':
                new_mu = np.clip(np.random.lognormal(scale=-0.7, sigma=0.5, size=self.setting.n_products), 0, 1)
            elif dist == 'exponential':
                new_mu = np.clip(np.random.exponential(scale=0.5, size=self.setting.n_products), 0, 1)
            else:
                raise ValueError(f"Unsupported distribution: {dist}")

            mu[start:end, :] = new_mu

        self.setting.dist_params = (n, mu.T)


    def reset(self):
        """
        Reset the environment and seller for a new trial.
        """
        self.seller.reset(self.setting)
        self.t = 0
        self.prices = np.zeros((self.setting.T, self.setting.n_products))
        self.purchases = np.zeros(
            (self.setting.T, self.setting.n_products),
             dtype=np.int32  # force correct numeric dtype
        )
        self.optimal_rewards = np.zeros(self.setting.T)
        self.ucb_history = np.zeros(
            (self.setting.T, self.seller.num_products, self.seller.num_prices)
        )
        self.regrets = np.zeros(self.setting.T)

    
    def round(self):
        try:
            actions = self.seller.pull_arm()
            print(f"DEBUG -- Using seller type: {type(self.seller)}")
            print("DEBUG -- actions:", actions)

            chosen_prices, chosen_indices = self.seller.yield_prices(actions)
            self.seller.history_chosen_prices.append(chosen_indices)
            self.prices[self.t] = chosen_prices

            print("DEBUG -- chosen_prices:", chosen_prices)
            print("type:", type(chosen_prices))
            print("shape:", np.shape(chosen_prices))

            # Handle non-stationary dist_params
            if self.setting.non_stationary in ['slightly', 'highly', 'manual']:
                if len(self.setting.dist_params) == 2:
                    dist_param_0 = self.setting.dist_params[0][self.t]
                    dist_param_1 = self.setting.dist_params[1][:, self.t]  # Fixed indexing
                    dist_params = (dist_param_0, dist_param_1)
                else:
                    dist_params = self.setting.dist_params[:, self.t]
            else:
                dist_params = self.setting.dist_params

            print("DEBUG -- dist_params used for buyer:", dist_params)
            print("DEBUG -- dist_params[1] shape:", np.shape(dist_params[1]))

            self.buyer = Buyer(
                name=f"Buyer at time {self.t}",
                setting=self.setting,
                dist_params=dist_params
            )

            demand = self.buyer.yield_demand(chosen_prices)
            purchased = self.seller.budget_constraint(np.array(demand))
            purchased_clean = np.asarray(purchased, dtype=np.int32).flatten()

            print("DEBUG -- purchased:", purchased)
            print("DEBUG -- purchased dtype:", purchased.dtype)
            print("DEBUG -- purchased_clean:", purchased_clean)
            print("DEBUG -- purchased_clean.shape:", purchased_clean.shape)

            if purchased_clean.ndim != 1 or purchased_clean.shape[0] != self.setting.n_products:
                raise ValueError(f"Malformed purchased_clean shape: {purchased_clean.shape}")

            self.purchases[self.t, :] = purchased_clean
            rewards = chosen_prices * purchased_clean
            
            if self.setting.algorithm == "primal_dual":
                self.seller.update(chosen_indices, rewards, purchased_clean)
            else:
                self.seller.update(chosen_indices, rewards)
                
            self.ucb_history[self.t] = self.seller.ucbs.copy()

            optimal_reward = self.compute_optimal_reward(self.buyer.valuations)
            self.optimal_rewards[self.t] = optimal_reward
            actual_reward = np.sum(rewards)
            self.regrets[self.t] = optimal_reward - actual_reward

            print(f"Round {self.t}: optimal={optimal_reward:.2f}, actual={actual_reward:.2f}, regret={self.regrets[self.t]:.2f}")
            print("DEBUG -- self.purchases row type:", type(self.purchases[self.t, :]))
            print("DEBUG -- purchased_clean type:", type(purchased_clean))

            self.t += 1

        except Exception as e:
            print(f"Error in round {self.t}: {e}")
            raise



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

        # --- Debug: Check UCB history for product 0 ---
        import matplotlib.pyplot as plt
        product_index = 0

        if hasattr(self, 'ucb_history') and isinstance(self.ucb_history, np.ndarray):
            try:
                ucb_data = self.ucb_history[:, product_index, :]  # shape: (T, num_prices)
                print("DEBUG -- ucb_history shape:", ucb_data.shape)

                if not np.any(ucb_data):
                    print("DEBUG -- ucb_history is all zeros or uninitialized!")
                else:
                    print("DEBUG -- Sample UCBs for product 0 at key steps:")
                    for t in [0, self.setting.T // 3, 2 * self.setting.T // 3, self.setting.T - 1]:
                        print(f"Round {t}: {ucb_data[t]}")

                    if plot:
                        # Plot UCBs of all prices for product 0
                        for price_idx in range(ucb_data.shape[1]):
                            plt.plot(ucb_data[:, price_idx], label=f"Price {price_idx}")
                        plt.xlabel("Step")
                        plt.ylabel("UCB Value")
                        plt.title(f"UCBs of Prices for Product {product_index} Over Time")
                        plt.legend()
                        plt.grid(True)
                        plt.show()

                        # Optional: quick plot for just one price
                        plt.plot(ucb_data[:, 0])
                        plt.title("UCB over time for Product 0, Price 0")
                        plt.xlabel("Time")
                        plt.ylabel("UCB Value")
                        plt.grid(True)
                        plt.show()
            except Exception as e:
                print(f"[ERROR] While plotting ucb_history: {e}")
        else:
            print("Warning: self.ucb_history not found or not a valid array.")

        if plot:
            plot_all(
                self.seller,
                self.optimal_rewards,
                self.regrets,
            )

    
    def run_simulation(self, n_trials=20, distributions=['gaussian'], algorithms=['ucb_sliding', 'primal_dual']):
        regrets_dict = {alg: {dist: [] for dist in distributions} for alg in algorithms}

        for trial in range(n_trials):
            for alg in algorithms:
                self.setting.algorithm = alg
                self.seller = self._select_seller(self.setting)
                self.reset()

                dist = distributions[trial % len(distributions)]
                self.setting.distribution = dist

                if self.setting.non_stationary == 'slightly':
                    self._generate_mu_schedule()

                self.play_all_rounds(plot=False)

                rewards = np.array(self.seller.history_rewards)
                if rewards.ndim > 1:
                    rewards = rewards.sum(axis=1)
                optimal_rewards = np.array(self.optimal_rewards)
                cumulative_regret = np.cumsum(optimal_rewards - rewards)

                regrets_dict[alg][dist].append(cumulative_regret)
                print(f"Trial {trial + 1} with {alg} finished, cumulative regret: {cumulative_regret[-1]:.2f}")

        # Plot
        for dist in distributions:
            for alg in algorithms:
                regrets_dict[alg][dist] = np.array(regrets_dict[alg][dist]).mean(axis=0)
        plot_cumulative_regret_by_distribution(self.setting.T, regrets_dict, n_trials)

    

    def _select_seller(self, setting):
        if setting.algorithm == "ucb_sliding":
            return SellerSliding(setting)
        else:
            return BaseSeller(setting)
        
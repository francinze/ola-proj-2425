from .seller import Seller
from .buyer import Buyer
from .setting import Setting
import numpy as np
from plotting import plot_all


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

    def round(self):
        """
        Play one round: seller chooses prices (or uses a_t if given),
        buyer responds, reward returned.
        :param a_t: Optional array of price indices (actions) for each product.
        :return: reward vector (or sum), chosen price indices
        """
        try:
            actions = self.seller.pull_arm()
            # Set seller's chosen prices to a_t
            chosen_prices = self.seller.yield_prices(actions)
            chosen_indices = actions
            self.seller.history_chosen_prices.append(chosen_indices)
            self.prices[self.t] = chosen_prices
            self.buyer = Buyer(
                name=f"Buyer at time {self.t}",
                setting=self.setting,
            )
            demand = self.buyer.yield_demand(chosen_prices)
            # Apply seller's inventory constraint before finalizing purchases
            purchased = self.seller.inventory_constraint(demand)
            self.seller.update(purchased, actions)
            self.purchases[self.t] = purchased
            self.t += 1
        except Exception as e:
            print(f"Error in round {self.t}: {e}")

    def play_all_rounds(self) -> None:
        '''Play all rounds of the simulation.'''
        for _ in range(self.setting.T):
            self.round()
        if self.setting.verbose == 'all':
            print("Simulation finished.")
            print(f"Final prices: {self.prices[self.t - 1]}")
            print(f"Purchases: {self.purchases[self.t - 1]}")

    def plot_seller_learning(self):
        """
        Plot the Seller's learning progress after the simulation.
        """
        plot_all(self.seller)

    def reset(self):
        """
        Reset the environment and seller for a new trial.
        """
        self.seller.reset()
        self.t = 0
        self.prices = np.ones((self.setting.T, self.setting.n_products))
        self.purchases = np.zeros(
            (self.setting.T, self.setting.n_products),
            dtype=int
        )

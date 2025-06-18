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
    def __init__(
        self, setting: Setting, distribution: str, verbose: bool = True
    ):
        self.seller = Seller(setting.products, setting.P, verbose=verbose)
        self.setting = setting
        self.distribution = distribution
        self.t = 0
        self.verbose = verbose

        # Collect log of results
        self.prices = np.ones((self.setting.T, len(self.setting.products)))
        self.purchases = np.zeros(
            (self.setting.T, len(self.setting.products)),
            dtype=int
        )

    def round(self, a_t=None):
        """
        Play one round: seller chooses prices (or uses a_t if given),
        buyer responds, reward returned.
        :param a_t: Optional array of price indices (actions) for each product.
        :return: reward vector (or sum), chosen price indices
        """
        if a_t is not None:
            # Set seller's chosen prices to a_t
            chosen_prices = np.array([
                self.seller.price_grid[i, a_t[i]]
                for i in range(self.seller.num_products)
            ])
            chosen_indices = a_t
            self.seller.history_chosen_prices.append(chosen_indices)
        else:
            chosen_prices, chosen_indices = self.seller.choose_prices()
        self.prices[self.t] = chosen_prices
        self.setting.P = chosen_prices
        self.buyer = Buyer(
            name=f"Buyer at time {self.t}",
            n_products=len(self.setting.products),
            distribution=self.distribution,
        )
        purchased = self.buyer.make_purchases(chosen_prices)
        # Convert purchases to reward vector (e.g., 1 if bought, 0 if not)
        rewards = np.zeros(len(chosen_prices))
        for i, price in enumerate(chosen_prices):
            if price in purchased:
                rewards[i] = price
        self.seller.update(rewards, chosen_indices)
        self.purchases[self.t] = rewards
        self.t += 1
        return rewards, chosen_indices

    def play_all_rounds(self) -> None:
        '''Play all rounds of the simulation.'''
        for _ in range(self.setting.T):
            try:
                actions = self.seller.pull_arm()
                self.round(a_t=actions)
            except Exception as e:
                print(f"Error during round {self.t}: {e}")
        if self.verbose:
            print("Simulation finished.")
            print(f"Final prices: {self.prices}")
            print(f"Purchases: {self.purchases}")

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
        self.prices = np.ones((self.setting.T, len(self.setting.products)))
        self.purchases = np.zeros(
            (self.setting.T, len(self.setting.products)),
            dtype=int
        )

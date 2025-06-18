from typing import List
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

    def round(self) -> None:
        '''Play one round of the simulation.'''
        # Get chosen prices and their indices for each product
        new_prices, chosen_indices = self.seller.choose_prices()
        self.prices[self.t] = new_prices

        self.setting.P = new_prices
        self.buyer: Buyer = Buyer(
            name=f"Buyer at time {self.t}",
            n_products=len(self.setting.products),
            distribution=self.distribution,
        )
        try:
            purchased: List[float] = self.buyer.make_purchases(new_prices)
            purchase_decision = np.array(purchased)  # purchases/product
            if self.verbose:
                print(f"{self.buyer} made purchases: {purchased}")
            # Calculate reward per product (e.g., revenue = price * quantity)
            if purchase_decision.shape[0] != len(new_prices):
                # If no purchases or shape mismatch, fill with zeros
                purchase_decision = np.zeros(len(new_prices))
            if self.verbose:
                print("reward for seller:", purchase_decision)
            # Update UCB statistics for each product
            self.seller.update_ucb(chosen_indices, purchase_decision)
            # Collect results in log
            self.purchases[self.t] = purchase_decision
        except ValueError as e:
            if self.verbose:
                print(
                    f"ValueError during round {self.t}: "
                    f"{self.buyer} could not make purchases.\n"
                    f"The prices were: {new_prices}\n"
                    f"His valuations were: {self.buyer.valuations}\n"
                    f"Error: {e}"
                )
            return
        self.t += 1

    def play_all_rounds(self) -> None:
        '''Play all rounds of the simulation.'''
        for _ in range(self.setting.T):
            try:
                self.round()
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

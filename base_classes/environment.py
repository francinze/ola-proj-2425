from typing import List
from seller import Seller
from buyer import Buyer
from setting import Setting
import numpy as np


class Environment:
    """
    Environment class for the simulation of a market.
    This class is responsible for managing the interaction between the seller
    and buyers, including the pricing strategy of the seller and the purchasing
    behavior of the buyers.
    It also handles the simulation of multiple rounds of interactions.
    """
    def __init__(self, setting: Setting, distribution: str):
        self.seller = Seller(setting.products, setting.P)
        self.setting = setting
        self.buyers: List[Buyer] = [
            Buyer(
                name=f"Buyer {i}",
                n_products=len(self.setting.products),
                distribution=distribution,
            )
            for i in range(self.setting.N)
        ]
        self.t = 0

        # Collect log of results
        self.prices = np.zeros((self.setting.T, len(self.setting.products)))
        self.purchases = np.zeros((self.setting.T), dtype=int)
        self.seller_utility = np.zeros((self.setting.T), dtype=float)

    def round(self) -> None:
        '''Play one round of the simulation.'''
        new_prices = self.seller.choose_prices()

        self.prices[self.t] = new_prices

        self.setting.P = new_prices
        buyer: Buyer = self.buyers[np.random.choice(
            self.setting.N, size=1, replace=False)[0]]
        print(f"Round {self.t}:")
        try:
            purchased: List[float] = buyer.make_purchases(new_prices)
            purchase_decision = sum(1 for item in purchased if item > 0)
            print("1")
            self.seller.inventory_constraint(purchase_decision)
            print("2")
            # Collect results in log
            self.purchases[self.t] = purchase_decision
            self.seller_utility[self.t] = self.seller.utility
            print(f"Purchased vector: {purchased}")
            print(f"Seller utility: {self.seller.utility}")
        except ValueError:
            print("Exceeds production capacity")
        self.t += 1

    def play_all_rounds(self) -> None:
        '''Play all rounds of the simulation.'''
        for _ in range(self.setting.T):
            self.round()
        print("Simulation finished.")
        print(f"Final prices: {self.prices}")
        print(f"Purchases: {self.purchases}")
        print(f"Seller utility: {self.seller_utility}")

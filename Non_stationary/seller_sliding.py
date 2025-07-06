import numpy as np
from collections import deque
from .setting import Setting

class SellerSliding:
    def __init__(self, setting: Setting):
        self.products = np.arange(setting.n_products)
        self.price_grid = np.linspace(0.1, 1.0, int(1 / setting.epsilon))
        self.price_grid = np.tile(self.price_grid, (setting.n_products, 1))
        if self.price_grid.ndim == 1:
            self.price_grid = self.price_grid.reshape((len(self.products), -1))

        self.num_products = len(self.products)
        self.num_prices = self.price_grid.shape[1]
        self.T = setting.T
        self.window_size = setting.sliding_window_size or 100  # Default to 100 if not provided

        # Sliding window memory for UCB
        self.history = deque(maxlen=self.window_size)  # stores (actions, rewards)

        # Initialize history for logging
        self.history_rewards = []
        self.history_chosen_prices = []

        # For UCB stats
        self.counts = np.zeros((self.num_products, self.num_prices))
        self.values = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)

    def yield_prices(self, actions):
        """Estimate UCBs from recent history."""
        # Compute UCB values using only the last window of history
        counts = np.zeros((self.num_products, self.num_prices))
        values = np.zeros((self.num_products, self.num_prices))

        for action, reward in self.history:
            for i, (a, r) in enumerate(zip(action, reward)):
                counts[i, a] += 1
                values[i, a] += r

        # Calculate means and UCBs
        means = np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)
        total_counts = np.sum(counts) + 1  # +1 to avoid log(0)
        ucbs = np.where(
            counts > 0,
            means + np.sqrt(2 * np.log(total_counts) / counts),
            np.inf  # Force exploration of untried arms
        )


        # Choose the best prices based on UCB
        chosen_indices = np.argmax(ucbs, axis=1)
        chosen_indices = np.asarray(chosen_indices, dtype=int).flatten()

        chosen_prices = np.array([
            self.price_grid[i, chosen_indices[i]]
            for i in range(self.num_products)
        ], dtype=np.float32)


        self.history_chosen_prices.append(chosen_indices)
        self.ucbs = ucbs.copy()  # <- ADD THIS!

        return chosen_prices, chosen_indices

    def update(self, actions, rewards):
        """Store actions and rewards in the sliding window history."""
        actions = np.asarray(actions, dtype=int).flatten()
        rewards = np.asarray(rewards, dtype=float).flatten()

        self.history.append((actions, rewards))
        self.history_rewards.append(rewards)
        
        # Update counts and values for UCB calculation
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            print(f"DEBUG -- i: {i}, action: {action}, type: {type(action)}")
            self.counts[i, action] += 1
            self.values[i, action] += reward

    def reset(self, setting: Setting):
        """Reset the seller for a new trial."""
        self.__init__(setting)  # Reinitialize with the new setting

    def pull_arm(self):
        """
        Returns dummy indices for now (0 for each product), just to keep environment happy.
        Environment uses these indices to pass to yield_prices().
        """
        #return [0] * self.num_products
        return np.argmax(self.ucbs, axis=1)

    def budget_constraint(self, demand):
        """
        Applies the budget constraint. By default, this is the identity function
        if constraint is lax. For strict constraints, limit the purchases.
        """
        demand = np.array(demand)
        binary_purchases = (demand > 0).astype(np.int32).reshape(-1)  # ✅ ensures int
        print("DEBUG -- budget_constraint returning:", binary_purchases)
        print("DEBUG -- budget_constraint return shape:", binary_purchases.shape)
        print("DEBUG -- budget_constraint return dtype:", binary_purchases.dtype)
        return binary_purchases

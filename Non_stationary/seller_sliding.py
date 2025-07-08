import numpy as np
from collections import deque
from .setting import Setting

# class SellerSliding:
#     def __init__(self, setting: Setting):
#         self.total_steps = 0
#         self.verbose = False
#         self.products = np.arange(setting.n_products)
#         self.price_grid = np.linspace(0.1, 1.0, int(1 / setting.epsilon))
#         self.price_grid = np.tile(self.price_grid, (setting.n_products, 1))
#         if self.price_grid.ndim == 1:
#             self.price_grid = self.price_grid.reshape((len(self.products), -1))

#         self.num_products = len(self.products)
#         self.num_prices = self.price_grid.shape[1]
#         self.T = setting.T
#         self.window_size = setting.sliding_window_size or 100  # Default to 100 if not provided

#         # Sliding window memory for UCB
#         self.history = deque(maxlen=self.window_size)  # stores (actions, rewards)
#         self.

#         # Initialize history for logging
#         self.history_rewards = []
#         self.history_chosen_prices = []

#         # For UCB stats
#         self.counts = np.ones((self.num_products, self.num_prices)) * 1e-3  
#         self.values = np.zeros((self.num_products, self.num_prices))
#         self.ucbs = np.full((self.num_products, self.num_prices), np.inf)
        
from collections import deque
import numpy as np

class SellerSliding:
    def __init__(self, setting: Setting):
        self.total_steps = 0
        self.verbose = False
        self.products = np.arange(setting.n_products)
        self.price_grid = np.linspace(0.1, 1.0, int(1 / setting.epsilon))
        self.price_grid = np.tile(self.price_grid, (setting.n_products, 1))
        if self.price_grid.ndim == 1:
            self.price_grid = self.price_grid.reshape((len(self.products), -1))

        self.num_products = len(self.products)
        self.num_prices = self.price_grid.shape[1]
        self.T = setting.T

        # Sliding window size (default to 100 if not set)
        self.window_size = getattr(setting, 'sliding_window_size', 100)

        # Sliding window storage
        self.history = deque(maxlen=self.window_size)  # stores tuples of (actions, rewards)

        # Logging only
        self.history_rewards = []
        self.history_chosen_prices = []

        # UCB statistics
        self.counts = np.ones((self.num_products, self.num_prices)) * 1e-3  # Avoid divide-by-zero
        self.values = np.zeros((self.num_products, self.num_prices))
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)

        
            
    def yield_prices(self, actions):
        if len(self.history) == 0:
            print("DEBUG -- yield_prices: history is empty, returning random prices")
            chosen_indices = np.random.randint(0, self.num_prices, size=self.num_products)
            chosen_prices = np.array([
                self.price_grid[i, chosen_indices[i]]
                for i in range(self.num_products)
            ], dtype=np.float32)
            self.history_chosen_prices.append(chosen_indices)
            self.ucbs = np.full((self.num_products, self.num_prices), np.inf)
            print("DEBUG -- chosen_prices:", chosen_prices)
            return chosen_prices, chosen_indices
        
        counts = np.zeros((self.num_products, self.num_prices))
        values = np.zeros((self.num_products, self.num_prices))
        
        for action, reward in self.history:
            for i, (a, r) in enumerate(zip(action, reward)):
                if i >= self.num_products:
                    raise IndexError(f"History index {i} exceeds num_products {self.num_products}")
                counts[i, a] += 1
                values[i, a] += r
        
        means = np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)
        total_counts = np.sum(counts)
        if total_counts <= 0:
            total_counts = 1
        
        epsilon = 1e-8
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = counts + epsilon
            exploration_term = np.sqrt(2 * np.log(total_counts + epsilon) / denom)
            ucb_raw = means + exploration_term
            ucb_raw = np.nan_to_num(ucb_raw, nan=0.0, posinf=1e6, neginf=0.0)
        
        if not np.all(np.isfinite(ucb_raw)):
            print("WARNING: Non-finite UCBs:", ucb_raw)
            ucb_raw = np.random.rand(*ucb_raw.shape)
        
        self.ucbs = ucb_raw.copy()
        chosen_indices = np.argmax(self.ucbs, axis=1).astype(int)
        chosen_prices = np.array([
            self.price_grid[i, chosen_indices[i]]
            for i in range(self.num_products)
        ], dtype=np.float32)
        
        self.history_chosen_prices.append(chosen_indices)
        print("DEBUG -- chosen_prices:", chosen_prices)
        return chosen_prices, chosen_indices


    
    
    
    def update(self, actions, rewards):
        self.total_steps += 1

        actions = np.asarray(actions, dtype=int).flatten()
        rewards = np.asarray(rewards, dtype=float).flatten()

        self.history.append((actions, rewards))
        self.history_rewards.append(rewards)

        # Step 1: Reset counts and values
        self.counts[:] = 1e-3  # Prevent divide-by-zero
        self.values[:] = 0.0

        # Step 2: Re-accumulate over sliding window
        for actions, rewards in self.history:
            for i in range(min(len(actions), self.num_products)):
                action = int(actions[i])
                reward = float(rewards[i])

                if action >= self.num_prices:
                    raise IndexError(f"Action {action} is out of bounds!")
                self.counts[i, action] += 1
                self.values[i, action] += reward

        # Step 3: Recalculate UCBs
        counts_safe = np.where(self.counts == 0, 1, self.counts)
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.divide(self.values, counts_safe)
            bonuses = np.sqrt(2 * np.log(self.total_steps + 1) / counts_safe)
            ucbs = means + bonuses
            ucbs = np.nan_to_num(ucbs, nan=0.0, posinf=np.inf, neginf=0.0)
            ucbs[self.counts == 0] = np.inf

        self.ucbs = ucbs.copy()


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
        binary_purchases = (demand > 0).astype(np.int32).reshape(-1)  # ensures int
        print("DEBUG -- budget_constraint returning:", binary_purchases)
        print("DEBUG -- budget_constraint return shape:", binary_purchases.shape)
        print("DEBUG -- budget_constraint return dtype:", binary_purchases.dtype)
        return binary_purchases

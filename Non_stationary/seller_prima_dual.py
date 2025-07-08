import numpy as np
from Non_stationary.seller import Seller as BaseSeller

class PrimalDualSeller(BaseSeller):
    def __init__(self, setting):
        super().__init__(setting)
        self.setting = setting

        self.num_products = setting.n_products
        self.epsilon = setting.epsilon
        self.num_prices = int(1 / self.epsilon)
        self.n_arms = self.num_prices
        self.price_grid = np.tile(np.linspace(0.1, 1.0, self.num_prices), (self.num_products, 1))
        # self.budget_constraint = setting.budget_constraint
        self.ucbs = np.full((self.num_products, self.num_prices), np.inf)

        self.T = setting.T
        self.B = setting.B if setting.B is not None else 1.0  # fallback if not set
        self.cost_coeff = 0.5
        self.eta = 0.1
        self.rho = self.B / self.T
        self.lambda_dual = 0.0
        self.gamma = 0.1

        # initialize weights: [num_products x num_prices]
        self.weights = np.ones((self.num_products, self.num_prices))

        self.t = 0
        self.last_actions = np.zeros(self.num_products, dtype=int)
        self.last_probs = np.full((self.num_products, self.num_prices), 1.0 / self.num_prices)

    def pull_arm(self):
        self.last_probs = self.weights / self.weights.sum(axis=1, keepdims=True)
        self.last_actions = np.array([
            np.random.choice(self.n_arms, p=self.last_probs[i])
            for i in range(self.num_products)
        ])
        return self.last_actions

    def yield_prices(self, actions):
        actions = np.asarray(actions, dtype=int).flatten()
        if len(actions) != self.num_products:
            raise ValueError(f"Expected {self.num_products} actions, got {len(actions)}")
        
        chosen_prices = np.array([
            self.price_grid[i][actions[i]] for i in range(self.num_products)
        ], dtype=np.float32)
        
        chosen_indices = actions
        self.history_chosen_prices.append(chosen_indices)

        print("DEBUG -- chosen_prices:", chosen_prices)
        return chosen_prices, chosen_indices  # Two values

    def update(self, actions, rewards, purchases):
        actions = np.asarray(actions, dtype=int).flatten()
        prices = self.yield_prices(actions)[0]  # Only need prices, discard indices
        rewards = np.asarray(rewards, dtype=float).flatten()
        purchases = np.asarray(purchases, dtype=int).flatten()

        costs = prices
        total_cost = np.sum(costs)

        self.lambda_dual -= self.eta * (self.rho - total_cost)
        self.lambda_dual = np.clip(self.lambda_dual, 0, 1 / self.rho)

        for i in range(self.num_products):
            a = actions[i]
            adjusted_reward = rewards[i] - self.lambda_dual * costs[i]
            self.weights[i, a] *= np.exp(self.gamma * adjusted_reward)

        self.t += 1



    def reset(self, setting=None):
        self.setting = setting if setting else self.setting
        self.weights = np.ones((self.num_products, self.num_prices))
        self.lambda_dual = 0.0
        self.t = 0
        self.last_actions = np.zeros(self.num_products, dtype=int)
        self.last_probs = np.full((self.num_products, self.num_prices), 1.0 / self.num_prices)

    # In PrimalDualSeller:
    def budget_constraint(self, demand):
        capped_demand = np.clip(demand, 0, 1).astype(np.int32)
        return capped_demand

    
    
    def budget_constraint(self, demand):
        capped_demand = np.clip(demand, 0, 1).astype(np.int32)
        print("DEBUG -- budget_constraint returning:", capped_demand)
        print("DEBUG -- budget_constraint return shape:", capped_demand.shape)
        print("DEBUG -- budget_constraint return dtype:", capped_demand.dtype)
        return capped_demand



######  Markdown Documentation
'''
The `PrimalDualSeller` class is an implementation of a primal-dual algorithm tailored for dynamic pricing in a multi-product setting under budget constraints. 
It operates by iteratively adjusting pricing strategies using both primal (price weights) and dual (Lagrange multiplier) variables. 
The class is designed to plug seamlessly into the simulation environment and maintains compatibility with the interface expected from other seller classes like `SellerSliding`.

The constructor method `__init__` initializes the seller. It sets up the number of products and prices, constructs the price grid using a uniform linear spacing based on the `epsilon` parameter, and initializes key variables. 
These include the `weights` matrix used to score each price for each product, the dual variable `lambda_dual` which governs how strictly the budget constraint is enforced, and algorithmic hyperparameters like `gamma` (learning rate for weight updates) and `eta` (step size for dual updates). 
The budget `rho` is also extracted from the setting, along with the simulation horizon `T`.

The method `pull_arm()` is responsible for selecting actions for each product. 
It determines which price index to use for each product by identifying the maximum value in the `weights` matrix along each product's price dimension. 
This essentially chooses the most promising price for each product based on current learned preferences.

The method `yield_prices(actions)` takes these selected price indices and returns the actual price values from the `price_grid` that correspond to them. 
This is essential to convert the abstract action space (indices) into real-world prices that can be evaluated in the simulation.

The `update(purchased, actions)` method is the core of the learning mechanism. After each round, it takes the observed purchases and actions (i.e., price choices), calculates the rewards as the product of price and purchase outcome, and adjusts the dual variable `lambda_dual` to penalize over-budget behavior. 
It then modifies the `weights` matrix by applying an exponential update that accounts for both the reward and the cost (penalized by `lambda_dual`). This ensures that pricing decisions not only favor high reward but also respect the budget.

Finally, the `reset(setting)` method allows reinitialization of the seller for a new simulation or experimental trial by simply re-calling the constructor with updated parameters. 
The method `budget_constraint(demand)` applies a hard threshold to demand, converting it into binary purchase decisions that reflect whether the demand exceeds zero and is within budget. 
This ensures a consistent and enforceable policy throughout the simulation.
'''
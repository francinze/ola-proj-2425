"""
This file contains the Setting class, which is used to define the parameters
of the simulation.
"""
import numpy as np


class Setting:
    def __init__(
        self,
        T: int = np.random.randint(99, 100),
        n_products: int = 10,
        epsilon: float = 0.1,
        B: int = None,
        distribution: str = "uniform",
        budget_constraint: str = "lax",
        verbose: str = 'all',
        non_stationary: str = 'no',
        dist_params: float = (0.1, 0.9)
    ):
        self.T = T
        self.n_products = n_products
        self.cost_coeff = 0.5
        if B is None:
            prices = np.linspace(0.1, 1.0, int(1 / epsilon))
            mean_cost = np.mean(prices)
            B = self.T/self.n_products * self.cost_coeff * mean_cost
        self.B = B  # Production capacity
        self.distribution = distribution  # Distribution type
        self.budget_constraint = budget_constraint
        self.verbose = verbose
        self.epsilon = epsilon
        self.non_stationary = non_stationary
        self.dist_params = self.create_params(dist_params)

    def create_params(self, dist_params):
        """
        Create stationary or non-stationary parameters for the simulation.
        """

        if self.non_stationary == 'slightly':
            # Decide how often we switch the distribution
            high_switch_slightly = np.log(self.T)/2
            switch_num = np.random.randint(low=1, high=high_switch_slightly.astype(int), size=self.n_products)

        elif self.non_stationary == 'highly':
            # Decide how often we switch the distribution
            low_switch_highly = np.log(self.T)
            high_switch_highly = np.log(self.T)*2
            switch_num = np.random.randint(low=low_switch_highly.astype(int), high=high_switch_highly.astype(int), size=self.n_products)

        else:
            return dist_params  # Return the provided parameters for stationary case
        

        # Decide when to switch
        switch_times = np.zeros((np.max(switch_num) + 2, self.n_products))
        for i in range(self.n_products):
            # Create a random sequence of switch times
            switch_times_temp = np.random.randint(low=0, high=self.T - 1, size=switch_num[i])
            # Sort the switch times
            switch_times_temp.sort()
            switch_times[0, i] = 0  # First switch at time 0
            switch_times[1:switch_num[i] + 1, i] = switch_times_temp
            # Fill the rest with T
            for j in range(switch_num[i] + 1, switch_times.shape[0]):
                switch_times[j, i] = self.T


        if self.distribution == "uniform":
            # Create high and low limit vectors
            high = np.zeros((self.T, self.n_products))
            low = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of values
                temp = np.random.uniform(
                    low=0, high=1, size=(switch_num[i] + 1, 2)
                )
                # Sort the values
                temp = np.sort(temp, axis=1)

                for j in range(0, switch_num[i] + 1):
                    high[switch_times[j, i].astype(int):switch_times[j + 1, i].astype(int), i] = temp[j, 1]
                    low[switch_times[j, i].astype(int):switch_times[j + 1, i].astype(int), i] = temp[j, 0]

            params = (high, low)

        elif self.distribution == "bernoulli":
            # Create mean values vector
            mu = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of means
                mu_temp = np.random.uniform(
                    low=0, high=1, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    mu[switch_times[j, i].astype(int):switch_times[j + 1, i].astype(int), i] = mu_temp[j]

            params = (np.ones((self.T, 1)), mu)

        elif self.distribution == "gaussian":
            # Create mean value and standard deviation vectors
            mu = np.zeros((self.T, self.n_products))
            sigma = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of means
                mu_temp = np.random.uniform(
                    low=0, high=1, size=switch_num[i] + 1
                )
                # Create a random sequence of standard deviations
                sigma_temp = np.random.uniform(
                    low=0.1, high=0.7, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    mu[switch_times[j, i].astype(int):switch_times[j + 1, i].astype(int), i] = mu_temp[j]
                    sigma[switch_times[j, i].astype(int):switch_times[j + 1, i].astype(int), i] = sigma_temp[j]

            params = (mu, sigma)

        elif self.distribution == "exponential":
            params = 0
        elif self.distribution == "beta":
            params = 0
        elif self.distribution == "lognormal":
            params = 0
        elif self.distribution == "test":
            params = 0
        elif self.distribution == "constant":
            # Create mean value and standard deviation vectors
            mu = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of values
                mu_temp = np.random.uniform(
                    low=0, high=1, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    mu[switch_times[j, i].astype(int):switch_times[j + 1, i].astype(int), i] = mu_temp[j]

            params = mu

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
            params = 0
        return params

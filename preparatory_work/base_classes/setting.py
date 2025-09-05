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
        verbose: str = 'all',
        non_stationary: str = 'no',
        dist_params: float = (0.1, 0.9),
        trend_flip_params: dict = None,
    ):
        self.T = T
        self.n_products = n_products
        self.cost_coeff = 0.5
        if B is None:
            prices = np.linspace(0.1, 1.0, int(1 / epsilon))
            mean_cost = np.mean(prices)
            if self.T is not None:
                B = self.T/self.n_products * self.cost_coeff * mean_cost
            else:
                # Default when T is None
                B = 100/self.n_products * self.cost_coeff * mean_cost
        self.B = B  # Production capacity
        self.distribution = distribution  # Distribution type
        self.verbose = verbose
        self.epsilon = epsilon
        self.non_stationary = non_stationary
        self.trend_flip_params = trend_flip_params
        self.dist_params = self.create_params(dist_params)

    def create_params(self, dist_params):
        """
        Create stationary or non-stationary parameters for the simulation.
        """

        if self.non_stationary == 'trend_flip':
            # Handle TrendFlipBetaEnvironment pattern
            return self._create_trend_flip_params(dist_params)

        elif self.non_stationary == 'highly':
            # Trend-flip implementation for highly non-stationary case
            return self._create_trend_flip_params(dist_params)

        elif self.non_stationary == 'slightly':
            # Decide how often we switch the distribution
            high_switch_slightly = max(2, np.log(self.T)/2)
            switch_num = np.random.randint(
                low=1, high=int(high_switch_slightly), size=self.n_products
            )

        else:
            # Return the provided parameters for stationary case
            return dist_params

        # The following logic applies only to 'slightly' non-stationary case

        # Decide when to switch
        switch_times = np.zeros((np.max(switch_num) + 2, self.n_products))
        for i in range(self.n_products):
            # Create a random sequence of switch times
            if switch_num[i] > 0 and self.T > 1:
                switch_times_temp = np.random.randint(
                    low=0, high=max(1, self.T - 1), size=switch_num[i]
                )
                # Sort the switch times
                switch_times_temp.sort()
                switch_times[0, i] = 0  # First switch at time 0
                switch_times[1:switch_num[i] + 1, i] = switch_times_temp
            else:
                switch_times[0, i] = 0
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
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    high[start_idx:end_idx, i] = temp[j, 1]
                    low[start_idx:end_idx, i] = temp[j, 0]

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
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    mu[start_idx:end_idx, i] = mu_temp[j]

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
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    mu[start_idx:end_idx, i] = mu_temp[j]
                    sigma[start_idx:end_idx, i] = sigma_temp[j]

            params = (mu, sigma)

        elif self.distribution == "exponential":
            # Create mean value and standard deviation vectors
            mean = np.zeros((self.T, self.n_products))
            scale = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of means
                mean_temp = np.random.uniform(
                    low=-0.5, high=1, size=switch_num[i] + 1
                )
                # Create a random sequence of standard deviations
                scale_temp = np.random.uniform(
                    low=0.1, high=1, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    mean[start_idx:end_idx, i] = mean_temp[j]
                    scale[start_idx:end_idx, i] = scale_temp[j]

            params = (mean, scale)

        elif self.distribution == "beta":
            # Create mean value and standard deviation vectors
            a = np.zeros((self.T, self.n_products))
            b = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of means
                a_temp = np.random.uniform(
                    low=0.5, high=5, size=switch_num[i] + 1
                )
                # Create a random sequence of standard deviations
                b_temp = np.random.uniform(
                    low=0.5, high=5, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    a[start_idx:end_idx, i] = a_temp[j]
                    b[start_idx:end_idx, i] = b_temp[j]

            params = (a, b)

        elif self.distribution == "lognormal":
            # Create mean value and standard deviation vectors
            mean = np.zeros((self.T, self.n_products))
            sigma = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of means
                mean_temp = np.random.uniform(
                    low=-1, high=0.5, size=switch_num[i] + 1
                )
                # Create a random sequence of standard deviations
                sigma_temp = np.random.uniform(
                    low=0.1, high=1, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    mean[start_idx:end_idx, i] = mean_temp[j]
                    sigma[start_idx:end_idx, i] = sigma_temp[j]

            params = (mean, sigma)

        elif self.distribution == "test":
            params = None

        elif self.distribution == "constant":
            # Create mean value and standard deviation vectors
            mu = np.zeros((self.T, self.n_products))
            for i in range(self.n_products):
                # Create a random sequence of values
                mu_temp = np.random.uniform(
                    low=0, high=1, size=switch_num[i] + 1
                )

                for j in range(0, switch_num[i] + 1):
                    start_idx = switch_times[j, i].astype(int)
                    end_idx = switch_times[j + 1, i].astype(int)
                    mu[start_idx:end_idx, i] = mu_temp[j]

            params = mu

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
            params = 0
        return params

    def _create_trend_flip_params(self, dist_params):
        """
        Create parameters for TrendFlip Beta environment.
        Expected trend_flip_params structure:
        {
            'u': float,              # per-round mean step
            'K': int,                # rounds per trend segment
            'mu0': float,            # start mean
            'v_target': float,       # variance upper bound
            'mu_bounds': tuple,      # (mu_min, mu_max) bounds
            'start_dir': int,        # +1 or -1 initial direction
            'seed': int              # RNG seed (optional)
        }
        """
        if self.trend_flip_params is None:
            # Use default parameters
            self.trend_flip_params = {
                'u': 0.015,
                'K': 25,
                'mu0': 0.2,
                'v_target': 0.02,
                'mu_bounds': (0.05, 0.95),
                'start_dir': 1,
            }
        
        u = self.trend_flip_params.get('u', 0.015)
        K = self.trend_flip_params.get('K', 25)
        mu0 = self.trend_flip_params.get('mu0', 0.2)
        v_target = self.trend_flip_params.get('v_target', 0.02)
        mu_bounds = self.trend_flip_params.get('mu_bounds', (0.05, 0.95))
        start_dir = self.trend_flip_params.get('start_dir', 1)
        
        # Validate parameters
        mu_min, mu_max = mu_bounds
        assert 0 < mu_min < mu_max < 1, "Invalid mu_bounds"
        assert 0 < u < 1, "Invalid step size u"
        assert K >= 1, "Invalid trend flip period K"
        assert 0.0 < v_target < 0.25, "Invalid target variance"
        
        # Fixed concentration from target variance
        c = 0.25 / v_target - 1.0
        
        # Generate mean trajectory with trend flips
        mu = np.clip(mu0, mu_min, mu_max)
        direction = 1 if start_dir >= 0 else -1
        mu_series = []
        
        for t in range(self.T):
            # Record current mu
            mu_series.append(float(mu))
            
            # Step mean for next round
            if t < self.T - 1:
                # Flip direction every K rounds
                if (t + 1) % K == 0:
                    direction *= -1
                mu = mu + direction * u
                # Reflect at bounds
                mu, direction = self._reflect_mu(mu, direction, mu_min, mu_max)
        
        # Convert to numpy array
        mu_array = np.array(mu_series)
        
        # Generate beta parameters (a, b) for each time step
        a_params = np.maximum(mu_array * c, 1e-9)
        b_params = np.maximum((1.0 - mu_array) * c, 1e-9)
        
        # Return as (T, n_products) arrays - broadcast mu to all products
        a_matrix = np.tile(a_params[:, np.newaxis], (1, self.n_products))
        b_matrix = np.tile(b_params[:, np.newaxis], (1, self.n_products))
        
        return (a_matrix, b_matrix)
    
    def _reflect_mu(self, mu, direction, mu_min, mu_max):
        """Reflect mean at bounds and flip direction if crossed."""
        if mu > mu_max:
            overshoot = mu - mu_max
            mu = mu_max - overshoot
            direction *= -1
        elif mu < mu_min:
            overshoot = mu_min - mu
            mu = mu_min + overshoot
            direction *= -1
        return mu, direction

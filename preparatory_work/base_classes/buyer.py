from .setting import Setting
from .logger import (log_buyer_valuations, log_buyer_demand, log_error)
import numpy as np
from typing import List, Any


class Buyer:
    """
    A buyer class that represents a buyer in the marketplace.
    The buyer has valuations for products and can yield demand based on prices.
    """
    def __init__(self, name: str, setting: Setting,
                 dist_params: np.ndarray[float, Any] = None):
        """
        Initialize a buyer with a name, setting, and distribution parameters.
        """
        self.name = name
        self.setting = setting
        self.T = setting.T  # Add T attribute from setting
        self.dist_params = dist_params  # Parameters for distribution

        # Generate valuations based on distribution
        self._generate_valuations()

        # Log buyer initialization
        log_buyer_valuations(self.name, self.valuations)

    def _generate_valuations(self):
        """Generate valuations based on the distribution and parameters."""
        if self.setting.distribution == "uniform":
            high = (self.dist_params[0] if self.dist_params is not None and
                    len(self.dist_params) > 0 else 1)
            low = (self.dist_params[1] if self.dist_params is not None and
                   len(self.dist_params) > 1 else 0)
            self.valuations = np.random.uniform(
                low=low, high=high, size=self.setting.n_products
            )
        elif self.setting.distribution == "gaussian":
            mean = (self.dist_params[0] if self.dist_params is not None and
                    len(self.dist_params) > 0 else 0.5)
            std = (self.dist_params[1] if self.dist_params is not None and
                   len(self.dist_params) > 1 else 0.2)
            self.valuations = np.clip(
                np.random.normal(
                    loc=mean, scale=std, size=self.setting.n_products
                ), 0, 1
            )
        elif self.setting.distribution == "bernoulli":
            p = (self.dist_params[0] if self.dist_params is not None and
                 len(self.dist_params) > 0 else 0.5)
            # Clip to [0,1] to ensure valid probability
            p = np.clip(p, 0, 1)
            # Bernoulli values are 0 or 1, but we map to continuous [0,1]
            bernoulli_values = np.random.binomial(
                1, p, self.setting.n_products)
            # Scale to create valuations between 0 and 1
            self.valuations = bernoulli_values * np.random.uniform(
                0.1, 1.0, self.setting.n_products
            )
        elif self.setting.distribution == "exponential":
            # exponential clipped to [0,1] for valuations
            mean = (self.dist_params[0] if self.dist_params is not None and
                    len(self.dist_params) > 0 else 0.5)
            scale = (self.dist_params[1] if self.dist_params is not None and
                     len(self.dist_params) > 1 else 0.5)
            self.valuations = np.clip(
                np.random.exponential(scale=scale,
                                      size=self.setting.n_products), 0, 1
            )
        elif self.setting.distribution == "beta":
            # Beta(2,5) is skewed toward 0, Beta(5,2) toward 1
            if self.dist_params is not None and len(self.dist_params) >= 2:
                a_param = self.dist_params[0]
                b_param = self.dist_params[1]
                
                # Handle different formats: scalar, vector, or default
                if np.isscalar(a_param):
                    a = np.full(self.setting.n_products, a_param)
                else:
                    a = np.asarray(a_param)
                    
                if np.isscalar(b_param):
                    b = np.full(self.setting.n_products, b_param)
                else:
                    b = np.asarray(b_param)
                    
                # Ensure vectors have correct length
                if len(a) != self.setting.n_products:
                    default_a = a[0] if len(a) > 0 else 2
                    a = np.full(self.setting.n_products, default_a)
                if len(b) != self.setting.n_products:
                    default_b = b[0] if len(b) > 0 else 5
                    b = np.full(self.setting.n_products, default_b)
            else:
                # Default parameters
                a = np.full(self.setting.n_products, 2)
                b = np.full(self.setting.n_products, 5)
                
            self.valuations = np.random.beta(a=a, b=b)
        elif self.setting.distribution == "lognormal":
            # lognormal with mean ~0.5, clipped to [0,1]
            mean = (self.dist_params[0] if self.dist_params is not None and
                    len(self.dist_params) > 0 else -0.7)
            sigma = (self.dist_params[1] if self.dist_params is not None and
                     len(self.dist_params) > 1 else 0.5)
            self.valuations = np.clip(np.random.lognormal(
                mean=mean, sigma=sigma, size=self.setting.n_products
            ), 0, 1)
        elif self.setting.distribution == "test":
            # set valuations in a fixed range for testing purposes
            if (self.dist_params is not None and
                    len(self.dist_params) > 0 and
                    hasattr(self.dist_params, 'shape') and
                    len(self.dist_params.shape) > 1 and
                    self.dist_params.shape[1] >= 2):
                high = self.dist_params[:, 0]
                low = self.dist_params[:, 1]
            else:
                high = np.linspace(0.2, 1, self.setting.n_products)
                low = np.linspace(0, 0.8, self.setting.n_products)
            v = np.zeros(self.setting.n_products)
            for i in range(self.setting.n_products):
                v[i] = np.random.uniform(
                    low=low[i], high=high[i]
                )
            self.valuations = np.clip(v, 0, 1)
        elif self.setting.distribution == "constant":
            # All valuations are the same constant value
            v = (self.dist_params if self.dist_params is not None and
                 len(self.dist_params) > 0 else
                 np.linspace(0.1, 1, self.setting.n_products))
            self.valuations = np.clip(v, 0, 1)
        else:
            raise ValueError(
                f"Unknown distribution: {self.setting.distribution}")

    def __str__(self):
        return f"Buyer(name={self.name}, valuations={self.valuations})"

    def __repr__(self):
        return f"Buyer number {self.name}"

    def yield_demand(self, prices: np.ndarray[float, Any]) -> List[float]:
        """
        Calculate demand based on prices and valuations.

        Project Requirement: "Buys all products priced below their respective
        valuations". This implements binary purchasing behavior as specified
        in the project requirements.

        Returns: Binary demand (1.0 if price < valuation, 0.0 otherwise)
        """
        demand = []
        log_buyer_demand(self.name, prices, "calculating demand")

        for i in range(len(prices)):
            if prices[i] is not None and self.valuations[i] is not None:
                # Binary demand model as per project requirements
                if prices[i] < self.valuations[i]:
                    demand.append(1.0)  # Buy the product
                else:
                    demand.append(0.0)  # Don't buy the product
            else:
                log_error(f"{self.name} encountered None value - "
                          f"price[{i}]: {prices[i]}, "
                          f"valuation[{i}]: {self.valuations[i]}")
                demand.append(0)  # No demand if price or valuation is None

        log_buyer_demand(self.name, prices, demand)
        return demand

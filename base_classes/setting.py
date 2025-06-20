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
        inventory_constraint: str = "lax",
        verbose: str = 'all'
    ):
        self.T = T
        self.n_products = n_products
        if B is None:
            B = np.random.randint(1, self.n_products)
        self.B = B  # Production capacity
        self.distribution = distribution  # Distribution type
        self.inventory_constraint = inventory_constraint
        self.verbose = verbose
        self.epsilon = epsilon

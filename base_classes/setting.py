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
        discretization: int = 10,
        B: int = np.random.randint(1, 1000),
    ):
        self.T = T
        self.products = np.arange(n_products)  # Products
        price_grid = np.linspace(0.1, 1.0, discretization)
        self.P = np.tile(price_grid, (n_products, 1))  # Price grid
        self.B = B  # Production capacity

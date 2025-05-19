"""
This file contains the Setting class, which is used to define the parameters
of the simulation.
"""
import numpy as np


class Setting:
    def __init__(
        self,
        T: int = np.random.randint(99, 100),
        N: int = np.random.randint(100, 1000),
        p_size: int = 10,
        B: int = np.random.randint(1, 1000),
    ):
        self.T = T
        self.N = N  # Number of buyers
        self.products = np.arange(p_size)  # Products
        self.P = np.zeros(p_size)  # Prices
        self.B = B  # Production capacity

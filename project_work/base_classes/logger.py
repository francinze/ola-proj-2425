"""
Logging system for the market simulation.

This module provides a comprehensive logging system that replaces the verbose
print statements throughout the codebase. It uses Python's standard logging
module for better control, formatting, and configurability.
"""

import logging
import sys
from typing import Optional, Any
from enum import Enum


class LogLevel(Enum):
    """Log levels for different components."""
    ENVIRONMENT = "environment"
    SELLER = "seller"
    BUYER = "buyer"
    SIMULATION = "simulation"
    ERROR = "error"


class MarketLogger:
    """
    Centralized logger for the market simulation system.

    This logger provides component-specific logging with configurable levels
    and formatters. It replaces the verbose print statements with proper
    logging that can be easily controlled and doesn't interfere with coverage.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MarketLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            MarketLogger._initialized = True

    def _setup_logging(self):
        """Set up the logging configuration."""
        # Create main logger
        self.logger = logging.getLogger('market_simulation')
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler with custom formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Component-specific loggers
        self.environment_logger = logging.getLogger(
            'market_simulation.environment')
        self.seller_logger = logging.getLogger('market_simulation.seller')
        self.buyer_logger = logging.getLogger('market_simulation.buyer')
        self.simulation_logger = logging.getLogger(
            'market_simulation.simulation')

        # Set initial state to disabled
        self.enabled = False
        self.component_levels = {
            LogLevel.ENVIRONMENT: False,
            LogLevel.SELLER: False,
            LogLevel.BUYER: False,
            LogLevel.SIMULATION: False,
            LogLevel.ERROR: True  # Errors always enabled
        }

    def configure(self, verbose: Optional[str] = None):
        """
        Configure logging based on verbose setting.

        Args:
            verbose: Verbose setting ('all', 'seller', 'buyer', 'environment',
                    'simulation', or None)
        """
        # Reset all component levels
        for level in self.component_levels:
            if level != LogLevel.ERROR:
                self.component_levels[level] = False

        self.enabled = verbose is not None

        if verbose == 'all':
            for level in self.component_levels:
                self.component_levels[level] = True
        elif verbose in ['seller', 'buyer', 'environment', 'simulation']:
            level_enum = LogLevel(verbose)
            self.component_levels[level_enum] = True

        # Update logger levels based on configuration
        if self.enabled:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def _should_log(self, component: LogLevel) -> bool:
        """Check if logging is enabled for a component."""
        return self.enabled and self.component_levels.get(component, False)

    def log_environment(self, message: str, level: str = 'info'):
        """Log environment-related messages."""
        if self._should_log(LogLevel.ENVIRONMENT):
            getattr(self.environment_logger, level)(message)

    def log_seller(self, message: str, level: str = 'info'):
        """Log seller-related messages."""
        if self._should_log(LogLevel.SELLER):
            getattr(self.seller_logger, level)(message)

    def log_buyer(self, message: str, level: str = 'info'):
        """Log buyer-related messages."""
        if self._should_log(LogLevel.BUYER):
            getattr(self.buyer_logger, level)(message)

    def log_simulation(self, message: str, level: str = 'info'):
        """Log simulation-related messages."""
        if self._should_log(LogLevel.SIMULATION):
            getattr(self.simulation_logger, level)(message)

    def log_error(self, message: str, level: str = 'error'):
        """Log error messages (always enabled)."""
        if self.component_levels[LogLevel.ERROR]:
            getattr(self.logger, level)(message)

    def disable(self):
        """Disable all logging."""
        self.enabled = False
        for level in self.component_levels:
            if level != LogLevel.ERROR:
                self.component_levels[level] = False

    def enable_for_component(self, component: str):
        """Enable logging for a specific component."""
        self.enabled = True
        if component in ['seller', 'buyer', 'environment', 'simulation']:
            level_enum = LogLevel(component)
            self.component_levels[level_enum] = True


# Global logger instance
market_logger = MarketLogger()


def configure_logging(verbose: Optional[str] = None):
    """
    Configure the market simulation logging.

    Args:
        verbose: Verbose setting ('all', 'seller', 'buyer', 'environment',
                'simulation', or None)
    """
    market_logger.configure(verbose)


def log_environment(message: str, level: str = 'info'):
    """Log environment-related messages."""
    market_logger.log_environment(message, level)


def log_seller(message: str, level: str = 'info'):
    """Log seller-related messages."""
    market_logger.log_seller(message, level)


def log_buyer(message: str, level: str = 'info'):
    """Log buyer-related messages."""
    market_logger.log_buyer(message, level)


def log_simulation(message: str, level: str = 'info'):
    """Log simulation-related messages."""
    market_logger.log_simulation(message, level)


def log_error(message: str, level: str = 'error'):
    """Log error messages."""
    market_logger.log_error(message, level)


def disable_logging():
    """Disable all logging."""
    market_logger.disable()


def enable_component_logging(component: str):
    """Enable logging for a specific component."""
    market_logger.enable_for_component(component)


# Add more specific logging functions for detailed operations


def log_buyer_valuations(name: str, valuations: Any):
    """Log buyer valuations."""
    log_buyer(f"{name} generated valuations: {valuations}")


def log_buyer_demand(name: str, prices: Any, demand: Any):
    """Log buyer demand decision."""
    log_buyer(f"{name} demand decision - prices: {prices}, demand: {demand}")


def log_seller_price_selection(price_indices: Any, prices: Any):
    """Log seller price selection."""
    log_seller(f"Selected price indices: {price_indices}, "
               f"actual prices: {prices}")


def log_seller_ucb_update(product: int, price_idx: int, count: float,
                          value: float, ucb: float):
    """Log detailed UCB update."""
    log_seller(f"UCB update - Product {product}, Price idx {price_idx}: "
               f"count={count:.2f}, value={value:.3f}, UCB={ucb:.3f}")


def log_inventory_details(purchases: Any, capacity: float,
                          constraint_applied: bool):
    """Log inventory constraint details."""
    if constraint_applied:
        log_seller(f"Inventory constraint applied - "
                   f"Original purchases: {purchases}, Capacity: {capacity}")
    else:
        log_seller(f"Inventory check - Purchases: {purchases}, "
                   f"Capacity: {capacity}, No constraint needed")


def log_budget_details(prices: Any, budget: float, constraint_applied: bool):
    """Log budget constraint details."""
    if constraint_applied:
        log_seller(f"Budget constraint applied - "
                   f"Prices: {prices}, Budget: {budget}")
    else:
        log_seller(f"Budget check - Prices: {prices}, Budget: {budget}, "
                   f"No constraint needed")


def log_round_summary(round_num: int, prices: Any, purchases: Any,
                      reward: float, regret: float):
    """Log comprehensive round summary."""
    log_simulation(f"Round {round_num} summary - Prices: {prices}, "
                   f"Purchases: {purchases}, Reward: {reward:.3f}, "
                   f"Regret: {regret:.3f}")


def log_distribution_params(distribution: str, params: Any):
    """Log distribution parameters."""
    log_environment(f"Using {distribution} distribution with "
                    f"parameters: {params}")


def log_nonstationary_update(round_num: int, old_params: Any,
                             new_params: Any):
    """Log non-stationary parameter updates."""
    log_environment(f"Round {round_num} - Parameter update from "
                    f"{old_params} to {new_params}")


def log_algorithm_choice(algorithm: str):
    """Log the chosen algorithm."""
    log_seller(f"Using {algorithm.upper()} algorithm for price selection")


def log_ucb1_update(product: int, price_idx: int, count: float,
                    value: float, ucb: float):
    """Log UCB1 algorithm update."""
    log_seller(f"UCB1 update - Product {product}, Price idx {price_idx}: "
               f"count={count:.2f}, value={value:.3f}, UCB={ucb:.3f}")


def log_primal_dual_update(lambda_val: float, cost: float, b_remaining: float):
    """Log primal-dual algorithm update."""
    log_seller(f"Primal-dual update - Lambda: {lambda_val:.3f}, "
               f"Cost: {cost:.3f}, B remaining: {b_remaining:.3f}")


def log_arm_selection(algorithm: str, step: int, chosen_indices: Any):
    """Log arm selection for the specified algorithm."""
    log_seller(f"{algorithm.upper()} arm selection at step {step}: "
               f"indices {chosen_indices}")

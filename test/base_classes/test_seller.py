"""
Comprehensive test suite for the Seller class.
Designed to achieve 100% code coverage.
"""
import numpy as np
import sys
import os

# Add the project_work directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project_work.base_classes.seller import Seller
from project_work.base_classes.setting import Setting


class TestSeller:
    """Test class for Seller with comprehensive coverage."""

    def setup_method(self):
        """Setup method called before each test."""
        np.random.seed(42)  # For reproducible tests

    def test_init_basic(self):
        """Test basic Seller initialization."""
        setting = Setting(
            n_products=3,
            T=50,
            B=10,
            epsilon=0.1,
            verbose='seller'
        )
        seller = Seller(setting)

        assert len(seller.products) == 3
        assert seller.B == 10
        assert seller.T == 50
        assert seller.num_products == 3
        assert seller.num_prices == 10  # 1/0.1
        assert seller.total_steps == 0
        assert seller.cost_coeff == 0.5
        assert seller.eta == 0.1

    def test_init_with_random_t(self):
        """Test Seller initialization when T is None."""
        setting = Setting(n_products=2, T=None)
        seller = Seller(setting)

        assert seller.T >= 99 and seller.T <= 100

    def test_init_verbose_all(self):
        """Test Seller initialization with verbose='all'."""
        setting = Setting(n_products=2, verbose='all')
        seller = Seller(setting)

        # Verbose behavior is now handled by logger, not stored as attribute
        assert seller.setting == setting

    def test_init_verbose_false(self):
        """Test Seller initialization with verbose not seller or all."""
        setting = Setting(n_products=2, verbose='buyer')
        seller = Seller(setting)

        # Verbose behavior is now handled by logger, not stored as attribute
        assert seller.setting == setting

    def test_price_grid_structure(self):
        """Test price grid structure and dimensions."""
        setting = Setting(n_products=3, epsilon=0.2)
        seller = Seller(setting)

        expected_prices = int(1 / 0.2)  # Should be 5
        assert seller.price_grid.shape == (3, expected_prices)
        assert np.all(seller.price_grid >= 0.1)
        assert np.all(seller.price_grid <= 1.0)

    def test_price_grid_single_product(self):
        """Test price grid with single product."""
        setting = Setting(n_products=1, epsilon=0.5)
        seller = Seller(setting)

        assert seller.price_grid.shape == (1, 2)  # 1/0.5 = 2

    def test_initial_ucbs_infinite(self):
        """Test that initial UCBs are set to infinity."""
        setting = Setting(n_products=2, epsilon=0.1)
        seller = Seller(setting)

        assert np.all(seller.ucbs == np.inf)

    def test_yield_prices(self):
        """Test yield_prices method."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        chosen_indices = np.array([0, 1])
        prices = seller.yield_prices(chosen_indices)

        assert len(prices) == 2
        assert len(seller.history_chosen_prices) == 1
        assert np.array_equal(seller.history_chosen_prices[0], chosen_indices)
        assert seller.cost[0] > 0  # Cost should be calculated

    def test_update_lambda(self):
        """Test update_lambda method."""
        setting = Setting(n_products=2, T=100, B=50)
        seller = Seller(setting)

        lambda_prev = 0.5
        eta = 0.1
        seller.total_steps = 0
        seller.cost[0] = 2.0

        new_lambda = seller.update_lambda(lambda_prev, eta)

        # Lambda should be clipped between 0 and T/B
        assert 0 <= new_lambda <= seller.T / seller.B

    def test_update_lambda_edge_cases(self):
        """Test update_lambda with edge cases."""
        setting = Setting(n_products=1, T=10, B=5)
        seller = Seller(setting)
        seller.total_steps = 0
        seller.cost[0] = 1.0

        # Test lower bound
        lambda_low = seller.update_lambda(-10, 0.1)
        assert lambda_low >= 0

        # Test upper bound
        lambda_high = seller.update_lambda(100, 0.1)
        assert lambda_high <= seller.T / seller.B

    def test_update_primal_dual(self):
        """Test update_primal_dual method."""
        setting = Setting(n_products=2, epsilon=0.5, verbose='seller')
        seller = Seller(setting)

        actions = np.array([0, 1])
        rewards = np.array([0.3, 0.7])

        initial_B = seller.B
        seller.update_primal_dual(actions, rewards)

        assert seller.total_steps == 1
        assert seller.B < initial_B  # B should decrease
        assert seller.counts[0, 0] == 1
        assert seller.counts[1, 1] == 1
        assert len(seller.history_rewards) == 1

    def test_update_primal_dual_multiple_calls(self):
        """Test update_primal_dual with multiple calls."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # First update
        actions1 = np.array([0, 1])
        rewards1 = np.array([0.2, 0.8])
        seller.update_primal_dual(actions1, rewards1)

        # Second update with same actions
        actions2 = np.array([0, 1])
        rewards2 = np.array([0.4, 0.6])
        seller.update_primal_dual(actions2, rewards2)

        assert seller.total_steps == 2
        assert seller.counts[0, 0] == 2
        assert seller.counts[1, 1] == 2

        # Test incremental mean update
        expected_value_0_0 = (0.2 + 0.4) / 2
        assert abs(seller.values[0, 0] - expected_value_0_0) < 1e-10

    def test_inventory_constraint_lax_exceeding(self):
        """Test inventory_constraint with lax rule and exceeding capacity."""
        setting = Setting(n_products=5, B=2, budget_constraint="lax")
        seller = Seller(setting)

        purchases = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 > 2 capacity
        result = seller.inventory_constraint(purchases)

        # Should have exactly B non-zero purchases
        non_zero_count = np.count_nonzero(result)
        assert non_zero_count <= seller.B

    def test_inventory_constraint_lax_verbose(self):
        """Test inventory_constraint with lax rule, verbose, and exceeding."""
        setting = Setting(
            n_products=4,
            B=2,
            budget_constraint="lax",
            verbose='seller'
        )
        seller = Seller(setting)

        purchases = np.array([1.0, 2.0, 3.0, 4.0])
        result = seller.inventory_constraint(purchases)

        non_zero_count = np.count_nonzero(result)
        assert non_zero_count <= seller.B

    def test_inventory_constraint_strict_exceeding(self):
        """Test inventory_constraint with strict rule and capacity exceeded."""
        setting = Setting(n_products=3, B=1, budget_constraint="strict")
        seller = Seller(setting)

        purchases = np.array([1.0, 2.0, 3.0])  # 3 > 1 capacity
        result = seller.inventory_constraint(purchases)

        # Should return all zeros
        assert np.array_equal(result, np.zeros_like(purchases))

    def test_inventory_constraint_strict_verbose(self):
        """Test inventory_constraint with strict rule, verbose, exceeding."""
        setting = Setting(
            n_products=3,
            B=1,
            budget_constraint="strict",
            verbose='seller'
        )
        seller = Seller(setting)

        purchases = np.array([1.0, 2.0, 3.0])
        result = seller.inventory_constraint(purchases)

        assert np.array_equal(result, np.zeros_like(purchases))

    def test_inventory_constraint_no_exceeding(self):
        """Test inventory_constraint when not exceeding capacity."""
        setting = Setting(n_products=3, B=5, budget_constraint="lax")
        seller = Seller(setting)

        purchases = np.array([1.0, 0, 2.0])  # Only 2 non-zero < 5 capacity
        result = seller.inventory_constraint(purchases)

        # Should return unchanged
        assert np.array_equal(result, purchases)

    def test_budget_constraint_lax_exceeding(self):
        """Test budget_constraint with lax rule and exceeding capacity."""
        setting = Setting(n_products=3, B=1, budget_constraint="lax")
        seller = Seller(setting)

        purchases = np.array([1.0, 2.0, 3.0])
        result = seller.budget_constraint(purchases)

        # Similar logic to inventory_constraint
        non_zero_count = np.count_nonzero(result)
        assert non_zero_count <= seller.B

    def test_budget_constraint_strict_exceeding(self):
        """Test budget_constraint with strict rule and exceeding capacity."""
        setting = Setting(n_products=3, B=1, budget_constraint="strict")
        seller = Seller(setting)

        purchases = np.array([1.0, 2.0, 3.0])
        result = seller.budget_constraint(purchases)

        # Should return all zeros when exceeding
        assert np.array_equal(result, np.zeros_like(purchases))

    def test_pull_arm_basic(self):
        """Test pull_arm method basic functionality."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        chosen_indices = seller.pull_arm()

        assert len(chosen_indices) == 2
        assert all(0 <= idx < seller.num_prices for idx in chosen_indices)

    def test_pull_arm_verbose(self):
        """Test pull_arm method with verbose output."""
        setting = Setting(n_products=2, epsilon=0.5, verbose='seller')
        seller = Seller(setting)

        chosen_indices = seller.pull_arm()

        assert len(chosen_indices) == 2

    def test_pull_arm_after_updates(self):
        """Test pull_arm after some UCB updates."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # Update some UCB values manually
        seller.counts[0, 0] = 1
        seller.values[0, 0] = 0.8
        seller.ucbs[0, 0] = 0.9

        seller.counts[1, 1] = 1
        seller.values[1, 1] = 0.6
        seller.ucbs[1, 1] = 0.7

        # Since other UCBs start at infinity, we need to make them finite too
        # or the algorithm will always choose the untried options
        for i in range(2):
            for j in range(seller.num_prices):
                if seller.counts[i, j] == 0:
                    seller.counts[i, j] = 1
                    seller.values[i, j] = 0.1
                    seller.ucbs[i, j] = 0.2

        # Now set our specific test values
        seller.ucbs[0, 0] = 0.9  # Highest for product 0
        seller.ucbs[1, 1] = 0.7  # Highest for product 1

        chosen_indices = seller.pull_arm()
        assert chosen_indices[0] == 0  # Should choose higher UCB
        assert chosen_indices[1] == 1

    def test_pull_arm_exception_handling(self):
        """Test pull_arm exception handling."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # Corrupt UCBs to force an exception
        seller.ucbs = None

        result = seller.pull_arm()
        # Should return safe default array instead of None
        assert result is not None
        assert len(result) == 2
        assert all(idx == 0 for idx in result)  # Safe default values

    def test_update_method(self):
        """Test update method."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        purchased = np.array([0.8, 1.2])  # Second value will be clipped
        actions = np.array([0, 1])

        seller.update(purchased, actions)

        assert seller.total_steps == 1
        assert len(seller.history_rewards) == 1

    def test_update_with_clipping(self):
        """Test update method with values that need clipping."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        purchased = np.array([-0.5, 1.5])  # Both values will be clipped
        actions = np.array([0, 1])

        seller.update(purchased, actions)

        assert seller.total_steps == 1

    def test_reset_method(self):
        """Test reset method."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # Make some changes
        seller.total_steps = 5
        seller.counts[0, 0] = 3
        seller.values[0, 0] = 0.7
        seller.history_rewards = [1, 2, 3]
        seller.history_chosen_prices = [[0, 1], [1, 0]]

        # Reset
        new_setting = Setting(n_products=3, epsilon=0.1)
        seller.reset(new_setting)

        assert seller.setting == new_setting
        assert seller.total_steps == 0
        assert np.all(seller.counts == 0)
        assert np.all(seller.values == 0)
        assert len(seller.history_rewards) == 0
        assert len(seller.history_chosen_prices) == 0

    def test_ucb_calculation(self):
        """Test UCB value calculation in update_primal_dual."""
        setting = Setting(n_products=1, epsilon=1.0, T=100)
        seller = Seller(setting)

        actions = np.array([0])
        rewards = np.array([0.5])

        seller.update_primal_dual(actions, rewards)

        # Check UCB formula: value + sqrt(2 * log(T) / n)
        expected_ucb = 0.5 + np.sqrt(2 * np.log(100) / 1)
        assert abs(seller.ucbs[0, 0] - expected_ucb) < 1e-10

    def test_cost_calculation(self):
        """Test cost calculation in yield_prices."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # First and second price for each product
        chosen_indices = np.array([0, 1])
        prices = seller.yield_prices(chosen_indices)

        expected_cost = np.sum(prices) * seller.cost_coeff
        assert abs(seller.cost[0] - expected_cost) < 1e-10

    def test_lambda_update_in_primal_dual(self):
        """Test lambda update within update_primal_dual."""
        setting = Setting(n_products=1, T=10, B=5)
        seller = Seller(setting)

        # Set initial lambda to a value that should change significantly
        seller.lambda_pd[0] = np.array([1.5])

        # Use a low-cost action to force lambda decrease
        actions = np.array([0])  # Use lowest price index (should be 0.1)
        rewards = np.array([0.3])

        initial_lambda_value = float(seller.lambda_pd[0][0])
        initial_total_steps = seller.total_steps

        seller.update_primal_dual(actions, rewards)

        # Check that total_steps was incremented
        assert seller.total_steps == initial_total_steps + 1

        # Lambda should be updated if the condition is met
        if initial_total_steps < seller.T - 1:
            # lambda_pd[1] should now be different from initial lambda_pd[0]
            new_lambda_value = float(seller.lambda_pd[1][0])
            assert abs(new_lambda_value - initial_lambda_value) > 1e-10
            # With low cost, rho_pd > cost, so lambda should decrease
            assert new_lambda_value < initial_lambda_value

    def test_b_decrease_in_primal_dual(self):
        """Test that B decreases correctly in update_primal_dual."""
        setting = Setting(n_products=1, B=10)
        seller = Seller(setting)

        # Set a cost manually
        seller.cost[0] = 2.0
        initial_B = seller.B

        actions = np.array([0])
        rewards = np.array([0.5])

        seller.update_primal_dual(actions, rewards)

        assert seller.B == initial_B - seller.cost[0]

    def test_incremental_mean_calculation(self):
        """Test incremental mean calculation in update_primal_dual."""
        setting = Setting(n_products=1, epsilon=1.0)
        seller = Seller(setting)

        # First update
        actions = np.array([0])
        rewards1 = np.array([0.2])
        seller.update_primal_dual(actions, rewards1)

        # Second update
        rewards2 = np.array([0.8])
        seller.update_primal_dual(actions, rewards2)

        # Mean should be (0.2 + 0.8) / 2 = 0.5
        expected_mean = 0.5
        assert abs(seller.values[0, 0] - expected_mean) < 1e-10

    def test_history_tracking(self):
        """Test that history is tracked correctly."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # First round
        actions1 = np.array([0, 1])
        rewards1 = np.array([0.3, 0.7])
        seller.update_primal_dual(actions1, rewards1)

        # Second round
        actions2 = np.array([1, 0])
        rewards2 = np.array([0.4, 0.6])
        seller.update_primal_dual(actions2, rewards2)

        assert len(seller.history_rewards) == 2
        assert seller.history_rewards[0] == np.sum(rewards1)
        assert seller.history_rewards[1] == np.sum(rewards2)

    def test_integer_conversion_in_pull_arm(self):
        """Test integer conversion in pull_arm method."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # Ensure argmax returns integers
        chosen_indices = seller.pull_arm()

        for idx in chosen_indices:
            assert isinstance(idx, (int, np.integer))

    def test_integer_conversion_in_update(self):
        """Test integer conversion in update_primal_dual method."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = Seller(setting)

        # Use float indices to test conversion
        actions = np.array([0.0, 1.0])
        rewards = np.array([0.5, 0.6])

        seller.update_primal_dual(actions, rewards)

        assert seller.counts[0, 0] == 1
        assert seller.counts[1, 1] == 1

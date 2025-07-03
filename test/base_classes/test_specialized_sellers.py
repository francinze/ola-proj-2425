"""
Test module for specialized seller classes.
Tests the new seller architecture that matches project requirements.
"""
import pytest
import numpy as np
import sys
import os

# Add the project work directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..',
                             'project_work'))

from base_classes.setting import Setting
from base_classes.specialized_sellers import (
    BaseSeller, UCB1Seller, CombinatorialUCBSeller,
    PrimalDualSeller, SlidingWindowUCBSeller,
    create_seller_for_requirement
)


class TestBaseSeller:
    """Test the base seller class."""
    
    def test_init_basic(self):
        """Test basic initialization of base seller."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = BaseSeller(setting)
        
        assert seller.num_products == 2
        assert seller.price_grid.shape == (2, 2)  # epsilon=0.5 -> 2 prices
        assert seller.counts.shape == (2, 2)
        assert seller.values.shape == (2, 2)
        assert seller.ucbs.shape == (2, 2)
        
    def test_calculate_price_weighted_rewards(self):
        """Test price-weighted reward calculation."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = BaseSeller(setting)
        
        # First price for product 0, second for product 1
        actions = np.array([0, 1])
        rewards = np.array([0.3, 0.7])

        price_weighted = seller.calculate_price_weighted_rewards(
            actions, rewards)
        
        # Price grid with epsilon=0.5: [0.1, 1.0]
        expected = np.array([0.1 * 0.3, 1.0 * 0.7])  # [0.03, 0.7]
        assert np.allclose(price_weighted, expected)


class TestUCB1Seller:
    """Test the UCB1 seller class."""
    
    def test_init_with_inventory_constraint(self):
        """Test UCB1 seller initialization with inventory constraint."""
        setting = Setting(n_products=2, epsilon=0.5, algorithm="ucb1")
        seller = UCB1Seller(setting, use_inventory_constraint=True)
        
        assert seller.algorithm == "ucb1"
        assert seller.use_inventory_constraint is True
        
    def test_init_without_inventory_constraint(self):
        """Test UCB1 seller initialization without inventory constraint."""
        setting = Setting(n_products=2, epsilon=0.5, algorithm="ucb1")
        seller = UCB1Seller(setting, use_inventory_constraint=False)
        
        assert seller.algorithm == "ucb1"
        assert seller.use_inventory_constraint is False
        
    def test_pull_arm_basic(self):
        """Test basic arm pulling mechanism."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = UCB1Seller(setting)
        
        # Initial UCBs should be infinite, so first choice is arbitrary
        actions = seller.pull_arm()
        assert len(actions) == 2
        # Valid price indices
        assert all(0 <= action < 2 for action in actions)
        
    def test_update_ucb1_single_step(self):
        """Test UCB1 update with a single step."""
        setting = Setting(n_products=1, epsilon=1.0)  # Single price [0.1]
        seller = UCB1Seller(setting)
        
        actions = np.array([0])
        rewards = np.array([0.5])
        
        seller.update_ucb1(actions, rewards)
        
        assert seller.total_steps == 1
        assert seller.counts[0, 0] == 1
        
        # Price-weighted reward = 0.1 * 0.5 = 0.05
        expected_value = 0.1 * 0.5
        assert abs(seller.values[0, 0] - expected_value) < 1e-10
        
        # UCB = value + sqrt(2 * log(t) / n)
        expected_ucb = expected_value + np.sqrt(2 * np.log(1) / 1)
        assert abs(seller.ucbs[0, 0] - expected_ucb) < 1e-10
        
    def test_update_with_inventory_constraint(self):
        """Test update method with inventory constraint."""
        # Only 1 item capacity
        setting = Setting(n_products=2, epsilon=0.5, B=1)
        seller = UCB1Seller(setting, use_inventory_constraint=True)

        actions = np.array([0, 1])
        # Both products purchased (exceeds capacity)
        purchases = np.array([1.0, 1.0])

        # Mock the budget_constraint method for testing
        original_budget_constraint = seller.budget_constraint

        def mock_budget_constraint(purchases):
            # Simulate constraint: only one purchase allowed
            return np.array([1.0, 0.0])

        seller.budget_constraint = mock_budget_constraint

        seller.update(purchases, actions)

        assert seller.total_steps == 1
        # Should have updated both products' statistics
        assert seller.counts[0, 0] == 1
        assert seller.counts[1, 1] == 1

        # Restore original method
        seller.budget_constraint = original_budget_constraint


class TestCombinatorialUCBSeller:
    """Test the Combinatorial-UCB seller class."""
    
    def test_init(self):
        """Test Combinatorial-UCB seller initialization."""
        setting = Setting(n_products=3, epsilon=0.5)
        seller = CombinatorialUCBSeller(setting)
        
        assert seller.algorithm == "combinatorial_ucb"
        assert seller.use_inventory_constraint is True  # Always enabled
        assert seller.num_products == 3
        
    def test_inherits_ucb1_functionality(self):
        """Test that Combinatorial-UCB inherits UCB1 functionality."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = CombinatorialUCBSeller(setting)
        
        # Should have all UCB1 methods
        assert hasattr(seller, 'pull_arm')
        assert hasattr(seller, 'update_ucb1')
        assert hasattr(seller, 'calculate_price_weighted_rewards')


class TestPrimalDualSeller:
    """Test the Primal-Dual seller class."""
    
    def test_init(self):
        """Test Primal-Dual seller initialization."""
        setting = Setting(n_products=2, epsilon=0.5, T=100, B=50)
        seller = PrimalDualSeller(setting)
        
        assert seller.algorithm == "primal_dual"
        assert seller.eta == 0.1
        assert seller.rho_pd == setting.B / setting.T
        assert hasattr(seller, 'lambda_pd')
        assert hasattr(seller, 'cost')
        
    def test_pull_arm_with_lambda_adjustment(self):
        """Test arm pulling with lambda adjustment."""
        setting = Setting(n_products=2, epsilon=0.5, T=100)
        seller = PrimalDualSeller(setting)
        
        # Initialize lambda to non-zero value
        seller.lambda_pd[0] = 0.1
        
        actions = seller.pull_arm()
        assert len(actions) == 2
        assert all(0 <= action < 2 for action in actions)
        
    def test_update_primal_dual_integration(self):
        """Test that update method calls primal_dual correctly."""
        setting = Setting(n_products=1, epsilon=1.0, T=100)
        seller = PrimalDualSeller(setting)
        
        actions = np.array([0])
        purchases = np.array([0.5])
        
        # Mock budget_constraint for testing
        original_budget_constraint = seller.budget_constraint
        seller.budget_constraint = lambda x: x  # Pass through
        
        seller.update(purchases, actions)
        
        assert seller.total_steps == 1
        # Verify update_primal_dual was called (check appropriate metrics)
        assert len(seller.cost_history) == 1
        assert len(seller.history_rewards) == 1
        # Check that price weights were updated
        assert seller.price_weights[0, 0] != 0.0
        
        # Restore original method
        seller.budget_constraint = original_budget_constraint


class TestSlidingWindowUCBSeller:
    """Test the Sliding Window UCB seller class."""
    
    def test_init_default_window(self):
        """Test initialization with default window size."""
        setting = Setting(n_products=2, epsilon=0.5, T=100)
        seller = SlidingWindowUCBSeller(setting)
        
        assert seller.algorithm == "sliding_window_ucb"
        expected_window = max(1, int(np.sqrt(100)))  # sqrt(100) = 10
        assert seller.window_size == expected_window
        assert hasattr(seller, 'reward_history')
        assert hasattr(seller, 'action_history')
        
    def test_init_custom_window(self):
        """Test initialization with custom window size."""
        setting = Setting(n_products=2, epsilon=0.5, T=100)
        seller = SlidingWindowUCBSeller(setting, window_size=20)
        
        assert seller.window_size == 20
        
    def test_inherits_combinatorial_ucb(self):
        """Test that Sliding Window UCB inherits Combinatorial-UCB."""
        setting = Setting(n_products=2, epsilon=0.5)
        seller = SlidingWindowUCBSeller(setting)
        
        # Should inherit from CombinatorialUCBSeller
        assert isinstance(seller, CombinatorialUCBSeller)
        assert seller.use_inventory_constraint is True


class TestSellerFactory:
    """Test the seller factory function."""
    
    def test_create_seller_requirement_1(self):
        """Test factory creates UCB1Seller for requirement 1."""
        setting = Setting(n_products=1, epsilon=0.5)
        seller = create_seller_for_requirement(1, setting)
        
        assert isinstance(seller, UCB1Seller)
        assert seller.use_inventory_constraint is True  # Default
        
    def test_create_seller_requirement_1_no_inventory(self):
        """Test factory creates UCB1Seller without inventory constraint."""
        setting = Setting(n_products=1, epsilon=0.5)
        seller = create_seller_for_requirement(
            1, setting, use_inventory_constraint=False)
        
        assert isinstance(seller, UCB1Seller)
        assert seller.use_inventory_constraint is False
        
    def test_create_seller_requirement_2(self):
        """Test factory creates CombinatorialUCBSeller for requirement 2."""
        setting = Setting(n_products=3, epsilon=0.5)
        seller = create_seller_for_requirement(2, setting)
        
        assert isinstance(seller, CombinatorialUCBSeller)
        
    def test_create_seller_requirement_3(self):
        """Test factory creates PrimalDualSeller for requirement 3."""
        setting = Setting(n_products=1, epsilon=0.5)
        seller = create_seller_for_requirement(3, setting)
        
        assert isinstance(seller, PrimalDualSeller)
        
    def test_create_seller_requirement_4(self):
        """Test factory creates PrimalDualSeller for requirement 4."""
        setting = Setting(n_products=3, epsilon=0.5)
        seller = create_seller_for_requirement(4, setting)
        
        assert isinstance(seller, PrimalDualSeller)
        
    def test_create_seller_requirement_5(self):
        """Test factory creates SlidingWindowUCBSeller for requirement 5."""
        setting = Setting(n_products=3, epsilon=0.5, T=100)
        seller = create_seller_for_requirement(5, setting)
        
        assert isinstance(seller, SlidingWindowUCBSeller)
        
    def test_create_seller_requirement_5_custom_window(self):
        """Test factory creates SlidingWindowUCBSeller with custom window."""
        setting = Setting(n_products=3, epsilon=0.5, T=100)
        seller = create_seller_for_requirement(5, setting, window_size=25)
        
        assert isinstance(seller, SlidingWindowUCBSeller)
        assert seller.window_size == 25
        
    def test_create_seller_invalid_requirement(self):
        """Test factory raises error for invalid requirement."""
        setting = Setting(n_products=2, epsilon=0.5)
        
        with pytest.raises(ValueError, match="Unknown requirement number"):
            create_seller_for_requirement(99, setting)


class TestIntegrationWithEnvironment:
    """Integration tests simulating environment interactions."""
    
    def test_ucb1_seller_full_cycle(self):
        """Test complete UCB1 seller interaction cycle."""
        setting = Setting(n_products=2, epsilon=0.5, T=10)
        seller = UCB1Seller(setting)
        
        # Simulate multiple rounds
        for t in range(3):
            actions = seller.pull_arm()
            # Simulate random purchases
            purchases = np.random.random(2)
            seller.update(purchases, actions)
            
        assert seller.total_steps == 3
        assert len(seller.history_rewards) == 3
        
    def test_primal_dual_seller_full_cycle(self):
        """Test complete Primal-Dual seller interaction cycle."""
        setting = Setting(n_products=2, epsilon=0.5, T=10, B=5)
        seller = PrimalDualSeller(setting)
        
        # Simulate multiple rounds
        for t in range(3):
            actions = seller.pull_arm()
            # Simulate random purchases
            purchases = np.random.random(2)
            seller.update(purchases, actions)
            
        assert seller.total_steps == 3
        assert len(seller.history_rewards) == 3
        # Lambda should have been updated
        assert not np.allclose(seller.lambda_pd[:3], 0)
        
    def test_price_weighted_rewards_consistency(self):
        """Test that all sellers consistently use price-weighted rewards."""
        setting = Setting(n_products=2, epsilon=0.5)
        
        sellers = [
            UCB1Seller(setting),
            CombinatorialUCBSeller(setting),
            PrimalDualSeller(setting),
            SlidingWindowUCBSeller(setting)
        ]
        
        actions = np.array([0, 1])
        rewards = np.array([0.4, 0.6])
        
        for seller in sellers:
            # Calculate expected price-weighted rewards
            expected_pw_rewards = seller.calculate_price_weighted_rewards(
                actions, rewards)
            
            # For this setting, price grid is [0.1, 1.0]
            # So expected = [0.1 * 0.4, 1.0 * 0.6] = [0.04, 0.6]
            assert np.allclose(expected_pw_rewards, [0.04, 0.6])

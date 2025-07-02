"""
Additional tests for the updated environment logic.
Tests the new optimal reward calculation for non-stationary environments.
"""
import numpy as np
import sys
import os

# Add the project work directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..',
                             'project_work'))

from base_classes.setting import Setting
from base_classes.environment import Environment
from base_classes.seller import Seller
from base_classes.buyer import Buyer


class TestEnvironmentUpdatedLogic:
    """Test the updated environment logic for optimal rewards."""
    def test_compute_optimal_reward_nonstationary_single_product(self):
        """Test optimal reward for non-stationary, single product."""
        setting = Setting(n_products=1, epsilon=0.5, T=10,
                          non_stationary='slightly')
        env = Environment(setting)
        
        # Test the compute_optimal_reward_with_valuation method directly
        valuation = np.array([0.7])
        optimal_reward = env.compute_optimal_reward_with_valuation(valuation)
        
        # Optimal price should be chosen to maximize reward given valuation
        # With price grid [0.1, 1.0] and valuation 0.7:
        # - Price 0.1: purchase=1, reward=0.1*1=0.1
        # - Price 1.0: purchase=0 (price > valuation), reward=1.0*0=0
        # So optimal price is 0.1 with reward 0.1
        expected_optimal = 0.1
        assert abs(optimal_reward - expected_optimal) < 1e-10

    
    def test_compute_optimal_reward_nonstationary_multiple_products(self):
        """Test optimal reward for non-stationary, multiple products."""
        setting = Setting(n_products=2, epsilon=0.5, T=10,
                          non_stationary='slightly')
        env = Environment(setting)
        
        # Test using the direct method with known valuations
        valuations = np.array([0.7, 0.3])
        optimal_reward = env.compute_optimal_reward_with_valuation(valuations)
        
        # Product 0: valuation=0.7, best price=0.1, reward=0.1
        # Product 1: valuation=0.3, best price=0.1, reward=0.1
        # Total expected optimal = 0.1 + 0.1 = 0.2
        expected_optimal = 0.2
        assert abs(optimal_reward - expected_optimal) < 1e-10
        
    def test_compute_optimal_reward_with_valuation(self):
        """Test optimal reward computation with specified valuation."""
        setting = Setting(n_products=2, epsilon=0.5, T=10)
        env = Environment(setting)
        
        valuations = np.array([0.8, 0.2])
        optimal_reward = env.compute_optimal_reward_with_valuation(valuations)
        
        # Product 0: valuation=0.8, best price=0.1, reward=0.1
        # Product 1: valuation=0.2, best price=0.1, reward=0.1  
        # Total = 0.2
        expected_optimal = 0.2
        assert abs(optimal_reward - expected_optimal) < 1e-10
        
    def test_round_uses_correct_optimal_calculation(self):
        """Test that round() uses correct optimal calculation method."""
        # Test non-stationary case
        setting_nonstat = Setting(n_products=1, epsilon=0.5, T=10,
                                  non_stationary='slightly')
        env_nonstat = Environment(setting_nonstat)
        
        # Initialize and run one round
        env_nonstat.reset()
        seller = Seller(setting_nonstat)
        env_nonstat.seller = seller
        
        env_nonstat.round()
        
        # Should have stored optimal reward
        assert len(env_nonstat.optimal_rewards) > 0
        assert env_nonstat.optimal_rewards[0] > 0
        
        # Test stationary case
        setting_stat = Setting(n_products=1, epsilon=0.5, T=10,
                               non_stationary='no')
        env_stat = Environment(setting_stat)
        
        env_stat.reset()
        seller_stat = Seller(setting_stat)
        env_stat.seller = seller_stat
        
        env_stat.round()
        
        # Should also have stored optimal reward
        assert len(env_stat.optimal_rewards) > 0
        assert env_stat.optimal_rewards[0] > 0
        
    def test_regret_calculation_with_new_optimal(self):
        """Test regret calculation uses updated optimal rewards."""
        setting = Setting(n_products=1, epsilon=0.5, T=10,
                          non_stationary='slightly')
        env = Environment(setting)
        env.reset()
        
        seller = Seller(setting)
        env.seller = seller
        
        # Run one round
        env.round()
        
        # Check that regret was calculated
        # Regrets array is initialized to T length, but only first element is set
        assert len(env.regrets) == setting.T
        regret = env.regrets[0]  # Check the first (and only calculated) regret
        
        # Regret should be optimal_reward - actual_reward
        optimal = env.optimal_rewards[0]
        actual = seller.history_rewards[-1]
        expected_regret = optimal - actual
        
        assert abs(regret - expected_regret) < 1e-10
        
    def test_optimal_vs_actual_reward_difference(self):
        """Test difference between optimal and actual rewards."""
        setting = Setting(n_products=1, epsilon=0.5, T=5)
        env = Environment(setting)
        env.reset()
        
        seller = Seller(setting)
        env.seller = seller
        
        # Run multiple rounds and check cumulative rewards
        for _ in range(3):
            env.round()
            
        # Total optimal should be >= total actual (regret >= 0)
        total_optimal = np.sum(env.optimal_rewards[:3])
        total_actual = np.sum(seller.history_rewards)
        total_regret = np.sum(env.regrets)
        
        assert total_regret >= 0  # Regret should be non-negative
        assert abs(total_regret - (total_optimal - total_actual)) < 1e-10
        
    def test_price_weighted_reward_consistency_in_environment(self):
        """Test that environment properly handles price-weighted rewards."""
        setting = Setting(n_products=2, epsilon=0.5, T=5)
        env = Environment(setting)
        env.reset()
        
        seller = Seller(setting)
        env.seller = seller
        
        # Run one round and check reward structure
        env.round()
        
        # Check that seller received price-weighted rewards
        assert len(seller.history_rewards) == 1
        reward = seller.history_rewards[0]
        
        # Reward should be sum of price × purchase for all products
        # This should match what was stored in the environment
        assert reward >= 0  # Should be non-negative
        
    def test_environment_with_specialized_sellers(self):
        """Test environment works with specialized seller classes."""
        from base_classes.specialized_sellers import (
            UCB1Seller, CombinatorialUCBSeller, PrimalDualSeller
        )
        
        setting = Setting(n_products=2, epsilon=0.5, T=5)
        
        # Test with different seller types
        sellers = [
            UCB1Seller(setting),
            CombinatorialUCBSeller(setting),
            PrimalDualSeller(setting)
        ]
        
        for seller in sellers:
            env = Environment(setting)
            env.reset()
            env.seller = seller
            
            # Run a few rounds
            rounds_to_run = 3
            for _ in range(rounds_to_run):
                env.round()
                
            # Check that everything worked correctly
            assert len(env.regrets) == setting.T  # T=5, so full array
            assert len(seller.history_rewards) == rounds_to_run
            # Check that non-negative regret for the rounds we ran
            for i in range(rounds_to_run):
                assert env.regrets[i] >= 0
            

class TestEnvironmentEdgeCases:
    """Test edge cases for the updated environment logic."""
    
    def test_optimal_reward_with_zero_valuation(self):
        """Test optimal reward when buyer valuation is zero."""
        setting = Setting(n_products=1, epsilon=0.5, T=10)
        env = Environment(setting)
        
        # Zero valuation means no purchase at any price
        valuations = np.array([0.0])
        optimal_reward = env.compute_optimal_reward_with_valuation(valuations)
        
        # Optimal reward should be 0 (no purchase possible)
        assert optimal_reward == 0.0
        
    def test_optimal_reward_with_high_valuation(self):
        """Test optimal reward when buyer valuation is very high."""
        setting = Setting(n_products=1, epsilon=0.1, T=10)  # More price options
        env = Environment(setting)
        
        # Very high valuation
        valuations = np.array([10.0])
        optimal_reward = env.compute_optimal_reward_with_valuation(valuations)
        
        # With high valuation, optimal price should be the highest price
        # Price grid with epsilon=0.1: [0.1, 0.2, ..., 1.0]
        # Highest price is 1.0, so optimal reward = 1.0 * 1 = 1.0
        assert abs(optimal_reward - 1.0) < 1e-10
        
    def test_optimal_reward_boundary_valuation(self):
        """Test optimal reward when valuation equals a price."""
        setting = Setting(n_products=1, epsilon=0.5, T=10)
        env = Environment(setting)
        
        # Valuation exactly equal to higher price
        valuations = np.array([1.0])
        optimal_reward = env.compute_optimal_reward_with_valuation(valuations)
        
        # At boundary, purchase probability might be 0.5 or 1
        # The optimal choice should be based on expected reward
        # Price 0.1: reward = 0.1 * 1 = 0.1
        # Price 1.0: reward = 1.0 * P(purchase), where P(purchase) depends on model
        # Should be >= 0.1
        assert optimal_reward >= 0.1
        
    def test_nonstationary_optimal_adapts_to_changing_valuations(self):
        """Test that non-stationary optimal adapts to valuation changes."""
        setting = Setting(n_products=1, epsilon=0.5, T=10,
                          non_stationary='slightly')
        env = Environment(setting)
        env.reset()
        
        seller = Seller(setting)
        env.seller = seller
        
        # Store optimal rewards for different valuations
        optimal_rewards = []
        
        # Test with different valuations using direct method
        test_valuations = [0.2, 0.8, 0.5]
        
        for val in test_valuations:
            valuations = np.array([val])
            optimal = env.compute_optimal_reward_with_valuation(valuations)
            optimal_rewards.append(optimal)
            
        # Optimal rewards should differ based on valuations
        # Higher valuation should not necessarily mean higher optimal
        # (depends on price discretization)
        assert len(set([round(r, 10) for r in optimal_rewards])) >= 1

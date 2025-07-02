"""
Comprehensive test suite for the Setting class.
Designed to achieve 100% code coverage.
"""
import pytest
import numpy as np
import sys
import os

# Add the project_work directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project_work.base_classes.setting import Setting


class TestSetting:
    """Test class for Setting with comprehensive coverage."""

    def test_init_default_parameters(self):
        """Test Setting initialization with default parameters."""
        setting = Setting()
        assert setting.T >= 99 and setting.T <= 99  # Random value between 99-100
        assert setting.n_products == 10
        assert setting.epsilon == 0.1
        assert setting.distribution == "uniform"
        assert setting.budget_constraint == "lax"
        assert setting.verbose == 'all'
        assert setting.non_stationary == 'no'
        assert setting.cost_coeff == 0.5
        assert setting.B is not None

    def test_init_custom_parameters(self):
        """Test Setting initialization with custom parameters."""
        T = 50
        n_products = 5
        epsilon = 0.05
        B = 100
        distribution = "gaussian"
        budget_constraint = "strict"
        verbose = "seller"
        non_stationary = "slightly"
        dist_params = (0.2, 0.8)

        setting = Setting(
            T=T,
            n_products=n_products,
            epsilon=epsilon,
            B=B,
            distribution=distribution,
            budget_constraint=budget_constraint,
            verbose=verbose,
            non_stationary=non_stationary,
            dist_params=dist_params
        )

        assert setting.T == T
        assert setting.n_products == n_products
        assert setting.epsilon == epsilon
        assert setting.B == B
        assert setting.distribution == distribution
        assert setting.budget_constraint == budget_constraint
        assert setting.verbose == verbose
        assert setting.non_stationary == non_stationary

    def test_b_calculation_when_none(self):
        """Test B calculation when B is None."""
        setting = Setting(T=100, n_products=5, epsilon=0.1, B=None)
        # B should be calculated as T/n_products * cost_coeff * mean_cost
        prices = np.linspace(0.1, 1.0, int(1 / 0.1))
        mean_cost = np.mean(prices)
        expected_B = 100 / 5 * 0.5 * mean_cost
        assert setting.B == expected_B

    def test_create_params_stationary(self):
        """Test create_params for stationary case."""
        setting = Setting(non_stationary='no', dist_params=(0.1, 0.9))
        assert setting.dist_params == (0.1, 0.9)

    def test_create_params_uniform_slightly_nonstationary(self):
        """Test create_params for uniform distribution with slightly non-stationary."""
        np.random.seed(42)  # For reproducible tests
        setting = Setting(
            T=50,
            n_products=3,
            distribution="uniform",
            non_stationary='slightly'
        )
        params = setting.dist_params
        assert len(params) == 2  # Should return (high, low) tuple
        assert params[0].shape == (50, 3)  # high values
        assert params[1].shape == (50, 3)  # low values

    def test_create_params_uniform_highly_nonstationary(self):
        """Test create_params for uniform distribution with highly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="uniform",
            non_stationary='highly'
        )
        params = setting.dist_params
        assert len(params) == 2
        assert params[0].shape == (50, 3)
        assert params[1].shape == (50, 3)

    def test_create_params_bernoulli_slightly_nonstationary(self):
        """Test create_params for Bernoulli distribution with slightly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="bernoulli",
            non_stationary='slightly'
        )
        params = setting.dist_params
        assert len(params) == 2
        assert params[0].shape == (50, 1)  # n values (ones)
        assert params[1].shape == (50, 3)  # p values

    def test_create_params_bernoulli_highly_nonstationary(self):
        """Test create_params for Bernoulli distribution with highly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="bernoulli",
            non_stationary='highly'
        )
        params = setting.dist_params
        assert len(params) == 2
        assert params[0].shape == (50, 1)
        assert params[1].shape == (50, 3)

    def test_create_params_gaussian_slightly_nonstationary(self):
        """Test create_params for Gaussian distribution with slightly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="gaussian",
            non_stationary='slightly'
        )
        params = setting.dist_params
        assert len(params) == 2
        assert params[0].shape == (50, 3)  # mu values
        assert params[1].shape == (50, 3)  # sigma values

    def test_create_params_gaussian_highly_nonstationary(self):
        """Test create_params for Gaussian distribution with highly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="gaussian",
            non_stationary='highly'
        )
        params = setting.dist_params
        assert len(params) == 2
        assert params[0].shape == (50, 3)
        assert params[1].shape == (50, 3)

    def test_create_params_constant_slightly_nonstationary(self):
        """Test create_params for constant distribution with slightly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="constant",
            non_stationary='slightly'
        )
        params = setting.dist_params
        assert params.shape == (50, 3)

    def test_create_params_constant_highly_nonstationary(self):
        """Test create_params for constant distribution with highly non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="constant",
            non_stationary='highly'
        )
        params = setting.dist_params
        assert params.shape == (50, 3)

    def test_create_params_exponential_nonstationary(self):
        """Test create_params for exponential distribution non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="exponential",
            non_stationary='slightly'
        )
        assert setting.dist_params == 0

    def test_create_params_beta_nonstationary(self):
        """Test create_params for beta distribution non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="beta",
            non_stationary='slightly'
        )
        assert setting.dist_params == 0

    def test_create_params_lognormal_nonstationary(self):
        """Test create_params for lognormal distribution non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="lognormal",
            non_stationary='slightly'
        )
        assert setting.dist_params == 0

    def test_create_params_test_nonstationary(self):
        """Test create_params for test distribution non-stationary."""
        np.random.seed(42)
        setting = Setting(
            T=50,
            n_products=3,
            distribution="test",
            non_stationary='slightly'
        )
        assert setting.dist_params == 0

    def test_create_params_unknown_distribution_error(self):
        """Test create_params raises error for unknown distribution."""
        with pytest.raises(ValueError, match="Unknown distribution: unknown"):
            Setting(
                T=50,
                n_products=3,
                distribution="unknown",
                non_stationary='slightly'
            )

    def test_edge_case_small_t(self):
        """Test with very small T values."""
        setting = Setting(T=5, n_products=2, non_stationary='slightly')
        assert setting.T == 5

    def test_edge_case_large_t(self):
        """Test with large T values."""
        setting = Setting(T=1000, n_products=10, non_stationary='highly')
        assert setting.T == 1000

    def test_different_epsilon_values(self):
        """Test with different epsilon values."""
        setting1 = Setting(epsilon=0.01)
        setting2 = Setting(epsilon=0.5)
        assert setting1.epsilon == 0.01
        assert setting2.epsilon == 0.5

    def test_all_verbose_options(self):
        """Test all verbose options."""
        verbose_options = ['all', 'seller', 'buyer', None]
        for verbose in verbose_options:
            setting = Setting(verbose=verbose)
            assert setting.verbose == verbose

    def test_all_budget_constraint_options(self):
        """Test all budget constraint options."""
        constraint_options = ['lax', 'strict']
        for constraint in constraint_options:
            setting = Setting(budget_constraint=constraint)
            assert setting.budget_constraint == constraint

    def test_all_non_stationary_options(self):
        """Test all non-stationary options."""
        non_stat_options = ['no', 'slightly', 'highly', 'manual']
        for non_stat in non_stat_options:
            if non_stat == 'manual':
                setting = Setting(non_stationary=non_stat)
                assert setting.non_stationary == non_stat
            else:
                setting = Setting(non_stationary=non_stat)
                assert setting.non_stationary == non_stat

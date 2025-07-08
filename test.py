from Non_stationary.setting import Setting
from Non_stationary.environment import Environment
from Non_stationary.seller_sliding import SellerSliding
from Non_stationary.seller_prima_dual import PrimalDualSeller
import matplotlib.pyplot as plt
import numpy as np
import copy

# Set seed for reproducibility
seed = 2012
print(f'Seed used for this run: {seed}')
np.random.seed(seed)

# Create shared setting
base_setting = Setting(
    T=300,
    n_products=3,
    epsilon=0.2,
    distribution='gaussian',
    verbose=None,
    B=None,
    budget_constraint="lax",
    non_stationary='slightly',
    dist_params=None,
    algorithm='ucb_sliding'  # will be overridden later for primal_dual
)


env = Environment(base_setting)


env.play_all_rounds(plot=True)

# === Sliding Window UCB Run ===
setting_sliding = copy.deepcopy(base_setting)
setting_sliding.algorithm = 'ucb_sliding'
seller_sliding = SellerSliding(setting_sliding)
env_sliding = Environment(setting_sliding)
env_sliding.seller = seller_sliding
env_sliding.reset()
env_sliding.play_all_rounds(plot=False)
cum_regret_sliding = np.cumsum(env_sliding.regrets)

# === Primal Dual Run ===
setting_primal = copy.deepcopy(base_setting)
setting_primal.algorithm = 'primal_dual'
seller_primal = PrimalDualSeller(setting_primal)
env_primal = Environment(setting_primal)
env_primal.seller = seller_primal
env_primal.reset()
env_primal.play_all_rounds(plot=False)
cum_regret_primal = np.cumsum(env_primal.regrets)

# === Comparison Plot ===
plt.figure(figsize=(10, 6))
plt.plot(cum_regret_sliding, label='Sliding Window UCB')
plt.plot(cum_regret_primal, label='Primal Dual')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret: Sliding Window UCB vs Primal Dual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === Comparison Plot ===
plt.figure(figsize=(10, 6))
plt.plot(cum_regret_sliding, label='Sliding Window UCB')
plt.plot(cum_regret_primal, label='Primal Dual')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret: Sliding Window UCB vs Primal Dual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all(seller, optimal_rewards, regrets, ucb_history=None):
    """
    Plot all relevant learning progress statistics for the seller agent.
    Only plots UCB if the seller actually uses UCB (UCB1/Combinatorial-UCB).
    """
    # Check if this seller type should have UCB plots
    has_ucb = (hasattr(seller, 'ucbs') and
               seller.ucbs is not None and
               hasattr(seller, 'algorithm') and
               seller.algorithm in ['ucb1', 'combinatorial_ucb',
                                    'sliding_window_ucb'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Top-left: Cumulative reward
    rewards = np.array(seller.history_rewards)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    cumulative_reward = np.cumsum(rewards.sum(axis=1))
    axes[0, 0].plot(cumulative_reward, label="Cumulative Reward", color='blue')
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Cumulative Reward")
    axes[0, 0].set_title("Cumulative Reward Over Time")
    axes[0, 0].grid()

    # Top-right: Heatmap
    sns.heatmap(
        seller.counts, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=[f"{p:.2f}" for p in seller.price_grid[0]],
        yticklabels=range(seller.num_products),
        ax=axes[0, 1]
    )
    axes[0, 1].set_xlabel("Price")
    axes[0, 1].set_ylabel("Product")
    axes[0, 1].set_title("Selection Frequency (Product vs. Price)")

    # Bottom-left: Regret
    x = np.arange(len(optimal_rewards))
    actual_rewards = optimal_rewards - regrets

    if len(optimal_rewards) > 50:
        window = max(10, len(optimal_rewards) // 10)
        if window % 2 == 0:
            window += 1

        smoothed_optimal = np.convolve(
            optimal_rewards, np.ones(window) / window, mode='valid'
        )
        smoothed_actual = np.convolve(
            actual_rewards, np.ones(window) / window, mode='valid'
        )
        x_smoothed = np.arange(window // 2, len(optimal_rewards) - window // 2)

        axes[1, 0].plot(
            x_smoothed, smoothed_optimal,
            label="Optimal Reward (Smoothed)", color='blue'
        )
        axes[1, 0].plot(
            x_smoothed, smoothed_actual,
            label="Actual Reward (Smoothed)", color='green'
        )
        axes[1, 0].fill_between(
            x_smoothed,
            smoothed_actual,
            smoothed_optimal,
            where=(smoothed_optimal > smoothed_actual),
            color='red',
            alpha=0.2,
            label="Regret"
        )
    else:
        axes[1, 0].plot(x, optimal_rewards, label="Optimal Reward",
                        color='blue')
        axes[1, 0].plot(x, actual_rewards, label="Actual Reward",
                        color='green')
        axes[1, 0].fill_between(
            x,
            actual_rewards,
            optimal_rewards,
            where=(optimal_rewards > actual_rewards),
            color='red',
            alpha=0.2,
            label="Regret"
        )

    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].set_title("Optimal vs Actual Reward Per Round (Regret Shaded)")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # Bottom-right: UCB or Price History
    if has_ucb and ucb_history is not None:
        n_prices = ucb_history.shape[2]
        ucb_product0 = ucb_history[:, 0, :]
        for i in range(n_prices):
            ucb = ucb_product0[:, i]
            is_finite = np.isfinite(ucb)
            if np.any(is_finite):
                first_finite = np.argmax(is_finite)
                axes[1, 1].plot(
                    np.arange(first_finite, len(ucb)),
                    ucb[first_finite:],
                    label=f"Price {i}"
                )
                if first_finite > 0:
                    y_range = (np.nanmax(ucb[is_finite]) -
                               np.nanmin(ucb[is_finite]) + 1)
                    y_top = ucb[first_finite] + 2 * y_range
                    axes[1, 1].plot(
                        [first_finite - 1, first_finite],
                        [y_top, ucb[first_finite]],
                        color=axes[1, 1].lines[-1].get_color(),
                        linestyle='dashed'
                    )
        axes[1, 1].set_xlim(0, ucb_history.shape[0] - 1)
        axes[1, 1].set_ylim(0, 0.5)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("UCB")
        axes[1, 1].set_title("UCBs of Prices for product 0 Over Time")
        axes[1, 1].legend()
        axes[1, 1].grid()
    else:
        # For non-UCB algorithms (like Primal-Dual), show price history instead
        if (hasattr(seller, 'history_chosen_prices') and
                len(seller.history_chosen_prices) > 0):
            price_history = np.array(seller.history_chosen_prices)
            if price_history.ndim == 1:
                price_history = price_history[:, None]

            for i in range(price_history.shape[1]):
                axes[1, 1].plot(price_history[:, i], label=f"Product {i}",
                                alpha=0.7)

            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Chosen Price")
            axes[1, 1].set_title("Price Selection History")
            axes[1, 1].legend()
            axes[1, 1].grid()
        else:
            algorithm_name = getattr(seller, "algorithm", "Unknown")
            text = f'No UCB data available\n(Algorithm: {algorithm_name})'
            axes[1, 1].text(
                0.5, 0.5, text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[1, 1].transAxes
            )
            axes[1, 1].set_title("UCB/Price Data")

    plt.tight_layout()
    plt.show()


def plot_cumulative_regret_by_distribution(T, regrets_dict, n_trials):
    """
    Plot cumulative regret with uncertainty intervals for each distribution.
    regrets_dict: dict {distribution: 2D np.array [n_trials, T]}
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(T)
    for dist, regrets in regrets_dict.items():
        avg_regret = regrets.mean(axis=0)
        regret_sd = regrets.std(axis=0)
        lower = avg_regret - regret_sd / np.sqrt(n_trials)
        upper = avg_regret + regret_sd / np.sqrt(n_trials)
        plt.plot(x, avg_regret, label=f"{dist}")
        plt.fill_between(x, lower, upper, alpha=0.2)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret with Uncertainty Intervals")
    plt.legend()
    plt.grid()
    plt.show()


def plot_ucb_product0_by_distribution(ucb_dict):
    """
    Plot UCBs of product 0 for each distribution over time.
    ucb_dict: dict {distribution: 2D np.array [T, n_prices]}
    """
    plt.figure(figsize=(10, 6))
    for dist, ucb_matrix in ucb_dict.items():
        max_ucb = ucb_matrix.max(axis=1)
        plt.plot(max_ucb, label=f"{dist}")
    plt.xlabel("Round")
    plt.ylabel("UCB (Product 0, max over prices)")
    plt.title("UCBs of Product 0 Over Time by Distribution")
    plt.legend()
    plt.grid()
    plt.show()


def plot_environment_results(environment):
    """
    Plot all results from an Environment instance after a single
    simulation run.

    Args:
        environment: Environment instance after running play_all_rounds()
    """
    # Check if the environment has the necessary data
    if (not hasattr(environment, 'optimal_rewards') or
            not hasattr(environment, 'regrets')):
        print("Warning: Environment does not have simulation results to plot")
        return

    # Only pass UCB history if the seller type supports it
    ucb_history = None
    if (hasattr(environment, 'ucb_history') and
        hasattr(environment.seller, 'algorithm') and
        environment.seller.algorithm in ['ucb1', 'combinatorial_ucb',
                                         'sliding_window_ucb']):
        ucb_history = getattr(environment, 'ucb_history', None)

    # Plot results from a single simulation run
    plot_all(
        environment.seller,
        environment.optimal_rewards,
        environment.regrets,
        ucb_history
    )


def plot_environment_simulation_results(environment, regrets_dict, ucb_dict,
                                        n_trials):
    """
    Plot results from Environment.run_simulation() with multiple trials.

    Args:
        environment: Environment instance used for simulation
        regrets_dict: Dictionary of regrets by distribution
                     {dist: [n_trials, T]}
        ucb_dict: Dictionary of UCB data by distribution
                 {dist: [n_trials, T, n_prices]}
        n_trials: Number of trials run
    """
    # Plot cumulative regret comparison across distributions
    plot_cumulative_regret_by_distribution(
        environment.setting.T,
        regrets_dict,
        n_trials
    )

    # Plot UCB comparison across distributions
    # Convert UCB dict to averaged format expected by plot function
    avg_ucb_dict = {}
    for dist, ucb_data in ucb_dict.items():
        if len(ucb_data) > 0:
            # Average across trials
            avg_ucb = ucb_data.mean(axis=0)
            # Make sure we have the right shape for product 0
            if avg_ucb.ndim == 3:
                # Shape is [T, n_products, n_prices] - take product 0
                avg_ucb_dict[dist] = avg_ucb[:, 0, :]
            elif avg_ucb.ndim == 2:
                # Shape is already [T, n_prices] for product 0
                avg_ucb_dict[dist] = avg_ucb
            else:
                print(f"Warning: Unexpected UCB shape for {dist}: "
                      f"{avg_ucb.shape}")

    if avg_ucb_dict:
        plot_ucb_product0_by_distribution(avg_ucb_dict)

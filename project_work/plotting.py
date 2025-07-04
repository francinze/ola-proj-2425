import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def cum_reward(ax, seller):
    rewards = np.array(seller.history_rewards)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    cumulative_reward = np.cumsum(rewards.sum(axis=1))
    ax.plot(cumulative_reward, label="Cumulative Reward", color='blue')
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Time")
    ax.grid()


def cum_regret(ax, environment):
    """
    Plot cumulative regret for the seller agent.
    """
    if not hasattr(environment, 'regrets'):
        print("Warning: Seller does not have regrets data to plot")
        return

    cumulative_regret = np.cumsum(environment.regrets)
    ax.plot(cumulative_regret, label="Cumulative Regret", color='red')
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret Over Time")
    ax.grid()


def product_selection_heatmap(ax, seller):
    price_history = np.array(seller.history_chosen_prices)
    if price_history.ndim == 1:
        price_history = price_history[:, None]

    # Create heatmap data from price history
    n_products = price_history.shape[1]
    if hasattr(seller, 'price_grid'):
        n_prices = len(seller.price_grid[0])
    else:
        n_prices = 10

    # Count frequency of each price for each product
    heatmap_data = np.zeros((n_products, n_prices))

    for product in range(n_products):
        for step in range(len(price_history)):
            price_idx = int(price_history[step, product])
            heatmap_data[product, price_idx] += 1

    # Create x-axis labels
    if hasattr(seller, 'price_grid'):
        x_labels = [f"{p:.2f}" for p in seller.price_grid[0]]
    else:
        x_labels = range(n_prices)

    sns.heatmap(
        heatmap_data, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=x_labels,
        yticklabels=range(n_products),
        ax=ax
    )
    ax.set_xlabel("Price")
    ax.set_ylabel("Product")
    ax.set_title("Selection Frequency (Product vs. Price)")


def plot_regret(ax, optimal_rewards, regrets):
    actual_rewards = optimal_rewards - regrets

    x, optimal, smoothed_opt = smooth_data(optimal_rewards)
    _, actual, smoothed_act = smooth_data(actual_rewards)

    ax.plot(
        x,
        optimal,
        label="Optimal Reward" + (" (Smoothed)" if smoothed_opt else ""),
        color='blue'
    )
    ax.plot(
        x,
        actual,
        label="Actual Reward" + (" (Smoothed)" if smoothed_act else ""),
        color='green'
    )
    ax.fill_between(
        x,
        actual,
        optimal,
        where=(optimal > actual),
        color='red',
        alpha=0.2,
        label="Regret"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Optimal vs Actual Reward Per Round (Regret Shaded)")
    ax.legend()
    ax.grid()


def plot_ucb(ax, ucb_history):
    n_prices = ucb_history.shape[2]
    ucb_product0 = ucb_history[:, 0, :]
    for i in range(n_prices):
        ucb = ucb_product0[:, i]
        is_finite = np.isfinite(ucb)
        if np.any(is_finite):
            first_finite = np.argmax(is_finite)
            ax.plot(
                np.arange(first_finite, len(ucb)),
                ucb[first_finite:],
                label=f"Price {i}"
            )
            if first_finite > 0:
                y_range = (
                    np.nanmax(ucb[is_finite]) - np.nanmin(ucb[is_finite]) + 1
                )
                y_top = ucb[first_finite] + 2 * y_range
                ax.plot(
                    [first_finite - 1, first_finite],
                    [y_top, ucb[first_finite]],
                    color=ax.lines[-1].get_color(),
                    linestyle='dashed'
                )
    ax.set_xlim(0, ucb_history.shape[0] - 1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("UCB")
    ax.set_title("UCBs of Prices for product 0 Over Time")
    ax.legend()
    ax.grid()


def plot_price_history(ax, seller):
    # For non-UCB algorithms (like Primal-Dual), show price history instead
    if (
        hasattr(seller, 'history_chosen_prices') and
        len(seller.history_chosen_prices) > 0
    ):
        price_history = np.array(seller.history_chosen_prices)
        if price_history.ndim == 1:
            price_history = price_history[:, None]

        for i in range(price_history.shape[1]):
            # Updated selection code:
            x_vals, y_vals, smoothed = smooth_data(price_history[:, i])
            ax.plot(
                x_vals,
                y_vals,
                label=f"Product {i}" + (" (Smoothed)" if smoothed else ""),
                alpha=0.7
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Chosen Price")
        ax.set_yticklabels(np.unique(price_history[:, :]))
        ax.set_title("Price Selection History")
        ax.legend()
        ax.grid()
    else:
        algorithm_name = getattr(seller, "algorithm", "Unknown")
        text = f'No UCB data available\n(Algorithm: {algorithm_name})'
        ax.text(
            0.5, 0.5, text,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
        ax.set_title("UCB/Price Data")


def plot_cumulative_regret_by_distribution(ax, T, regrets_dict, n_trials):
    """
    Plot cumulative regret with uncertainty intervals for each distribution.
    regrets_dict: dict {distribution: 2D np.array [n_trials, T]}
    """
    x = np.arange(T)
    for dist, regrets in regrets_dict.items():
        avg_regret = regrets.mean(axis=0)
        regret_sd = regrets.std(axis=0)
        lower = avg_regret - regret_sd / np.sqrt(n_trials)
        upper = avg_regret + regret_sd / np.sqrt(n_trials)
        # Updated selection code:
        x_vals, y_vals, smoothed = smooth_data(avg_regret)
        ax.plot(
            x_vals,
            y_vals,
            label=f"{dist}" + (" (Smoothed)" if smoothed else ""),
            alpha=0.7
        )
        ax.fill_between(x, lower, upper, alpha=0.2)
    ax.set_xlabel("Round")
    ax.set_xlabel("Cumulative Regret")
    ax.set_title("Cumulative Regret with Uncertainty Intervals")
    ax.legend()
    ax.grid()


def plot_all(environment):
    """
    Plot all relevant learning progress statistics for the seller agent.
    Only plots UCB if the seller actually uses UCB (UCB1/Combinatorial-UCB).
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

    # Check if this seller type should have UCB plots
    has_ucb = (
        hasattr(environment.seller, 'ucbs') and
        environment.seller.ucbs is not None and
        hasattr(environment.seller, 'algorithm') and
        environment.seller.algorithm in [
            'ucb1', 'combinatorial_ucb', 'sliding_window_ucb'
        ]
    )

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)

    # Top-left: Cumulative reward
    cum_reward(axes[0, 0], environment.seller)

    # Top-right: Cumulative regret
    cum_regret(axes[0, 1], environment)

    # Middle-left: Heatmap
    product_selection_heatmap(axes[1, 0], environment.seller)

    # Middle-right: UCB or Price History
    if has_ucb and ucb_history is not None:
        plot_ucb(axes[1, 1], ucb_history)
    else:
        plot_price_history(axes[1, 1], environment.seller)

    # Bottom-left: Regret
    plot_regret(axes[2, 0], environment.optimal_rewards, environment.regrets)

    # Bottom-right: Cumulative regret
    plot_cumulative_regret_by_distribution(
        axes[2, 1],
        T=len(environment.optimal_rewards),
        regrets_dict={'All': np.array([environment.regrets])},
        n_trials=1
    )

    plt.suptitle(
        f"Learning Progress of {environment.seller.algorithm} Seller",
        fontsize=16
    )
    plt.show()


def smooth_data(data, threshold=50, window_divisor=5, min_window=10):
    """
    Generic smoothing function for time series data.

    Args:
        data: 1D array of data to smooth
        threshold: minimum length to apply smoothing
        window_divisor: divisor for calculating window size
        min_window: minimum window size

    Returns:
        tuple: (x_values, smoothed_data) or (original_indices, original_data)
    """
    if len(data) > threshold:
        window = max(min_window, len(data) // window_divisor)
        if window % 2 == 0:
            window += 1

        smoothed = np.convolve(data, np.ones(window) / window, mode='same')
        x_smoothed = np.arange(len(data))
        return x_smoothed, smoothed, True
    else:
        return np.arange(len(data)), data, False

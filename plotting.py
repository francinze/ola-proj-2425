import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cum_reward(seller):
    """
    Plot cumulative reward over time.
    """
    rewards = np.array(seller.history_rewards)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    cumulative_reward = np.cumsum(rewards.sum(axis=1))

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_reward, label="Cumulative Reward", color='blue')
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.grid()
    plt.show()


def plot_heatmap(seller):
    """
    Plot heatmap of product-price selection frequency.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        seller.counts, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=[f"{p:.2f}" for p in seller.price_grid[0]],
        yticklabels=range(seller.num_products)
    )
    plt.xlabel("Price")
    plt.ylabel("Product")
    plt.title("Selection Frequency (Product vs. Price)")
    plt.show()


def plot_ucb(ucb_history):
    """
    Plot UCBs of all products over time.
    UCBs that start at infinity are shown as lines coming from above
    when they first become finite.
    """
    plt.figure(figsize=(10, 6))
    n_prices = ucb_history.shape[2]
    ucb_product0 = ucb_history[:, 0, :]
    for i in range(n_prices):
        ucb = ucb_product0[:, i]
        is_finite = np.isfinite(ucb)
        if np.any(is_finite):
            first_finite = np.argmax(is_finite)
            # Plot the finite part
            plt.plot(
                np.arange(first_finite, len(ucb)),
                ucb[first_finite:],
                label=f"Price {i}"
            )
            # If the first value is inf, draw a vertical line from far above
            if first_finite > 0:
                y_top = ucb[first_finite] + 2 * (
                    np.nanmax(ucb[is_finite]) - np.nanmin(ucb[is_finite]) + 1
                )
                plt.plot(
                    [first_finite - 1, first_finite],
                    [y_top, ucb[first_finite]],
                    color=plt.gca().lines[-1].get_color(),
                    linestyle='dashed'
                )
    plt.xlim(0, ucb_history.shape[0] - 1)
    plt.ylim(0, 0.5)
    plt.xlabel("Step")
    plt.ylabel("UCB")
    plt.title("UCBs of Prices for product 0 Over Time")
    plt.legend()
    plt.grid()
    plt.show()


def plot_regret(optimal_rewards, regrets):
    """
    Plot per-round (non-cumulative) regret over time,
    showing optimal and actual rewards.
    The area between the two lines is shaded to represent regret.
    For long series (>50), apply curve smoothing (same as average reward).
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(optimal_rewards))
    actual_rewards = optimal_rewards - regrets

    if len(optimal_rewards) > 50:
        # Dynamically set window size: 1/10th of series length, at least 10
        window = max(10, len(optimal_rewards) // 10)
        if window % 2 == 0:
            window += 1  # Ensure window is odd for centering

        # Smooth both optimal and actual rewards
        smoothed_optimal = np.convolve(
            optimal_rewards, np.ones(window) / window, mode='valid'
        )
        smoothed_actual = np.convolve(
            actual_rewards, np.ones(window) / window, mode='valid'
        )
        x_smoothed = np.arange(window // 2, len(optimal_rewards) - window // 2)

        plt.plot(
            x_smoothed, smoothed_optimal,
            label="Optimal Reward (Smoothed)", color='blue'
        )
        plt.plot(
            x_smoothed, smoothed_actual,
            label="Actual Reward (Smoothed)", color='green'
        )
        plt.fill_between(
            x_smoothed,
            smoothed_actual,
            smoothed_optimal,
            where=(smoothed_optimal > smoothed_actual),
            color='red',
            alpha=0.2,
            label="Regret"
        )
    else:
        plt.plot(x, optimal_rewards, label="Optimal Reward", color='blue')
        plt.plot(x, actual_rewards, label="Actual Reward", color='green')
        plt.fill_between(
            x,
            actual_rewards,
            optimal_rewards,
            where=(optimal_rewards > actual_rewards),
            color='red',
            alpha=0.2,
            label="Regret"
        )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Optimal vs Actual Reward Per Round (Regret Shaded)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_all(seller, optimal_rewards, regrets, ucb_history=None):
    """
    Plot all relevant learning progress statistics for the seller agent.
    """
    plot_cum_reward(seller)
    plot_heatmap(seller)
    plot_regret(optimal_rewards, regrets)

    if ucb_history is not None:
        plot_ucb(ucb_history)


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

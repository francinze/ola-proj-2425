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


def plot_avg_reward(seller):
    """
    Plot reward per round. For long series (>50), apply curve smoothing.
    """
    rewards = np.array(seller.history_rewards)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    avg_rewards = rewards.sum(axis=1)

    x = np.arange(len(avg_rewards))
    if len(avg_rewards) > 50:
        # Dynamically set window size: 1/10th of series length, at least 10
        window = max(10, len(avg_rewards) // 10)
        if window % 2 == 0:
            window += 1  # Ensure window is odd for centering
        smoothed = np.convolve(
            avg_rewards, np.ones(window) / window, mode='same'
        )
        plt.plot(
            x,
            smoothed,
            label="Smoothed Reward per Round",
            color='green'
        )
    else:
        plt.plot(x, avg_rewards, label="Reward per Round", color='green')

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Per Round")
    plt.grid()
    plt.show()


def plot_selection_frequency(seller):
    """
    Plot selection frequency of products and prices.
    """
    product_counts = np.sum(seller.counts, axis=1)
    price_counts = np.sum(seller.counts, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Product selection frequency
    axs[0].bar(range(seller.num_products), product_counts)
    axs[0].set_xlabel("Product ID")
    axs[0].set_ylabel("Total Times Any Price Chosen")
    axs[0].set_title("Product Selection Frequency")
    axs[0].set_xticks(range(seller.num_products))
    axs[0].set_xticklabels([f"{i}" for i in range(seller.num_products)])

    # Price selection frequency
    price_indices = np.arange(len(seller.price_grid[0]))
    axs[1].bar(price_indices, price_counts)
    axs[1].set_xlabel("Price")
    axs[1].set_ylabel("Total Times Chosen Across Products")
    axs[1].set_title("Price Selection Frequency")
    x_ticks = [f"{p:.2f}" for p in seller.price_grid[0]]
    axs[1].set_xticks(price_indices)
    axs[1].set_xticklabels(x_ticks, rotation=45)

    plt.tight_layout()
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


def plot_all(seller):
    """
    Plot all relevant learning progress statistics for the seller agent.
    """
    plot_cum_reward(seller)
    plot_avg_reward(seller)
    plot_selection_frequency(seller)
    plot_heatmap(seller)

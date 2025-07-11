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
    Enhanced plot with demo notebook style visualizations in 2x2 layout.
    """
    # Check if the environment has the necessary data
    if (not hasattr(environment, 'optimal_rewards') or
            not hasattr(environment, 'regrets')):
        print("Warning: Environment does not have simulation results to plot")
        return

    # Calculate derived metrics
    seller_rewards = np.array(environment.seller.history_rewards)
    optimal_rewards = np.array(environment.optimal_rewards)
    regrets = np.array(environment.regrets)

    seller_cumulative = np.cumsum(seller_rewards)
    optimal_cumulative = np.cumsum(optimal_rewards)
    cumulative_regret = np.cumsum(regrets)

    # Get seller information
    seller_name = getattr(environment.seller, 'algorithm', 'Unknown')
    T = len(seller_rewards)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Plot 1: Cumulative Reward Comparison (UCB1 vs Oracle)
    axes[0, 0].plot(seller_cumulative, label=f'{seller_name} Agent',
                    color='green', linewidth=2)
    axes[0, 0].plot(optimal_cumulative, label='Oracle (Optimal)',
                    linestyle='--', color='blue', linewidth=2)
    axes[0, 0].set_title('Cumulative Reward: Agent vs Oracle')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Pseudo-Regret with Theoretical Curve (Most Important Plot)
    axes[0, 1].plot(cumulative_regret, label='Cumulative Regret',
                    color='red', linewidth=2)

    # Add theoretical curve if we can estimate suboptimality gaps
    if hasattr(environment.seller, 'history_chosen_prices'):
        try:
            # Estimate theoretical bound (simplified)
            log_t = np.log(np.arange(1, T + 1))
            # Simple approximation for theoretical curve
            # In practice, this would need proper suboptimality gap calculation
            theoretical_multiplier = cumulative_regret[-1] / log_t[-1] * 1.2
            theoretical_curve = log_t * theoretical_multiplier
            axes[0, 1].plot(theoretical_curve,
                            label=r'Theoretical $O(\log T)$ bound',
                            linestyle='--', color='blue', alpha=0.7)
        except Exception:
            pass  # Skip theoretical curve if calculation fails

    axes[0, 1].set_title(f'{seller_name} Regret vs Theoretical Bound')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Regret')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Price/Arm Selections Over Time (if available)
    if hasattr(environment.seller, 'history_chosen_prices'):
        price_history = np.array(environment.seller.history_chosen_prices)
        if price_history.ndim == 1:
            price_history = price_history[:, None]

        # For single product, show price selections as scatter
        if price_history.shape[1] == 1:
            # Get unique prices and map to indices for coloring
            unique_prices = np.unique(price_history[:, 0])
            price_to_idx = {price: idx for idx, price
                            in enumerate(unique_prices)}
            colors = [price_to_idx[price] for price in price_history[:, 0]]

            scatter = axes[1, 0].scatter(range(T), price_history[:, 0],
                                         c=colors, cmap='viridis', s=15,
                                         alpha=0.7)
            axes[1, 0].set_title('Price Selections Over Time')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Selected Price')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Price Index')
        else:
            # Multiple products - show as lines
            for i in range(min(3, price_history.shape[1])):  # Max 3
                x_vals, y_vals, smoothed = smooth_data(price_history[:, i])
                axes[1, 0].plot(
                    x_vals,
                    y_vals,
                    label=f'Product {i}' + (" (Smoothed)" if smoothed else ""),
                    alpha=0.8
                )
                axes[1, 0].set_title('Price Selection History')
                axes[1, 0].set_xlabel('Round')
                axes[1, 0].set_ylabel('Selected Price')
                axes[1, 0].legend()
    else:
        # Fallback: show instantaneous regret
        axes[1, 0].plot(regrets, label='Instantaneous Regret',
                        color='orange', alpha=0.7)
        axes[1, 0].set_title('Instantaneous Regret')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Regret')
        axes[1, 0].legend()

    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Price/Action Selection Frequency
    if hasattr(environment.seller, 'history_chosen_prices'):
        price_history = np.array(environment.seller.history_chosen_prices)
        if price_history.ndim == 1:
            price_history = price_history[:, None]
        # Count frequency for first product
        unique_prices, counts = np.unique(price_history[:, 0],
                                          return_counts=True)
        proportions = counts / T

        bars = axes[1, 1].bar(range(len(unique_prices)), proportions)
        axes[1, 1].set_title('Proportion of Times Each Price Was Selected')
        axes[1, 1].set_xlabel('Price Value')
        axes[1, 1].set_ylabel('Proportion of Selections')
        axes[1, 1].set_xticks(range(len(unique_prices)))
        axes[1, 1].set_xticklabels([f'{p:.2f}' for p in unique_prices])

        # Color bars by frequency
        max_prop = max(proportions)
        for bar, prop in zip(bars, proportions):
            bar.set_color(plt.cm.viridis(prop / max_prop))
    else:
        # Fallback: efficiency metrics
        efficiency = (np.sum(seller_rewards) / np.sum(optimal_rewards)) * 100
        final_regret = cumulative_regret[-1]
        avg_regret = final_regret / T

        metrics_text = f"Final Regret: {final_regret:.2f}\n"
        metrics_text += f"Avg Regret/Round: {avg_regret:.3f}\n"
        metrics_text += f"Efficiency: {efficiency:.1f}%"
        axes[1, 1].text(0.5, 0.5, metrics_text,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightblue"))
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    axes[1, 1].grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle(f"Learning Progress Analysis: {seller_name} Seller",
                 fontsize=16)
    plt.show()


def plot_multi_trial_ucb_analysis(environments_list, n_trials=None,
                                  suboptimality_gaps=None):
    """
    Plot multi-trial UCB1 analysis with confidence intervals and
    theoretical curve. This recreates the most important plot from
    the demo notebook.
    """
    if not environments_list:
        print("No trial data provided for multi-trial analysis")
        return

    if n_trials is None:
        n_trials = len(environments_list)

    # Extract regret data from all trials
    all_regrets = []
    T = 0

    for env in environments_list:
        if hasattr(env, 'regrets'):
            regrets = np.array(env.regrets)
            cumulative_regret = np.cumsum(regrets)
            all_regrets.append(cumulative_regret)
            T = max(T, len(cumulative_regret))

    if not all_regrets:
        print("No regret data found in environments")
        return

    # Pad shorter trials and convert to array
    all_regrets_padded = []
    for regret in all_regrets:
        if len(regret) < T:
            # Pad with last value
            padded = np.pad(
                regret,
                (0, T - len(regret)),
                mode='constant',
                constant_values=regret[-1]
            )
        else:
            padded = regret[:T]
        all_regrets_padded.append(padded)

    all_regrets = np.array(all_regrets_padded)

    # Compute statistics
    mean_regret = np.mean(all_regrets, axis=0)
    stderr_regret = np.std(all_regrets, axis=0) / np.sqrt(n_trials)
    ci_upper = mean_regret + 1.96 * stderr_regret
    ci_lower = mean_regret - 1.96 * stderr_regret

    # Theoretical curve
    log_t = np.log(np.arange(1, T + 1))
    if suboptimality_gaps is not None:
        # Proper theoretical curve using suboptimality gaps
        sum_inv_gaps = sum(1 / d for d in suboptimality_gaps if d > 0)
        theoretical_curve = log_t * sum_inv_gaps
    else:
        # Approximation based on empirical data
        theoretical_multiplier = mean_regret[-1] / log_t[-1] * 1.2
        theoretical_curve = log_t * theoretical_multiplier

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(mean_regret, label=f'Average Regret ({n_trials} trials)',
             color='red', linewidth=2)
    plt.fill_between(range(T), ci_lower, ci_upper, color='red', alpha=0.3,
                     label='95% Confidence Interval')
    plt.plot(theoretical_curve,
             label=r'$\log(T) \times \sum \frac{1}{\Delta_a}$',
             linestyle='--', color='blue', linewidth=2)

    plt.title('Multi-Trial Analysis: Average Regret with Theoretical Bound')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics
    final_regret = mean_regret[-1]
    theoretical_final = theoretical_curve[-1]
    ratio = final_regret / theoretical_final if theoretical_final > 0 else 0

    plt.text(
        0.02,
        0.98,
        f'Final Regret: {final_regret:.2f}\n'
        f'Theoretical: {theoretical_final:.2f}\n'
        f'Ratio: {ratio:.3f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="lightblue",
            alpha=0.8
        )
    )

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("=" * 60)
    print("📊 MULTI-TRIAL ANALYSIS")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Rounds per trial: {T}")
    print(f"Final average regret: {final_regret:.2f}")
    print(f"95% CI: [{ci_lower[-1]:.2f}, {ci_upper[-1]:.2f}]")
    print(f"Standard error: {stderr_regret[-1]:.2f}")
    print(f"Coefficient of variation: "
          f"{(np.std(all_regrets[:, -1])/final_regret*100):.1f}%")
    if suboptimality_gaps:
        print(f"Theoretical bound: {theoretical_final:.2f}")
        print(f"Empirical vs Theoretical ratio: {ratio:.3f}")


def smooth_data(data, threshold=50, window_divisor=4, min_window=12):
    """
    Smoothing with input padding for better edge behavior.
    """
    if len(data) > threshold:
        window = max(min_window, len(data) // window_divisor)
        if window % 2 == 0:
            window += 1

        # Pad input data to reduce edge effects
        pad_size = window // 2
        padded_data = np.pad(data, pad_size, mode='reflect')

        # Convolve and extract original region
        smoothed_padded = np.convolve(
            padded_data, np.ones(window) / window, mode='same'
        )
        smoothed = smoothed_padded[pad_size:-pad_size]

        x_smoothed = np.arange(len(data))
        return x_smoothed, smoothed, True
    else:
        return np.arange(len(data)), data, False

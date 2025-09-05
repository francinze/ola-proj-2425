#!/usr/bin/env python3
"""
Requirement 2 Parameter Tuning: Fine-tuning CombinatorialUCBSeller
Goal: Find optimal parameters for CombinatorialUCBSeller

This script tunes three key parameters of the CombinatorialUCBSeller:

1. cost_coeff (0.01-1.0): Controls the cost calculation in the algorithm
   - Lower values reduce cost impact, higher values increase cost sensitivity
   
2. confidence_multiplier (0.5-3.0): Scales the confidence bounds in UCB/LCB
   - Lower values = less exploration, higher values = more exploration
   
3. softmax_temperature (0.1-10.0): Controls the softmax distribution sharpness
   - Lower values = sharper (more greedy), higher values = softer (more random)

The algorithm is tested on Requirement 2's setting: multiple products (5) in a
stochastic environment using CombinatorialUCB with inventory constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
from datetime import datetime
import itertools
import warnings

from base_classes.setting import Setting
from base_classes.environment import Environment
from base_classes.specialized_sellers import CombinatorialUCBSeller
warnings.filterwarnings('ignore')


class Req2ParameterTuner:
    """Parameter tuning focused on fine-tuning CombinatorialUCBSeller"""

    def __init__(self):
        self.results = []
        self.start_time = None

    def define_parameter_grids(self):
        """Define parameter grids for CombinatorialUCBSeller fine-tuning"""

        # Expanded CombinatorialUCBSeller parameters for fine-tuning
        self.improved_params = {
            'cost_coeff': [
                0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0
            ],
            'confidence_multiplier': [
                0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0
            ],
            'softmax_temperature': [
                0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
            ]
        }

        # Generate all parameter combinations
        self.improved_combinations = list(itertools.product(
            *self.improved_params.values()
        ))

        print("📊 Fine-tuning Parameter Grid for CombinatorialUCBSeller:")
        print(
            f"   Cost coefficients: {len(self.improved_params['cost_coeff'])}"
        )
        print(
            f"   Confidence multipliers: {
                len(self.improved_params['confidence_multiplier'])
            }"
        )
        print(
            f"   Softmax temperatures: {
                len(self.improved_params['softmax_temperature'])
            }"
        )
        print(f"   Total combinations: {len(self.improved_combinations)}")

    def run_single_experiment(self, params, setting, seed):
        """Run a single experiment with given parameters"""

        # Set random seed
        np.random.seed(seed)

        # Create environment
        env = Environment(setting)

        # Create enhanced CombinatorialUCBSeller with specific parameters
        seller = EnhancedCombinatorialUCBSeller(
            setting=setting,
            cost_coeff=params[0],
            confidence_multiplier=params[1],
            softmax_temperature=params[2]
        )

        # Replace seller and run
        env.seller = seller
        env.play_all_rounds()

        # Calculate metrics
        rewards = np.array(env.seller.history_rewards)
        regrets = env.optimal_rewards - rewards
        cum_regret = np.cumsum(regrets)

        theoretical_bound = np.sqrt(setting.T * np.log(setting.T))

        # Performance metrics
        purchase_decisions = np.sum([1 for r in rewards if r > 0])
        metrics = {
            'total_rewards': np.sum(rewards),
            'final_regret': cum_regret[-1],
            'avg_regret': np.mean(regrets),
            'theoretical_bound': theoretical_bound,
            'regret_ratio': cum_regret[-1] / theoretical_bound,
            'efficiency': (
                np.sum(rewards) / np.sum(env.optimal_rewards)
            ) * 100,
            'regret_compliance': cum_regret[-1] <= 2 * theoretical_bound,
            'learning_trend': self.calculate_learning_trend(regrets),
            'total_purchases': purchase_decisions,
            'stability': self.calculate_stability(regrets)
        }

        return metrics

    def calculate_learning_trend(self, regrets):
        """Calculate if algorithm is learning (regret decreasing over time)"""
        T = len(regrets)
        early_regret = np.mean(regrets[:T//4])
        late_regret = np.mean(regrets[-T//4:])
        if early_regret > 0:
            return (early_regret - late_regret) / early_regret
        else:
            return 0

    def calculate_stability(self, regrets):
        """Calculate stability as inverse of regret variance"""
        return 1.0 / (np.var(regrets) + 1e-6)

    def run_comprehensive_tuning(self, seeds=[42, 123, 789, 456, 999]):
        """Run comprehensive parameter tuning for CombinatorialUCBSeller"""

        self.start_time = time.time()
        print(
            f"🚀 Starting CombinatorialUCBSeller fine-tuning at {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }"
        )
        print("   Goal: Find optimal parameters for CombinatorialUCBSeller")
        print(f"   Seeds: {seeds}")
        print("=" * 70)

        # Define experimental setting for R2: Multiple products + stochastic
        setting = Setting(
            T=1000,
            n_products=5,  # Multiple products for R2
            epsilon=0.2,
            distribution='exponential',  # Stochastic environment
            dist_params=(50, 15),
            non_stationary='no',  # Stochastic (stationary)
            algorithm="combinatorial_ucb",
            verbose='silent'
        )

        # Calculate total experiments
        total_experiments = len(self.improved_combinations) * len(seeds)

        print(f"📈 Total experiments to run: {total_experiments}")
        print(f"   Estimated time: {total_experiments * 4 / 60:.1f} minutes")
        print()

        # Main progress bar
        bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        with tqdm(
            total=total_experiments,
            desc="🔬 Fine-tuning Progress",
            bar_format=bar_fmt
        ) as pbar:

            # Test CombinatorialUCBSeller combinations
            for i, params in enumerate(self.improved_combinations):
                param_dict = dict(zip(self.improved_params.keys(), params))

                for seed in seeds:
                    pbar.set_description(
                        f"🚀 Config {i+1}/{
                            len(self.improved_combinations)
                        } seed={seed}"
                    )

                    try:
                        metrics = self.run_single_experiment(
                            params, setting, seed
                        )

                        # Store results
                        result = {
                            'algorithm': 'CombinatorialUCBSeller',
                            'seed': seed,
                            'params': param_dict.copy(),
                            **metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.results.append(result)

                    except Exception as e:
                        print(f"\n❌ Error in params={param_dict} seed={seed}: {e}")

                    pbar.update(1)

        elapsed_time = time.time() - self.start_time
        print("\n✅ CombinatorialUCBSeller fine-tuning completed!")
        print(f"   Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Results collected: {len(self.results)}")

    def analyze_results(self):
        """Analyze fine-tuning results"""

        if not self.results:
            print("❌ No results to analyze!")
            return

        print("\n📊 COMBINATORIAL-UCB FINE-TUNING ANALYSIS")
        print("=" * 60)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)

        # Expand parameter dictionaries into columns
        param_cols = pd.json_normalize(df['params'])
        df = pd.concat([df.drop('params', axis=1), param_cols], axis=1)

        print(f"Total experiments completed: {len(df)}")
        print(f"Seeds used: {sorted(df['seed'].unique())}")

        # Define comprehensive performance score
        df['performance_score'] = (
            df['regret_compliance'].astype(int) * 40 +  # 40% weight on compliance
            df['efficiency'] * 0.3 +  # 30% weight on efficiency
            df['learning_trend'] * 100 * 0.15 +  # 15% weight on learning
            (1 / (df['regret_ratio'] + 0.1)) * 10 +  # 10% weight on regret ratio
            df['stability'] * 5  # 5% weight on stability
        )

        # Best parameter analysis
        self.best_parameter_analysis(df)

        # Parameter sensitivity analysis
        self.parameter_sensitivity_analysis(df)

        # Create analysis plots
        self.create_analysis_plots(df)

        # Save results
        self.save_results(df)

        return df

    def best_parameter_analysis(self, df):
        """Analyze best parameters"""

        print("\n🎯 BEST PARAMETER CONFIGURATIONS")
        print("=" * 50)

        # Group by parameters and average across seeds
        param_columns = ['cost_coeff', 'confidence_multiplier',
                         'softmax_temperature']
        grouped = df.groupby(param_columns).agg({
            'performance_score': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'final_regret': ['mean', 'std'],
            'regret_ratio': ['mean', 'std'],
            'regret_compliance': 'mean',
            'learning_trend': 'mean',
            'stability': 'mean'
        }).round(3)

        # Flatten column names
        grouped.columns = [
            '_'.join(col).strip() for col in grouped.columns.values
        ]

        # Sort by performance score
        grouped = grouped.sort_values(
            'performance_score_mean', ascending=False
        )

        print("🏆 TOP 5 PARAMETER CONFIGURATIONS:")
        print("-" * 80)

        for i, (params, row) in enumerate(grouped.head(5).iterrows()):
            perf_mean = row['performance_score_mean']
            perf_std = row['performance_score_std']
            print(f"#{i+1} Performance Score: {perf_mean:.2f} ± {perf_std:.2f}")
            print(f"     Cost Coefficient: {params[0]}")
            print(f"     Confidence Multiplier: {params[1]}")
            print(f"     Softmax Temperature: {params[2]}")
            eff_mean = row['efficiency_mean']
            eff_std = row['efficiency_std']
            print(f"     Efficiency: {eff_mean:.1f}% ± {eff_std:.1f}")
            reg_mean = row['final_regret_mean']
            reg_std = row['final_regret_std']
            print(f"     Final Regret: {reg_mean:.2f} ± {reg_std:.2f}")
            comp_rate = row['regret_compliance_mean']
            print(f"     Compliance Rate: {comp_rate:.0%}")
            print(f"     Learning Trend: {row['learning_trend_mean']:.3f}")
            print(f"     Stability: {row['stability_mean']:.3f}")
            print()

    def parameter_sensitivity_analysis(self, df):
        """Analyze parameter sensitivity"""

        print("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 50)

        for param in ['cost_coeff', 'confidence_multiplier', 'softmax_temperature']:
            param_analysis = df.groupby(param).agg({
                'performance_score': ['mean', 'std'],
                'efficiency': 'mean',
                'regret_compliance': 'mean'
            }).round(3)

            param_analysis.columns = [
                '_'.join(col).strip() for col in param_analysis.columns.values
            ]
            param_analysis = param_analysis.sort_values(
                'performance_score_mean', ascending=False
            )

            print(f"\n📈 {param.upper()} Impact:")
            print(f"Best value: {param_analysis.index[0]} (Score: {param_analysis.iloc[0]['performance_score_mean']:.2f})")
            print(f"Worst value: {param_analysis.index[-1]} (Score: {param_analysis.iloc[-1]['performance_score_mean']:.2f})")
            print(
                f"Range impact: {
                    param_analysis.iloc[0]['performance_score_mean'] -
                    param_analysis.iloc[-1]['performance_score_mean']:.2f
                }"
            )

    def create_analysis_plots(self, df):
        """Create comprehensive analysis plots"""

        print("\n📈 Creating analysis plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CombinatorialUCBSeller Fine-tuning Analysis',
                     fontsize=16, fontweight='bold')

        # Plot 1: Performance Score Distribution
        axes[0, 0].hist(
            df['performance_score'], bins=30, alpha=0.7, edgecolor='black'
        )
        axes[0, 0].set_title('Performance Score Distribution')
        axes[0, 0].set_xlabel('Performance Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cost Coefficient Impact
        cost_coeffs = sorted(df['cost_coeff'].unique())
        cost_data = [df[
            df['cost_coeff'] == cc
        ]['performance_score'] for cc in cost_coeffs]
        axes[0, 1].boxplot(
            cost_data, labels=[f'{cc:.2f}' for cc in cost_coeffs]
        )
        axes[0, 1].set_title('Cost Coefficient Impact')
        axes[0, 1].set_xlabel('Cost Coefficient')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Softmax Temperature Impact
        temperatures = sorted(df['softmax_temperature'].unique())
        temp_data = [df[
            df['softmax_temperature'] == temp
        ]['performance_score'] for temp in temperatures]
        axes[0, 2].boxplot(
            temp_data, labels=[f'{temp}' for temp in temperatures]
        )
        axes[0, 2].set_title('Softmax Temperature Impact')
        axes[0, 2].set_xlabel('Softmax Temperature')
        axes[0, 2].set_ylabel('Performance Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Efficiency vs Final Regret
        axes[1, 0].scatter(
            df['final_regret'],
            df['efficiency'],
            alpha=0.6,
            c=df['performance_score'],
            cmap='viridis'
        )
        axes[1, 0].set_title('Efficiency vs Final Regret')
        axes[1, 0].set_xlabel('Final Regret')
        axes[1, 0].set_ylabel('Efficiency (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Compliance Rate by Parameter Ranges
        cc_bins = pd.cut(df['cost_coeff'], bins=5)
        compliance_by_cc = df.groupby(cc_bins)['regret_compliance'].mean() * 100
        axes[1, 1].bar(range(len(compliance_by_cc)), compliance_by_cc.values)
        axes[1, 1].set_title('Compliance Rate by Cost Coefficient Range')
        axes[1, 1].set_ylabel('Compliance Rate (%)')
        axes[1, 1].set_xticks(range(len(compliance_by_cc)))
        axes[1, 1].set_xticklabels([
            f'{interval.left:.2f}-{interval.right:.2f}'
            for interval in compliance_by_cc.index
        ], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Performance Score Heatmap (Cost Coeff vs Temperature)
        pivot_data = df.groupby(
            ['cost_coeff', 'softmax_temperature']
        )['performance_score'].mean().unstack()
        im = axes[1, 2].imshow(pivot_data.values, cmap='viridis', aspect='auto')
        axes[1, 2].set_title('Performance Heatmap (Cost Coeff vs Temperature)')
        axes[1, 2].set_xlabel('Softmax Temperature Index')
        axes[1, 2].set_ylabel('Cost Coefficient Index')
        plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"combinatorial_ucb_fine_tuning_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Analysis plots saved to: {plot_filename}")

    def save_results(self, df):
        """Save results to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV
        csv_filename = f"combinatorial_ucb_fine_tuning_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)

        # Get best configuration
        best_config = df.loc[df['performance_score'].idxmax()]

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(df),
            'total_time_minutes': (time.time() - self.start_time) / 60,
            'seeds': df['seed'].unique().tolist(),
            'best_configuration': {
                'cost_coeff': best_config['cost_coeff'],
                'confidence_multiplier': best_config['confidence_multiplier'],
                'softmax_temperature': best_config['softmax_temperature'],
                'performance_score': best_config['performance_score'],
                'efficiency': best_config['efficiency'],
                'final_regret': best_config['final_regret'],
                'regret_compliance': bool(best_config['regret_compliance'])
            },
            'overall_statistics': {
                'mean_performance_score': df['performance_score'].mean(),
                'std_performance_score': df['performance_score'].std(),
                'compliance_rate': df['regret_compliance'].mean(),
                'mean_efficiency': df['efficiency'].mean()
            }
        }

        json_filename = f"combinatorial_ucb_summary_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print("💾 Results saved to:")
        print(f"   Detailed data: {csv_filename}")
        print(f"   Summary: {json_filename}")

        print("\n🏆 OPTIMAL CONFIGURATION FOUND:")
        print(f"   Cost Coefficient: {best_config['cost_coeff']}")
        print(f"   Confidence Multiplier: {best_config['confidence_multiplier']}")
        print(f"   Softmax Temperature: {best_config['softmax_temperature']}")
        print(f"   Performance Score: {best_config['performance_score']:.2f}")
        print(f"   Efficiency: {best_config['efficiency']:.1f}%")


class EnhancedCombinatorialUCBSeller(CombinatorialUCBSeller):
    """Enhanced CombinatorialUCBSeller with tunable parameters"""
    def __init__(self, setting: Setting, cost_coeff: float = 0.1,
                 confidence_multiplier: float = 1.0,
                 softmax_temperature: float = 1.0):
        """Initialize enhanced CombinatorialUCB seller with parameters."""
        super().__init__(setting, cost_coeff=cost_coeff)
        self.confidence_multiplier = confidence_multiplier
        self.softmax_temperature = softmax_temperature
    
    def compute_ucb_lcb_bounds(self):
        """Enhanced UCB/LCB computation with tunable confidence multiplier."""
        ucb_rewards = np.zeros((self.num_products, self.num_prices))
        lcb_costs = np.zeros((self.num_products, self.num_prices))

        for i in range(self.num_products):
            for j in range(self.num_prices):
                n = self.counts[i, j]
                if n > 0:
                    # Enhanced confidence with multiplier
                    confidence = self.confidence_multiplier * np.sqrt(2 * np.log(self.total_steps) / n)
                    ucb_rewards[i, j] = self.values[i, j] + confidence
                    lcb_costs[i, j] = self.cost_values[i, j] - confidence
                else:
                    # Optimistic initialization
                    ucb_rewards[i, j] = np.inf
                    lcb_costs[i, j] = 0.0

        return ucb_rewards, lcb_costs
    
    def solve_lp_for_distribution(self, ucb_rewards, lcb_costs):
        """Enhanced LP distribution with tunable temperature."""
        # Compute expected profit for each price combination
        expected_profits = np.zeros((self.num_products, self.num_prices))

        for i in range(self.num_products):
            for j in range(self.num_prices):
                # Expected profit = UCB_reward - LCB_cost
                expected_profits[i, j] = ucb_rewards[i, j] - lcb_costs[i, j]

        # Apply softmax with temperature to get probabilities for each product
        gamma_t = np.zeros((self.num_products, self.num_prices))
        for i in range(self.num_products):
            # Apply temperature scaling
            profits = expected_profits[i] / self.softmax_temperature

            # Replace inf values with large finite values
            profits = np.where(np.isinf(profits), 1e10, profits)
            profits = np.where(np.isnan(profits), 0, profits)

            # Softmax computation with numerical stability
            max_val = np.max(profits)
            if np.isfinite(max_val):
                exp_vals = np.exp(profits - max_val)
                sum_exp = np.sum(exp_vals)
                if sum_exp > 0:
                    gamma_t[i] = exp_vals / sum_exp
                else:
                    gamma_t[i] = np.ones(self.num_prices) / self.num_prices
            else:
                gamma_t[i] = np.ones(self.num_prices) / self.num_prices

        return gamma_t


def main():
    """Main execution function"""

    print("🎯 COMBINATORIAL-UCB FINE-TUNING")
    print("=" * 50)
    print("Goal: Find optimal parameters for CombinatorialUCBSeller")
    print("Setting: Multiple products, stochastic environment (R2)")
    print()

    # Initialize tuner
    tuner = Req2ParameterTuner()

    # Define parameter grids
    tuner.define_parameter_grids()

    # Ask user for confirmation
    total_combos = len(tuner.improved_combinations)
    estimated_time = total_combos * 5 * 0.4 / 60  # 5 seeds * 0.4 seconds

    print(f"\n⚠️  This will run approximately {estimated_time:.0f} minutes.")
    response = input("   Continue? (y/N): ").strip().lower()

    if response not in ['y', 'yes']:
        print("❌ Fine-tuning cancelled.")
        return

    print()

    try:
        # Run tuning
        tuner.run_comprehensive_tuning()

        # Analyze results
        tuner.analyze_results()

        print("\n🎉 CombinatorialUCBSeller fine-tuning completed!")
        print("   Check the generated files for optimal parameter configuration.")

    except KeyboardInterrupt:
        print("\n⏹️  Fine-tuning interrupted by user.")
        if tuner.results:
            print(f"   Partial results available: {len(tuner.results)} experiments")
            tuner.analyze_results()

    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

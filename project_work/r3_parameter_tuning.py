#!/usr/bin/env python3
"""
Requirement 3 Parameter Tuning: Fine-tuning PrimalDualSeller
Goal: Find optimal parameters for PrimalDualSeller
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
from base_classes.specialized_sellers import PrimalDualSeller
warnings.filterwarnings('ignore')


class Req3ParameterTuner:
    """Parameter tuning focused on fine-tuning PrimalDualSeller"""

    def __init__(self):
        self.results = []
        self.start_time = None

    def define_parameter_grids(self):
        """Define expanded parameter grids for PrimalDualSeller fine-tuning"""

        # Expanded PrimalDualSeller parameters for fine-tuning
        self.improved_params = {
            'learning_rate': [
                0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05
            ],
            'regret_learning_rate': [
                0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.03, 0.05, 0.07, 0.1
            ],
            'base_temperature': [
                0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0
            ]
        }

        # Generate all parameter combinations
        self.improved_combinations = list(itertools.product(
            *self.improved_params.values()
        ))

        print("📊 Fine-tuning Parameter Grid for PrimalDualSeller:")
        print(
            f"   Learning rates: {len(self.improved_params['learning_rate'])}"
        )
        print(
            f"   Regret learning rates: {
                len(self.improved_params['regret_learning_rate'])
            }"
        )
        print(
            f"   Base temperatures: {
                len(self.improved_params['base_temperature'])
            }"
        )
        print(f"   Total combinations: {len(self.improved_combinations)}")

    def run_single_experiment(self, params, setting, seed):
        """Run a single experiment with given parameters"""

        # Set random seed
        np.random.seed(seed)

        # Create environment
        env = Environment(setting)

        # Create PrimalDualSeller with specific parameters
        seller = PrimalDualSeller(
            setting=setting,
            learning_rate=params[0],
            regret_learning_rate=params[1],
            base_temperature=params[2]
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
        return (early_regret - late_regret) / early_regret if early_regret > 0 else 0

    def calculate_stability(self, regrets):
        """Calculate stability as inverse of regret variance"""
        return 1.0 / (np.var(regrets) + 1e-6)

    def run_comprehensive_tuning(self, seeds=[42, 123, 789, 456, 999]):
        """Run comprehensive parameter tuning for PrimalDualSeller"""

        self.start_time = time.time()
        print(
            f"🚀 Starting PrimalDualSeller fine-tuning at {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }"
        )
        print("   Goal: Find optimal parameters for PrimalDualSeller")
        print(f"   Seeds: {seeds}")
        print("=" * 70)

        # Define experimental setting
        setting = Setting(
            T=1000, n_products=1, epsilon=0.2,
            distribution='gaussian', dist_params=(50, 15),
            non_stationary='highly', algorithm="improved_primal_dual",
            verbose='silent'
        )

        # Calculate total experiments
        total_experiments = len(self.improved_combinations) * len(seeds)

        print(f"📈 Total experiments to run: {total_experiments}")
        print(f"   Estimated time: {total_experiments * 3 / 60:.1f} minutes")
        print()

        # Main progress bar
        bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        with tqdm(
            total=total_experiments,
            desc="🔬 Fine-tuning Progress",
            bar_format=bar_fmt
        ) as pbar:

            # Test PrimalDualSeller combinations
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
                            'algorithm': 'PrimalDualSeller',
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
        print(f"\n✅ PrimalDualSeller fine-tuning completed!")
        print(f"   Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Results collected: {len(self.results)}")

    def analyze_results(self):
        """Analyze fine-tuning results"""

        if not self.results:
            print("❌ No results to analyze!")
            return

        print(f"\n📊 IMPROVED PRIMAL-DUAL FINE-TUNING ANALYSIS")
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

        print(f"\n🎯 BEST PARAMETER CONFIGURATIONS")
        print("=" * 50)

        # Group by parameters and average across seeds
        param_columns = ['learning_rate', 'regret_learning_rate', 'base_temperature']
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
            print(f"#{i+1} Performance Score: {row['performance_score_mean']:.2f} ± {row['performance_score_std']:.2f}")
            print(f"     Learning Rate: {params[0]}")
            print(f"     Regret Learning Rate: {params[1]}")
            print(f"     Base Temperature: {params[2]}")
            print(f"     Efficiency: {row['efficiency_mean']:.1f}% ± {row['efficiency_std']:.1f}")
            print(f"     Final Regret: {row['final_regret_mean']:.2f} ± {row['final_regret_std']:.2f}")
            print(f"     Compliance Rate: {row['regret_compliance_mean']:.0%}")
            print(f"     Learning Trend: {row['learning_trend_mean']:.3f}")
            print(f"     Stability: {row['stability_mean']:.3f}")
            print()

    def parameter_sensitivity_analysis(self, df):
        """Analyze parameter sensitivity"""

        print("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 50)

        for param in ['learning_rate', 'regret_learning_rate', 'base_temperature']:
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
        fig.suptitle('PrimalDualSeller Fine-tuning Analysis',
                     fontsize=16, fontweight='bold')

        # Plot 1: Performance Score Distribution
        axes[0, 0].hist(
            df['performance_score'], bins=30, alpha=0.7, edgecolor='black'
        )
        axes[0, 0].set_title('Performance Score Distribution')
        axes[0, 0].set_xlabel('Performance Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Learning Rate Impact (fixed the boxplot issue)
        learning_rates = sorted(df['learning_rate'].unique())
        lr_data = [df[
            df['learning_rate'] == lr
        ]['performance_score'] for lr in learning_rates]
        axes[0, 1].boxplot(
            lr_data, labels=[f'{lr:.3f}' for lr in learning_rates]
        )
        axes[0, 1].set_title('Learning Rate Impact')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Base Temperature Impact
        temperatures = sorted(df['base_temperature'].unique())
        temp_data = [df[
            df['base_temperature'] == temp
        ]['performance_score'] for temp in temperatures]
        axes[0, 2].boxplot(
            temp_data, labels=[f'{temp}' for temp in temperatures]
        )
        axes[0, 2].set_title('Base Temperature Impact')
        axes[0, 2].set_xlabel('Base Temperature')
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
        lr_bins = pd.cut(df['learning_rate'], bins=5)
        compliance_by_lr = df.groupby(lr_bins)['regret_compliance'].mean() * 100
        axes[1, 1].bar(range(len(compliance_by_lr)), compliance_by_lr.values)
        axes[1, 1].set_title('Compliance Rate by Learning Rate Range')
        axes[1, 1].set_ylabel('Compliance Rate (%)')
        axes[1, 1].set_xticks(range(len(compliance_by_lr)))
        axes[1, 1].set_xticklabels([
            f'{interval.left:.3f}-{interval.right:.3f}'
            for interval in compliance_by_lr.index
        ], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Performance Score Heatmap (Learning Rate vs Base Temperature)
        pivot_data = df.groupby(
            ['learning_rate', 'base_temperature']
        )['performance_score'].mean().unstack()
        axes[1, 2].imshow(pivot_data.values, cmap='viridis', aspect='auto')
        axes[1, 2].set_title('Performance Heatmap (LR vs Temperature)')
        axes[1, 2].set_xlabel('Base Temperature Index')
        axes[1, 2].set_ylabel('Learning Rate Index')

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"improved_primal_dual_fine_tuning_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Analysis plots saved to: {plot_filename}")

    def save_results(self, df):
        """Save results to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV
        csv_filename = f"improved_primal_dual_fine_tuning_{timestamp}.csv"
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
                'learning_rate': best_config['learning_rate'],
                'regret_learning_rate': best_config['regret_learning_rate'],
                'base_temperature': best_config['base_temperature'],
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

        json_filename = f"improved_primal_dual_summary_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print("💾 Results saved to:")
        print(f"   Detailed data: {csv_filename}")
        print(f"   Summary: {json_filename}")

        print("\n🏆 OPTIMAL CONFIGURATION FOUND:")
        print(f"   Learning Rate: {best_config['learning_rate']}")
        print(f"   Regret Learning Rate: {best_config['regret_learning_rate']}")
        print(f"   Base Temperature: {best_config['base_temperature']}")
        print(f"   Performance Score: {best_config['performance_score']:.2f}")
        print(f"   Efficiency: {best_config['efficiency']:.1f}%")


def main():
    """Main execution function"""

    print("🎯 IMPROVED PRIMAL-DUAL FINE-TUNING")
    print("=" * 50)
    print("Goal: Find optimal parameters for PrimalDualSeller")
    print("Setting: Single product, highly non-stationary environment")
    print()

    # Initialize tuner
    tuner = Req3ParameterTuner()

    # Define parameter grids
    tuner.define_parameter_grids()

    # Ask user for confirmation
    total_combos = len(tuner.improved_combinations)
    estimated_time = total_combos * 5 * 0.3 / 60  # 5 seeds * 0.3 seconds

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
        results_df = tuner.analyze_results()

        print("\n🎉 PrimalDualSeller fine-tuning completed!")
        print("   Check the generated files for optimal parameter configuration.")

    except KeyboardInterrupt:
        print("\n⏹️  Fine-tuning interrupted by user.")
        if tuner.results:
            print(f"   Partial results available: {len(tuner.results)} experiments")
            tuner.analyze_results()

    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        traceback.print_exc()
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

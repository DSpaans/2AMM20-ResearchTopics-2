"""
Evaluation Script: Compare Unconstrained PPO vs PPO-Lagrangian

This script evaluates and compares the performance of:
1. Unconstrained PPO (pure profit maximization)
2. PPO-Lagrangian (profit with constraints)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, NamedTuple
import matplotlib.pyplot as plt
from pathlib import Path

from chargax import Chargax, get_electricity_prices


class EpisodeMetrics(NamedTuple):
    """Metrics collected during an episode."""
    total_profit: float
    capacity_violations: float
    uncharged_kw: float
    rejected_customers: int
    battery_degradation: float
    total_charged: float
    total_discharged: float
    charged_overtime: float
    charged_undertime: float


def evaluate_episode(
    env: Chargax,
    policy_fn,
    rng: jax.random.PRNGKey,
) -> EpisodeMetrics:
    """Evaluate a single episode and collect metrics."""

    # Reset environment
    obs, state = env.reset(rng)

    # Track metrics
    done = False
    step = 0
    max_steps = env.max_episode_steps

    while not done and step < max_steps:
        # Get action (would use trained policy)
        rng, action_rng = jax.random.split(rng)
        action = env.action_space.sample(action_rng)

        # Step environment
        rng, step_rng = jax.random.split(rng)
        timestep, state = env.step(step_rng, state, action)

        obs = timestep.observation
        done = timestep.terminated or timestep.truncated
        step += 1

    # Extract final metrics
    logging_data = timestep.info.get('logging_data', {})

    return EpisodeMetrics(
        total_profit=float(logging_data.get('profit', 0.0)),
        capacity_violations=float(logging_data.get('exceeded_capacity', 0.0)),
        uncharged_kw=float(logging_data.get('uncharged_kw', 0.0)),
        rejected_customers=int(logging_data.get('rejected_customers', 0)),
        battery_degradation=float(logging_data.get('total_discharged_kw', 0.0)),
        total_charged=float(logging_data.get('total_charged_kw', 0.0)),
        total_discharged=float(logging_data.get('total_discharged_kw', 0.0)),
        charged_overtime=float(logging_data.get('charged_overtime', 0.0)),
        charged_undertime=float(logging_data.get('charged_undertime', 0.0)),
    )


def evaluate_multiple_episodes(
    env: Chargax,
    policy_fn,
    num_episodes: int = 50,
    seed: int = 42,
) -> List[EpisodeMetrics]:
    """Evaluate multiple episodes and return metrics."""

    print(f"Evaluating {num_episodes} episodes...")

    rng = jax.random.PRNGKey(seed)
    results = []

    for i in range(num_episodes):
        rng, eval_rng = jax.random.split(rng)
        metrics = evaluate_episode(env, policy_fn, eval_rng)
        results.append(metrics)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_episodes} episodes")

    return results


def compute_statistics(metrics_list: List[EpisodeMetrics]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for all metrics."""

    stats = {}

    for field in EpisodeMetrics._fields:
        values = [getattr(m, field) for m in metrics_list]
        stats[field] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    return stats


def print_comparison_table(
    unconstrained_stats: Dict,
    lagrangian_stats: Dict,
    constraint_thresholds: Dict[str, float],
):
    """Print a comparison table of both methods."""

    print("\n" + "=" * 100)
    print("EVALUATION RESULTS: Unconstrained PPO vs PPO-Lagrangian")
    print("=" * 100)

    # Profit comparison
    print("\n📊 PROFIT (Primary Objective)")
    print("-" * 100)
    print(f"{'Metric':<30} {'Unconstrained PPO':<25} {'PPO-Lagrangian':<25} {'Difference':<20}")
    print("-" * 100)

    unc_profit = unconstrained_stats['total_profit']['mean']
    lag_profit = lagrangian_stats['total_profit']['mean']
    profit_diff = lag_profit - unc_profit
    profit_pct = (profit_diff / unc_profit) * 100 if unc_profit != 0 else 0

    print(f"{'Total Profit (€)':<30} "
          f"{unc_profit:>10.2f} ± {unconstrained_stats['total_profit']['std']:>8.2f}  "
          f"{lag_profit:>10.2f} ± {lagrangian_stats['total_profit']['std']:>8.2f}  "
          f"{profit_diff:>10.2f} ({profit_pct:>+6.1f}%)")

    # Constraint violations
    print("\n⚠️  CONSTRAINT VIOLATIONS (Lower is Better)")
    print("-" * 100)
    print(f"{'Constraint':<30} {'Unconstrained':<25} {'Lagrangian':<25} {'Threshold':<20}")
    print("-" * 100)

    constraints = [
        ('Capacity Exceeded (kW)', 'capacity_violations', 'capacity_exceeded', 10.0),
        ('Uncharged Energy (kWh)', 'uncharged_kw', 'uncharged_satisfaction', 50.0),
        ('Rejected Customers', 'rejected_customers', 'rejected_customers', 2.0),
        ('Battery Degradation (kWh)', 'battery_degradation', 'battery_degradation', 100.0),
    ]

    for display_name, metric_key, threshold_key, default_threshold in constraints:
        unc_val = unconstrained_stats[metric_key]['mean']
        lag_val = lagrangian_stats[metric_key]['mean']
        threshold = constraint_thresholds.get(threshold_key, default_threshold)

        unc_status = "✓" if unc_val <= threshold else "✗"
        lag_status = "✓" if lag_val <= threshold else "✗"

        print(f"{display_name:<30} "
              f"{unc_status} {unc_val:>10.2f} ± {unconstrained_stats[metric_key]['std']:>8.2f}  "
              f"{lag_status} {lag_val:>10.2f} ± {lagrangian_stats[metric_key]['std']:>8.2f}  "
              f"≤ {threshold:>10.2f}")

    # Operational metrics
    print("\n📈 OPERATIONAL METRICS")
    print("-" * 100)
    print(f"{'Metric':<30} {'Unconstrained PPO':<25} {'PPO-Lagrangian':<25}")
    print("-" * 100)

    operations = [
        ('Total Charged (kWh)', 'total_charged'),
        ('Total Discharged (kWh)', 'total_discharged'),
        ('Overtime Minutes', 'charged_overtime'),
        ('Undertime Minutes', 'charged_undertime'),
    ]

    for display_name, metric_key in operations:
        unc_val = unconstrained_stats[metric_key]['mean']
        lag_val = lagrangian_stats[metric_key]['mean']

        print(f"{display_name:<30} "
              f"{unc_val:>10.2f} ± {unconstrained_stats[metric_key]['std']:>8.2f}  "
              f"{lag_val:>10.2f} ± {lagrangian_stats[metric_key]['std']:>8.2f}")

    print("=" * 100)

    # Summary
    print("\n📋 SUMMARY")
    print("-" * 100)

    # Count constraint violations
    unc_violations = sum([
        1 for _, metric_key, threshold_key, default_threshold in constraints
        if unconstrained_stats[metric_key]['mean'] > constraint_thresholds.get(threshold_key, default_threshold)
    ])

    lag_violations = sum([
        1 for _, metric_key, threshold_key, default_threshold in constraints
        if lagrangian_stats[metric_key]['mean'] > constraint_thresholds.get(threshold_key, default_threshold)
    ])

    print(f"Unconstrained PPO:")
    print(f"  Profit:      €{unc_profit:.2f}")
    print(f"  Violations:  {unc_violations}/4 constraints violated")
    print(f"  Rating:      {'❌ UNSAFE' if unc_violations > 0 else '✅ SAFE'}")

    print(f"\nPPO-Lagrangian:")
    print(f"  Profit:      €{lag_profit:.2f} ({profit_pct:+.1f}%)")
    print(f"  Violations:  {lag_violations}/4 constraints violated")
    print(f"  Rating:      {'❌ UNSAFE' if lag_violations > 0 else '✅ SAFE'}")

    if lag_violations < unc_violations:
        print(f"\n✅ PPO-Lagrangian successfully reduced constraint violations!")
    elif lag_violations == 0 and unc_violations == 0:
        print(f"\n✅ Both methods satisfied all constraints!")

    print("=" * 100)


def plot_comparison(
    unconstrained_results: List[EpisodeMetrics],
    lagrangian_results: List[EpisodeMetrics],
    save_path: str = "results/comparison.png",
):
    """Create comparison plots."""

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Unconstrained PPO vs PPO-Lagrangian', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('total_profit', 'Total Profit (€)', 'higher'),
        ('capacity_violations', 'Capacity Violations (kW)', 'lower'),
        ('uncharged_kw', 'Uncharged Energy (kWh)', 'lower'),
        ('rejected_customers', 'Rejected Customers', 'lower'),
        ('total_charged', 'Total Energy Charged (kWh)', 'neutral'),
        ('battery_degradation', 'Battery Degradation (kWh)', 'lower'),
    ]

    for idx, (metric, title, better) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        unc_values = [getattr(m, metric) for m in unconstrained_results]
        lag_values = [getattr(m, metric) for m in lagrangian_results]

        ax.boxplot([unc_values, lag_values], labels=['Unconstrained', 'Lagrangian'])
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Color code based on which is better
        if better == 'higher':
            if np.mean(lag_values) > np.mean(unc_values):
                ax.set_facecolor('#e8f5e9')  # Light green
        elif better == 'lower':
            if np.mean(lag_values) < np.mean(unc_values):
                ax.set_facecolor('#e8f5e9')  # Light green

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {save_path}")
    plt.close()


def main():
    """Run the full evaluation pipeline."""

    print("=" * 100)
    print("CHARGAX EVALUATION: Unconstrained PPO vs PPO-Lagrangian")
    print("=" * 100)

    # Define constraint thresholds
    constraint_thresholds = {
        'capacity_exceeded': 10.0,
        'uncharged_satisfaction': 50.0,
        'rejected_customers': 2.0,
        'battery_degradation': 100.0,
    }

    # Create environment
    print("\n1. Creating environment...")
    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
        minutes_per_timestep=5,
        num_discretization_levels=10,
        num_chargers=16,
        num_dc_groups=10,
        elec_customer_sell_price=0.75,
    )

    # Dummy policy (random) - replace with trained policies
    def random_policy(obs, rng):
        return env.action_space.sample(rng)

    # Evaluate unconstrained
    print("\n2. Evaluating Unconstrained PPO...")
    unconstrained_results = evaluate_multiple_episodes(env, random_policy, num_episodes=50, seed=42)
    unconstrained_stats = compute_statistics(unconstrained_results)

    # Evaluate lagrangian
    print("\n3. Evaluating PPO-Lagrangian...")
    lagrangian_results = evaluate_multiple_episodes(env, random_policy, num_episodes=50, seed=43)
    lagrangian_stats = compute_statistics(lagrangian_results)

    # Print comparison
    print_comparison_table(unconstrained_stats, lagrangian_stats, constraint_thresholds)

    # Create plots
    print("\n4. Creating visualizations...")
    plot_comparison(unconstrained_results, lagrangian_results)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()


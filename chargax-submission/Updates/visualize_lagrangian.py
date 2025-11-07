"""
Visualization Tools for PPO-Lagrangian Training

This script provides visualization tools to understand:
1. How Lagrange multipliers evolve during training
2. Constraint violation trends
3. Profit vs constraint satisfaction tradeoff
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict


def plot_lambda_evolution(
    lambda_history: Dict[str, List[float]],
    save_path: str = "results/lambda_evolution.png",
):
    """
    Plot how Lagrange multipliers evolve during training.

    Args:
        lambda_history: Dict mapping constraint names to lists of lambda values over time
    """

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Lagrange Multiplier Evolution During Training',
                 fontsize=14, fontweight='bold')

    constraint_names = [
        ('capacity_exceeded', 'Grid Capacity'),
        ('uncharged_satisfaction', 'Customer Satisfaction'),
        ('rejected_customers', 'Queue Management'),
        ('battery_degradation', 'Battery Health'),
    ]

    for idx, (key, title) in enumerate(constraint_names):
        ax = axes[idx // 2, idx % 2]

        if key in lambda_history:
            values = lambda_history[key]
            ax.plot(values, linewidth=2, color='#1976d2')
            ax.fill_between(range(len(values)), values, alpha=0.3, color='#1976d2')
            ax.set_title(f'λ_{title}', fontweight='bold')
            ax.set_xlabel('Training Update')
            ax.set_ylabel('Lambda Value')
            ax.grid(True, alpha=0.3)

            # Add annotation for final value
            final_val = values[-1]
            ax.annotate(f'Final: {final_val:.4f}',
                       xy=(len(values)-1, final_val),
                       xytext=(len(values)*0.7, final_val*1.1),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Lambda evolution plot saved to: {save_path}")
    plt.close()


def plot_constraint_violations(
    violation_history: Dict[str, List[float]],
    thresholds: Dict[str, float],
    save_path: str = "results/constraint_violations.png",
):
    """
    Plot constraint violations over training with threshold lines.
    """

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Constraint Violations During Training',
                 fontsize=14, fontweight='bold')

    constraints = [
        ('capacity_exceeded', 'Grid Capacity (kW)', '#e53935'),
        ('uncharged_satisfaction', 'Unmet Demand (kWh)', '#fb8c00'),
        ('rejected_customers', 'Rejected Customers', '#fdd835'),
        ('battery_degradation', 'Battery Cycling (kWh)', '#43a047'),
    ]

    for idx, (key, title, color) in enumerate(constraints):
        ax = axes[idx // 2, idx % 2]

        if key in violation_history:
            values = violation_history[key]
            threshold = thresholds.get(key, 0)

            ax.plot(values, linewidth=2, color=color, label='Actual')
            ax.axhline(y=threshold, color='red', linestyle='--',
                      linewidth=2, label=f'Threshold ({threshold:.1f})')

            # Shade violations
            violations = np.array(values) > threshold
            if violations.any():
                ax.fill_between(range(len(values)), values, threshold,
                               where=violations, alpha=0.3, color='red',
                               label='Violation')

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Training Update')
            ax.set_ylabel('Violation Magnitude')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Constraint violations plot saved to: {save_path}")
    plt.close()


def plot_profit_vs_constraints(
    profit_history: List[float],
    total_violations: List[float],
    save_path: str = "results/profit_vs_constraints.png",
):
    """
    Plot the tradeoff between profit and constraint satisfaction.
    """

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Profit vs Constraint Satisfaction Tradeoff',
                 fontsize=14, fontweight='bold')

    # Plot 1: Both metrics over time
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(profit_history, color='#2e7d32', linewidth=2, label='Profit')
    ax1.set_xlabel('Training Update')
    ax1.set_ylabel('Profit (€)', color='#2e7d32')
    ax1.tick_params(axis='y', labelcolor='#2e7d32')
    ax1.grid(True, alpha=0.3)

    line2 = ax1_twin.plot(total_violations, color='#d32f2f', linewidth=2, label='Total Violations')
    ax1_twin.set_ylabel('Total Constraint Violations', color='#d32f2f')
    ax1_twin.tick_params(axis='y', labelcolor='#d32f2f')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.set_title('Temporal Evolution', fontweight='bold')

    # Plot 2: Scatter plot
    ax2.scatter(total_violations, profit_history, alpha=0.6, s=50, c='#1976d2')
    ax2.set_xlabel('Total Constraint Violations')
    ax2.set_ylabel('Profit (€)')
    ax2.set_title('Tradeoff Space', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add pareto frontier approximation
    sorted_idx = np.argsort(total_violations)
    sorted_violations = np.array(total_violations)[sorted_idx]
    sorted_profit = np.array(profit_history)[sorted_idx]

    # Simple pareto frontier (points where increasing profit requires increasing violations)
    pareto_points = []
    max_profit = -np.inf
    for v, p in zip(sorted_violations, sorted_profit):
        if p > max_profit:
            pareto_points.append((v, p))
            max_profit = p

    if pareto_points:
        pareto_v, pareto_p = zip(*pareto_points)
        ax2.plot(pareto_v, pareto_p, 'r--', linewidth=2, label='Pareto Frontier', alpha=0.7)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Profit vs constraints plot saved to: {save_path}")
    plt.close()


def plot_training_summary(
    metrics_history: Dict[str, List[float]],
    save_path: str = "results/training_summary.png",
):
    """
    Create a comprehensive training summary dashboard.
    """

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('PPO-Lagrangian Training Dashboard', fontsize=16, fontweight='bold')

    # Reward evolution
    ax1 = fig.add_subplot(gs[0, :])
    if 'reward_mean' in metrics_history:
        ax1.plot(metrics_history['reward_mean'], label='Original Reward', linewidth=2)
    if 'penalized_reward_mean' in metrics_history:
        ax1.plot(metrics_history['penalized_reward_mean'],
                label='Penalized Reward', linewidth=2, linestyle='--')
    ax1.set_title('Reward Evolution', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Training Update')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lambda values
    ax2 = fig.add_subplot(gs[1, 0])
    lambda_keys = [k for k in metrics_history.keys() if k.startswith('lambda_')]
    for key in lambda_keys:
        label = key.replace('lambda_', 'λ_')
        ax2.plot(metrics_history[key], label=label, linewidth=2)
    ax2.set_title('Lagrange Multipliers', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Training Update')
    ax2.set_ylabel('Lambda Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Capacity violations
    ax3 = fig.add_subplot(gs[1, 1])
    if 'capacity_violation' in metrics_history:
        ax3.plot(metrics_history['capacity_violation'], color='#e53935', linewidth=2)
        ax3.axhline(y=10.0, color='red', linestyle='--', label='Threshold')
    ax3.set_title('Capacity Violations', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Training Update')
    ax3.set_ylabel('Violation (kW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Customer satisfaction
    ax4 = fig.add_subplot(gs[1, 2])
    if 'uncharged_violation' in metrics_history:
        ax4.plot(metrics_history['uncharged_violation'], color='#fb8c00', linewidth=2)
        ax4.axhline(y=50.0, color='red', linestyle='--', label='Threshold')
    ax4.set_title('Unmet Demand', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Training Update')
    ax4.set_ylabel('Violation (kWh)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Rejected customers
    ax5 = fig.add_subplot(gs[2, 0])
    if 'rejected_violation' in metrics_history:
        ax5.plot(metrics_history['rejected_violation'], color='#fdd835', linewidth=2)
        ax5.axhline(y=2.0, color='red', linestyle='--', label='Threshold')
    ax5.set_title('Rejected Customers', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Training Update')
    ax5.set_ylabel('Count')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Battery degradation
    ax6 = fig.add_subplot(gs[2, 1])
    if 'battery_violation' in metrics_history:
        ax6.plot(metrics_history['battery_violation'], color='#43a047', linewidth=2)
        ax6.axhline(y=100.0, color='red', linestyle='--', label='Threshold')
    ax6.set_title('Battery Cycling', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Training Update')
    ax6.set_ylabel('Violation (kWh)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = "Training Summary\n" + "="*30 + "\n\n"

    if 'reward_mean' in metrics_history:
        final_reward = metrics_history['reward_mean'][-1]
        summary_text += f"Final Reward: {final_reward:.2f}\n"

    if lambda_keys:
        summary_text += "\nFinal Lambda Values:\n"
        for key in lambda_keys:
            final_lambda = metrics_history[key][-1]
            name = key.replace('lambda_', '')
            summary_text += f"  {name}: {final_lambda:.4f}\n"

    violation_keys = [k for k in metrics_history.keys() if 'violation' in k]
    if violation_keys:
        summary_text += "\nFinal Violations:\n"
        thresholds = {
            'capacity_violation': 10.0,
            'uncharged_violation': 50.0,
            'rejected_violation': 2.0,
            'battery_violation': 100.0,
        }
        for key in violation_keys:
            final_val = metrics_history[key][-1]
            threshold = thresholds.get(key, 0)
            status = "✓" if final_val <= threshold else "✗"
            name = key.replace('_violation', '')
            summary_text += f"  {status} {name}: {final_val:.2f}\n"

    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training summary dashboard saved to: {save_path}")
    plt.close()


def generate_example_visualizations():
    """Generate example visualizations with synthetic data."""

    print("Generating example visualizations...")
    print("(In production, use actual training metrics)")

    # Generate synthetic data
    num_updates = 1000

    # Lambda evolution (should increase when violations occur, then stabilize)
    lambda_history = {
        'capacity_exceeded': [0.01 + 0.05 * (1 - np.exp(-i/200)) for i in range(num_updates)],
        'uncharged_satisfaction': [0.01 + 0.08 * (1 - np.exp(-i/150)) for i in range(num_updates)],
        'rejected_customers': [0.01 + 0.03 * (1 - np.exp(-i/250)) for i in range(num_updates)],
        'battery_degradation': [0.01 + 0.02 * (1 - np.exp(-i/300)) for i in range(num_updates)],
    }

    # Constraint violations (should decrease as lambdas increase)
    violation_history = {
        'capacity_exceeded': [15 * np.exp(-i/200) + np.random.normal(0, 1) for i in range(num_updates)],
        'uncharged_satisfaction': [60 * np.exp(-i/150) + np.random.normal(0, 2) for i in range(num_updates)],
        'rejected_customers': [5 * np.exp(-i/250) + np.random.normal(0, 0.5) for i in range(num_updates)],
        'battery_degradation': [120 * np.exp(-i/300) + np.random.normal(0, 3) for i in range(num_updates)],
    }

    thresholds = {
        'capacity_exceeded': 10.0,
        'uncharged_satisfaction': 50.0,
        'rejected_customers': 2.0,
        'battery_degradation': 100.0,
    }

    # Profit (should increase initially, then stabilize)
    profit_history = [100 + 50 * (1 - np.exp(-i/100)) + np.random.normal(0, 5)
                     for i in range(num_updates)]

    total_violations = [sum([violation_history[k][i] for k in violation_history.keys()])
                       for i in range(num_updates)]

    # Metrics for dashboard
    metrics_history = {
        'reward_mean': profit_history,
        'penalized_reward_mean': [p - 0.5*v for p, v in zip(profit_history, total_violations)],
        **{f'lambda_{k}': v for k, v in lambda_history.items()},
        **{f'{k.replace("_", "_")}': v for k, v in violation_history.items()},
    }

    # Generate plots
    plot_lambda_evolution(lambda_history)
    plot_constraint_violations(violation_history, thresholds)
    plot_profit_vs_constraints(profit_history, total_violations)
    plot_training_summary(metrics_history)

    print("\n✓ Example visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - results/lambda_evolution.png")
    print("  - results/constraint_violations.png")
    print("  - results/profit_vs_constraints.png")
    print("  - results/training_summary.png")


if __name__ == "__main__":
    generate_example_visualizations()


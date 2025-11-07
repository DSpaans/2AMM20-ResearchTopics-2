import jax
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from chargax import Chargax, get_electricity_prices, PPOLagrangianConfig, PPOLagrangian

def create_env(traffic_level, grid_capacity_mult=1.0, num_chargers=16, num_dc_groups=10):
    """Create environment with specified configuration."""
    return Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,

        user_profiles="shopping",
        arrival_frequency=traffic_level,

        minutes_per_timestep=5,
        num_discretization_levels=10,
        num_chargers=num_chargers,
        num_dc_groups=num_dc_groups,

        elec_customer_sell_price=0.75,
        grid_capacity_multiplier=grid_capacity_mult,

        # No manual penalties
        capacity_exceeded_alpha=0.0,
        charged_satisfaction_alpha=0.0,
        time_satisfaction_alpha=0.0,
        rejected_customers_alpha=0.0,
        battery_degredation_alpha=0.0,
    )

def run_experiment(env, config, exp_name):
    """Run training experiment and return metrics."""
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"{'='*80}")

    lagrangian = PPOLagrangian(env, config)
    metrics = lagrangian.train()

    # Print final results
    print(f"\nFinal Results for {exp_name}:")
    print(f"   • Reward: {metrics['mean_reward'][-1]:.2f}")
    print(f"   • Profit: {metrics['mean_profit'][-1]:.2f}")
    print(f"   • Capacity Exceeded: {metrics['mean_capacity_exceeded'][-1]:.2f} kW")
    print(f"   • Rejected Customers: {metrics['mean_rejected_customers'][-1]:.2f}")
    print(f"   • Unmet Demand: {metrics['mean_uncharged_kw'][-1]:.2f} kWh")
    print(f"   • Battery Degradation: {metrics['mean_battery_degradation'][-1]:.2f} kWh")

    return metrics

def plot_comparison(results_dict, output_dir="results"):
    """Generate comparison plots across different scenarios."""
    Path(output_dir).mkdir(exist_ok=True)

    # Plot 1: Profit comparison across traffic levels
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Profit
    ax = axes[0, 0]
    for name, metrics in results_dict.items():
        ax.plot(metrics['mean_profit'], label=name, alpha=0.8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Profit (€)')
    ax.set_title('Profit Across Traffic Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Capacity Exceeded
    ax = axes[0, 1]
    for name, metrics in results_dict.items():
        ax.plot(metrics['mean_capacity_exceeded'], label=name, alpha=0.8)
    ax.axhline(y=2.0, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Capacity Exceeded (kW)')
    ax.set_title('Grid Capacity Violations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rejected Customers
    ax = axes[1, 0]
    for name, metrics in results_dict.items():
        ax.plot(metrics['mean_rejected_customers'], label=name, alpha=0.8)
    ax.axhline(y=0.3, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Rejected Customers')
    ax.set_title('Customer Rejections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Unmet Demand
    ax = axes[1, 1]
    for name, metrics in results_dict.items():
        ax.plot(metrics['mean_uncharged_kw'], label=name, alpha=0.8)
    ax.axhline(y=10.0, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Unmet Demand (kWh)')
    ax.set_title('Unmet Energy Demand')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/traffic_level_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir}/traffic_level_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("=" * 80)
    print("PPO-LAGRANGIAN: SHOPPING PROFILE ACROSS TRAFFIC LEVELS")
    print("=" * 80)

    # STRICT constraint thresholds
    THRESHOLDS = {
        'capacity_exceeded': 2.0,
        'uncharged_satisfaction': 10.0,
        'rejected_customers': 0.3,
        'battery_degradation': 25.0,
    }

    # Configuration for experiments
    base_config = {
        # PPO parameters
        'num_steps': 300,
        'num_envs': 12,
        'num_epochs': 4,
        'num_minibatches': 4,
        'learning_rate': 2.5e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_coef': 0.2,
        'clip_coef_vf': 10.0,
        'ent_coef': 0.01,
        'vf_coef': 0.25,
        'max_grad_norm': 100.0,

        # Lagrangian parameters
        'lambda_lr': 0.035,
        'lambda_init': 0.01,
        'constraint_thresholds': THRESHOLDS,
        'penalty_update_frequency': 10,
        'use_penalty_annealing': True,
    }

    # Training settings (separate from config)
    total_timesteps = 3_000_000

    # Experiments to run
    experiments = [
        # Baseline: Shopping + Medium traffic (standard)
        {
            'name': 'Shopping-Medium (Baseline)',
            'traffic': 100,
            'grid_mult': 1.0,
            'chargers': 16,
            'dc_groups': 10,
        },
        # Shopping + Low traffic
        {
            'name': 'Shopping-Low',
            'traffic': 50,
            'grid_mult': 1.0,
            'chargers': 16,
            'dc_groups': 10,
        },
        # Shopping + High traffic (challenging)
        {
            'name': 'Shopping-High',
            'traffic': 250,
            'grid_mult': 1.0,
            'chargers': 16,
            'dc_groups': 10,
        },
        # Shopping + High traffic + Reduced capacity (VERY challenging)
        {
            'name': 'Shopping-High-ReducedGrid',
            'traffic': 250,
            'grid_mult': 0.20,
            'chargers': 12,
            'dc_groups': 5,
        },
    ]

    # Store results
    all_results = {}

    # Run all experiments
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Experiment: {exp['name']}")
        print(f"   Traffic: {exp['traffic']} cars/day")
        print(f"   Grid capacity: {exp['grid_mult']*100:.0f}% (~{exp['grid_mult']*800:.0f} kW)")
        print(f"   Chargers: {exp['chargers']}")
        print(f"   DC groups: {exp['dc_groups']}")
        print(f"{'='*80}")

        # Create environment
        env = create_env(
            traffic_level=exp['traffic'],
            grid_capacity_mult=exp['grid_mult'],
            num_chargers=exp['chargers'],
            num_dc_groups=exp['dc_groups']
        )

        # Create config
        config = PPOLagrangianConfig(**base_config)

        # Run experiment
        metrics = run_experiment(env, config, exp['name'])
        all_results[exp['name']] = metrics

    # Generate comparison plots
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    plot_comparison(all_results)

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL EXPERIMENTS")
    print("=" * 80)
    print(f"\n{'Experiment':<35} {'Profit':<12} {'Capacity':<12} {'Rejected':<12} {'Unmet':<12}")
    print("-" * 80)

    for name, metrics in all_results.items():
        profit = metrics['mean_profit'][-1]
        capacity = metrics['mean_capacity_exceeded'][-1]
        rejected = metrics['mean_rejected_customers'][-1]
        unmet = metrics['mean_uncharged_kw'][-1]

        print(f"{name:<35} {profit:<12.2f} {capacity:<12.2f} {rejected:<12.2f} {unmet:<12.2f}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


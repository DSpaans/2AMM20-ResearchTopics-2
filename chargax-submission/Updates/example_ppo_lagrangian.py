"""
Example: Using PPO-Lagrangian for Constrained EV Charging Optimization

This example demonstrates how to train an agent that maximizes profit
while respecting operational constraints using PPO-Lagrangian.
"""

import jax
import jax.numpy as jnp
from chargax import Chargax, get_electricity_prices
from chargax.ppo_lagrangian import create_ppo_lagrangian, PPOLagrangianConfig


def main():
    """Train a constrained RL agent for EV charging station control."""

    # Create CHALLENGING environment to force constraint violations
    print("Initializing Chargax environment...")
    print("⚠️  Using CHALLENGING configuration to force constraint violations:")
    print("   - Grid capacity: 25% of normal (~200 kW instead of ~800 kW)")
    print("   - Car arrivals: 250 cars/day (HIGH frequency)")
    print("   - Chargers: 12 (reduced from 16)")
    print("   - DC groups: 6 (fewer high-power chargers)")

    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,

        minutes_per_timestep=5,
        num_discretization_levels=10,

        # REDUCED RESOURCES - Forces scarcity
        num_chargers=12,            # Reduced from 16 (fewer charging spots)
        num_dc_groups=6,            # Reduced from 10 (fewer high-power chargers)

        # INCREASED DEMAND - More cars competing for resources
        arrival_frequency=250,      # HIGH frequency (was 100, now 250 cars/day)

        elec_customer_sell_price=0.75,

        # SEVERELY REDUCED GRID CAPACITY - Creates bottleneck
        # With only 25% capacity, agent MUST make hard choices
        grid_capacity_multiplier=0.25,  # ~200 kW instead of ~800 kW

        # Note: With PPO-Lagrangian, we keep these at 0 and let
        # the Lagrange multipliers handle constraint enforcement
        capacity_exceeded_alpha=0.0,
        charged_satisfaction_alpha=0.0,
        time_satisfaction_alpha=0.0,
        rejected_customers_alpha=0.0,
        battery_degredation_alpha=0.0,
    )

    # Define VERY STRICT constraint thresholds
    # With the challenging environment above, these WILL be violated
    constraint_thresholds = {
        # Grid capacity: Allow max 3 kW violations (very strict!)
        # With 25% capacity and high demand, this will be constantly challenged
        'capacity_exceeded': 3.0,

        # Customer satisfaction: Allow max 15 kWh unmet demand (very tight)
        # With 250 cars/day and limited chargers, agent must prioritize carefully
        'uncharged_satisfaction': 15.0,

        # Queue management: Allow ZERO rejected customers (ideal service)
        # With high arrivals and few chargers, this is extremely difficult
        'rejected_customers': 0.5,  # Less than 1 per episode

        # Battery health: Allow max 30 kWh battery cycling (protect battery)
        # Forces minimal battery degradation strategy
        'battery_degradation': 30.0,
    }

    print("\nConstraint Configuration:")
    print("-" * 60)
    for name, threshold in constraint_thresholds.items():
        print(f"  {name:30s}: {threshold:8.2f}")
    print("-" * 60)

    # Create PPO-Lagrangian agent with custom configuration
    config = PPOLagrangianConfig(
        # Standard PPO parameters (matching paper)
        num_steps=300,
        num_envs=12,
        num_epochs=4,
        num_minibatches=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        clip_coef_vf=10.0,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=100.0,

        # Lagrangian-specific parameters
        lambda_lr=0.035,           # Learning rate for Lagrange multipliers
        lambda_init=0.01,          # Initial penalty strength
        penalty_update_frequency=10,  # Update lambdas every 10 PPO updates
        use_penalty_annealing=True,

        constraint_thresholds=constraint_thresholds,
    )

    agent = create_ppo_lagrangian(env, **config._asdict())

    # Initialize random key
    rng = jax.random.PRNGKey(42)

    print("\nStarting PPO-Lagrangian Training...")
    print("This will automatically adjust constraint penalties during training.")
    print("=" * 60)

    # Train the agent
    # Note: This is a simplified example. In practice, you would integrate
    # this with your full PPO implementation or use Jaxnasium
    agent.train(rng, env, total_timesteps=int(1e7))

    # Print final Lagrangian state
    print("\n" + "=" * 60)
    print("Final Lagrangian State:")
    print("=" * 60)
    lag_info = agent.get_lagrangian_info()

    print("\nLearned Penalty Weights (Lambda values):")
    for name, value in lag_info['lambda_values'].items():
        print(f"  λ_{name:30s}: {value:8.4f}")

    print("\nFinal Constraint Violations (Running Average):")
    for name, value in lag_info['constraint_violations'].items():
        threshold = lag_info['constraint_thresholds'][name]
        status = "✓" if value <= threshold else "✗"
        print(f"  {status} {name:30s}: {value:8.2f} / {threshold:8.2f}")

    print("\nTotal Updates:", lag_info['update_count'])
    print("=" * 60)


def compare_unconstrained_vs_constrained():
    """
    Compare unconstrained PPO (pure profit maximization) vs
    constrained PPO-Lagrangian.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Unconstrained vs Constrained RL")
    print("=" * 60)

    print("\n1. UNCONSTRAINED PPO (Current main.py approach):")
    print("   - Objective: Maximize profit only")
    print("   - All constraint alphas = 0")
    print("   - May violate capacity limits")
    print("   - May leave customers unsatisfied")
    print("   - Maximum profit but potentially unsafe operation")

    print("\n2. MANUAL PENALTY TUNING:")
    print("   - Objective: Profit - α₁·capacity - α₂·uncharged - ...")
    print("   - Requires careful tuning of α values")
    print("   - Hard to balance profit vs constraints")
    print("   - Fixed penalties throughout training")

    print("\n3. PPO-LAGRANGIAN (This Implementation):")
    print("   - Objective: Maximize profit")
    print("   - Constraints: Hard limits on violations")
    print("   - Automatically learns penalty weights (λ)")
    print("   - Adapts penalties during training")
    print("   - Guarantees constraint satisfaction in expectation")

    print("\n" + "=" * 60)

    print("\nKey Advantage of PPO-Lagrangian:")
    print("  → You specify WHAT constraints to satisfy (thresholds)")
    print("  → Algorithm learns HOW MUCH to penalize violations (λ)")
    print("  → No manual tuning of penalty weights!")

    print("\nWhen to use PPO-Lagrangian:")
    print("  ✓ Need to guarantee safety/operational constraints")
    print("  ✓ Don't want to manually tune penalty weights")
    print("  ✓ Constraints are critical (grid capacity, customer SLA)")
    print("  ✓ Want interpretable constraint satisfaction metrics")

    print("\nWhen unconstrained PPO is sufficient:")
    print("  ✓ Pure optimization problem without hard constraints")
    print("  ✓ Environment naturally limits bad behavior")
    print("  ✓ Matching paper's pure profit maximization setting")

    print("=" * 60)


if __name__ == "__main__":
    # Run the main training example
    main()

    # Show comparison
    compare_unconstrained_vs_constrained()

    print("\n" + "=" * 60)
    print("INTEGRATION NOTES")
    print("=" * 60)
    print("""
To fully integrate PPO-Lagrangian with your existing setup:

1. Update __init__.py to export PPO-Lagrangian:
   from .ppo_lagrangian import create_ppo_lagrangian, PPOLagrangianConfig

2. Integrate with Jaxnasium PPO:
   - Modify reward computation to include Lagrangian penalties
   - Add constraint tracking to rollout collection
   - Add lambda update logic after PPO updates

3. For production use:
   - Set appropriate constraint thresholds based on:
     * Grid connection capacity
     * Customer SLA requirements
     * Battery warranty specifications
   - Monitor lambda values during training
   - Adjust lambda_lr if penalties converge too slowly/quickly

4. Evaluation:
   - Track both profit AND constraint violations
   - Compare against baseline (unconstrained PPO)
   - Verify constraints are satisfied in held-out episodes
""")
    print("=" * 60)


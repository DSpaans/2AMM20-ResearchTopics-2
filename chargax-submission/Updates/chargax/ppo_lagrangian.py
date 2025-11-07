"""
PPO-Lagrangian: Constrained Reinforcement Learning for EV Charging Station Control

This implements PPO with Lagrangian relaxation to handle constraints during training.
The agent learns to maximize profit while respecting operational constraints such as:
- Grid capacity limits
- Customer satisfaction (charging requirements)
- Response time constraints
- Battery health constraints

Reference:
Ray et al. "Benchmarking Safe Exploration in Deep Reinforcement Learning" (2019)
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, NamedTuple, Callable
from functools import partial
import equinox as eqx
import jymkit as jym


class LagrangianState(NamedTuple):
    """State for Lagrangian multipliers and constraint tracking."""
    lambda_values: jnp.ndarray  # Lagrange multipliers for each constraint
    constraint_violations: jnp.ndarray  # Running average of constraint violations
    update_count: int  # Number of updates performed


class PPOLagrangianConfig(NamedTuple):
    """Configuration for PPO-Lagrangian algorithm."""
    # Standard PPO parameters
    num_steps: int = 300
    num_envs: int = 12
    num_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0
    ent_coef: float = 0.01
    vf_coef: float = 0.25
    max_grad_norm: float = 100.0

    # Lagrangian-specific parameters
    lambda_lr: float = 0.035  # Learning rate for Lagrange multipliers
    lambda_init: float = 0.01  # Initial value for Lagrange multipliers
    constraint_thresholds: Dict[str, float] = None  # Max allowed violation per constraint
    penalty_update_frequency: int = 10  # Update lambdas every N PPO updates
    use_penalty_annealing: bool = True  # Gradually increase penalty strength

    # Constraint settings
    constraint_names: Tuple[str, ...] = (
        'capacity_exceeded',
        'uncharged_satisfaction',
        'rejected_customers',
        'battery_degradation'
    )


class ConstraintBuffer(NamedTuple):
    """Buffer for tracking constraint violations during rollouts."""
    capacity_exceeded: jnp.ndarray
    uncharged_kw: jnp.ndarray
    rejected_customers: jnp.ndarray
    battery_degradation: jnp.ndarray

    @staticmethod
    def empty(num_steps: int, num_envs: int):
        """Create an empty constraint buffer."""
        return ConstraintBuffer(
            capacity_exceeded=jnp.zeros((num_steps, num_envs)),
            uncharged_kw=jnp.zeros((num_steps, num_envs)),
            rejected_customers=jnp.zeros((num_steps, num_envs)),
            battery_degradation=jnp.zeros((num_steps, num_envs)),
        )

    def add_step(self, step_idx: int, info_dict: Dict) -> 'ConstraintBuffer':
        """Add constraint values from a timestep."""
        logging_data = info_dict.get('logging_data', {})

        return ConstraintBuffer(
            capacity_exceeded=self.capacity_exceeded.at[step_idx].set(
                logging_data.get('exceeded_capacity', 0.0)
            ),
            uncharged_kw=self.uncharged_kw.at[step_idx].set(
                logging_data.get('uncharged_kw', 0.0)
            ),
            rejected_customers=self.rejected_customers.at[step_idx].set(
                logging_data.get('rejected_customers', 0.0)
            ),
            battery_degradation=self.battery_degradation.at[step_idx].set(
                logging_data.get('total_discharged_kw', 0.0)
            ),
        )

    def get_mean_violations(self) -> jnp.ndarray:
        """Compute mean constraint violations across the buffer."""
        return jnp.array([
            self.capacity_exceeded.mean(),
            self.uncharged_kw.mean(),
            self.rejected_customers.mean(),
            self.battery_degradation.mean(),
        ])


def initialize_lagrangian_state(config: PPOLagrangianConfig) -> LagrangianState:
    """Initialize Lagrange multipliers and constraint tracking."""
    num_constraints = len(config.constraint_names)
    return LagrangianState(
        lambda_values=jnp.full(num_constraints, config.lambda_init),
        constraint_violations=jnp.zeros(num_constraints),
        update_count=0,
    )


def compute_lagrangian_penalty(
    constraint_violations: jnp.ndarray,
    lambda_values: jnp.ndarray,
    constraint_thresholds: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the Lagrangian penalty term.

    Penalty = sum_i lambda_i * max(0, constraint_i - threshold_i)
    """
    excess_violations = jnp.maximum(0.0, constraint_violations - constraint_thresholds)
    penalty = jnp.sum(lambda_values * excess_violations)
    return penalty


def update_lagrange_multipliers(
    lag_state: LagrangianState,
    constraint_violations: jnp.ndarray,
    constraint_thresholds: jnp.ndarray,
    lambda_lr: float,
) -> LagrangianState:
    """
    Update Lagrange multipliers using gradient ascent on the dual problem.

    lambda_new = max(0, lambda_old + lr * (constraint - threshold))
    """
    excess_violations = constraint_violations - constraint_thresholds

    # Gradient ascent on dual problem
    new_lambdas = lag_state.lambda_values + lambda_lr * excess_violations

    # Project to non-negative values (Lagrange multipliers must be >= 0)
    new_lambdas = jnp.maximum(0.0, new_lambdas)

    # Update running average of constraint violations with exponential moving average
    alpha = 0.9  # Smoothing factor
    new_violations = alpha * lag_state.constraint_violations + (1 - alpha) * constraint_violations

    return LagrangianState(
        lambda_values=new_lambdas,
        constraint_violations=new_violations,
        update_count=lag_state.update_count + 1,
    )


def compute_penalized_reward(
    reward: jnp.ndarray,
    constraint_buffer: ConstraintBuffer,
    lambda_values: jnp.ndarray,
    constraint_thresholds: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the penalized reward for each timestep.

    r_penalized = r_original - sum_i lambda_i * max(0, c_i - threshold_i)
    """
    # Get constraint values at each timestep
    constraint_values = jnp.stack([
        constraint_buffer.capacity_exceeded,
        constraint_buffer.uncharged_kw,
        constraint_buffer.rejected_customers,
        constraint_buffer.battery_degradation,
    ], axis=-1)  # Shape: (num_steps, num_envs, num_constraints)

    # Compute excess violations
    excess = jnp.maximum(0.0, constraint_values - constraint_thresholds)

    # Compute penalty at each timestep
    penalty = jnp.sum(lambda_values * excess, axis=-1)  # Shape: (num_steps, num_envs)

    # Apply penalty to rewards
    penalized_reward = reward - penalty

    return penalized_reward


class PPOLagrangian:
    """PPO with Lagrangian relaxation for constrained RL."""

    def __init__(self, config: PPOLagrangianConfig):
        self.config = config

        # Set default constraint thresholds if not provided
        if config.constraint_thresholds is None:
            self.constraint_thresholds = jnp.array([
                10.0,   # capacity_exceeded (kW)
                50.0,   # uncharged_kw
                2.0,    # rejected_customers
                100.0,  # battery_degradation (kWh)
            ])
        else:
            self.constraint_thresholds = jnp.array([
                config.constraint_thresholds.get(name, 0.0)
                for name in config.constraint_names
            ])

        # Initialize Lagrangian state
        self.lag_state = initialize_lagrangian_state(config)

    def collect_rollout(
        self,
        rng: jax.random.PRNGKey,
        env: jym.Environment,
        policy_fn: Callable,
        value_fn: Callable,
    ) -> Tuple[Dict, ConstraintBuffer]:
        """
        Collect a rollout while tracking constraints.

        Returns:
            rollout_data: Standard PPO rollout data (obs, actions, rewards, etc.)
            constraint_buffer: Constraint violations at each step
        """
        # Initialize
        num_steps = self.config.num_steps
        num_envs = self.config.num_envs

        # Reset environments
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        # Storage
        observations = []
        actions_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        log_probs_list = []

        constraint_buffer = ConstraintBuffer.empty(num_steps, num_envs)

        # Collect rollout
        for step in range(num_steps):
            rng, action_key = jax.random.split(rng)

            # Get action from policy
            action, log_prob = policy_fn(obs, action_key)
            value = value_fn(obs)

            # Step environment
            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, num_envs)
            timestep, env_state = jax.vmap(env.step)(step_keys, env_state, action)

            # Store data
            observations.append(obs)
            actions_list.append(action)
            rewards_list.append(timestep.reward)
            dones_list.append(timestep.terminated | timestep.truncated)
            values_list.append(value)
            log_probs_list.append(log_prob)

            # Track constraints
            constraint_buffer = constraint_buffer.add_step(step, timestep.info)

            # Update observation
            obs = timestep.observation

        # Stack everything
        rollout_data = {
            'observations': jnp.stack(observations),
            'actions': jnp.stack(actions_list),
            'rewards': jnp.stack(rewards_list),
            'dones': jnp.stack(dones_list),
            'values': jnp.stack(values_list),
            'log_probs': jnp.stack(log_probs_list),
        }

        return rollout_data, constraint_buffer

    def compute_advantages(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        next_value: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute GAE advantages."""
        num_steps = len(rewards)
        advantages = jnp.zeros_like(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae_lam
            advantages = advantages.at[t].set(last_gae_lam)

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        rng: jax.random.PRNGKey,
        rollout_data: Dict,
        constraint_buffer: ConstraintBuffer,
        policy_params: any,
        value_params: any,
        optimizer_state: any,
    ) -> Tuple[any, any, any, Dict]:
        """Perform one PPO update with Lagrangian penalties."""

        # Apply Lagrangian penalties to rewards
        penalized_rewards = compute_penalized_reward(
            rollout_data['rewards'],
            constraint_buffer,
            self.lag_state.lambda_values,
            self.constraint_thresholds,
        )

        # Compute advantages with penalized rewards
        rollout_data['rewards'] = penalized_rewards

        # Standard PPO update would go here
        # (This is a simplified version - full implementation would include
        #  the actual PPO update logic with minibatch sampling, etc.)

        # Update Lagrange multipliers periodically
        if self.lag_state.update_count % self.config.penalty_update_frequency == 0:
            mean_violations = constraint_buffer.get_mean_violations()
            self.lag_state = update_lagrange_multipliers(
                self.lag_state,
                mean_violations,
                self.constraint_thresholds,
                self.config.lambda_lr,
            )

        # Compute metrics
        metrics = {
            'penalized_reward_mean': penalized_rewards.mean(),
            'original_reward_mean': rollout_data['rewards'].mean(),
            'capacity_violation': constraint_buffer.capacity_exceeded.mean(),
            'uncharged_violation': constraint_buffer.uncharged_kw.mean(),
            'rejected_violation': constraint_buffer.rejected_customers.mean(),
            'battery_violation': constraint_buffer.battery_degradation.mean(),
            'lambda_capacity': self.lag_state.lambda_values[0],
            'lambda_uncharged': self.lag_state.lambda_values[1],
            'lambda_rejected': self.lag_state.lambda_values[2],
            'lambda_battery': self.lag_state.lambda_values[3],
        }

        return policy_params, value_params, optimizer_state, metrics

    def train(
        self,
        rng: jax.random.PRNGKey,
        env: jym.Environment,
        total_timesteps: int = int(1e7),
    ) -> 'PPOLagrangian':
        """
        Train the agent using PPO-Lagrangian.

        This is a high-level training loop. In practice, you would integrate this
        with your existing PPO implementation or use a library like Jaxnasium.
        """
        print("=" * 60)
        print("PPO-Lagrangian Training")
        print("=" * 60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Num environments: {self.config.num_envs}")
        print(f"Steps per rollout: {self.config.num_steps}")
        print(f"Constraint thresholds: {dict(zip(self.config.constraint_names, self.constraint_thresholds))}")
        print(f"Initial lambda values: {self.lag_state.lambda_values}")
        print("=" * 60)

        # This would contain the full training loop
        # For now, we'll return self to maintain compatibility
        return self

    def get_lagrangian_info(self) -> Dict:
        """Get current state of Lagrange multipliers and constraints."""
        return {
            'lambda_values': dict(zip(self.config.constraint_names, self.lag_state.lambda_values)),
            'constraint_violations': dict(zip(self.config.constraint_names, self.lag_state.constraint_violations)),
            'update_count': self.lag_state.update_count,
            'constraint_thresholds': dict(zip(self.config.constraint_names, self.constraint_thresholds)),
        }


def create_ppo_lagrangian(
    env: jym.Environment,
    constraint_thresholds: Dict[str, float] = None,
    **kwargs
) -> PPOLagrangian:
    """
    Convenience function to create a PPO-Lagrangian agent.

    Args:
        env: The Chargax environment
        constraint_thresholds: Dict mapping constraint names to max allowed violations
        **kwargs: Additional config parameters

    Example:
        >>> from chargax import Chargax, get_electricity_prices
        >>> from chargax.ppo_lagrangian import create_ppo_lagrangian
        >>>
        >>> env = Chargax(
        ...     elec_grid_buy_price=get_electricity_prices("2023_NL"),
        ...     elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
        ... )
        >>>
        >>> # Define constraint thresholds
        >>> constraints = {
        ...     'capacity_exceeded': 5.0,      # Max 5 kW capacity violation
        ...     'uncharged_satisfaction': 30.0, # Max 30 kWh unmet demand
        ...     'rejected_customers': 1.0,      # Max 1 rejected customer per episode
        ...     'battery_degradation': 50.0,    # Max 50 kWh battery cycling
        ... }
        >>>
        >>> agent = create_ppo_lagrangian(env, constraint_thresholds=constraints)
        >>> agent.train(jax.random.PRNGKey(42), env)
    """
    config = PPOLagrangianConfig(
        constraint_thresholds=constraint_thresholds,
        **kwargs
    )
    return PPOLagrangian(config)


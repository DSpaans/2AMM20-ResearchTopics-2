
import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import optax
from typing import NamedTuple, List, Dict, Tuple
from dataclasses import replace

from chargax import Chargax, LogWrapper, NormalizeVecObservation
from chargax.algorithms.networks import ActorNetworkMultiDiscrete, CriticNetwork
import wandb


def create_ppo_networks(
    key,
    in_shape: int,
    actor_features: List[int],
    critic_features: List[int],
    actions_nvec: int,
):
    actor_key, critic1_key = jax.random.split(key)
    actor = ActorNetworkMultiDiscrete(actor_key, in_shape, actor_features, actions_nvec)
    critic = CriticNetwork(critic1_key, in_shape, critic_features)
    return actor, critic


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array


class PPOConfig(NamedTuple):
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    total_timesteps: int = int(5e6)
    num_envs: int = 12
    num_steps: int = 300
    num_minibatches: int = 4
    update_epochs: int = 4

    seed: int = 42
    debug: bool = False
    evaluate_deterministically: bool = False

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs

    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches


class LagrangianConfig(NamedTuple):
    # Lagrangian hyperparams
    lambda_lr: float = 0.05
    lambda_init: float = 0.01
    penalty_update_frequency: int = 1  # update lambdas every K PPO iterations
    # Constraints (episode-level thresholds unless noted)
    # thresholds: Dict[str, float] = {
    #     "exceeded_capacity": 5.0,        # kW per step (instantaneous)
    #     "uncharged_kw": 100.0,            # kWh per episode
    #     "rejected_customers": 20.0,       # count per episode
    #     "total_discharged_kw": 4500.0,    # kWh per episode (proxy for degradation)
    # }
    thresholds: Dict[str, float] | None = None


class TrainStateLag(NamedTuple):
    actor: eqx.Module
    critic: eqx.Module
    optimizer_state: optax.OptState
    lambdas: jnp.ndarray           # shape (4,)
    iter_count: jnp.ndarray        # scalar


def _extract_logging_series(info_tree) -> Dict[str, jnp.ndarray]:
    """""""""Return series with shape (T, N) for each metric from the batched trajectory info.
    Expected keys inside info['logging_data']: exceeded_capacity (per-step),
    uncharged_kw (cumulative), rejected_customers (cumulative), total_discharged_kw (cumulative).
    """""""""
    ld = info_tree["""logging_data"""]
    out = {
        """exceeded_capacity""": ld["""exceeded_capacity"""],        # (T, N)
        """uncharged_kw""": ld["""uncharged_kw"""],                  # (T, N) cumulative
        """rejected_customers""": ld["""rejected_customers"""],      # (T, N) cumulative
        """total_discharged_kw""": ld["""total_discharged_kw"""],    # (T, N) cumulative
    }
    return out


def _first_diff_along_time(x: jnp.ndarray) -> jnp.ndarray:
    # x: (T, N) -> return per-step increments with x[0] assumed 0 previous
    zeros = jnp.zeros_like(x[:1])
    prev = jnp.concatenate([zeros, x[:-1]], axis=0)
    return x - prev


def _per_step_constraints(info_tree, num_steps: int) -> jnp.ndarray:
    """""""""Build (T, N, 4) per-step constraint array in fixed order:
    [exceeded_capacity, uncharged_kw, rejected_customers, total_discharged_kw].
    exceeded_capacity is already per-step; others are converted to per-step increments via diff.
    """""""""
    series = _extract_logging_series(info_tree)
    exceeded = series["""exceeded_capacity"""]
    uncharged = _first_diff_along_time(series["""uncharged_kw"""])
    rejected = _first_diff_along_time(series["""rejected_customers"""])
    degr = _first_diff_along_time(series["""total_discharged_kw"""])
    return jnp.stack([exceeded, uncharged, rejected, degr], axis=-1)  # (T, N, 4)


def _episode_totals_from_series(info_tree) -> jnp.ndarray:
    """""""""Return (N, 4) episode totals to update lambdas. Uses the same order as above."""""""""
    series = _extract_logging_series(info_tree)
    exceeded_total = series["""exceeded_capacity"""].sum(axis=0)         # sum over time
    # for cumulative metrics, just take the last time step
    uncharged_total = series["""uncharged_kw"""][-1]
    rejected_total = series["""rejected_customers"""][-1]
    degr_total = series["""total_discharged_kw"""][-1]
    return jnp.stack([exceeded_total, uncharged_total, rejected_total, degr_total], axis=-1)  # (N, 4)


def build_ppo_lagrangian_trainer(
    env: Chargax,
    ppo_config: Dict = {},
    lag_config: Dict = {},
    baselines: Dict = {},
):
    # """""""""Drop-in alternative for build_ppo_trainer with Lagrangian penalties.

    # Usage (in train.py):
    #     from ppo_lagrangian_chargax import build_ppo_lagrangian_trainer
    #     train, cfg = build_ppo_lagrangian_trainer(env, ppo_config, {
    #         """thresholds""": {
    #             """exceeded_capacity""": 0.0,
    #             """uncharged_kw""": 40.0,
    #             """rejected_customers""": 1.0,
    #             """total_discharged_kw""": 80.0,
    #         },
    #         """lambda_lr""": 0.05,
    #         """penalty_update_frequency""": 1,
    #     })
    #     trained_state, logs = train()

    # Notes:
    # - We assume rollout length equals episode length (as in your current PPO: num_steps == env.episode_length).
    # - Thresholds for cumulative metrics are per episode; for per-step penalty we divide by num_steps.
    # - Lagrange multipliers are updated every `penalty_update_frequency` PPO iterations using episode totals.
    # """""""""

    # Wrap env (same as PPO)
    env = LogWrapper(env)
    env = NormalizeVecObservation(env)
    observation_space = env.observation_space
    action_space = env.action_space
    logging_baselines = baselines

    cfg = PPOConfig(**ppo_config)
    lagcfg = LagrangianConfig(**lag_config) if len(lag_config) else LagrangianConfig()

    # rng & networks
    rng = jax.random.PRNGKey(cfg.seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)

    actor, critic = create_ppo_networks(
        key=network_key,
        in_shape=observation_space.shape[0],
        actor_features=[256, 256],
        critic_features=[256, 256],
        actions_nvec=action_space.nvec,
    )

    # optimizer & lr schedule
    def linear_schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_iterations
        return cfg.learning_rate * frac

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=linear_schedule if cfg.anneal_lr else cfg.learning_rate, eps=1e-5),
    )
    optimizer_state = optimizer.init({"""actor""": actor, """critic""": critic})

    # Lagrange multipliers (order fixed to 4 constraints)
    default_thresholds = {
        "exceeded_capacity": 0.0,   # per step
        "uncharged_kw": 90.0,       # per episode
        "rejected_customers": 40.0, # per episode
        "total_discharged_kw": 100.0,  # per episode
    }
    thresholds_dict = default_thresholds.copy()
    if lagcfg.thresholds is not None:
        thresholds_dict.update(lagcfg.thresholds)

    thresholds_vec = jnp.array(
        [
            float(thresholds_dict["exceeded_capacity"]),
            float(thresholds_dict["uncharged_kw"]),
            float(thresholds_dict["rejected_customers"]),
            float(thresholds_dict["total_discharged_kw"]),
        ],
        dtype=jnp.float32,
    )

    # Helpful print once at startup so you can see what’s actually used
    print(
        "[PPO-Lagrangian] thresholds used (episode level): "
        f"exceeded_capacity={thresholds_vec[0]}, "
        f"uncharged_kw={thresholds_vec[1]}, "
        f"rejected_customers={thresholds_vec[2]}, "
        f"total_discharged_kw={thresholds_vec[3]}"
    )

    train_state = TrainStateLag(
        actor=actor,
        critic=critic,
        optimizer_state=optimizer_state,
        lambdas=jnp.full((4,), lagcfg.lambda_init, dtype=jnp.float32),
        iter_count=jnp.array(0, dtype=jnp.int32),
    )

    # Reset vectorized envs
    rng, key = jax.random.split(rng)
    reset_keys = jax.random.split(key, cfg.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

    def eval_func(train_state: TrainStateLag, rng):
        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action_dist = train_state.actor(obs)
            if cfg.evaluate_deterministically:
                action = jnp.argmax(action_dist.logits, axis=-1)
            else:
                action = action_dist.sample(seed=action_key)
            (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)
            episode_reward += reward
            return (rng, obs, env_state, done, episode_reward)

        def cond_func(carry):
            return jnp.logical_not(carry[3])

        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key)
        done = False
        episode_reward = 0.0

        rng, obs, env_state, done, episode_reward = jax.lax.while_loop(cond_func, step_env, (rng, obs, env_state, done, episode_reward))
        return episode_reward

    def train_func(rng_outer=rng):

        # ------- Inner helpers (inside jit/scan) -------
        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # policy
            action_dist = jax.vmap(train_state.actor)(last_obs)
            value = jax.vmap(train_state.critic)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)

            # step envs
            rng, key = jax.random.split(rng)
            vstep_keys = jax.random.split(key, cfg.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = jax.vmap(env.step, in_axes=(0, 0, 0))(vstep_keys, env_state, action)
            done = jnp.logical_or(terminated, truncated)

            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info,
            )
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        def _compute_advantages_from_arrays(rewards, values, dones, last_value):
            # rewards/values/dones: (T, N) ; last_value: (N,)
            def scan_fn(carry, t_inputs):
                gae = carry
                r_t, v_t, d_t, nv_t = t_inputs
                delta = r_t + cfg.gamma * nv_t * (1 - d_t) - v_t
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - d_t) * gae
                return gae, gae + v_t

            # reverse over time
            next_values = jnp.concatenate([values[1:], last_value[None, ...]], axis=0)
            _, returns_rev = jax.lax.scan(
                lambda carry, ts: scan_fn(carry, ts),
                jnp.zeros_like(last_value),
                (rewards[::-1], values[::-1], dones[::-1], next_values[::-1]),
            )
            returns = returns_rev[::-1]
            advantages = returns - values
            return advantages, returns

        def _update_epoch(update_state, _):

            @eqx.filter_value_and_grad(has_aux=True)
            def _loss_fn(params, trajectory_mb, advantages_mb, returns_mb):
                action_dist = jax.vmap(params["""actor"""])(trajectory_mb.observation)
                log_prob = action_dist.log_prob(trajectory_mb.action).sum(axis=-1)
                entropy = action_dist.entropy().mean()
                value = jax.vmap(params["""critic"""])(trajectory_mb.observation)

                # clipped policy loss
                ratio = jnp.exp(log_prob - trajectory_mb.log_prob.sum(axis=-1))
                adv_norm = (advantages_mb - advantages_mb.mean()) / (advantages_mb.std() + 1e-8)
                loss1 = adv_norm * ratio
                loss2 = jnp.clip(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_norm
                actor_loss = -jnp.minimum(loss1, loss2).mean()

                # clipped value loss
                vpred_clip = trajectory_mb.value + jnp.clip(value - trajectory_mb.value, -cfg.clip_coef_vf, cfg.clip_coef_vf)
                v_loss = jnp.maximum(jnp.square(value - returns_mb), jnp.square(vpred_clip - returns_mb)).mean()

                total_loss = actor_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                return total_loss, (actor_loss, v_loss, entropy)

            def _apply_minibatch(train_state, minibatch):
                traj_mb, adv_mb, ret_mb = minibatch
                (total_loss, _), grads = _loss_fn(
                    {"""actor""": train_state.actor, """critic""": train_state.critic},
                    traj_mb, adv_mb, ret_mb
                )
                updates, new_opt_state = optimizer.update(grads, train_state.optimizer_state)
                new_networks = optax.apply_updates({"""actor""": train_state.actor, """critic""": train_state.critic}, updates)
                return TrainStateLag(
                    actor=new_networks["""actor"""],
                    critic=new_networks["""critic"""],
                    optimizer_state=new_opt_state,
                    lambdas=train_state.lambdas,
                    iter_count=train_state.iter_count,
                ), total_loss

            train_state, trajectory_batch, advantages, returns, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, cfg.batch_size)
            batch = (trajectory_batch, advantages, returns)
            batch = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), batch)
            shuffled = jax.tree.map(lambda x: jnp.take(x, batch_idx, axis=0), batch)
            minibatches = jax.tree.map(lambda x: x.reshape((cfg.num_minibatches, -1) + x.shape[1:]), shuffled)

            train_state, total_loss = jax.lax.scan(_apply_minibatch, train_state, minibatches)
            return (train_state, trajectory_batch, advantages, returns, rng), total_loss

        def train_step(runner_state, _):

            # 1) Rollout
            runner_state, trajectory_batch = jax.lax.scan(_env_step, runner_state, None, cfg.num_steps)

            # 2) Build per-step penalties
            # (T, N, 4) instantaneous constraint values
            constraint_steps = _per_step_constraints(trajectory_batch.info, cfg.num_steps)

            # thresholds per step (exceeded_capacity is per-step already; others per-episode / T)
            per_step_thresholds = jnp.array([
                thresholds_vec[0],
                thresholds_vec[1] / cfg.num_steps,
                thresholds_vec[2] / cfg.num_steps,
                thresholds_vec[3] / cfg.num_steps,
            ])  # (4,)

            # excess violations (T, N, 4)
            excess = jnp.maximum(0.0, constraint_steps - per_step_thresholds)

            # penalty per (t, env)
            penalty = (runner_state[0].lambdas * excess).sum(axis=-1)  # (T, N)

            # penalized rewards
            rewards = trajectory_batch.reward - penalty

            # 3) Compute advantages/returns from penalized rewards
            train_state, env_state, last_obs, rng = runner_state
            last_value = jax.vmap(train_state.critic)(last_obs)  # (N,)
            values = trajectory_batch.value  # (T, N)
            dones = trajectory_batch.done   # (T, N)
            advantages, returns = _compute_advantages_from_arrays(rewards, values, dones, last_value)

            # 4) PPO update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, cfg.update_epochs)

            # 5) Lagrange multiplier update using episode totals
            new_train_state = update_state[0]
            totals = _episode_totals_from_series(trajectory_batch.info).mean(axis=0)  # (4,)
            def _update_lambdas(ts: TrainStateLag):
                lambdas = jnp.maximum(0.0, ts.lambdas + lagcfg.lambda_lr * (totals - thresholds_vec))
                return TrainStateLag(
                    actor=ts.actor, critic=ts.critic, optimizer_state=ts.optimizer_state,
                    lambdas=lambdas, iter_count=ts.iter_count + 1,
                )

            new_train_state = jax.lax.cond(
                (new_train_state.iter_count % lagcfg.penalty_update_frequency) == 0,
                _update_lambdas,
                lambda ts: ts,
                new_train_state
            )

            # 6) Logging & eval (same style as PPO)
            # 6) Logging & eval (same style as PPO)
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            rng = update_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(new_train_state, eval_key)
            per_step_mean = constraint_steps.mean(axis=(0, 1))  # (4,)

            # Package our extra fields so we can log them via debug.callback (outside JIT)
            metric["eval_rewards"] = eval_rewards
            metric["lagrangian"] = {
                "lambdas": new_train_state.lambdas,
                "per_step_mean": per_step_mean,
                "episode_totals": totals,
                "thresholds": thresholds_vec,
            }

            def callback(info):
                if cfg.debug:
                    print(f'timestep={(info["train_timestep"][-1][0] * cfg.num_envs)}, eval rewards={info["eval_rewards"]}')
                if wandb.run:
                    if "logging_data" not in info:
                        info["logging_data"] = {}
                    finished_episodes = info["returned_episode"]
                    if finished_episodes.any():
                        info["logging_data"] = jax.tree.map(
                            lambda x: x[finished_episodes].mean(), info["logging_data"]
                        )
                        # Unpack our lagrangian fields
                        lag = info["lagrangian"]
                        lambdas = lag["lambdas"]
                        step_mean = lag["per_step_mean"]
                        totals = lag["episode_totals"]
                        thresholds = lag["thresholds"]
                        wandb.log({
                            "timestep": info["train_timestep"][-1][0] * cfg.num_envs,
                            "eval_rewards": info["eval_rewards"],
                            "lambda/exceeded_capacity": lambdas[0],
                            "lambda/uncharged_kw": lambdas[1],
                            "lambda/rejected_customers": lambdas[2],
                            "lambda/total_discharged_kw": lambdas[3],
                            "constraint_step_mean/exceeded_capacity": step_mean[0],
                            "constraint_step_mean/uncharged_kw": step_mean[1],
                            "constraint_step_mean/rejected_customers": step_mean[2],
                            "constraint_step_mean/total_discharged_kw": step_mean[3],
                            "constraint_episode_mean/exceeded_capacity": totals[0],
                            "constraint_episode_mean/uncharged_kw": totals[1],
                            "constraint_episode_mean/rejected_customers": totals[2],
                            "constraint_episode_mean/total_discharged_kw": totals[3],
                            "thresholds/exceeded_capacity": thresholds[0],
                            "thresholds/uncharged_kw": thresholds[1],
                            "thresholds/rejected_customers": thresholds[2],
                            "thresholds/total_discharged_kw": thresholds[3],
                            **info["logging_data"],
                            **logging_baselines
                        })
            jax.debug.callback(callback, metric)

            runner_state = (new_train_state, env_state, last_obs, rng)
            return runner_state, _


        # initial runner state
        rng, key = jax.random.split(rng_outer)
        runner_state = (train_state, env_state, obsv, key)
        trained_runner_state, train_metrics = jax.lax.scan(train_step, runner_state, None, cfg.num_iterations)
        return trained_runner_state, train_metrics

    return train_func, cfg

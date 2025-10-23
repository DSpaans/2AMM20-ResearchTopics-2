import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import optax
from typing import NamedTuple, List, Sequence, Union, Mapping, Literal
from dataclasses import replace, field
from typing import Sequence, Dict, List, NamedTuple

from chargax import Chargax, LogWrapper, NormalizeVecObservation
from chargax.algorithms.networks import ActorNetworkMultiDiscrete, CriticNetwork
import wandb

def create_ppo_networks(
    key,
    in_shape: int,
    actor_features: List[int],
    critic_features: List[int],
    actions_nvec: int,
    num_costs: int,
):
    """Create PPO networks (actor, reward critic V(s), and cost criticV^C(s))"""
    actor_key, critic_key, cost_root_key = jax.random.split(key, 3)
    actor = ActorNetworkMultiDiscrete(actor_key, in_shape, actor_features, actions_nvec)
    critic = CriticNetwork(critic_key, in_shape, critic_features) # V(s) for reward
    # K independent cost critics
    cost_keys = jax.random.split(cost_root_key, num_costs)
    cost_critics = tuple(CriticNetwork(k, in_shape, critic_features) for k in cost_keys) # V^C(s) for cost
    return actor, critic, cost_critics

@chex.dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    ent_coef: float = 0.01
    vf_coef: float = 0.25 # Reward value lost rate

    total_timesteps: int = 5e6
    num_envs: int = 12
    num_steps: int = 300 # steps per environment
    num_minibatches: int = 4 # Number of mini-batches
    update_epochs: int = 4 # K epochs to update the policy

    seed: int = 42
    debug: bool = False
    evaluate_deterministically: bool = False
    
    # ADDED Extra parameters for lagrangian PPO
    cost_keys: Sequence[str] = (
        "charged_satisfaction",
        "time_satisfaction",
        "rejected_customers",
        "capacity_exceeded",
        "battery_degradation",
    )
    # 22/10 change: cost_limits can be array or mapping, units per step or episode
    #cost_limits: jnp.ndarray = field(default_factory=lambda: jnp.array((0.0, 0.05, 0.0, 0.0, 0.0), dtype=jnp.float32))
    cost_limits: Union[jnp.ndarray, Mapping[str, float]] = field(
        default_factory=lambda: jnp.array((0.0, 0.05, 0.0, 0.0, 0.0), dtype=jnp.float32)
    )
    cost_limit_units: Literal["per_step", "per_episode"] = "per_step"
    
    alpha_init: float = 0.0 #Might have to change this
    alpha_lr: float = 1e-3 #Conservative
    alpha_max: float = 1e6

    # Value function weight
    cost_vf_coef: float = 0.25 #Cost value lost weight

    # Cost GAE
    cost_gamma: float = 0.99
    cost_gae_lambda: float = 0.95
    

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs
    
    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches
    

# Define a simple tuple to hold the state of the environment. 
# This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array
    # ADDED extras for Lagrangian
    costs: chex.Array # shape: (num_envs, num_costs)
    profit_delta: chex.Array # per-step profit change

class TrainState(NamedTuple):
    actor: eqx.Module
    critic: eqx.Module
    cost_critics: tuple # Tuple[eqx.Module, ...] Added
    optimizer_state: optax.OptState
    alphas: chex.Array # TODO shape: (num_costs,)

def _extract_cost_deltas(curr_logging: Dict[str, chex.Array],
                         prev_logging: Dict[str, chex.Array],
                         cost_keys: Sequence[str],
                         beta: float
                         ) -> chex.Array:
    """
    Build per-env cost increments for each requested cost key, in order.
    Supports:
      - "capacity_exceeded"        -> logging "exceeded_capacity"
      - "charged_satisfaction"     -> logging "uncharged_kw"            (penalize unmet energy)
      - "time_satisfaction"        -> (charged_overtime - beta * charged_undertime)
      - "rejected_customers"       -> logging "rejected_customers"
      - "battery_degradation"      -> logging "total_discharged_kw"
    Also accepts direct logging names ("uncharged_kw","exceeded_capacity",...) as cost_keys.
    Output shape: (num_envs, num_costs)
    """
    direct_map = {
        "capacity_exceeded": "exceeded_capacity",
        "charged_satisfaction": "uncharged_kw",
        "rejected_customers": "rejected_customers",
        "battery_degradation": "total_discharged_kw",
    }
    deltas = []
    for k in cost_keys:
        if k == "time_satisfaction":
            co_curr = curr_logging["charged_overtime"]
            co_prev = prev_logging["charged_overtime"]
            cu_curr = curr_logging["charged_undertime"]
            cu_prev = prev_logging["charged_undertime"]
            delta = (co_curr - co_prev) - beta * (cu_curr - cu_prev)
        else:
            # allow either friendly name or exact logging key
            key = direct_map.get(k, k)
            delta = (curr_logging[key] - prev_logging[key]).astype(jnp.float32)
        deltas.append(delta.astype(jnp.float32))
    return jnp.stack(deltas, axis=-1)


# Jit the returned function, not this function
def build_ppo_lagrangian_trainer(
        env: Chargax,
        config_params: dict = {},
        baselines: dict = {}, # Will be inserted every wandb log step
    ):

    # setup env (wrappers) and config
    env = LogWrapper(env)
    env = NormalizeVecObservation(env)
    # env = FlattenObservationWrapper(env)
    observation_space = env.observation_space
    action_space = env.action_space
    num_actions = action_space.n
    logging_baselines = baselines

    config = PPOConfig(**config_params)

    # Choose one primary constraint for the policy loss
    #PRIMARY = tuple(config.cost_keys).index("battery_degradation")   To pick your primary - OLD
    
    # ADDED cost info
    beta = getattr(env, "beta", 0.0)
    # 22/10 change:
    # num_costs = len(tuple(config.cost_keys))
    # cost_limits = jnp.asarray(config.cost_limits, dtype=jnp.float32).reshape((num_costs,))
    
    num_costs = len(tuple(config.cost_keys))

    # Turn cost_limits into a vector aligned with cost_keys
    if isinstance(config.cost_limits, dict):
        big = 1e9  # effectively unconstrained if not provided
        limits_vec = jnp.array(
            [config.cost_limits.get(k, big) for k in config.cost_keys],
            dtype=jnp.float32
        )
    else:
        limits_vec = jnp.asarray(config.cost_limits, dtype=jnp.float32).reshape((num_costs,))

    # If user gave episode limits, convert to per-step averages for optimization
    if config.cost_limit_units == "per_episode":
        limits_vec = limits_vec / env.episode_length  # same as dividing sum by T

    cost_limits_step = limits_vec
    cost_limits_episode = limits_vec * env.episode_length  # for logging


    # rng keys
    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)

    # networks
    actor, critic, cost_critics = create_ppo_networks(
        key=network_key, 
        in_shape=observation_space.shape[0],
        actor_features=[256, 256], 
        critic_features=[256, 256], 
        actions_nvec=action_space.nvec,
        num_costs=num_costs,
    )

    # optimizer
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_iterations
        )
        return config.learning_rate * frac
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_lr else config.learning_rate,
            eps=1e-5
        ),
    )
    optimizer_state = optimizer.init({
        "actor": actor,
        "critic": critic,
        "cost_critics": cost_critics,
    })

    train_state = TrainState(
        actor=actor,
        critic=critic,
        cost_critics=cost_critics,
        optimizer_state=optimizer_state,
        alphas=jnp.ones((num_costs,), dtype=jnp.float32) * jnp.float32(config.alpha_init),
    )

    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_key)

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action_dist = train_state.actor(obs)
            if config.evaluate_deterministically:
                action = jnp.argmax(action_dist.logits, axis=-1)
            else:
                action = action_dist.sample(seed=action_key)
            (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)
            
            
            
            episode_reward += reward
            return (rng, obs, env_state, done, episode_reward)
        
        def cond_func(carry):
            _, _, _, done, _ = carry
            return jnp.logical_not(done)
        
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key)
        done = False
        episode_reward = 0.0

        rng, obs, env_state, done, episode_reward = jax.lax.while_loop(cond_func, step_env, (rng, obs, env_state, done, episode_reward))

        return episode_reward

    def train_func(rng=rng):
        
        # functions prepended with _ are called in jax.lax.scan of train_step

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng, prev_logging = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            action_dist = jax.vmap(train_state.actor)(last_obs)
            value = jax.vmap(train_state.critic)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, config.num_envs)
            (obsv, reward, terminated, truncated, info), env_state= jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)
            
            # ADDED Lagrangian reward calculation
            raw_logging = info["logging_data"]
            curr_logging = {
                k: jnp.asarray(raw_logging.get(k, prev_logging[k]))
                    .astype(prev_logging[k].dtype)
                    .reshape(prev_logging[k].shape)
                for k in prev_logging.keys()
            }
            costs = _extract_cost_deltas(curr_logging, prev_logging, config.cost_keys, beta)  # (num_envs, num_costs)
            profit_delta = (curr_logging["profit"] - prev_logging["profit"]).astype(jnp.float32)

            # jax.debug.breakpoint()

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info,
                costs=costs,
                profit_delta=profit_delta,
            )

            runner_state = (train_state, env_state, obsv, rng, curr_logging)
            return runner_state, transition
        
        def _calculate_gae(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            value, reward, done = (
                transition.value,
                transition.reward,
                transition.done,
            )
            delta = reward + config.gamma * next_value * (1 - done) - value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, value), (gae, gae + value)
        
        # OLD: Old Helper that referenced Primary constrain
        """ def _calculate_cost_gae(cost_gae_and_next_value, carry_tuple):
            # GAE for costs using the cost critic. carry_tuple = (transition, next_cost_value)
            gae, next_value_c = cost_gae_and_next_value
            transition, = carry_tuple
            done = transition.done
            # current cost value V^C(s_t)
            value_c_t = jax.vmap(train_state.cost_critic)(transition.observation)
            # use per-step (scalar) cost for PRIMARY constraint
            step_cost = transition.costs[..., PRIMARY]  # [num_envs]
            delta = step_cost + config.cost_gamma * next_value_c * (1 - done) - value_c_t
            gae = delta + config.cost_gamma * config.cost_gae_lambda * (1 - done) * gae
            return (gae, value_c_t), (gae, gae + value_c_t) """
        
        #NEW: Helper for GAE (Generic)
        def _gae_T(values, rewards, dones, gamma, lam):
            # values: [T+1,B], rewards: [T,B], dones: [T,B]
            T = rewards.shape[0]
            gae_acc = jnp.zeros_like(rewards[0])
            def _scan(carry, t):
                gae = carry
                delta = rewards[t] + gamma * values[t+1] * (1.0 - dones[t]) - values[t]
                gae = delta + gamma * lam * (1.0 - dones[t]) * gae
                return gae, gae
            adv = jax.lax.scan(_scan, gae_acc, jnp.arange(T-1, -1, -1), reverse=True)[1]
            ret = adv + values[:-1]
            return adv, ret
        
        def _update_epoch(update_state, _):
            """ Do one epoch of update"""

            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_los_fn(params, alpha_vec, trajectory_minibatch,
                             advantages, returns, cost_advantages, cost_returns):
                # actor
                action_dist = jax.vmap(params["actor"])(trajectory_minibatch.observation)
                log_prob = action_dist.log_prob(trajectory_minibatch.action).sum(axis=-1)
                entropy = action_dist.entropy().mean()

                # critics
                value = jax.vmap(params["critic"])(trajectory_minibatch.observation)
                # OLD: cost_value = jax.vmap(params["cost_critic"])(trajectory_minibatch.observation)

                # cost values for all K -> [B, K]
                cost_value = jnp.stack(
                    [jax.vmap(ck)(trajectory_minibatch.observation) for ck in params["cost_critics"]],
                    axis=-1
                )

                # ratios
                ratio = jnp.exp(log_prob - trajectory_minibatch.log_prob.sum(axis=-1))

                # normalize both advantages
                adv_r = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # normalize every cost advantage per k (over the minibatch)
                mean_c = cost_advantages.mean(axis=0)
                std_c  = cost_advantages.std(axis=0) + 1e-8
                adv_c = (cost_advantages - mean_c) / std_c  # shape [N, K]

                # combined advantage: A - sum_k alpha_k * A_c_k
                penalty = jnp.sum(jax.lax.stop_gradient(alpha_vec) * adv_c, axis=-1)  # [N]
                comb = adv_r - penalty

                # Actor loss
                actor_loss = -jnp.minimum(
                    comb * ratio,
                    jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * comb
                ).mean()

                # reward value loss (clipped)
                value_pred_clipped = trajectory_minibatch.value + jnp.clip(
                    value - trajectory_minibatch.value, -config.clip_coef_vf, config.clip_coef_vf
                )
                value_loss = jnp.maximum(
                    jnp.square(value - returns),
                    jnp.square(value_pred_clipped - returns)
                ).mean()

                # cost value loss (no clipping is fine)
                cost_value_loss = jnp.square(cost_value - cost_returns).mean()

                total_loss = (
                    actor_loss
                    + config.vf_coef * value_loss
                    + config.cost_vf_coef * cost_value_loss
                    - config.ent_coef * entropy
                )
                aux = (actor_loss, value_loss, cost_value_loss, entropy)
                return total_loss, aux

            def __update_over_minibatch(train_state: TrainState, minibatch):
                # Unpack minibatch leaves (now FIVE tensors)
                trajectory_mb, advantages_mb, returns_mb, cost_adv_mb, cost_ret_mb = minibatch

                # OLD: Use the PRIMARY constraint’s alpha for the combined advantage
                #alpha_primary = train_state.alphas[PRIMARY]

                # Compute loss + grads
                (total_loss, aux), grads = __ppo_los_fn(
                    {
                        "actor": train_state.actor,
                        "critic": train_state.critic,
                        "cost_critics": train_state.cost_critics,
                    },
                    train_state.alphas,
                    trajectory_mb,
                    advantages_mb, returns_mb,
                    cost_adv_mb, cost_ret_mb,
                )

                # Optimizer step
                updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state)
                new_params = optax.apply_updates(
                    {
                        "actor": train_state.actor,
                        "critic": train_state.critic,
                        "cost_critics": train_state.cost_critics,
                    },
                    updates
                )

                # Write back
                train_state = TrainState(
                    actor=new_params["actor"],
                    critic=new_params["critic"],
                    cost_critics=new_params["cost_critics"],
                    optimizer_state=optimizer_state,
                    alphas=train_state.alphas,
                )

                return train_state, total_loss


            train_state, trajectory_batch, advantages, returns, cost_advantages, cost_returns, rng, prev_logging = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.batch_size)
            batch = (trajectory_batch, advantages, returns, cost_advantages, cost_returns)
            
            # reshape (flatten over first dimension)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
            )
            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            # split in minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
            )
            # update over minibatches
            train_state, total_loss = jax.lax.scan(
                __update_over_minibatch, train_state, minibatches
            )
            
            # Lagrangian
            # trajectory_batch.costs must be collected during rollout: shape (num_steps, num_envs, num_costs)
            # costs_batch = trajectory_batch.costs
            # mean_costs = costs_batch.mean(axis=(0, 1))       # (num_costs,)
            # violations = mean_costs - cost_limits            # (num_costs,)

            # new_alphas = jnp.clip(
            #     train_state.alphas + config.alpha_lr * violations,
            #     0.0,
            #     config.alpha_max,
            # )

            # # write alphas back into TrainState
            # train_state = TrainState(
            #     actor=train_state.actor,
            #     critic=train_state.critic,
            #     cost_critics=train_state.cost_critics,
            #     optimizer_state=train_state.optimizer_state,
            #     alphas=new_alphas,
            # )
            
            update_state = (train_state, trajectory_batch, advantages, returns, cost_advantages, cost_returns, rng, prev_logging)
            return update_state, total_loss

        def train_step(runner_state, _):

            # Do rollout of single trajactory
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # calculate gae
            train_state, env_state, last_obs, rng, prev_logging = runner_state
            last_value = jax.vmap(train_state.critic)(last_obs)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
            
            # OLD: For only Primary constraint
            """ # last cost value V^C(s_T)
            last_cost_value = jax.vmap(train_state.cost_critic)(last_obs)
            
            # Running GAE over costs (treat per-step PRIMARY cost as "reward" in this stream)
            _, (cost_advantages, cost_returns) = jax.lax.scan(
                _calculate_cost_gae,
                (jnp.zeros_like(last_cost_value), last_cost_value),
                (trajectory_batch,),
                reverse=True,
                unroll=16
            ) """

            # NEW: Multi-K Constraints
            # costs: [T, B, K]
            costs = trajectory_batch.costs
            T, B, K = costs.shape

            # V^C_k(s) for all k -> values_c: [T+1, B, K]
            def _values_c_for_obs(obs):  # obs: [B, ...] -> [B,K]
                vals = [jax.vmap(ck)(obs) for ck in train_state.cost_critics]
                return jnp.stack(vals, axis=-1)

            values_c_t = jax.vmap(_values_c_for_obs)(trajectory_batch.observation)  # [T, B, K]
            last_values_c = _values_c_for_obs(last_obs)                         # [B,K]
            values_c = jnp.concatenate([values_c_t, last_values_c[None, ...]], axis=0)  # [T+1,B,K]

            # GAE per constraint k (vmap over last axis)
            cost_advantages, cost_returns = jax.vmap(
                lambda vCk, Ck: _gae_T(vCk, Ck, trajectory_batch.done, config.cost_gamma, config.cost_gae_lambda),
                in_axes=(2, 2), out_axes=(2, 2)
            )(values_c, costs)  # both [T,B,K]
    
            # Do update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, cost_advantages, cost_returns, rng, prev_logging)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            (train_state,
            trajectory_batch,
            advantages,
            returns,
            cost_advantages,
            cost_returns,
            rng,
            prev_logging) = update_state

            metric = trajectory_batch.info
            metric["loss_info"] = loss_info


            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(train_state, eval_key)
            metric["eval_rewards"] = eval_rewards
            
            costs_batch = trajectory_batch.costs
            mean_costs_step = costs_batch.mean(axis=(0, 1))
            
            finished = metric["returned_episode"]
            log = metric["logging_data"]
            episode_len = env.episode_length
            
            def masked_mean(x):
                num = finished.sum()
                # avoid NaN
                return jnp.where(
                    num > 0,
                    (jnp.where(finished, x, 0.0).sum()) / num,
                    jnp.nan,
                )
            
            # Map cost_keys -> episodic means using final logging values on finished steps
            episodic_means_list = []
            for name in config.cost_keys:
                if name == "time_satisfaction":
                    over = masked_mean(log["charged_overtime"])
                    under = masked_mean(log["charged_undertime"])
                    episodic_means_list.append(over - beta * under)
                elif name == "charged_satisfaction":
                    episodic_means_list.append(masked_mean(log["uncharged_kw"]))
                elif name == "capacity_exceeded":
                    episodic_means_list.append(masked_mean(log["exceeded_capacity"]))
                elif name == "rejected_customers":
                    episodic_means_list.append(masked_mean(log["rejected_customers"]))
                elif name == "battery_degradation":
                    episodic_means_list.append(masked_mean(log["total_discharged_kw"]))
                else:
                    # Allow direct logging metric names as cost_keys
                    episodic_means_list.append(masked_mean(log[name]))
            mean_costs_episode = jnp.stack(episodic_means_list, axis=0)
            
            num_finished = finished.sum()
            def update_alpha(alphas, mean_ep, limits_ep):
                violations_ep = mean_ep - limits_ep
                new_alphas = jnp.clip(alphas + config.alpha_lr * violations_ep,
                                    0.0, config.alpha_max)
                return new_alphas

            new_alphas = jax.lax.cond(
                num_finished > 0,
                lambda _: update_alpha(train_state.alphas, mean_costs_episode, cost_limits_episode),
                lambda _: train_state.alphas, # no change if no complete episodes
                operand=None,
            )

            train_state = TrainState(
                actor=train_state.actor,
                critic=train_state.critic,
                cost_critics=train_state.cost_critics,
                optimizer_state=train_state.optimizer_state,
                alphas=new_alphas,
            )

            def callback(info, alphas, mean_costs_step_arg, mean_costs_ep_arg,
                        cost_limits_step_arg, cost_limits_ep_arg):
                if config.debug:
                    print(f'timestep={(info["train_timestep"][-1][0] * config.num_envs)}, eval rewards={info["eval_rewards"]}')
                if wandb.run:
                    if "logging_data" not in info:
                        info["logging_data"] = {}
                    finished_episodes = info["returned_episode"]
                    if finished_episodes.any():
                        info["logging_data"] = jax.tree.map(
                            lambda x: x[finished_episodes].mean(), info["logging_data"]
                        )
                        log_dict = {
                            "timestep": info["train_timestep"][-1][0] * config.num_envs,
                            "eval_rewards": info["eval_rewards"],
                            **info["logging_data"],
                            **logging_baselines,
                        }
                        for i, name in enumerate(config.cost_keys):
                            # alphas
                            log_dict[f"alpha/{name}"] = alphas[i]
                            # per-step view
                            log_dict[f"cost/{name}_mean_per_step"] = mean_costs_step_arg[i]
                            log_dict[f"limit/{name}_per_step"] = cost_limits_step_arg[i]
                            # per-episode view (true episodic mean)
                            log_dict[f"cost/{name}_mean_per_episode"] = mean_costs_ep_arg[i]
                            log_dict[f"limit/{name}_per_episode"] = cost_limits_ep_arg[i]
                        wandb.log(log_dict)

            jax.debug.callback(
                callback,
                metric,
                train_state.alphas,
                mean_costs_step,
                mean_costs_episode,
                cost_limits_step,
                cost_limits_episode,
            )

            runner_state = (train_state, env_state, last_obs, rng, prev_logging)
            return runner_state, _ 

        rng, key = jax.random.split(rng)
        
        zeros_logging = {
            "profit": jnp.zeros((config.num_envs,)),
            "exceeded_capacity": jnp.zeros((config.num_envs,)),
            "uncharged_kw": jnp.zeros((config.num_envs,)),
            "charged_overtime": jnp.zeros((config.num_envs,)),
            "charged_undertime": jnp.zeros((config.num_envs,)),
            "rejected_customers": jnp.zeros((config.num_envs,)),
            "total_discharged_kw": jnp.zeros((config.num_envs,)),
        }
        
        runner_state = (train_state, env_state, obsv, key, zeros_logging)
        trained_runner_state, train_metrics = jax.lax.scan(
            train_step, runner_state, None, config.num_iterations
        )

        return trained_runner_state, train_metrics

    return train_func, config
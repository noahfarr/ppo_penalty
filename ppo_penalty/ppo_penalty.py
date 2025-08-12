from typing import Any

import chex
import flax.linen as nn
import gymnax
from gymnax.wrappers import LogWrapper, FlattenObservationWrapper
import jax
import jax.numpy as jnp
import distrax
import optax
from flax import core


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        logits = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        return distrax.Categorical(logits=logits)


class Critic(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        v = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(v, axis=-1)


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    info: chex.Array
    log_prob: chex.Array
    value: chex.Array
    logits: chex.Array


@chex.dataclass
class PPOConfig:
    num_envs: int = 4
    num_eval_envs: int = 4
    num_steps: int = 128
    total_timesteps: int = int(5e5)
    num_train_steps: int = 100_000
    normalize_advantage: bool = True
    clip_vloss: bool = True
    num_evaluation_steps: int = 10_000
    learning_rate: float = 2.5e-4
    update_epochs: int = 4
    num_minibatches: int = 4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    target_kl: float = 0.01
    kl_beta_init: float = 1.0
    kl_adapt_scale: float = 1.5
    kl_beta_up: float = 2.0
    kl_beta_down: float = 0.5

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps


@chex.dataclass(frozen=True)
class PPOState:
    step: int
    obs: chex.Array
    env_state: chex.Array
    actor_params: core.FrozenDict[str, Any]
    actor_optimizer_state: optax.OptState
    critic_params: core.FrozenDict[str, Any]
    critic_optimizer_state: optax.OptState
    kl_beta: float


@chex.dataclass(frozen=True)
class PPO:
    cfg: PPOConfig
    env: gymnax.environments.environment.Environment
    env_params: gymnax.EnvParams
    actor: Actor
    critic: Critic
    optimizer: optax.GradientTransformation

    def init(self, key):
        key, env_key, actor_key, critic_key = jax.random.split(key, 4)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )

        actor_params = self.actor.init(actor_key, obs)
        actor_optimizer_state = self.optimizer.init(actor_params)

        critic_params = self.critic.init(critic_key, obs)
        critic_optimizer_state = self.optimizer.init(critic_params)

        return (
            key,
            PPOState(
                step=0,  # type: ignore
                obs=obs,  # type: ignore
                env_state=env_state,  # type: ignore
                actor_params=actor_params,  # type: ignore
                actor_optimizer_state=actor_optimizer_state,  # type: ignore
                critic_params=critic_params,  # type: ignore
                critic_optimizer_state=critic_optimizer_state,  # type: ignore
                kl_beta=self.cfg.kl_beta_init,  # type: ignore
            ),
        )

    def train(self, key, state, num_steps):

        def update_step(carry: tuple, _):

            def compute_gae(
                gamma: float, gae_lambda: float, final_value: jax.Array, transitions
            ):
                """Compute Generalized Advantage Estimates (GAE) for a trajectory."""

                def f(carry, transition):
                    advantage, value = carry
                    delta = (
                        transition.reward
                        + gamma * value * (1 - transition.done)
                        - transition.value
                    )
                    advantage = (
                        delta + gamma * gae_lambda * (1 - transition.done) * advantage
                    )
                    return (advantage, transition.value), advantage

                _, advantages = jax.lax.scan(
                    f,
                    (jnp.zeros_like(final_value), final_value),
                    transitions,
                    reverse=True,
                )
                returns = advantages + transitions.value
                return advantages, returns

            def step(carry: tuple, _):
                key, state = carry

                key, action_key, step_key = jax.random.split(key, 3)

                probs = self.actor.apply(state.actor_params, state.obs)
                action = probs.sample(seed=action_key)
                log_prob = probs.log_prob(action)

                value = self.critic.apply(state.critic_params, state.obs)

                step_key = jax.random.split(step_key, self.cfg.num_envs)
                next_obs, env_state, reward, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None)
                )(step_key, state.env_state, action, self.env_params)

                transition = Transition(
                    observation=state.obs,  # type: ignore
                    action=action,  # type: ignore
                    reward=reward,  # type: ignore
                    done=done,  # type: ignore
                    info=info,  # type: ignore
                    log_prob=log_prob,  # type: ignore
                    value=value,  # type: ignore
                    logits=probs.logits,  # type: ignore
                )

                state = state.replace(
                    step=state.step + self.cfg.num_envs,
                    obs=next_obs,
                    env_state=env_state,
                )
                carry = (
                    key,
                    state,
                )
                return carry, transition

            (key, state), transitions = jax.lax.scan(
                step,
                carry,
                length=self.cfg.num_steps,
            )
            final_value = self.critic.apply(state.critic_params, state.obs)

            advantages, returns = compute_gae(
                self.cfg.gamma,
                self.cfg.gae_lambda,
                final_value,
                transitions,
            )

            if self.cfg.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            batch = (transitions, advantages, returns)

            def update_epoch(carry: tuple, _):
                key, state, batch = carry

                def update_minibatch(state, minibatch: tuple):
                    transitions, advantages, returns = minibatch

                    def actor_loss_fn(params, transitions, advantages, beta):
                        probs = self.actor.apply(params, transitions.observation)
                        log_prob = probs.log_prob(transitions.action)
                        entropy = probs.entropy().mean()

                        ratio = jnp.exp(log_prob - transitions.log_prob)

                        old_probs = distrax.Categorical(logits=transitions.logits)
                        kl = old_probs.kl_divergence(probs).mean()

                        actor_loss = (
                            -(ratio * advantages).mean()
                            + beta * kl
                            - self.cfg.ent_coef * entropy
                        )
                        return actor_loss, {"kl": kl, "entropy": entropy}

                    def critic_loss_fn(params, transitions, advantages, returns):
                        value = self.critic.apply(params, transitions.observation)

                        if self.cfg.clip_vloss:
                            critic_loss = jnp.square(value - returns)
                            clipped_value = transitions.value + jnp.clip(
                                (value - transitions.value),
                                -self.cfg.clip_coef,
                                self.cfg.clip_coef,
                            )
                            clipped_critic_loss = jnp.square(clipped_value - returns)
                            critic_loss = (
                                0.5
                                * jnp.maximum(critic_loss, clipped_critic_loss).mean()
                            )
                        else:
                            critic_loss = 0.5 * jnp.square(value - returns).mean()

                        return self.cfg.vf_coef * critic_loss

                    def update_beta(kl):
                        high = self.cfg.target_kl * self.cfg.kl_adapt_scale
                        low = self.cfg.target_kl / self.cfg.kl_adapt_scale
                        new_beta = jnp.where(
                            kl > high,
                            state.kl_beta * self.cfg.kl_beta_up,
                            jnp.where(
                                kl < low,
                                state.kl_beta * self.cfg.kl_beta_down,
                                state.kl_beta,
                            ),
                        )
                        return new_beta

                    (actor_loss, aux), actor_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=True
                    )(state.actor_params, transitions, advantages, state.kl_beta)
                    actor_updates, actor_optimizer_state = self.optimizer.update(
                        actor_grads, state.actor_optimizer_state, state.actor_params
                    )
                    actor_params = optax.apply_updates(
                        state.actor_params, actor_updates
                    )

                    critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                        state.critic_params, transitions, advantages, returns
                    )
                    critic_updates, critic_optimizer_state = self.optimizer.update(
                        critic_grads, state.critic_optimizer_state, state.critic_params
                    )
                    critic_params = optax.apply_updates(
                        state.critic_params, critic_updates
                    )

                    kl_beta = update_beta(
                        aux["kl"],
                    )

                    state = state.replace(
                        actor_params=actor_params,
                        actor_optimizer_state=actor_optimizer_state,
                        critic_params=critic_params,
                        critic_optimizer_state=critic_optimizer_state,
                        kl_beta=kl_beta,
                    )
                    return state, (actor_loss, critic_loss)

                key, permutation_key = jax.random.split(key)

                permutation = jax.random.permutation(
                    permutation_key, self.cfg.batch_size
                )
                flattened_batch = jax.tree.map(
                    lambda x: x.reshape(-1, *x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), flattened_batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [self.cfg.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                state, loss = jax.lax.scan(
                    update_minibatch,
                    state,
                    minibatches,
                )

                return (
                    key,
                    state,
                    batch,
                ), loss

            (key, state, batch), (actor_loss, critic_loss) = jax.lax.scan(
                update_epoch,
                (key, state, batch),
                length=self.cfg.update_epochs,
            )
            transitions, *_ = batch

            return (key, state), transitions.info

        (key, state), info = jax.lax.scan(
            update_step,
            (key, state),
            length=num_steps // self.cfg.num_envs // self.cfg.num_steps,
        )

        return key, state, info

    def evaluate(self, key, state, num_steps):
        key, reset_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_eval_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )

        def step(carry, _):
            key, obs, env_state = carry

            probs = self.actor.apply(state.actor_params, obs)
            action = jnp.argmax(probs.logits, axis=-1)

            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.cfg.num_eval_envs)
            obs, env_state, _, _, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, self.env_params)

            return (key, obs, env_state), info

        (key, obs, env_state), info = jax.lax.scan(
            step, (key, obs, env_state), length=num_steps
        )

        return key, info


if __name__ == "__main__":
    seed = 0
    cfg = PPOConfig()
    env, env_params = gymnax.make("CartPole-v1")
    env = LogWrapper(env)
    env = FlattenObservationWrapper(env)

    actor = Actor(
        action_dim=env.action_space(env_params).n,
    )

    critic = Critic()

    if cfg.anneal_lr:
        learning_rate = optax.linear_schedule(
            init_value=cfg.learning_rate,
            end_value=0.0,
            transition_steps=(cfg.total_timesteps // cfg.num_envs // cfg.num_steps)
            * cfg.update_epochs
            * cfg.num_minibatches,
        )
    else:
        learning_rate = cfg.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )

    agent = PPO(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
    )

    key = jax.random.key(seed)

    key, state = agent.init(key)

    key, info = agent.evaluate(key, state, cfg.num_evaluation_steps)

    episodic_returns = info["returned_episode_returns"][info["returned_episode"]].mean()
    episodic_lengths = info["returned_episode_lengths"][info["returned_episode"]].mean()

    print(
        f"Step: {state.step} | Episodic return: {episodic_returns} | Episodic length: {episodic_lengths}"
    )

    def cond_fn(carry):
        _, state = carry
        return state.step < cfg.total_timesteps

    def body_fn(carry):
        key, state = carry
        key, state, info = agent.train(key, state, cfg.num_train_steps)
        key, info = agent.evaluate(key, state, cfg.num_evaluation_steps)

        def callback(info, step):
            episodic_returns = info["returned_episode_returns"][
                info["returned_episode"]
            ].mean()
            episodic_lengths = info["returned_episode_lengths"][
                info["returned_episode"]
            ].mean()
            print(
                f"Step: {step} | Episodic return: {episodic_returns} | Episodic length: {episodic_lengths}"
            )

        jax.debug.callback(callback, info, state.step)

        return (key, state)

    jax.lax.while_loop(cond_fn, body_fn, (key, state))

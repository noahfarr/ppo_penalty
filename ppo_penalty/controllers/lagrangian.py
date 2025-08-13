import chex
import jax
import jax.numpy as jnp
import optax

from ppo_penalty.controllers.controller import (
    KLControllerConfig,
    KLControllerState,
    KLController,
)

@chex.dataclass(frozen=True)
class LagrangianKLControllerConfig(KLControllerConfig):
    target: float = 0.01
    initial_log_beta: float = 0.0
    lr: float = 1e-3

@chex.dataclass(frozen=True)
class LagrangianKLControllerState(KLControllerState):
    log_beta: jnp.ndarray
    optimizer_state: optax.OptState

    @property
    def beta(self):
        return jnp.exp(self.log_beta)

@chex.dataclass(frozen=True)
class LagrangianKLController(KLController):

    optimizer: optax.GradientTransformation = optax.adam(1e-3)

    def init(self) -> KLControllerState:
        log_beta = jnp.asarray(self.config.initial_log_beta)
        optimizer_state = self.optimizer.init(log_beta)
        return LagrangianKLControllerState(
            log_beta=log_beta,
            optimizer_state=optimizer_state,
        )

    def update(self, state: KLControllerState, kl: jnp.ndarray) -> KLControllerState:
        def loss_fn(log_beta: jnp.ndarray):
            beta = jnp.exp(log_beta)
            loss = beta * (self.config.target - kl)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.log_beta)

        updates, optimizer_state = self.optimizer.update(grads, state.optimizer_state, params=state.log_beta)
        log_beta = optax.apply_updates(state.log_beta, updates)

        return state.replace(
            log_beta=log_beta,
            optimizer_state=optimizer_state,
        )


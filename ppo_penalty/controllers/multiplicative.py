import chex
import jax.numpy as jnp

from ppo_penalty.controllers.controller import (
    KLControllerConfig,
    KLControllerState,
    KLController,
)


@chex.dataclass(frozen=True)
class MultiplicativeKLControllerConfig(KLControllerConfig):
    """Configuration for the multiplicative KL controller."""

    target: float = 0.01
    initial_beta: float = 1.0
    adapt_scale: float = 1.5
    beta_up: float = 2.0
    beta_down: float = 0.5


@chex.dataclass(frozen=True)
class MultiplicativeKLControllerState(KLControllerState):
    """State for the multiplicative KL controller."""

    beta: float


@chex.dataclass(frozen=True)
class MultiplicativeKLController(KLController):
    """The multiplicative KL controller."""

    def init(self) -> KLControllerState:
        state = MultiplicativeKLControllerState(
            beta=self.config.initial_beta,
        )
        return state

    def update(self, state: KLControllerState, kl: jnp.ndarray) -> KLControllerState:
        """Update the controller."""
        high = self.config.target * self.config.adapt_scale
        low = self.config.target / self.config.adapt_scale

        beta = jnp.where(
            kl > high,
            state.beta * self.config.beta_up,
            jnp.where(kl < low, state.beta * self.config.beta_down, state.beta),
        )

        return state.replace(beta=beta)  # type: ignore

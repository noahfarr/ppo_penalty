from abc import ABC, abstractmethod
import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class KLControllerConfig: ...


@chex.dataclass(frozen=True)
class KLControllerState: ...


@chex.dataclass(frozen=True)
class KLController(ABC):

    config: KLControllerConfig

    @abstractmethod
    def init(self, cfg: KLControllerConfig) -> KLControllerState: ...

    @abstractmethod
    def update(
        self, state: KLControllerState, kl: jnp.ndarray
    ) -> KLControllerState: ...

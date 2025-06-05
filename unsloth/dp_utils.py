import math
from typing import Iterable, List, Optional

import numpy as np
import torch


class RDPAccountant:
    """Simple Rényi Differential Privacy accountant."""

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        orders: Optional[Iterable[float]] = None,
        target_delta: float = 1e-6,
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.target_delta = target_delta
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)]
        self.orders = np.array(list(orders), dtype=float)
        self.rdp = np.zeros_like(self.orders)
        self.steps = 0

    def _compute_rdp(self) -> np.ndarray:
        if self.noise_multiplier == 0:
            return np.inf * np.ones_like(self.orders)
        return (
            (self.sample_rate ** 2)
            * self.orders
            / (2.0 * self.noise_multiplier ** 2)
        )

    def step(self) -> None:
        self.steps += 1
        self.rdp += self._compute_rdp()

    def get_epsilon(self) -> float:
        if self.steps == 0:
            return 0.0
        eps = self.rdp - math.log(self.target_delta) / (self.orders - 1)
        return float(np.min(eps))


class DPOptimizer(torch.optim.Optimizer):
    """Wrap an optimizer to add DP gradient clipping and noise."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        noise_multiplier: float,
        sample_rate: float,
        accountant: Optional[RDPAccountant] = None,
    ) -> None:
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = self.optimizer.state

        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.accountant = accountant

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        parameters: List[torch.Tensor] = [
            p for group in self.param_groups for p in group["params"] if p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        std = self.noise_multiplier * self.max_grad_norm
        for p in parameters:
            noise = torch.randn_like(p.grad) * std
            p.grad.add_(noise)
        result = self.optimizer.step(closure)
        if self.accountant is not None:
            self.accountant.step()
        return result

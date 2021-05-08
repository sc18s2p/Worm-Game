from typing import List

import torch

from simple_worm.controls import CONTROL_KEYS, ControlsFenics, ControlsNumpy
from simple_worm.util_torch import f2t, t2f, t2n


def to_torch(self) -> 'ControlsTorch':
    return ControlsTorch(
        e10=f2t(self.e10),
        e20=f2t(self.e20),
        alpha=f2t(self.alpha),
        beta=f2t(self.beta),
        gamma=f2t(self.gamma),
    )


# Extend ControlsFenics with a helper method to convert to torch tensors
ControlsFenics.to_torch = to_torch


class ControlsTorch:
    def __init__(
            self,
            e10: torch.Tensor = None,
            e20: torch.Tensor = None,
            alpha: torch.Tensor = None,
            beta: torch.Tensor = None,
            gamma: torch.Tensor = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1,
            optim_e0: bool = False,
            optim_abg: bool = False,
    ):
        assert (worm is None
                and e10 is not None and e20 is not None
                and alpha is not None and beta is not None and gamma is not None) \
               or worm is not None

        if worm is not None:
            frame_shape = (3, worm.N + 1)
            e10 = torch.zeros(frame_shape, requires_grad=optim_e0)
            e20 = torch.zeros(frame_shape, requires_grad=optim_e0)
            ab_shape = (n_timesteps, worm.N + 1)
            alpha = torch.zeros(ab_shape, requires_grad=optim_abg)
            beta = torch.zeros(ab_shape, requires_grad=optim_abg)
            g_shape = (n_timesteps, worm.N)
            gamma = torch.zeros(g_shape, requires_grad=optim_abg)

        assert e10.shape == e20.shape
        assert alpha.shape == beta.shape
        assert e10.shape[-1] == alpha.shape[-1] == gamma.shape[-1] + 1

        self.e10 = e10
        self.e20 = e20
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def parameters(self, as_dict=False):
        if as_dict:
            return {k: getattr(self, k) for k in CONTROL_KEYS}
        else:
            return [getattr(self, k) for k in CONTROL_KEYS]

    def requires_grad(self):
        return self.e10.requires_grad or self.e20.requires_grad \
               or self.alpha.requires_grad or self.beta.requires_grad or self.gamma.requires_grad

    def to_fenics(self, worm: 'Worm'):
        return ControlsFenics(
            e10=t2f(self.e10, fs=worm.V3, name='e10'),
            e20=t2f(self.e20, fs=worm.V3, name='e20'),
            alpha=[t2f(a, fs=worm.V, name=f'alpha_t{t}') for t, a in enumerate(self.alpha)],
            beta=[t2f(b, fs=worm.V, name=f'beta_t{t}') for t, b in enumerate(self.beta)],
            gamma=[t2f(g, fs=worm.Q, name=f'gamma_t{t}') for t, g in enumerate(self.gamma)],
        )

    def to_numpy(self):
        return ControlsNumpy(
            e10=t2n(self.e10),
            e20=t2n(self.e20),
            alpha=t2n(self.alpha),
            beta=t2n(self.beta),
            gamma=t2n(self.gamma),
        )


class ControlsBatchTorch(ControlsTorch):
    def __init__(
            self,
            e10: torch.Tensor = None,
            e20: torch.Tensor = None,
            alpha: torch.Tensor = None,
            beta: torch.Tensor = None,
            gamma: torch.Tensor = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1,
            batch_size: int = 1,
            optim_e0: bool = False,
            optim_abg: bool = False,
    ):
        super().__init__(e10, e20, alpha, beta, gamma, worm, n_timesteps, optim_e0, optim_abg)
        if worm is not None:
            for k in CONTROL_KEYS:
                c = getattr(self, k)
                if batch_size > 1:
                    c.data = c.unsqueeze(0).expand(batch_size, *c.shape).clone()
                else:
                    c.data = c.unsqueeze(0)

    def __getitem__(self, i):
        args = {k: getattr(self, k)[i] for k in CONTROL_KEYS}
        return ControlsTorch(**args)

    @staticmethod
    def from_list(Cs: List[ControlsTorch]):
        args = {
            k: torch.stack([getattr(C, k) for C in Cs])
            for k in CONTROL_KEYS
        }
        return ControlsBatchTorch(**args)

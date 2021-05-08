from typing import List

import numpy as np
from fenics import *

try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    pass

from simple_worm.util import v2f

FRAME_KEYS = ['e10', 'e20']
DRIVE_KEYS = ['alpha', 'beta', 'gamma']
CONTROL_KEYS = ['e10', 'e20', 'alpha', 'beta', 'gamma']


class ControlsFenics:
    def __init__(
            self,
            e10: Function = None,
            e20: Function = None,
            alpha: List[Function] = None,
            beta: List[Function] = None,
            gamma: List[Function] = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1,
    ):
        assert (worm is None
                and e10 is not None and e20 is not None
                and alpha is not None and beta is not None and gamma is not None) \
               or worm is not None

        if worm is not None:
            e10 = v2f(val=worm.e10_default, fs=worm.V3, name='e10')
            e20 = v2f(val=worm.e20_default, fs=worm.V3, name='e20')
            alpha = []
            beta = []
            gamma = []
            for i in range(n_timesteps):
                alpha.append(v2f(val=worm.alpha_pref_default, fs=worm.V, name=f'alpha_t{i}'))
                beta.append(v2f(val=worm.beta_pref_default, fs=worm.V, name=f'beta_t{i}'))
                gamma.append(v2f(val=worm.gamma_pref_default, fs=worm.Q, name=f'gamma_t{i}'))

        assert len(alpha) == len(beta) == len(gamma)
        self.e10 = e10
        self.e20 = e20
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __getitem__(self, i):
        """
        Returns a copy with only the requested time index of drive variables.
        """
        args = {
            **{k: getattr(self, k) for k in FRAME_KEYS},
            **{k: [getattr(self, k)[i]] for k in DRIVE_KEYS}
        }
        return ControlsFenics(**args)


class ControlsNumpy:
    def __init__(
            self,
            e10: np.ndarray = None,
            e20: np.ndarray = None,
            alpha: np.ndarray = None,
            beta: np.ndarray = None,
            gamma: np.ndarray = None,
            worm: 'Worm' = None,
            n_timesteps: int = 1,
    ):
        assert (worm is None
                and e10 is not None and e20 is not None
                and alpha is not None and beta is not None and gamma is not None) \
               or worm is not None

        if worm is not None:
            frame_shape = (3, worm.N + 1)
            e10 = np.zeros(frame_shape)
            e20 = np.zeros(frame_shape)
            ab_shape = (n_timesteps, worm.N + 1)
            alpha = np.zeros(ab_shape)
            beta = np.zeros(ab_shape)
            g_shape = (n_timesteps, worm.N)
            gamma = np.zeros(g_shape)

        assert e10.shape == e20.shape
        assert alpha.shape == beta.shape
        assert e10.shape[-1] == alpha.shape[-1] == gamma.shape[-1] + 1

        self.e10 = e10
        self.e20 = e20
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def to_fenics(self, worm: 'Worm'):
        return ControlsFenics(
            e10=v2f(self.e10, fs=worm.V3, name='e10'),
            e20=v2f(self.e20, fs=worm.V3, name='e20'),
            alpha=[v2f(a, fs=worm.V, name=f'alpha_t{t}') for t, a in enumerate(self.alpha)],
            beta=[v2f(b, fs=worm.V, name=f'beta_t{t}') for t, b in enumerate(self.beta)],
            gamma=[v2f(g, fs=worm.Q, name=f'gamma_t{t}') for t, g in enumerate(self.gamma)],
        )

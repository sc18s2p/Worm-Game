from typing import List
from typing import Union

import numpy as np
import torch
from fenics import *
from fenics_adjoint import *
from sklearn.decomposition import PCA

from simple_worm.util import f2n, v2f


def t2n(var: torch.Tensor) -> np.ndarray:
    """
    Torch to numpy
    """
    return var.detach().numpy()


def f2t(var: Union[Function, List[Function]]) -> torch.Tensor:
    """
    Fenics to torch
    Returns a torch tensor containing fenics function values
    """
    return torch.from_numpy(f2n(var))


def t2f(
        val: torch.Tensor,
        var: Function = None,
        fs: FunctionSpace = None,
        name: str = None
) -> Function:
    """
    Torch to fenics
    Set a value to a new or existing fenics variable.
    """
    val = t2n(val)
    return v2f(val, var, fs, name)


def calculate_e0_batch(worm_len, x0):
    print('calculating e0 batch')
    e10 = []
    e20 = []
    for x0i in x0:
        e10i, e20i = calculate_e0_single(worm_len, x0i)
        e10.append(e10i)
        e20.append(e20i)
    e10 = torch.stack(e10)
    e20 = torch.stack(e20)
    return e10, e20


def calculate_e0_single(worm_len, x0):
    # Calculate a reference frame using all the body coordinates
    pca = PCA()
    pca.fit(x0.T)
    components_ref = pca.components_

    # Find and plot frames
    window_size = 20
    e10 = []
    e20 = []
    for i in range(worm_len):
        # Sliding window PCA
        window_start = max(0, int(i - window_size / 2))
        window_end = min(worm_len - 1, int(i + window_size / 2))
        pts = x0[:, window_start:window_end + 1].T
        pca = PCA()
        pca.fit(pts)
        components_i = pca.components_

        # Correlate components with reference window (should correct for sign-flips)
        for j in range(1, 3):
            if np.dot(components_i[j], components_ref[j]) < 0:
                components_i[j] *= -1

        e10.append(components_i[1])
        e20.append(components_i[2])

    e10 = torch.from_numpy(np.stack(e10, axis=1))
    e20 = torch.from_numpy(np.stack(e20, axis=1))

    return e10, e20

import numpy as np
import torch

from simple_worm.worm_torch import WormModule, ControlsBatchTorch


def generate_test_target(
        N=10,
        T=0.1,
        dt=0.1,
        batch_size=1,
        e10_val=torch.tensor([0, 1, 0]),
        e20_val=torch.tensor([0, 0, 1]),
        alpha_pref_freq=1,
        beta_pref_freq=0,
):
    print('--- Generating test target')
    worm = WormModule(N - 1, dt=dt, batch_size=batch_size)
    n_timesteps = int(T / dt)
    C = ControlsBatchTorch(
        worm=worm.worm_solver,
        batch_size=batch_size,
        n_timesteps=n_timesteps
    )

    # Set ICs
    x0 = torch.zeros((batch_size, 3, N), dtype=torch.float64)
    x0[:, 0] = torch.linspace(start=0, end=1, steps=N)
    # x0[:, 0] = torch.linspace(start=0, end=1 / np.sqrt(3), steps=N)
    # x0[:, 1] = torch.linspace(start=0, end=1 / np.sqrt(3), steps=N)
    # x0[:, 2] = torch.linspace(start=1 / np.sqrt(3), end=0, steps=N)
    C.e10[:] = e10_val.unsqueeze(0).unsqueeze(-1)
    C.e20[:] = e20_val.unsqueeze(0).unsqueeze(-1)

    # Set alpha/beta to propagating sine waves
    offset = 0.
    for i in range(n_timesteps):
        if alpha_pref_freq > 0:
            C.alpha[:, i] = torch.sin(
                alpha_pref_freq * 2 * np.pi * (torch.linspace(start=0, end=1, steps=N) + offset)
            )
        if beta_pref_freq > 0:
            C.beta[:, i] = torch.sin(
                beta_pref_freq * 2 * np.pi * (torch.linspace(start=0, end=1, steps=N) + offset)
            )
        offset += dt

    eps = 1e-2
    C.gamma[:] = torch.linspace(start=-eps, end=eps, steps=N - 1)

    # Run the model forward to generate the output
    X, E1, E2 = worm.forward(x0, C)

    return x0, C, X, E1, E2

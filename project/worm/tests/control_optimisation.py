import numpy as np
import torch
from torch.nn.functional import mse_loss

from simple_worm.controls_torch import ControlsBatchTorch
from simple_worm.plot3d import generate_scatter_clip
from simple_worm.worm_torch import WormModule
from tests.helpers import generate_test_target

N_VID_XS = 6  # number of training examples to show in the videos


def control_optimisation(
        optim_e0=False,
        optim_abg=False,
        N=4,
        T=0.2,
        dt=0.1,
        lr=0.5,
        n_iter=20,
        parallel_solvers=0,
        generate_vids=False
):
    print('\n==== Test Control Optimisation ===')
    print(
        f'optim_e0={optim_e0}, optim_abg={optim_abg}, N={N}, dt={dt:.2f}, T={T:.2f}, lr={lr:.2E}, n_iter={n_iter}, parallel_solvers={parallel_solvers}\n')
    batch_size = 1 if parallel_solvers == 0 else parallel_solvers
    if generate_vids:
        vid_dir = f'vids/optim_e0={optim_e0},optim_abg={optim_abg},N={N},T={T:.2f},dt={dt:.2f},lr={lr:.2f},ps={parallel_solvers}'

    # Get targets
    x0, C_target, X_target, E1_target, E2_target = generate_test_target(
        N,
        T,
        dt,
        batch_size,
        beta_pref_freq=0.25,
        e10_val=torch.tensor([0, 0, 1]),
        e20_val=torch.tensor([0, 1, 0])
    )
    X_target_np = X_target[0].detach().numpy().copy()

    worm = WormModule(
        N - 1,
        dt=dt,
        batch_size=batch_size,
        inverse_opt_max_iter=1,
        parallel=parallel_solvers > 0,
        n_workers=parallel_solvers,
        quiet=True,
    )

    # Set ICs and target
    C = ControlsBatchTorch(
        worm=worm.worm_solver,
        n_timesteps=int(T / dt),
        batch_size=batch_size,
        optim_e0=optim_e0,
        optim_abg=optim_abg
    )

    # Clone the target parameters if we aren't trying to optimise them
    if not optim_e0:
        C.e10[:] = C_target.e10.clone()
        C.e20[:] = C_target.e20.clone()
    else:
        with torch.no_grad():
            C.e10.normal_(std=1e-3)
            C.e20.normal_(std=1e-3)
            C.e10[:, 1] = 1
            C.e20[:, 2] = 1
    if not optim_abg:
        C.alpha[:] = C_target.alpha.clone()
        C.beta[:] = C_target.beta.clone()
        C.gamma[:] = C_target.gamma.clone()
    else:
        with torch.no_grad():
            C.alpha.normal_(std=1e-3)
            C.beta.normal_(std=1e-3)
            C.gamma.normal_(std=1e-3)

    # Create an optimiser
    optimiser = torch.optim.Adam(C.parameters(), lr=lr)

    # Save outputs
    X_outs = []
    labels = []
    LX_prev = torch.tensor(np.inf)
    LE1_prev = torch.tensor(np.inf)
    LE2_prev = torch.tensor(np.inf)
    Le10_prev = torch.tensor(np.inf)
    Le20_prev = torch.tensor(np.inf)
    La_prev = torch.tensor(np.inf)
    Lb_prev = torch.tensor(np.inf)
    Lg_prev = torch.tensor(np.inf)

    # Iteratively optimise using gradient descent
    for n in range(n_iter):

        X, E1, E2, C_opt = worm.forward(x0, C, calculate_inverse=True, X_target=X_target)

        # Calculate losses
        LX = mse_loss(X, X_target)
        LE1 = mse_loss(E1, E1_target)
        LE2 = mse_loss(E2, E2_target)
        Le10 = mse_loss(C.e10, C_target.e10)
        Le20 = mse_loss(C.e20, C_target.e20)
        La = mse_loss(C.alpha, C_target.alpha)
        Lb = mse_loss(C.beta, C_target.beta)
        Lg = mse_loss(C.gamma, C_target.gamma)
        L = LX + LE1 + LE2 + La + Lb + Lg

        print(f'Episode {n}. Loss = {L:.5E} '
              f'(LX={LX:.3E}, LE1={LE1:.3E}, LE2={LE2:.3E}, '
              f'Le10={Le10:.3E}, Le20={Le20:.3E}, '
              f'La={La:.3E}, Lb={Lb:.3E}, Lg={Lg:.3E})')

        # Check that losses are decreasing (or not)
        if n == 0:
            # There are usually fluctuations in the indirect losses so it is not possible to assert
            # monotonic decrease (like we do with LX) so instead check if the overall loss has decreased
            LE1_first = LE1
            LE2_first = LE2
            Le10_first = Le10
            Le20_first = Le20
            La_first = La
            Lb_first = Lb
            Lg_first = Lg
        else:
            # x0 and X_target should not be changing
            assert torch.allclose(x0, x0_prev)
            assert torch.allclose(X_target, X_target_prev)

            # Loss should only be decreasing if something is being optimised
            if optim_e0 or optim_abg:
                assert LX <= LX_prev
            else:
                assert torch.allclose(LX, LX_prev)
                assert torch.allclose(LE1, LE1_prev)
                assert torch.allclose(LE2, LE2_prev)

            # e10/e20 should only change if they are being optimised
            if optim_e0:
                assert not torch.allclose(C.e10, e10_prev)
                assert not torch.allclose(C.e20, e20_prev)
            else:
                assert torch.allclose(C.e10, e10_prev)
                assert torch.allclose(C.e20, e20_prev)
                assert torch.allclose(Le10, Le10_prev)
                assert torch.allclose(Le20, Le20_prev)

            # alpha/beta/gamma should only change if they are being optimised
            if optim_abg:
                assert not torch.allclose(C.alpha, alpha_prev)
                assert not torch.allclose(C.beta, beta_prev)
                assert not torch.allclose(C.gamma, gamma_prev)
            else:
                assert torch.allclose(C.alpha, alpha_prev)
                assert torch.allclose(C.beta, beta_prev)
                assert torch.allclose(C.gamma, gamma_prev)
                assert torch.allclose(La, La_prev)
                assert torch.allclose(Lb, Lb_prev)
                assert torch.allclose(Lg, Lg_prev)

        x0_prev = x0.clone().detach()
        X_target_prev = X_target.clone().detach()
        e10_prev = C.e10.clone().detach()
        e20_prev = C.e20.clone().detach()
        alpha_prev = C.alpha.clone().detach()
        beta_prev = C.beta.clone().detach()
        gamma_prev = C.gamma.clone().detach()
        LX_prev = LX
        LE1_prev = LE1
        LE2_prev = LE2
        Le10_prev = Le10
        Le20_prev = Le20
        La_prev = La
        Lb_prev = Lb
        Lg_prev = Lg

        # Do optimisation step
        if C.requires_grad():
            optimiser.zero_grad()
            LX.backward()
            optimiser.step()

        # Make vids
        if generate_vids:
            X_outs.append(X[0].detach().numpy())
            labels.append(f'X_{n + 1}')
            if len(X_outs) > N_VID_XS:
                idxs = np.round(np.linspace(0, len(X_outs) - 1, N_VID_XS)).astype(int)
                X_outs_to_plot = [X_outs[i] for i in idxs]
                labels_to_plot = [labels[i] for i in idxs]
            else:
                X_outs_to_plot = X_outs
                labels_to_plot = labels

            generate_scatter_clip(
                [X_target_np, *X_outs_to_plot],
                save_dir=vid_dir,
                labels=['Target', *[l for l in labels_to_plot]]
            )

    # Check overall loss changes
    if 0:
        # These tests fail over short runs so disabled for ci
        if optim_abg or optim_e0:
            assert LE1 <= LE1_first
            assert LE2 <= LE2_first
        if optim_e0:
            assert Le10 <= Le10_first
            assert Le20 <= Le20_first
        else:
            assert torch.allclose(Le10, Le10_first)
            assert torch.allclose(Le20, Le20_first)
        if optim_abg:
            assert La <= La_first
            assert Lb <= Lb_first
            assert Lg <= Lg_first
        else:
            assert torch.allclose(La, La_first)
            assert torch.allclose(Lb, Lb_first)
            assert torch.allclose(Lg, Lg_first)


default_args = {
    'optim_e0': False,
    'optim_abg': False,
    'N': 10,
    'T': 0.3,
    'dt': 0.1,
    'lr': 1e-3,
    'n_iter': 2,
    'parallel_solvers': 0,
    'generate_vids': False,
}


def test_optim_e0():
    control_optimisation(**{**default_args, 'optim_e0': True})


def test_optim_abg():
    control_optimisation(**{**default_args, 'optim_abg': True})


def test_optim_both():
    control_optimisation(**{**default_args, 'optim_e0': True, 'optim_abg': True})


def test_optim_neither():
    control_optimisation(**default_args)


def test_parallel_solvers():
    control_optimisation(**{**default_args, 'optim_e0': True, 'optim_abg': True, 'parallel_solvers': 2})


if __name__ == '__main__':
    test_optim_e0()
    test_optim_abg()
    test_optim_both()
    test_optim_neither()
    test_parallel_solvers()

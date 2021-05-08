import os
import shutil
import time
from datetime import timedelta

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from simple_worm.controls_torch import ControlsBatchTorch, ControlsTorch
from simple_worm.plot3d import generate_scatter_clip, plot_3d_frames
from simple_worm.util_torch import t2n
from simple_worm.worm_torch import WormModule

LOGS_PATH = 'logs'
N_VID_EGS = 1  # number of training examples to show in the videos
START_TIMESTAMP = time.strftime('%Y-%m-%d_%H%M%S')


class Trainer:
    def __init__(
            self,
            N: int = 10,
            T: float = 1.,
            dt: float = 0.1,
            optim_e0: bool = False,
            optim_abg: bool = True,
            target_params: dict = {},
            lr: float = 0.1,
            sgd_momentum: float = 0.9,
            reg_weights: dict = {},
            inverse_opt_max_iter: int = 2,
            inverse_opt_tol: float = 1e-8,
            parallel_solve: bool = False,
            parallel_solve_workers: int = 1,
            save_videos=False,
            save_plots=False,
    ):
        # Domain
        self.N = N
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)

        # Optimiser parameters
        self.optim_e0 = optim_e0
        self.optim_abg = optim_abg
        self.lr = lr
        self.sgd_momentum = sgd_momentum

        # Inverse optimiser parameters
        self.reg_weights = reg_weights
        self.inverse_opt_max_iter = inverse_opt_max_iter
        self.inverse_opt_tol = inverse_opt_tol

        # Training params
        self.start_step = 1
        self.global_step = 0
        self.best_loss = 1.e10
        self.save_plots = save_plots
        self.save_videos = save_videos

        # Worm module
        self.worm = WormModule(
            N - 1,
            dt=dt,
            batch_size=1,
            reg_weights=reg_weights,
            inverse_opt_max_iter=inverse_opt_max_iter,
            inverse_opt_tol=inverse_opt_tol,
            parallel=parallel_solve,
            n_workers=parallel_solve_workers,
            quiet=True
        )
        self.X_outs = []
        self.X_labels = []

        self._init_params(optim_e0, optim_abg, target_params)
        self._build_optimiser()

    @property
    def logs_path(self) -> str:
        return LOGS_PATH + f'/N={self.N},' \
                           f'T={self.T:.2f},' \
                           f'dt={self.dt:.2f}' \
                           f'/{START_TIMESTAMP}_' \
                           f'lr={self.lr:.1E},' \
                           f'rw={self.reg_weights},' \
                           f'ii={self.inverse_opt_max_iter},' \
                           f'it={self.inverse_opt_tol:.1E}'

    def _init_loggers(self):
        self.logger = SummaryWriter(self.logs_path, flush_secs=5)

    def _init_params(self, optim_e0: bool, optim_abg: bool, target_params: dict = {}):
        # Generate targets
        self.x0, self.C_target, self.X_target, self.E1_target, self.E2_target \
            = self._generate_test_target(**target_params)

        # Generate optimisable controls
        self.C = ControlsTorch(
            worm=self.worm.worm_solver,
            n_timesteps=self.n_steps,
            optim_e0=optim_e0,
            optim_abg=optim_abg
        )

        # Clone the target parameters if we aren't trying to optimise them
        if not optim_e0:
            self.C.e10[:] = self.C_target.e10.clone()
            self.C.e20[:] = self.C_target.e20.clone()
        else:
            with torch.no_grad():
                self.C.e10[1] = 1
                self.C.e20[2] = 1
        if not optim_abg:
            self.C.alpha[:] = self.C_target.alpha.clone()
            self.C.beta[:] = self.C_target.beta.clone()
            self.C.gamma[:] = self.C_target.gamma.clone()
        else:
            # Add some noise
            with torch.no_grad():
                self.C.alpha.normal_(std=1e-3)
                self.C.beta.normal_(std=1e-3)
                self.C.gamma.normal_(std=1e-5)

    def _generate_test_target(
            self,
            e10_val: torch.Tensor = torch.tensor([0, 1, 0]),
            e20_val: torch.Tensor = torch.tensor([0, 0, 1]),
            alpha_pref_freq: float = 1.,
            beta_pref_freq: float = 0.,
    ):
        print('Generating test target')
        C = ControlsBatchTorch(worm=self.worm.worm_solver, n_timesteps=self.n_steps)

        # Set ICs
        x0 = torch.zeros((1, 3, self.N), dtype=torch.float64)
        x0[:, 0] = torch.linspace(start=0, end=1, steps=self.N)
        C.e10[:] = e10_val.unsqueeze(0).unsqueeze(-1)
        C.e20[:] = e20_val.unsqueeze(0).unsqueeze(-1)

        # Set alpha/beta to propagating sine waves
        offset = 0.
        for i in range(self.n_steps):
            if alpha_pref_freq > 0:
                C.alpha[:, i] = torch.sin(
                    alpha_pref_freq * 2 * np.pi * (torch.linspace(start=0, end=1, steps=self.N) + offset)
                )
            if beta_pref_freq > 0:
                C.beta[:, i] = torch.sin(
                    beta_pref_freq * 2 * np.pi * (torch.linspace(start=0, end=1, steps=self.N) + offset)
                )
            offset += self.dt

        # Add a slight twist along the body
        eps = 1e-2
        C.gamma[:] = torch.linspace(start=-eps, end=eps, steps=self.N - 1)

        # Run the model forward to generate the output
        X, E1, E2 = self.worm.forward(x0, C)

        # Remove batch dimension
        return x0[0], C[0], X[0], E1[0], E2[0]

    def _build_optimiser(self):
        self.LX = nn.MSELoss()
        self.LE1 = nn.MSELoss()
        self.LE2 = nn.MSELoss()
        self.Le10 = nn.MSELoss()
        self.Le20 = nn.MSELoss()
        self.La = nn.MSELoss()
        self.Lb = nn.MSELoss()
        self.Lg = nn.MSELoss()
        self.optimiser = optim.Adam(
            self.C.parameters(),
            lr=self.lr
        )

    def configure_paths(self, renew_logs):
        if renew_logs:
            print('Removing previous log files...')
            shutil.rmtree(self.logs_path, ignore_errors=True)
        os.makedirs(self.logs_path, exist_ok=True)

    def train(self, n_steps):
        self._init_loggers()  # need to call this here in case paths have changed
        final_step = self.start_step + n_steps - 1

        # Initial plots
        self._plot_e0()
        self._plot_controls()

        for step in range(self.start_step, final_step + 1):
            start_time = time.time()
            self._train_step(step, final_step)
            time_per_step = time.time() - start_time
            seconds_left = float((final_step - step) * time_per_step)
            print('Time per step: {}, Est. complete in: {}'.format(
                str(timedelta(seconds=time_per_step)),
                str(timedelta(seconds=seconds_left))))

    def _train_step(self, step, final_step):
        # Forward simulation
        X, E1, E2, CO, XO = self.worm.forward(
            self.x0.unsqueeze(0),
            ControlsBatchTorch.from_list([self.C]),
            calculate_inverse=True,
            calculate_X_opt=True,
            X_target=self.X_target.unsqueeze(0)
        )
        # Remove batch dims
        X, E1, E2, CO, XO = X[0], E1[0], E2[0], CO[0], XO[0]

        self.X = X
        self.X_outs.append(X.detach().numpy())
        self.X_labels.append(f'X_{step}')
        self.E1 = E1
        self.E2 = E2

        # Calculate losses
        LX = self.LX(X, self.X_target)
        LE1 = self.LE1(E1, self.E1_target)
        LE2 = self.LE2(E2, self.E2_target)
        Le10 = self.Le10(self.C.e10, self.C_target.e10)
        Le20 = self.Le20(self.C.e20, self.C_target.e20)
        La = self.La(self.C.alpha, self.C_target.alpha)
        Lb = self.Lb(self.C.beta, self.C_target.beta)
        Lg = self.Lg(self.C.gamma, self.C_target.gamma)
        L = LX + LE1 + LE2 + Le10 + Le20 + La + Lb + Lg

        # Calculate gradients and do optimisation step
        if self.C.requires_grad():
            self.optimiser.zero_grad()
            LX.backward()
            self.optimiser.step()

        # Increment global step counter
        self.global_step += 1

        # Calculate norms
        parameter_norm_sum = 0.
        for p in self.C.parameters():
            parameter_norm_sum += p.norm()

        # Write debug
        self.logger.add_scalar('loss/step', L, self.global_step)
        self.logger.add_scalar('loss/LX', LX, self.global_step)
        self.logger.add_scalar('loss/LE1', LE1, self.global_step)
        self.logger.add_scalar('loss/LE2', LE2, self.global_step)
        self.logger.add_scalar('loss/Le10', Le10, self.global_step)
        self.logger.add_scalar('loss/Le20', Le20, self.global_step)
        self.logger.add_scalar('loss/La', La, self.global_step)
        self.logger.add_scalar('loss/Lb', Lb, self.global_step)
        self.logger.add_scalar('loss/Lg', Lg, self.global_step)
        self.logger.add_scalar('loss/norm', parameter_norm_sum, self.global_step)

        print(f'[{step}/{final_step}]. Loss = {L:.5E} '
              f'(LX={LX:.3E}, LE1={LE1:.3E}, LE2={LE2:.3E}, '
              f'Le10={Le10:.3E}, Le20={Le20:.3E}, '
              f'La={La:.3E}, Lb={Lb:.3E}, Lg={Lg:.3E})')

        self._plot_X()
        self._plot_3d_frames()
        self._plot_e0()
        self._plot_controls()
        self._make_vids()

        return L

    def _make_vids(self):
        if not self.save_videos:
            return

        # Make vids
        if N_VID_EGS == 1:
            idxs = [len(self.X_outs) - 1]
            X_outs_to_plot = [self.X_outs[-1]]
            labels_to_plot = [self.X_labels[-1]]
        elif len(self.X_outs) > N_VID_EGS:
            idxs = np.round(np.linspace(0, len(self.X_outs) - 1, N_VID_EGS)).astype(int)
            X_outs_to_plot = [self.X_outs[i] for i in idxs]
            labels_to_plot = [self.X_labels[i] for i in idxs]
        else:
            idxs = list(range(len(self.X_outs)))
            X_outs_to_plot = self.X_outs
            labels_to_plot = self.X_labels
        print(f'Generating scatter clip with idxs={idxs}')

        generate_scatter_clip(
            [t2n(self.X_target), *X_outs_to_plot],
            save_dir=self.logs_path + '/vids',
            save_fn=str(self.global_step),
            labels=['Target', *[l for l in labels_to_plot]]
        )

    def _plot_3d_frames(self):
        X = t2n(self.X)
        X_target = t2n(self.X_target)
        E1 =t2n(self.E1)
        E2 =t2n(self.E2)
        E1_target= t2n(self.E1_target)
        E2_target= t2n(self.E2_target)

        fig = plot_3d_frames(
            Xs=[X, X_target],
            E1s=[E1, E1_target],
            E2s=[E2, E2_target],
            Cs=[self.C.to_numpy(), self.C_target.to_numpy()],
            labels=['Attempt', 'Target']
        )

        self._save_plot('3d')
        self.logger.add_figure(f'3d', fig, self.global_step)
        self.logger.flush()
        plt.close(fig)

    def _plot_X(self):
        X = t2n(self.X)
        X_target = t2n(self.X_target)

        # Get MSE losses
        LX = np.square(X - X_target)

        # Determine common scales
        X_vmin = min(X.min(), X_target.min())
        X_vmax = max(X.max(), X_target.max())
        L_vmax = LX.max()

        fig, axes = plt.subplots(3, 3, figsize=(14, 8))

        for row_idx, M in enumerate([X, LX, X_target]):
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]

                # Select xyz component and transpose so columns are frames
                Mc = M[:, col_idx].T
                if row_idx in [0, 2]:
                    vmin = X_vmin
                    vmax = X_vmax
                    m = ax.matshow(
                        Mc,
                        cmap=plt.cm.PRGn,
                        clim=(vmin, vmax),
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                        aspect='auto'
                    )
                else:
                    if L_vmax > 0:
                        norm = colors.LogNorm(vmin=1e-7, vmax=L_vmax)
                    else:
                        norm = None
                    m = ax.matshow(Mc, cmap=plt.cm.Reds, aspect='auto', norm=norm)

                if row_idx == 0:
                    ax.set_title(['x', 'y', 'z'][col_idx])
                if col_idx == 0:
                    ax.set_ylabel(['X', 'LX', 'X_target'][row_idx])
                if col_idx == 2:
                    fig.colorbar(m, ax=ax, format='%.5f')

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        fig.tight_layout()
        self._save_plot('X')
        self.logger.add_figure(f'LX', fig, self.global_step)
        self.logger.flush()
        plt.close(fig)


    def _plot_e0(self):
        if not self.optim_e0:
            return

        # Convert controls to numpy
        C = self.C.to_numpy()
        C_target = self.C_target.to_numpy()

        # Get MSE losses
        Le10 = np.square(C.e10 - C_target.e10)
        Le20 = np.square(C.e20 - C_target.e20)

        # Determine common scales
        e_vmin = min(C.e10.min(), C.e20.min(), C_target.e10.min(), C_target.e20.min())
        e_vmax = max(C.e10.max(), C.e20.max(), C_target.e10.max(), C_target.e20.max())
        L_vmin = min(Le10.min(), Le20.min())
        L_vmax = max(Le10.max(), Le20.max())

        fig, axes = plt.subplots(3, 3, figsize=(12, 7))

        M1 = [C.e10, Le10, C_target.e10]
        M2 = [C.e20, Le20, C_target.e20]

        for row_idx in range(3):
            for col_idx in range(2):
                ax = axes[row_idx, col_idx * 2]
                M = [M1, M2][col_idx][row_idx]

                if row_idx in [0, 2, 3, 5]:
                    vmin = e_vmin
                    vmax = e_vmax
                    m = ax.matshow(
                        M,
                        cmap=plt.cm.PRGn,
                        clim=(vmin, vmax),
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                        aspect='auto'
                    )
                else:
                    m = ax.matshow(M, cmap=plt.cm.Reds, aspect='auto', vmin=L_vmin, vmax=L_vmax)

                if row_idx == 0:
                    ax.set_title(['e10', 'e20'][col_idx])
                if col_idx == 0:
                    ax.set_ylabel(['e', 'Le', 'e_target'][row_idx])
                if col_idx == 1:
                    fig.colorbar(m, ax=ax, format='%.3f')

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        # Calculate magnitudes
        mags = np.stack([np.linalg.norm(C.e10, axis=0), np.linalg.norm(C.e20, axis=0)])
        ax = axes[0, 1]
        m = ax.matshow(mags, cmap=plt.cm.bwr, aspect='auto', vmin=0, vmax=max(2, mags.max()))
        ax.set_title('Magnitudes')
        fig.colorbar(m, ax=ax, format='%.3f')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Calculate orthogonality
        dp = np.array([[np.dot(C.e10[:, i], C.e20[:, i]) for i in range(C.e10.shape[-1])]])
        ax = axes[1, 1]
        m = ax.matshow(dp, cmap=plt.cm.bwr, aspect='auto')
        ax.set_title('Orthogonality (dot products)')
        fig.colorbar(m, ax=ax, format='%.3f')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        fig.tight_layout()
        self._save_plot('e0')
        self.logger.add_figure(f'LE', fig, self.global_step)
        self.logger.flush()
        plt.close(fig)

    def _plot_controls(self):
        if not self.optim_abg:
            return

        # Convert controls to numpy
        C = self.C.to_numpy()
        C_target = self.C_target.to_numpy()

        # Get MSE losses
        La = np.square(C.alpha - C_target.alpha)
        Lb = np.square(C.beta - C_target.beta)
        Lg = np.square(C.gamma - C_target.gamma)

        # Determine common scales
        X_vmin = min(C.alpha.min(), C_target.alpha.min(), C.beta.min(),
                     C_target.beta.min(), C.gamma.min(), C_target.gamma.min())
        X_vmax = max(C.alpha.max(), C_target.alpha.max(), C.beta.max(),
                     C_target.beta.max(), C.gamma.max(), C_target.gamma.max())
        L_vmax = max(La.max(), Lb.max(), Lg.max())

        fig, axes = plt.subplots(3, 3, figsize=(12, 7))

        for row_idx, Ms in enumerate([
            [C.alpha, C.beta, C.gamma],
            [La, Lb, Lg],
            [C_target.alpha, C_target.beta, C_target.gamma]
        ]):
            for col_idx in range(3):
                M = Ms[col_idx].T
                ax = axes[row_idx, col_idx]
                if row_idx in [0, 2]:
                    vmin = X_vmin
                    vmax = X_vmax
                    m = ax.matshow(
                        M,
                        cmap=plt.cm.PRGn,
                        clim=(vmin, vmax),
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                        aspect='auto'
                    )
                else:
                    if L_vmax > 0:
                        norm = colors.LogNorm(vmin=1e-7, vmax=L_vmax)
                    else:
                        norm = None
                    m = ax.matshow(M, cmap=plt.cm.Reds, aspect='auto', norm=norm)

                if row_idx == 0:
                    ax.set_title(['alpha', 'beta', 'gamma'][col_idx])
                if col_idx == 0:
                    ax.set_ylabel(['attempt', 'L', 'target'][row_idx])
                if col_idx == 2:
                    fig.colorbar(m, ax=ax, format='%.3f')

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        fig.tight_layout()
        self._save_plot('abg')
        self.logger.add_figure(f'LC', fig, self.global_step)
        self.logger.flush()
        plt.close(fig)

    def _save_plot(self, plot_type):
        if self.save_plots:
            save_dir = self.logs_path + f'/plots/{plot_type}'
            os.makedirs(save_dir, exist_ok=True)
            path = save_dir + f'/{self.global_step:04d}.svg'
            plt.savefig(path, bbox_inches='tight')


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

import os
import time
from typing import List

import matplotlib.animation as manimation
import numpy as np
from matplotlib import cm
from matplotlib import gridspec
# mpl.use('agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from simple_worm.controls import ControlsNumpy

FPS = 5


def cla(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])


def interactive():
    import matplotlib
    gui_backend = 'Qt5Agg'
    matplotlib.use(gui_backend, force=True)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def generate_scatter_clip(Xs, save_dir, save_fn=None, labels=None):
    """
    Pass any number of midline trajectories (Xs).
    Each X in Xs should have shape: [n_timesteps, 3, worm_length]
    """
    os.makedirs(save_dir, exist_ok=True)
    save_fn = save_fn if save_fn is not None else time.strftime('%Y-%m-%d_%H%M%S')
    save_path = save_dir + '/' + save_fn + '.mp4'
    # print('save_path', save_path)

    # [n_timesteps, 3, worm_length]
    worm_length = Xs[0].shape[2]
    n_frames = Xs[0].shape[0]

    if labels is None:
        labels = [f'X_{i + 1}' for i in range(len(Xs))]

    # Colourmap / facecolors
    cmap = cm.get_cmap('plasma_r')
    fc = cmap((np.arange(worm_length) + 0.5) / worm_length)

    # Get common scale
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for X in Xs:
        mins = np.minimum(mins, X.min(axis=(0, 2)))
        maxs = np.maximum(maxs, X.max(axis=(0, 2)))
    max_range = max(maxs - mins)
    means = mins + (maxs - mins) / 2
    mins = means - max_range / 2
    maxs = means + max_range / 2

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(len(Xs) * 2, 8))
    gs = gridspec.GridSpec(3, len(Xs))
    ax_idxs = np.array([0, 1, 2])
    axes = [[], [], []]
    scts = [[], [], []]
    for row_idx in range(3):
        for col_idx in range(len(Xs)):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d', elev=5, azim=1)
            cla(ax)
            if row_idx == 0:
                ax.set_title(labels[col_idx])

            pts0 = np.zeros((3, worm_length))
            sct = ax.scatter(pts0[0], pts0[1], pts0[2], s=3, alpha=0.7, c=fc, depthshade=True)
            scts[row_idx].append(sct)

            ax.set_xlim(mins[row_idx], maxs[row_idx])
            ax.set_ylim(mins[(row_idx - 1) % 3], maxs[(row_idx - 1) % 3])
            ax.set_zlim(mins[(row_idx - 2) % 3], maxs[(row_idx - 2) % 3])
            axes[row_idx].append(ax)

    def update(i):
        all_scts = ()
        for col_idx, X in enumerate(Xs):
            X = X[i]
            for row_idx in range(3):
                pts = np.array([X[j] for j in np.roll(ax_idxs, shift=row_idx)])
                sct = scts[row_idx][col_idx]
                sct.set_offsets(pts[:2].T)
                sct.set_3d_properties(pts[2], zdir='z')
                all_scts += (sct,)
        return all_scts

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        blit=True
    )

    # Save
    metadata = dict(title=save_path, artist='leeds_worm_lab', comment='...')
    ani.save(save_path, writer='ffmpeg', fps=FPS, metadata=metadata)
    plt.close(fig)


def plot_3d_frames(
        Xs: List[np.ndarray],
        E1s: List[np.ndarray],
        E2s: List[np.ndarray],
        Cs: List[ControlsNumpy],
        labels=None
):
    # interactive()

    # [n_timesteps, 3, worm_length]
    worm_length = Xs[0].shape[2]
    n_frames = Xs[0].shape[0]
    if labels is None:
        labels = [f'X_{i + 1}' for i in range(len(Xs))]

    # Colourmap / facecolors
    cmap = cm.get_cmap('plasma_r')
    fc = cmap((np.arange(worm_length) + 0.5) / worm_length)
    cmap_quiver_a = cm.get_cmap('OrRd')
    cmap_quiver_b = cm.get_cmap('BuPu')

    # Get common scale
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for X in Xs:
        mins = np.minimum(mins, X.min(axis=(0, 2)))
        maxs = np.maximum(maxs, X.max(axis=(0, 2)))
    max_range = max(maxs - mins)
    means = mins + (maxs - mins) / 2
    mins = means - max_range / 3
    maxs = means + max_range / 3

    alpha_max = 0
    beta_max = 0
    for C in Cs:
        alpha_max = max(alpha_max, C.alpha.max())
        beta_max = max(beta_max, C.beta.max())
    # alpha_max = alpha_max * 1.1
    # beta_max = beta_max * 1.1

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(n_frames * 4, len(Xs) * 5))
    gs = gridspec.GridSpec(len(Xs), n_frames)
    for row_idx in range(len(Xs)):
        for col_idx in range(n_frames):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d')
            cla(ax)
            if row_idx == 0:
                ax.set_title(f'frame={col_idx + 1}')
            if col_idx == 0:
                ax.text2D(-0.1, 0.5, labels[row_idx], transform=ax.transAxes, rotation='vertical')

            # Get frame data
            X = Xs[row_idx][col_idx]
            E1 = E1s[row_idx][col_idx]
            E2 = E2s[row_idx][col_idx]

            # Scale frame vectors by forces
            frame_vector_scale = 0.3
            C = Cs[row_idx]
            if C is not None:
                alpha = C.alpha[col_idx]
                beta = C.beta[col_idx]
                E1 = E1 * alpha * frame_vector_scale
                E2 = E2 * beta * frame_vector_scale

                fc_quiver_alpha = cmap_quiver_a(np.abs(alpha) / alpha_max)
                fc_quiver_beta = cmap_quiver_b(np.abs(beta) / beta_max)

                arrow_opts = {
                    'mutation_scale': 5,
                    'arrowstyle': '-|>',
                    'linewidth': 1,
                    'alpha': 0.7
                }

                for i in range(worm_length):
                    ae1 = Arrow3D(
                        xs=[X[0, i], X[0, i] + E1[0, i]],
                        ys=[X[1, i], X[1, i] + E1[1, i]],
                        zs=[X[2, i], X[2, i] + E1[2, i]],
                        color=fc_quiver_alpha[i],
                        **arrow_opts
                    )
                    be2 = Arrow3D(
                        xs=[X[0, i], X[0, i] + E2[0, i]],
                        ys=[X[1, i], X[1, i] + E2[1, i]],
                        zs=[X[2, i], X[2, i] + E2[2, i]],
                        color=fc_quiver_beta[i],
                        **arrow_opts
                    )
                    ax.add_artist(ae1)
                    ax.add_artist(be2)

            # Scatter plot of midline
            ax.scatter(X[0], X[1], X[2], s=20, alpha=0.9, c=fc, depthshade=False)

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0)

    # plt.show()
    # exit()

    return fig

import os

import time

from simple_worm.plot3d import generate_scatter_clip
from tests.helpers import generate_test_target


def test_scatter_clip():
    # Make a single video clip
    x0, C, X, E1, E2 = generate_test_target(
        N=20,
        T=1,
        dt=0.1
    )
    save_dir = 'vids'
    save_fn = time.strftime('%Y-%m-%d_%H%M%S')
    generate_scatter_clip(
        Xs=[X[0].numpy()],
        save_dir=save_dir,
        save_fn=save_fn
    )

    # Check the video file was created
    assert os.path.exists(save_dir + '/' + save_fn + '.mp4')


if __name__ == '__main__':
    test_scatter_clip()

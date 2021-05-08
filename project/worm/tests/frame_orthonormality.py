import torch

from simple_worm.plot3d import generate_scatter_clip
from tests.helpers import generate_test_target


# todo: replace this test and write better ones to check orthonormality of the frame is preserved
def test_e1_magnitude():
    N = 20
    dt = 0.1
    T = 0.1

    e10s = torch.tensor([
        [3, 0, 0],
        [2, 0, 0],
        [1, 0, 0],
    ])
    e20s = torch.tensor([
        [1, 1, 0],
        [0, 2, 0],
        [1, 1, 1],
    ])
    Xs = []

    for i in range(len(e10s)):
        x0, C, X, E1, E2 = generate_test_target(
            N=N,
            dt=dt,
            T=T,
            e10_val=e10s[i],
            e20_val=e20s[i]
        )
        Xs.append(X[0].numpy())

    generate_scatter_clip(
        Xs=Xs,
        labels=[f'e10={e10}' for e10 in e10s],
        save_dir='vids'
    )


if __name__ == '__main__':
    test_e1_magnitude()

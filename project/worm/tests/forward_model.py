import numpy as np
from fenics import *

from simple_worm.controls import ControlsFenics, ControlsNumpy
from simple_worm.util import v2f, f2n
from simple_worm.worm import Worm

try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    # Needs importing here to line up with the imports in worm.py when both modules
    # are installed (even though not strictly required here).
    pass

# Parameters
N = 10
T = 0.03
dt = 0.01
n_timesteps = int(T / dt)


def _make_test_worm():
    # Initialise worm object with number of points and time step
    return Worm(N - 1, dt)


def _check_output(X, E1, E2, expect='same'):
    assert expect in ['same', 'different']
    assert len(X) == len(E1) == len(E2)
    X = f2n(X)
    E1 = f2n(E1)
    E2 = f2n(E2)
    n_timesteps = len(X)
    for i in range(n_timesteps):
        for j in range(i + 1, n_timesteps):
            if expect == 'same':
                assert np.allclose(X[i], X[j])
                assert np.allclose(E1[i], E1[j])
                assert np.allclose(E2[i], E2[j])
            elif expect == 'different':
                assert not np.allclose(X[i], X[j])
                assert not np.allclose(E1[i], E1[j])
                assert not np.allclose(E2[i], E2[j])


def test_defaults():
    print('\n\n----test_defaults')
    worm = _make_test_worm()
    X, E1, E2 = worm.solve(T)
    _check_output(X, E1, E2, expect='same')


def test_solve_twice():
    print('\n\n----test_solve_twice')
    worm = _make_test_worm()
    worm.solve(T)
    assert worm.t == T

    C = ControlsFenics(worm=worm, n_timesteps=n_timesteps)
    worm.solve(T, C=C, reset=False)
    assert np.allclose(worm.t, 2 * T)


def test_stepwise_solve():
    print('\n\n----test_stepwise_solve')
    worm = _make_test_worm()
    worm.initialise()
    print(f't={worm.t:.2f}')

    C_t1 = ControlsFenics(worm=worm)
    x_t1, e1_t1, e2_t1 = worm.update_solution(C_t1)
    print(f't={worm.t:.2f}')
    assert worm.t == dt

    C_t2 = ControlsFenics(worm=worm)
    x_t2, e1_t2, e2_t2 = worm.update_solution(C_t2)
    print(f't={worm.t:.2f}')
    assert worm.t == dt * 2

    _check_output([x_t1, x_t2], [e1_t1, e1_t2], [e2_t1, e2_t2], expect='same')


def test_manual_ics():
    print('\n\n----test_manual_ics')
    worm = _make_test_worm()

    # Set initial conditions manually
    x0 = np.zeros((3, N))
    x0[:][0] = np.linspace(1, 0, N, endpoint=True)

    C = ControlsNumpy(worm=worm, n_timesteps=n_timesteps)
    C.e10[:][1] = np.ones((N,))
    C.e20[:][2] = np.ones((N,))

    # Must convert x0 and C to fenics before passing to solver
    X, E1, E2 = worm.solve(
        T,
        x0=v2f(x0, fs=worm.V3, name='x0'),
        C=C.to_fenics(worm)
    )
    _check_output(X, E1, E2, expect='same')


def test_expression_ics():
    print('\n\n----test_expression_ics')
    worm = _make_test_worm()

    # Set initial conditions with fenics expressions
    # Online documentation missing but you can use simple functions like sin,
    # cos, etc. and variables with values passed as kwargs. For spatial
    # coordinate use x[0].
    x0 = Expression(('x[0]', '0', '0'), degree=1)
    e10 = Expression(('0', '1', '0'), degree=1)
    e20 = Expression(('0', '0', '1'), degree=0)

    C = ControlsFenics(worm=worm, n_timesteps=n_timesteps)
    C.e10 = v2f(e10, C.e10)
    C.e20 = v2f(e20, C.e20)

    X, E1, E2 = worm.solve(
        T,
        x0=v2f(x0, fs=worm.V3, name='x0'),  # convert the expression into a fenics variable
        C=C
    )
    _check_output(X, E1, E2, expect='same')


def test_manual_forcing():
    print('\n\n----test_manual_forcing')
    worm = _make_test_worm()

    # Create forcing functions (preferred curvatures) for each timestep
    C = ControlsFenics(worm=worm, n_timesteps=n_timesteps)
    C.alpha = [v2f(2 * np.ones((N,)), fs=worm.V) for _ in range(n_timesteps)]
    C.beta = [v2f(3 * np.ones((N,)), fs=worm.V) for _ in range(n_timesteps)]
    C.gamma = [v2f(5 * np.ones((N - 1,)), fs=worm.Q) for _ in range(n_timesteps)]

    X, E1, E2 = worm.solve(T, C=C)
    _check_output(X, E1, E2, expect='different')


def test_expression_forcing():
    print('\n\n----test_expression_forcing')
    worm = _make_test_worm()

    # Create forcing functions (preferred curvatures) using expressions
    # Online documentation missing but you can use simple functions like sin,
    # cos, etc. and variables with values passed as kwargs. For spatial
    # coordinate use x[0].
    alpha_pref = Expression('v', degree=1, v=1)
    beta_pref = Expression('v', degree=1, v=1)
    gamma_pref = Expression('v', degree=0, v=1)

    C = ControlsFenics(worm=worm, n_timesteps=n_timesteps)
    C.alpha = [v2f(alpha_pref, fs=worm.V) for _ in range(n_timesteps)]
    C.beta = [v2f(beta_pref, fs=worm.V) for _ in range(n_timesteps)]
    C.gamma = [v2f(gamma_pref, fs=worm.Q) for _ in range(n_timesteps)]

    X, E1, E2 = worm.solve(T, C=C)
    _check_output(X, E1, E2, expect='different')


if __name__ == '__main__':
    test_defaults()
    test_solve_twice()
    test_stepwise_solve()
    test_manual_ics()
    test_expression_ics()
    test_manual_forcing()
    test_expression_forcing()

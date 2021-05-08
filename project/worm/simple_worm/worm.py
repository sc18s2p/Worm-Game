import warnings
from typing import Tuple

import numpy as np
from fenics import *

from simple_worm.controls import ControlsFenics
from simple_worm.util import v2f, f2n

try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    pass
from ffc.quadrature.deprecation \
    import QuadratureRepresentationDeprecationWarning

set_log_level(100)
warnings.simplefilter('ignore', QuadratureRepresentationDeprecationWarning)

# global geometry helpers
dxL = dx(scheme='vertex', degree=1,
         metadata={'representation': 'quadrature',
                   'degree': 1})


def grad(function): return Dx(function, 0)


class Worm:
    """
    Class for holding all the information about the worm geometry and to
    update the geometry.
    """

    def __init__(
            self,
            N: int,
            dt: float,
            quiet=False
    ):
        # Domain
        self.N = N
        self.dt = dt
        self.t = 0.
        self.quiet = quiet

        # Constants
        self.K = 40.0
        self.K_rot = 1.0
        self.A = 1.0
        self.B = 0.0
        self.C = 1.0
        self.D = 0.0

        # Default initial conditions
        self.x0_default = Expression(('x[0]', '0', '0'), degree=1)
        self.e10_default = Expression(('0', '1', '0'), degree=1)
        self.e20_default = Expression(('0', '0', '1'), degree=0)

        # Default forces
        self.alpha_pref_default = Expression('0', degree=1)
        self.beta_pref_default = Expression('0', degree=1)
        self.gamma_pref_default = Expression('0', degree=0)

        # Set up function spaces
        self._init_spaces()
        self.F_op = None
        self.L = None
        self.bc = None

    # ---------- Init functions ----------

    def initialise(
            self,
            x0: Function = None,
            C: ControlsFenics = None,
            n_timesteps: int = 1,
    ) -> Tuple[Function, ControlsFenics]:
        """
        Initialise/reset the simulation.
        """
        self.t = 0
        self._init_solutions()

        # Set default initial position
        if x0 is None:
            x0 = v2f(val=self.x0_default, fs=self.V3, name='x0')

        # Set default controls
        if C is None:
            C = ControlsFenics(worm=self, n_timesteps=n_timesteps)

        self._compute_initial_values(x0, C.e10, C.e20)
        self._init_forms()

        return x0, C

    def _init_spaces(self):
        """
        Set up function spaces.
        """
        mesh = UnitIntervalMesh(self.N)
        P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
        P0 = FiniteElement('DP', mesh.ufl_cell(), 0)
        P1_3 = MixedElement([P1] * 3)
        self.V = FunctionSpace(mesh, P1)
        self.V3 = FunctionSpace(mesh, P1_3)
        self.Q = FunctionSpace(mesh, P0)
        self.VV = [self.V3, self.V3, self.V3, self.V, self.Q, self.Q, self.Q]
        self.W = FunctionSpace(
            mesh,
            MixedElement([P1_3, P1_3, P1_3, P1, P0, P0, P0])
        )

    def _init_solutions(self):
        """
        Set up functions to hold state and solutions.
        """
        self.u_n = Function(self.W)
        self.e1_n = Function(self.V3)
        self.e2_n = Function(self.V3)
        self.alpha_pref = Function(self.V)
        self.beta_pref = Function(self.V)
        self.gamma_pref = Function(self.Q)

    # ---------- Main methods ----------

    def solve(
            self,
            T: float,
            x0: Function = None,
            C: ControlsFenics = None,
            reset: bool = True
    ) -> Tuple[Function, Function, Function]:
        """
        Run the forward model for T seconds.
        """
        n_timesteps = int(T / self.dt)
        if reset:
            x0, C = self.initialise(x0, C, n_timesteps)
        assert len(C.alpha) == len(C.beta) == len(C.gamma) == n_timesteps, \
            'Controls not available for every simulation step.'
        self._print(f'Solve forward (t={self.t:.2f}..{self.t + T:.2f} / n_steps={n_timesteps})')

        X = []
        E1 = []
        E2 = []

        for i in range(n_timesteps):
            self._print(f't={self.t:.2f}')
            x_t, e1_t, e2_t = self.update_solution(C[i])
            X.append(x_t)
            E1.append(e1_t)
            E2.append(e2_t)

        return X, E1, E2

    # ---------- Initial state and conditions ----------

    def _fix_frame(self, e1, e2):
        # Orthogonalise e2 against e1
        e2 = e2 - dot(e2, e1) / dot(e1, e1) * e1
        e2 = project(e2, self.V3)

        # Normalise frame
        e1 = project(e1 / sqrt(dot(e1, e1)), self.V3)
        e2 = project(e2 / sqrt(dot(e2, e2)), self.V3)

        return e1, e2

    def _compute_initial_values(self, x0, e10, e20):
        e10, e20 = self._fix_frame(e10, e20)
        mu0 = sqrt(dot(grad(x0), grad(x0)))
        kappa0 = self._compute_initial_curvature(x0, mu0)
        gamma0 = self._compute_initial_twist(e10, e20, mu0)

        # Initialize global solution variables
        fa = FunctionAssigner(self.W, self.VV)
        fa.assign(self.u_n, [x0, Function(self.V3), kappa0, Function(self.V),
                             Function(self.Q), gamma0, Function(self.Q)])
        self.e1_n.assign(e10)
        self.e2_n.assign(e20)
        self.mu0 = mu0

    def _compute_initial_curvature(self, x0, mu0):
        # Set up problem for initial curvature
        kappa_trial = TrialFunction(self.V3)
        kappa_test = TestFunction(self.V3)
        F0_kappa = dot(kappa_trial, kappa_test) * mu0 * dxL \
                   + inner(grad(x0), grad(kappa_test)) / mu0 * dx
        a0_kappa, L0_kappa = lhs(F0_kappa), rhs(F0_kappa)
        kappa0 = Function(self.V3)
        solve(a0_kappa == L0_kappa, kappa0)
        if np.isnan(kappa0.vector().sum()):
            raise RuntimeError('kappa0 contains NaNs')
        return kappa0

    def _compute_initial_twist(self, e10, e20, mu0):
        # Set up problem for initial twist
        gamma = TrialFunction(self.Q)
        v = TestFunction(self.Q)
        F_gamma0 = (gamma - dot(grad(e10), e20) / mu0) * v * dx
        a_gamma0, L_gamma0 = lhs(F_gamma0), rhs(F_gamma0)
        gamma0 = Function(self.Q)
        solve(a_gamma0 == L_gamma0, gamma0)
        if np.isnan(gamma0.vector().sum()):
            raise RuntimeError('gamma0 contains NaNs')
        return gamma0

    def _init_forms(self):
        dt = self.dt

        # Geometry
        x_n, y_n, kappa_n, m_n, z_n, gamma_n, p_n = split(self.u_n)

        mu = sqrt(inner(grad(x_n), grad(x_n)))
        tau = grad(x_n) / mu
        tauv_tilde = project(tau, self.V3)
        tauv = tauv_tilde / sqrt(dot(tauv_tilde, tauv_tilde))

        tau_cross_kappa = cross(tau, kappa_n)
        tauv_cross_kappa = cross(tauv, kappa_n)

        tautau = outer(tau, tau)
        P = Identity(3) - tautau
        Pv = Identity(3) - outer(tauv, tauv)

        # Define variational problem
        u = TrialFunction(self.W)
        v = TestFunction(self.W)

        # Split test and trial functions
        x, y, kappa, m, z, gamma, p = split(u)
        phi_x, phi_y, phi_kappa, phi_m, phi_z, phi_gamma, phi_p = split(v)

        # Parameters
        KK = self.K * P + tautau
        K_rot = self.K_rot
        A, B, C, D = self.A, self.B, self.C, self.D

        # Variational form
        F_x = 1.0 / dt * dot(KK * (x - x_n), phi_x) * mu * dx \
              - p * dot(tau, grad(phi_x)) * dx \
              - dot(P * grad(y), grad(phi_x)) / mu * dx \
              - z * dot(tau_cross_kappa, grad(phi_x)) * dx

        F_y = dot(y - A * (kappa - self.alpha_pref * self.e1_n - self.beta_pref * self.e2_n)
                  - B * (Pv * (kappa - kappa_n) / dt - m * tauv_cross_kappa),
                  phi_y) * mu * dx  # TODO wrong measure

        F_w = inner(grad(x), grad(phi_kappa)) / mu * dx \
              + dot(kappa, phi_kappa) * mu * dxL

        F_m = -K_rot * m * phi_m * mu * dx \
              - z * grad(phi_m) * dx \
              + dot(y, tauv_cross_kappa) * phi_m * mu * dx

        F_z = (z - C * (gamma - self.gamma_pref) - D / dt *
               (gamma - gamma_n)) * phi_z * mu * dx

        F_gamma = 1.0 / dt * (gamma - gamma_n) * phi_gamma * mu * dx \
                  - grad(m) * phi_gamma * dx \
                  + dot(tau_cross_kappa, 1.0 / dt * grad(x - x_n)) \
                  * phi_gamma * dx

        F_p = (dot(tau, grad(x)) - self.mu0) * phi_p * dx

        F = F_x + F_y + F_w + F_m + F_z + F_gamma + F_p
        self.F_op, self.L = lhs(F), rhs(F)

        # boundary conditions
        kappa_space = self.W.split()[2]
        kappa_b = project(self.alpha_pref * self.e1_n + self.beta_pref * self.e2_n, self.V3)
        self.bc = DirichletBC(kappa_space, kappa_b, lambda x, o: o)

    # ---------- Main algorithm ----------

    def update_solution(
            self,
            C: ControlsFenics
    ) -> Tuple[Function, Function, Function]:
        """
        Run the model forward a single timestep.
        """

        # Update time
        dt = self.dt
        self.t += dt

        # Update driving forces
        assert len(C.alpha) == 1  # should only receive controls for a single timestep
        self.alpha_pref.assign(C.alpha[0])
        self.beta_pref.assign(C.beta[0])
        self.gamma_pref.assign(C.gamma[0])

        # Compute solution
        u = Function(self.W)
        solve(self.F_op == self.L, u, bcs=self.bc)
        if np.isnan(u.vector().sum()):
            raise RuntimeError('solution u contains NaNs')

        # Extract previous geometry
        x_n, y_n, kappa_n, m_n, z_n, gamma_n, p_n = split(self.u_n)
        mu = sqrt(inner(grad(x_n), grad(x_n)))
        tau = grad(x_n) / mu
        tauv_tilde = project(tau, self.V3)
        tauv = tauv_tilde / sqrt(dot(tauv_tilde, tauv_tilde))

        # Updated geometry
        x, y, kappa, m, z, gamma, p = split(u)
        mu_new = sqrt(inner(grad(x), grad(x)))
        tau_new = grad(x) / mu_new
        tauv_tilde = project(tau_new, self.V3)
        new_tauv = tauv_tilde / sqrt(dot(tauv_tilde, tauv_tilde))

        # Update e1, e2
        def rotated_frame(v):
            k = cross(tauv, new_tauv)
            c = dot(tauv, new_tauv)
            varphi = dt * m

            tmp = v * c \
                  + cross(k, v) \
                  + dot(v, k) / (1 + c) * k
            ret = tmp * cos(varphi) \
                  + cross(new_tauv, tmp) * sin(varphi) \
                  + dot(tmp, new_tauv) * (1 - cos(varphi)) * new_tauv
            return project(ret, self.V3)

        e1 = rotated_frame(self.e1_n)
        if np.isnan(e1.vector().sum()):
            raise RuntimeError('solution e1 contains NaNs')
        e2 = rotated_frame(self.e2_n)
        if np.isnan(e2.vector().sum()):
            raise RuntimeError('solution e2 contains NaNs')

        # Renormalise and re-orthogonalise frame
        e1, e2 = self._fix_frame(e1, e2)

        # Update solution
        self.u_n.assign(u)
        self.e1_n.assign(e1)
        self.e2_n.assign(e2)

        # Re-project variables to return (is this necessary?)
        x_n = project(x_n, self.V3)
        e1_n = project(self.e1_n, self.V3)
        e2_n = project(self.e2_n, self.V3)

        return x_n, e1_n, e2_n

    # ---------- Helpers ----------

    def _print(self, s):
        if not self.quiet:
            print(s)

    # ---------- Getters ----------

    def get_x(self) -> np.ndarray:
        """
        Returns the position of mid-line points as a numpy array.
        """
        x = split(self.u_n)[0]
        x = project(x, self.V3)
        return f2n(x)

    def get_e1(self) -> np.ndarray:
        """
        Returns the first component of cross section frame.
        """
        return f2n(self.e1_n)

    def get_e2(self) -> np.ndarray:
        """
        Returns the second component of cross section frame.
        """
        return f2n(self.e2_n)

    def get_alpha(self) -> np.ndarray:
        """
        Returns the curvature in the direction e1 (first frame direction).
        """
        kappa = self.u_n.split(deepcopy=True)[2]
        alpha_expr = dot(kappa, self.e1_n)
        alpha = project(alpha_expr, self.V)
        return f2n(alpha)

    def get_beta(self) -> np.ndarray:
        """
        Returns the curvature in the direction e2 (second frame direction).
        """
        kappa = self.u_n.split(deepcopy=True)[2]
        beta_expr = dot(kappa, self.e2_n)
        beta = project(beta_expr, self.V)
        return f2n(beta)

    def get_gamma(self) -> np.ndarray:
        """
        Return the twist of the frame about the mid-line.
        """
        gamma = self.u_n.split(deepcopy=True)[5]
        return f2n(gamma)

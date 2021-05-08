import warnings
from typing import Union

import numpy as np

from fenics import *
try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    pass
from ffc.quadrature.deprecation \
    import QuadratureRepresentationDeprecationWarning

set_log_level(100)
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

# global geometry helpers
dxL = dx(scheme='vertex', degree=1,
         metadata={'representation': 'quadrature',
                   'degree': 1})


def grad(function): return Dx(function, 0)


class Geometry:
    """
    Holder for geometric information derived from solution.
    """

    def __init__(self, u, V3):
        x, y, kappa, m, z, gamma, p = split(u)

        # position
        self.x = x
        # vector curvature
        self.kappa = kappa
        # twist
        self.gamma = gamma
        # tangential angular momentum
        self.m = m

        # length element
        self.mu = sqrt(inner(grad(x), grad(x)))
        # tangent vector (element-wise)
        self.tau = Dx(x, 0) / self.mu
        # tangent vector (vertex-wise)
        tauv_tilde = project(self.tau, V3)
        self.tauv = tauv_tilde / sqrt(dot(tauv_tilde, tauv_tilde))

        # cross product of tau and kappa
        self.tau_cross_kappa = cross(self.tau, self.kappa)
        # cross product of tauv and kappa
        self.tauv_cross_kappa = cross(self.tauv, self.kappa)

        # outer product
        self.tautau = outer(self.tau, self.tau)

        # projection away from tangent operator
        self.P = Identity(3) - self.tautau

        # projection away from tangent operator
        self.Pv = Identity(3) - outer(self.tauv, self.tauv)


class Worm:
    """
    Class for holding all the information about the worm geometry and to
    update the geometry.
    """

    def __init__(self, N: int, dt: float):
        # discretisation parameters
        self.N = N
        self.dt = dt

        # set parameters
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

        # set up function spaces
        self._init_spaces()

        # initialise values
        self.set_initial_values()

    # ---------- init functions ----------

    def _init_spaces(self):
        """
        Set up function spaces which hold solutions
        """
        # mesh and spaces
        mesh = UnitIntervalMesh(self.N)
        P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P0 = FiniteElement("DP", mesh.ufl_cell(), 0)
        P1_3 = MixedElement([P1] * 3)

        V = FunctionSpace(mesh, P1)
        V3 = FunctionSpace(mesh, P1_3)
        Q = FunctionSpace(mesh, P0)

        self.V = V
        self.V3 = V3
        self.Q = Q
        self.VV = [V3, V3, V3, V, Q, Q, Q]
        self.W = FunctionSpace(mesh,
                               MixedElement([P1_3, P1_3, P1_3,
                                             P1, P0, P0, P0]))

    def set_initial_values(
            self,
            x0: Union[np.ndarray, Expression, Function] = None,
            e10: Union[np.ndarray, Expression, Function] = None,
            e20: Union[np.ndarray, Expression, Function] = None
    ):
        """
        Set the initial values of the system. kwargs are used to
        set non-default options for x0, e10 and e20.
        """
        V3 = self.VV[0]
        self.t = 0.
        self._set_param('x0', V3, x0)
        self._set_param('e10', V3, e10)
        self._set_param('e20', V3, e20)
        self._compute_initial_values()

    def _compute_initial_values(self):
        """
        Compute the initial conditions
        """
        # find split spaces
        V3 = self.V3
        V = self.V
        Q = self.Q

        # initial position and frame
        x0 = self.x0
        e1_n = self.e10
        e2_n = self.e20

        # initial geometry
        mu0 = sqrt(dot(grad(x0), grad(x0)))

        # set up problem for initial curvature
        kappa_trial = TrialFunction(V3)
        kappa_test = TestFunction(V3)
        F0_kappa = dot(kappa_trial, kappa_test) * mu0 * dxL \
                   + inner(grad(x0), grad(kappa_test)) / mu0 * dx
        a0_kappa, L0_kappa = lhs(F0_kappa), rhs(F0_kappa)

        # solve problem for initial curvature
        kappa0 = Function(V3)
        solve(a0_kappa == L0_kappa, kappa0)
        if np.isnan(kappa0.vector().sum()):
            raise RuntimeError('kappa0 contains NaNs')

        # set up problem for initial twist
        gamma = TrialFunction(Q)
        v = TestFunction(Q)
        F_gamma0 = (gamma - dot(grad(e1_n), e2_n) / mu0) * \
                   v * dx()
        a_gamma0, L_gamma0 = lhs(F_gamma0), rhs(F_gamma0)
        gamma0 = Function(Q)
        solve(a_gamma0 == L_gamma0, gamma0)
        if np.isnan(gamma0.vector().sum()):
            raise RuntimeError('gamma0 contains NaNs')

        # initialize global old solution variable
        fa = FunctionAssigner(self.W, self.VV)
        u_n = Function(self.W)
        fa.assign(u_n, [x0, Function(V3), kappa0, Function(V),
                         Function(Q), gamma0, Function(Q)])

        self.geometry = Geometry(u_n, self.V3)
        self.u_n = u_n
        self.e1_n = e1_n
        self.e2_n = e2_n
        self.mu0 = self.geometry.mu

    # ---------- main algorithm ----------

    def update(
            self,
            alpha_pref: Union[np.ndarray, Expression, Function] = None,
            beta_pref: Union[np.ndarray, Expression, Function] = None,
            gamma_pref: Union[np.ndarray, Expression, Function] = None
    ):
        """
        Update the solution to the next time step.  kwargs are used to
        update forcing functions: alpha_pref, beta_pref and gamma_pref
        """
        self._set_param('alpha_pref', self.V, alpha_pref)
        self._set_param('beta_pref', self.V, beta_pref)
        self._set_param('gamma_pref', self.VV[5], gamma_pref)
        return self._update_solution()

    def _update_solution(self):
        # extract helper space
        V3 = self.VV[0]

        # update time
        dt = self.dt
        self.t += dt

        # get data
        my_alpha = self.alpha_pref
        my_beta = self.beta_pref
        my_gamma = self.gamma_pref

        # extract previous geometry
        geo = self.geometry
        x_n = geo.x
        kappa_n = geo.kappa
        gamma_n = geo.gamma
        mu = geo.mu
        tau = geo.tau
        tauv = geo.tauv
        tautau = geo.tautau
        tau_cross_kappa = geo.tau_cross_kappa
        tauv_cross_kappa = geo.tauv_cross_kappa
        P = geo.P
        Pv = geo.Pv

        # Define variational problem
        u = TrialFunction(self.W)
        v = TestFunction(self.W)

        # Split test and trial functions
        x, y, kappa, m, z, gamma, p = split(u)
        phi_x, phi_y, phi_kappa, phi_m, phi_z, phi_gamma, phi_p = split(v)

        # parameters
        KK = self.K * P + tautau
        K_rot = self.K_rot
        A = self.A
        B = self.B
        C = self.C
        D = self.D

        # Variational form
        F_x = 1.0 / dt * dot(KK * (x - x_n), phi_x) * mu * dx \
              - p * dot(tau, grad(phi_x)) * dx \
              - dot(P * grad(y), grad(phi_x)) / mu * dx \
              - z * dot(tau_cross_kappa, grad(phi_x)) * dx

        F_y = dot(y - A * (kappa - my_alpha * self.e1_n - my_beta * self.e2_n)
                  - B * (Pv * (kappa - kappa_n) / dt - m * tauv_cross_kappa),
                  phi_y) * mu * dx  # TODO wrong measure

        F_w = inner(grad(x), grad(phi_kappa)) / mu * dx \
              + dot(kappa, phi_kappa) * mu * dxL

        F_m = -K_rot * m * phi_m * mu * dx \
              - z * grad(phi_m) * dx \
              + dot(y, tauv_cross_kappa) * phi_m * mu * dx

        F_z = (z - C * (gamma - my_gamma) - D / dt *
               (gamma - gamma_n)) * phi_z * mu * dx()

        F_gamma = 1.0 / dt * (gamma - gamma_n) * phi_gamma * mu * dx() \
                  - grad(m) * phi_gamma * dx() \
                  + dot(tau_cross_kappa, 1.0 / dt * grad(x - x_n)) \
                  * phi_gamma * dx()

        F_p = (dot(tau, grad(x)) - self.mu0) * phi_p * dx

        F = F_x + F_y + F_w + F_m + F_z + F_gamma + F_p
        F_op, L = lhs(F), rhs(F)

        # boundary conditions
        kappa_space = self.W.split()[2]
        # kappa_space = self.W.sub(2)
        kappa_b = project(my_alpha * self.e1_n + my_beta * self.e2_n, V3)
        bc = DirichletBC(kappa_space,
                         kappa_b,
                         lambda x, o: o)

        # compute solution
        u = Function(self.W)
        solve(F_op == L, u, bcs=bc)
        if np.isnan(u.vector().sum()):
            raise RuntimeError('solution u contains NaNs')

        # updated geometry
        new_geo = Geometry(u, self.V3)
        new_tauv = new_geo.tauv
        new_m = new_geo.m
        self.geometry = new_geo

        # update e1, e2
        def rotated_frame(v):
            k = cross(tauv, new_tauv)
            c = dot(tauv, new_tauv)
            varphi = dt * new_m

            tmp = v * c \
                  + cross(k, v) \
                  + dot(v, k) / (1 + c) * k
            ret = tmp * cos(varphi) \
                  + cross(new_tauv, tmp) * sin(varphi) \
                  + dot(tmp, new_tauv) * (1 - cos(varphi)) * new_tauv
            return project(ret, V3)

        e1 = rotated_frame(self.e1_n)
        if np.isnan(e1.vector().sum()):
            raise RuntimeError('solution e1 contains NaNs')
        e2 = rotated_frame(self.e2_n)
        if np.isnan(e2.vector().sum()):
            raise RuntimeError('solution e2 contains NaNs')

        # update old solution
        self.u_n.assign(u)
        self.e1_n = interpolate(e1, V3)
        self.e2_n = interpolate(e2, V3)
        x_n = project(new_geo.x, self.V3)

        return x_n, e1, e2

    # ---------- helpers ----------

    def _set_param(
            self,
            name: str,
            fs: FunctionSpace,
            val: Union[np.ndarray, Expression, Function] = None
    ):
        """
        Sets values or an expression to a model parameter
        """

        # If no value passed, reset to default
        if val is None:
            default = getattr(self, name + '_default')
            var = interpolate(default, fs)

        # If numpy array passed, set these as the function values
        elif isinstance(val, np.ndarray):
            var = Function(fs, name=name)
            self._set_vals_from_numpy(var, val)

        # If an expression is passed, interpolate on the space
        elif isinstance(val, Expression):
            var = interpolate(val, fs)

        # If a function is passed, just assign
        elif isinstance(val, Function):
            var = interpolate(val, val.function_space())

        # Set the variable as an object property
        setattr(self, name, var)

    def _set_vals_from_numpy(self, var: Function, values: np.ndarray):
        """
        Sets the vertex-values (or between-vertex-values) of a variable from a numpy array
        """
        fs = var.function_space()
        dof_maps = self._dof_maps(fs)
        assert values.shape == dof_maps.shape
        n_subspaces = fs.num_sub_spaces()
        vector = var.vector()

        if n_subspaces == 0:
            n_subspaces = 1
            values = values[np.newaxis, :]
            dof_maps = dof_maps[np.newaxis, :]

        for d in range(n_subspaces):
            for i in range(dof_maps.shape[1]):
                vector[dof_maps[d, i]] = values[d, i]

    @staticmethod
    def _dof_maps(fs: FunctionSpace) -> np.ndarray:
        """
        Returns a numpy array for the dof maps of the function space
        """
        n_subspaces = fs.num_sub_spaces()
        if n_subspaces > 0:
            dof_map = [fs.sub(d).dofmap().dofs() for d in range(n_subspaces)]
        else:
            dof_map = fs.dofmap().dofs()
        return np.array(dof_map)

    def _to_ndarray(self, var: Function) -> np.ndarray:
        """
        Returns a numpy array containing the function values
        """
        fs = var.function_space()
        dof_maps = self._dof_maps(fs)
        if fs.num_sub_spaces() == 0:
            arr = np.array([var.vector()[dof_maps[i]]
                            for i in range(dof_maps.shape[0])])
        else:
            arr = np.array([[var.vector()[dof_maps[d][i]]
                             for d in range(dof_maps.shape[0])]
                            for i in range(dof_maps.shape[1])])
        return arr

    # ---------- getters ----------

    def get_x(self) -> np.ndarray:
        """
        Returns the position of mid-line points as a numpy array.
        """
        x = project(self.geometry.x, self.V3)
        return self._to_ndarray(x)

    def get_e1(self) -> np.ndarray:
        """
        Returns the first component of cross section frame.
        """
        return self._to_ndarray(self.e1_n)

    def get_e2(self) -> np.ndarray:
        """
        Returns the second component of cross section frame.
        """
        return self._to_ndarray(self.e2_n)

    def get_alpha(self) -> np.ndarray:
        """
        Returns the curvature in the direction e1 (first frame direction).
        """
        # get alpha as function
        kappa = self.u_n.split(deepcopy=True)[2]
        alpha_expr = dot(kappa, self.e1_n)
        alpha = project(alpha_expr, self.V)
        return self._to_ndarray(alpha)

    def get_beta(self) -> np.ndarray:
        """
        Returns the curvature in the direction e2 (second frame direction).
        """
        # get beta as function
        kappa = self.u_n.split(deepcopy=True)[2]
        beta_expr = dot(kappa, self.e2_n)
        beta = project(beta_expr, self.V)
        return self._to_ndarray(beta)

    def get_gamma(self) -> np.ndarray:
        """
        Return the twist of the frame about the mid-line.
        """
        # get gamma as function
        gamma = self.u_n.split(deepcopy=True)[5]
        return self._to_ndarray(gamma)


################################################
#                   TESTS                      #
################################################

def _make_test_worm(worm_len=10, dt=0.1):
    # initialise worm object with number of points and time step
    return Worm(worm_len - 1, dt)


def _sim_test_worm(worm, alpha_pref=None, beta_pref=None, gamma_pref=None, t=2.0):
    # run simulation loop
    while worm.t < t:
        print(f't={worm.t:.2f}')
        worm.update(
            alpha_pref=alpha_pref,
            beta_pref=beta_pref,
            gamma_pref=gamma_pref
        )


def _print_output(worm):
    # getters for output in physical coordinates
    print(worm.get_x())
    print(worm.get_e1())
    print(worm.get_e2())

    # getters for output in intrinsic coordinates
    print(worm.get_alpha())
    print(worm.get_beta())
    print(worm.get_gamma())


def test_no_ics():
    print('\n\n----test_no_ics')
    N = 10
    worm = _make_test_worm(worm_len=N)
    # option 1: no arguments
    # can call to change reset initial values to defaults, but doesn't require calling otherwise
    # worm.set_initial_values()
    _sim_test_worm(worm, t=1)
    _print_output(worm)


def test_numpy_ics():
    print('\n\n----test_numpy_ics')
    N = 10
    worm = _make_test_worm(worm_len=N)

    # option 2: set initial conditions with three numpy ndarrays
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    x0 = np.zeros((3, N))
    x0[:][0] = np.linspace(0, 1, N, endpoint=True)
    e10 = np.zeros((3, N))
    e10[:][1] = np.ones((N,))
    e20 = np.zeros((3, N))
    e20[:][2] = np.ones((N,))
    worm.set_initial_values(x0=x0, e10=e10, e20=e20)

    _sim_test_worm(worm, t=2)
    _print_output(worm)


def test_expression_ics():
    print('\n\n----test_expression_ics')
    worm = _make_test_worm()

    # option 3: set initial conditions with three expressions
    # Online documentation missing but you can use simple functions like sin,
    # cos, etc. and variables with values passed as kwargs. For spatial
    # coordinate use x[0].
    x0 = Expression(('x[0]', '0', '0'), degree=1)
    e10 = Expression(('0', '1', '0'), degree=1)
    e20 = Expression(('0', '0', '1'), degree=0)
    worm.set_initial_values(x0=x0, e10=e10, e20=e20)

    _sim_test_worm(worm)
    _print_output(worm)


def test_numpy_forcing():
    print('\n\n----test_numpy_forcing')
    N = 10
    worm = _make_test_worm(worm_len=N)

    # create forcing functions (preferred curvatures) using numpy
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    # these examples show the required sizes
    alpha_pref = 2 * np.ones((N,))
    beta_pref = 3 * np.ones((N,))
    gamma_pref = 5 * np.ones((N - 1,))

    _sim_test_worm(
        worm,
        alpha_pref=alpha_pref,
        beta_pref=beta_pref,
        gamma_pref=gamma_pref
    )
    _print_output(worm)


def test_expression_forcing():
    print('\n\n----test_expression_forcing')
    worm = _make_test_worm()

    # create forcing functions (preferred curvatures) using expressions
    # Online documentation missing but you can use simple functions like sin,
    # cos, etc. and variables with values passed as kwargs. For spatial
    # coordinate use x[0].
    alpha_pref = Expression('v', degree=1, v=1)
    beta_pref = Expression('v', degree=1, v=1)
    gamma_pref = Expression('v', degree=0, v=1)

    _sim_test_worm(
        worm,
        alpha_pref=alpha_pref,
        beta_pref=beta_pref,
        gamma_pref=gamma_pref
    )
    _print_output(worm)


if __name__ == '__main__':
    test_no_ics()
    test_numpy_ics()
    test_expression_ics()
    test_numpy_forcing()
    test_expression_forcing()

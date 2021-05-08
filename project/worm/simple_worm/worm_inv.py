from typing import Tuple

from fenics import *
from fenics_adjoint import *

from simple_worm.controls import ControlsFenics, DRIVE_KEYS
from simple_worm.worm import dx, Worm


class WormInv(Worm):
    """
    Extends the forward model to provide an inverse solver.
    """

    def __init__(
            self,
            N: int,
            dt: float,
            reg_weights: dict = {},
            inverse_opt_max_iter: int = 4,
            inverse_opt_tol: float = 1e-8,
            quiet: bool = False
    ):
        super().__init__(
            N=N,
            dt=dt,
            quiet=quiet
        )

        # Inverse optimisation parameters
        self.inverse_opt_ctrls = True
        self.inverse_opt_ics = True
        self.reg_weights = reg_weights  # how much regularisation to apply when optimising the controls
        self.inverse_opt_max_iter = inverse_opt_max_iter
        self.inverse_opt_tol = inverse_opt_tol

    def initialise(
            self,
            x0: Function = None,
            C: ControlsFenics = None,
            n_timesteps: int = 1,
    ) -> Tuple[Function, ControlsFenics]:
        """
        Clear the adjoint tape and reset the state.
        """
        tape = get_working_tape()
        tape.clear_tape()
        return super().initialise(x0, C, n_timesteps)

    # ---------- Main methods ----------

    def solve_both(
            self,
            T: float,
            x0: Function,
            C: ControlsFenics,
            X_target: Function
    ) -> Tuple[Function, Function, Function, ControlsFenics]:
        """
        Run the forward model for T seconds and then solve the inverse problem.
        """
        # Solve forwards
        X, E1, E2 = super().solve(T, x0, C)

        # Solve inverse
        C_opt = self.solve_inverse(T, C, X, X_target)

        return X, E1, E2, C_opt

    def solve_inverse(
            self,
            T: float,
            C: ControlsFenics,
            X: Function,
            X_target: Function
    ) -> ControlsFenics:
        """
        Solve the inverse problem.
        """
        n_timesteps = int(T / self.dt)
        assert len(X) == len(X_target) == n_timesteps, 'X or X_target incorrect size.'
        self._print('Solve inverse')

        # Data loss
        L_data = 0
        for t in range(n_timesteps):
            # Implement a trapezoidal rule
            if t == n_timesteps - 1:
                weight = 0.5
            else:
                weight = 1
            L_data += weight * self.dt * assemble((X[t] - X_target[t])**2 * dx)

        # Build the functional to minimise
        ctrls_ic = [Control(C.e10), Control(C.e20)]
        ctrls_prefs = []
        m = []

        # todo...
        # # Add soft-penalty to ics having magnitude away from 1
        # reg_terms.append((1-dot(e10, e10))**2*dx)
        # reg_terms.append((1-dot(e20, e20))**2*dx)
        #
        # # Add smoothness regularisation on ics
        # reg_terms.append(inner(grad(e10), grad(e10))*dx)
        # reg_terms.append(inner(grad(e20), grad(e20))*dx)

        # Set up regularisation losses and weightings, defaulting to 0 everywhere
        reg_weights = {}
        reg_losses = {}
        for k in ['L2', 'grad_t', 'grad_x']:
            reg_weights[k] = {}
            reg_losses[k] = {}
            for abg in DRIVE_KEYS:
                w = 0
                if k in self.reg_weights and abg in self.reg_weights[k]:
                    w = self.reg_weights[k][abg]
                reg_weights[k][abg] = w
                reg_losses[k][abg] = 0

        for k, abg in {'alpha': C.alpha, 'beta': C.beta, 'gamma': C.gamma}.items():
            ctrls_prefs.extend([Control(c) for c in abg])
            L2_abg = 0
            grad_t = 0
            grad_x = 0
            for t in range(n_timesteps):
                mu = 1  # todo: sqrt(dot(grad(X[t]), grad(X[t])))

                # L2 penalty - smaller forcings are preferable
                L2_abg += abg[t]**2 * mu * self.dt * dx

                # Smoothing in time
                if t < n_timesteps - 1:
                    grad_t += (abg[t + 1] - abg[t])**2 * mu / self.dt * dx

                # Smoothing in space
                if k != 'gamma':
                    # Can't take gradient of gamma -- pw-constant!
                    grad_x += grad(abg[t])**2 / mu * self.dt * dx

            reg_losses['L2'][k] = L2_abg
            reg_losses['grad_t'][k] = grad_t
            reg_losses['grad_x'][k] = grad_x

        # Sum up the regularisation terms
        regularisation = 0
        for rk, rv in reg_losses.items():
            for fk, fv in rv.items():
                regularisation += Constant(reg_weights[rk][fk]) * fv

        if self.inverse_opt_ics:
            m.extend(ctrls_ic)
        if self.inverse_opt_ctrls:
            m.extend(ctrls_prefs)

        # Build the reduced functional
        J = L_data + assemble(regularisation)
        rf = ReducedFunctional(J, m)

        # Minimise functional to find the optimal controls
        opt_inputs = minimize(
            rf,
            method='L-BFGS-B',
            options={
                'maxiter': self.inverse_opt_max_iter,
                'gtol': self.inverse_opt_tol,
                'disp': not self.quiet
            }
        )

        # Split up results
        if self.inverse_opt_ics:
            e10_opt = opt_inputs[0]
            e20_opt = opt_inputs[1]
            opt_inputs = opt_inputs[2:]
        else:
            e10_opt = ctrls_ic[0]
            e20_opt = ctrls_ic[1]

        if self.inverse_opt_ctrls:
            opt_ctrls = opt_inputs
        else:
            opt_ctrls = ctrls_prefs
        alpha_opt = opt_ctrls[0:n_timesteps]
        beta_opt = opt_ctrls[n_timesteps:n_timesteps * 2]
        gamma_opt = opt_ctrls[n_timesteps * 2:]

        return ControlsFenics(
            e10=e10_opt,
            e20=e20_opt,
            alpha=alpha_opt,
            beta=beta_opt,
            gamma=gamma_opt,
        )

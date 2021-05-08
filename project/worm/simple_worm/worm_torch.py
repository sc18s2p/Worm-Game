from multiprocessing import Pool
from typing import Tuple

import torch
import torch.nn as nn

from simple_worm.controls import CONTROL_KEYS
from simple_worm.controls_torch import ControlsTorch, ControlsBatchTorch
from simple_worm.util_torch import calculate_e0_single, f2t, t2f
from simple_worm.worm_inv import WormInv


class DummyContext(object):
    """
    Used to simulate a autograd ctx object when parallel processing.
    """
    pass


class WormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            worm: WormInv,
            x0: torch.Tensor,
            e10: torch.Tensor,
            e20: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            gamma: torch.Tensor,
            calculate_e0: bool = False,
            calculate_inverse: bool = False,
            calculate_X_opt: bool = False,
            X_target: torch.Tensor = None
    ):
        if calculate_inverse:
            assert X_target is not None
        if calculate_X_opt:
            assert calculate_inverse is not None

        # Simulation run time
        T = worm.dt * len(alpha)

        # Store inputs for backwards pass
        C = ControlsTorch(e10, e20, alpha, beta, gamma)
        ctx.C = C

        # Calculate initial frame from initial midline
        if calculate_e0:
            C.e10, C.e20 = calculate_e0_single(worm.N + 1, x0)

        # Convert tensor inputs to fenics variables
        x0f = t2f(x0, fs=worm.V3, name='x0')
        Cf = C.to_fenics(worm)

        if calculate_inverse:
            # Convert targets to fenics variables
            X_target = [t2f(xtt, fs=worm.V3, name=f'xt_t{t}') for t, xtt in enumerate(X_target)]

            # Execute forward and backwards pass in solver
            X, E1, E2, Cf_opt = worm.solve_both(T, x0f, Cf, X_target)

            # If required, run the model forwards again with the optimal controls for comparison
            if calculate_X_opt:
                X_opt, _, _ = worm.solve(T, x0f, Cf_opt)

            # Convert fenics controls to torch tensors
            C_opt = Cf_opt.to_torch()
            ctx.C_opt = C_opt
        else:
            # Dummy controls
            C_opt = ControlsTorch(worm=worm)

            # Only execute forward solver
            X, E1, E2 = worm.solve(T, x0f, Cf)

        # Convert forward model outputs to torch tensors
        X, E1, E2 = f2t(X), f2t(E1), f2t(E2)
        if calculate_X_opt:
            X_opt = f2t(X_opt)
        else:
            X_opt = torch.zeros_like(X)

        return X, E1, E2, *C_opt.parameters(), X_opt

    @staticmethod
    def backward(
            ctx,
            X_grad,
            E1_grad,
            E2_grad,
            e10_grad,
            e20_grad,
            alpha_pref_grad,
            beta_pref_grad,
            gamma_pref_grad,
            X_opt_grad
    ):
        assert hasattr(ctx, 'C_opt'), 'Optimal controls not in ctx, has inverse problem been solved?'

        # Gradients taken simply as the difference with the optimals found by the solver
        x0_grad = None
        e10_grad = ctx.C.e10 - ctx.C_opt.e10
        e20_grad = ctx.C.e20 - ctx.C_opt.e20
        alpha_pref_grad = ctx.C.alpha - ctx.C_opt.alpha
        beta_pref_grad = ctx.C.beta - ctx.C_opt.beta
        gamma_pref_grad = ctx.C.gamma - ctx.C_opt.gamma

        # Return gradients wrt controls
        return None, x0_grad, e10_grad, e20_grad, alpha_pref_grad, beta_pref_grad, gamma_pref_grad, None, None, None, None


class WormFunctionParallel(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            worm_mod: 'WormModule',
            x0: torch.Tensor,
            e10: torch.Tensor,
            e20: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            gamma: torch.Tensor,
            calculate_e0=False,
            calculate_inverse=False,
            calculate_X_opt=False,
            X_target=None
    ):
        print(f'Starting simulation pool (n_workers={worm_mod.n_workers})')

        # Pass arguments to recreate an identical worm module in the separate processes, more robust than sharing or serialising
        worm = worm_mod.worm_solver
        worm_args = {
            'N': worm.N,
            'dt': worm.dt,
            'reg_weights': worm.reg_weights,
            'inverse_opt_max_iter': worm.inverse_opt_max_iter,
            'inverse_opt_tol': worm.inverse_opt_tol,
            'quiet': True
        }
        C = ControlsBatchTorch(e10, e20, alpha, beta, gamma)
        with Pool(worm_mod.n_workers) as pool:
            args = []
            for i in range(worm_mod.batch_size):
                args_i = {
                    'batch_size': worm_mod.batch_size,
                    'worm_args': worm_args,
                    'x0': x0[i],
                    'C': C[i],
                    'calculate_e0': calculate_e0,
                    'calculate_inverse': calculate_inverse,
                    'calculate_X_opt': calculate_X_opt,
                    'X_target': None if not calculate_inverse else X_target[i],
                }
                args.append(args_i)

            outs = pool.map(
                WormFunctionParallel.solve_single,
                [[i, args[i]] for i in range(worm_mod.batch_size)]
            )
        X, E1, E2, C_opt, X_opt = zip(*outs)

        # Stack outputs
        X, E1, E2 = torch.stack(X), torch.stack(E1), torch.stack(E2)
        if calculate_X_opt:
            X_opt = torch.stack(X_opt)
        else:
            X_opt = torch.zeros_like(X)

        if calculate_inverse:
            # Stack inputs
            C_opt = ControlsBatchTorch.from_list(C_opt)

            # Store in ctx
            ctx.C = C
            ctx.C_opt = C_opt
        else:
            # Dummy controls
            C_opt = ControlsBatchTorch(worm=worm, batch_size=worm_mod.batch_size)

        return X, E1, E2, *C_opt.parameters(), X_opt

    @staticmethod
    def solve_single(args):
        batch_idx = args[0]
        fn_args = args[1]
        batch_size = fn_args['batch_size']
        print(f'#{batch_idx + 1}/{batch_size} started')
        worm = WormInv(**fn_args['worm_args'])
        ctx = DummyContext()

        # Solve using single-process implementation
        X, E1, E2, e10_opt, e20_opt, alpha_opt, beta_opt, gamma_opt, X_opt = \
            WormFunction.forward(
                ctx,
                worm,
                x0=fn_args['x0'],
                **fn_args['C'].parameters(as_dict=True),
                calculate_e0=fn_args['calculate_e0'],
                calculate_inverse=fn_args['calculate_inverse'],
                calculate_X_opt=fn_args['calculate_X_opt'],
                X_target=fn_args['X_target'],
            )
        C_opt = ControlsTorch(e10_opt, e20_opt, alpha_opt, beta_opt, gamma_opt)

        print(f'#{batch_idx + 1}/{batch_size} finished')

        return X, E1, E2, C_opt, X_opt

    @staticmethod
    def backward(*args):
        return WormFunction.backward(*args)


class WormModule(nn.Module):
    def __init__(
            self,
            N: int,
            dt: float,
            batch_size: int,
            reg_weights: dict = {},
            inverse_opt_max_iter=4,
            inverse_opt_tol=1e-8,
            parallel: bool = False,
            n_workers: int = 2,
            quiet=False
    ):
        super().__init__()

        # Initialise worm solver
        self.worm_solver = WormInv(
            N=N,
            dt=dt,
            reg_weights=reg_weights,
            inverse_opt_max_iter=inverse_opt_max_iter,
            inverse_opt_tol=inverse_opt_tol,
            quiet=quiet
        )

        # Process batches in parallel
        self.batch_size = batch_size
        self.parallel = parallel
        self.n_workers = n_workers

    def forward(
            self,
            x0: torch.Tensor,
            C: ControlsBatchTorch,
            calculate_e0: bool = False,
            calculate_inverse: bool = False,
            calculate_X_opt: bool = False,
            X_target: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ControlsBatchTorch, torch.Tensor]:
        if self.parallel:
            args = self, x0, *C.parameters(), calculate_e0, calculate_inverse, calculate_X_opt
            if calculate_inverse:
                args += (X_target,)
            outs = WormFunctionParallel.apply(*args)
        else:
            outs = []
            for i in range(self.batch_size):
                args = self.worm_solver, x0[i], *C[i].parameters(), calculate_e0, calculate_inverse, calculate_X_opt
                if calculate_inverse:
                    args += (X_target[i],)
                out = WormFunction.apply(*args)
                outs.append(out)

            # Rearrange by output index and stack over number of input sets
            outs = tuple(torch.stack(out) for out in zip(*outs))

        # Prepare outputs
        X = outs[0]
        E1 = outs[1]
        E2 = outs[2]

        # Set the outputs as requiring grad if the inputs require it
        if C.requires_grad():
            X.requires_grad_(True)
            E1.requires_grad_(True)
            E2.requires_grad_(True)
        ret = X, E1, E2

        if calculate_inverse:
            C_opts = {k: outs[3 + i] for i, k in enumerate(CONTROL_KEYS)}
            ret += (ControlsBatchTorch(**C_opts),)

        if calculate_X_opt:
            XO = outs[-1]
            ret += (XO,)

        return ret

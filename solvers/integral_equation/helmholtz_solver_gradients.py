# helmholtz_solver_gradients.py: adjoint-state gradient computation and the
# pytorch-autograd wrapper for HelmholtzSolverBicgstab (see
# helmholtz_solver_bicgstab.py).
#
# This is a behavior-preserving split of HelmholtzSolverDifferentiable.py's
# Helmholtz_gradient_exterior method and PytorchPDESolver class: gmres support
# is dropped (bicgstab only), and the gradient computation is expressed as a
# standalone function that takes a HelmholtzSolverBicgstab instance as its
# first argument (rather than being a bound method), matching the existing
# pattern of PytorchPDESolver.apply(q, solver_obj, config) taking the solver
# as a parameter.
#
# This file contains:
# - helmholtz_gradient_exterior (adjoint state method)
# - PytorchPDESolver, a pytorch autograd Function wrapping the forward and
#   adjoint solves -- kept for future use even though nothing in the current
#   pipeline needs to backprop through the solver right now.

import logging
from typing import Dict, Tuple

import numpy as np
import torch

from solvers.integral_equation.helmholtz_solver_bicgstab import (
    HelmholtzSolverBicgstab,
    DEFAULT_RTOL,
    TORCH_CDTYPE,
    to_cfloat,
    _check_linsys_solver,
    _dump_nan_debug,
)
from solvers.integral_equation.bicgstab_batch import bicgstab_batch

# Keys in a PytorchPDESolver config dict that are specific to one direction
# (forward vs. adjoint) and must not leak into the other's solve kwargs.
_FORWARD_ONLY_CONFIG_KEYS = {"fwd_linsys_solver"}
_ADJOINT_ONLY_CONFIG_KEYS = {
    "adj_linsys_solver", "adj_batch_size", "adj_rtol", "adj_max_iter",
    "adj_use_half_grid", "adj_half_grid_tol_ratio",
}


def _cfg(config: Dict, key: str, fallback_key: str, default):
    """config[key] if present, else config[fallback_key], else default. Used
    for the adjoint settings that default to the corresponding forward
    setting (e.g. adj_rtol falls back to rtol)."""
    return config.get(key, config.get(fallback_key, default))


def _half_grid_lambda_init(
    solver: HelmholtzSolverBicgstab,
    qflat_hg: torch.Tensor,
    rhs_hg_chunk: torch.Tensor,
    N_x: int,
    k: float,
    rtol_hg: float,
    max_iter: int,
    restart: int,
    verbose: bool,
    convergence_by_dir: bool,
    report_status: bool,
    debug_arrays: Dict,
    **kwargs,
) -> torch.Tensor | None:
    """Multigrid warm-start helper for the adjoint solve -- the adjoint-system
    analogue of HelmholtzSolverBicgstab._half_grid_sigma_init. Solves the
    adjoint system on the half-resolution grid and interpolates the result up
    to a lambda initializer for the full-grid adjoint solve. Returns None
    -- instead of raising -- if the coarse solve itself produced a NaN, after
    dumping debug info to scratch_dir/.

    Args:
        qflat_hg (torch.Tensor): the scattering potential downsampled to the
            half grid, shape ((N_x//2)**2, 1)
        rhs_hg_chunk (torch.Tensor): this chunk's adjoint-system RHS,
            downsampled to the half grid; shape ((N_x//2)**2, j_range)
        debug_arrays (Dict): extra arrays (q, rhs, grad_output, ...) to
            include in the NaN debug dump, if one is triggered
    """
    def hg_adjoint_matvec(x: torch.Tensor) -> torch.Tensor:
        """Apply (I-diag(q)k^2 G_k)* (adjoint) to vector x (half-grid version)"""
        in_shape = x.shape
        x_shaped = x.reshape((N_x // 2) ** 2, -1)
        g_out = solver.half_grid_solver._G_apply(torch.multiply(qflat_hg, x_shaped), adj=True)
        term2 = (k**2) * g_out
        return (x_shaped + to_cfloat(term2)).reshape(in_shape)

    j_range = rhs_hg_chunk.shape[0]  # rhs_hg_chunk has shape (j_range, (N_x//2)**2)
    lam_init_hg, out_info = bicgstab_batch(
        hg_adjoint_matvec,
        rhs_hg_chunk.T.unsqueeze(0), # shape: (1, (N_x//2)**2, j_range)
        rtol=rtol_hg,
        maxiter=max_iter,
        restart=restart,
        verbose=verbose,
        convergence_by_dir=convergence_by_dir,
        log_resid_norm=kwargs.get("log_resid_norm", True),
    )
    if report_status:
        status = "optimal" if out_info["optimal"] else "not optimal"
        logging.debug(f"bicgstab exited after {out_info['niter']} iterations with status {status}")
        print(f"bicgstab exited after {out_info['niter']} iterations with status {status}")

    lam_init_hg = lam_init_hg.reshape(-1, j_range).T # flip to (j_range, (N_x//2)**2)
    lam_init_tmp = solver.half_to_full_grid(
        lam_init_hg.reshape(j_range, N_x//2, N_x//2)
    ).reshape(j_range, N_x**2).to(solver.device)
    lam_init = lam_init_tmp.T.unsqueeze(0) # shape: (1, N_x**2, j_range)

    if torch.any(torch.isnan(lam_init)):
        debug_fp = _dump_nan_debug(
            "debug_adj_hg_solver_nan",
            q_flat_hg=qflat_hg,
            rhs_hg=rhs_hg_chunk,
            lam_init_hg=lam_init_hg,
            lam_init_fg=lam_init,
            **debug_arrays,
        )
        msg = (
            f"BiCGSTAB's coarse grid lam_init contains a NaN! Discarding "
            f"and saving debug outputs to {debug_fp}."
        )
        logging.info(msg)
        print(msg)
        return None

    return lam_init


def helmholtz_gradient_exterior(
    solver: HelmholtzSolverBicgstab,
    q: torch.Tensor,
    grad_p_h: torch.Tensor,
    Gk_sigma: torch.Tensor,
    batch_size: int=100,
    source_dirs: np.ndarray=None,
    usin: torch.Tensor=None,
    adj_linsys_solver: str=None,
    rtol: float=DEFAULT_RTOL,
    max_iter: int=100,
    restart: int = 50,
    verbose: bool=False,
    error_unless_converged: bool=False,
    adj_use_half_grid: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Uses the Adjoint State Method to differentiate through solution to the
    Helmholtz Equation on the exterior ring (i.e., the Helmholtz_solve_exterior
    method of HelmholtzSolverBicgstab).
    Assumes that the output, p=F_k[q], is fed into some function h(p) that outputs a real scalar
    and that we are given the gradient of h(p) with respect to p.
    This function calculates the gradient of h(F_k[q]) with respect to q.

    The PDE solve process involves solving some equation
        A(q) σ_s = b_s(q), s=1,2,...,N_s
    where different sources have different RHS values b_s(q), and the output is
        p := F_k[q]
        (F_k[q])_s := (M_k σ_s) for s=1,2,...,N_s
    as the far-field measurements of the scattered wave for a given right hand side b_s(q).
    This function handles the different sources in batches.

    Computes the gradient using the adjoint state method:
    1. Calculate (to find the the RHS)
        dh/dσ_s = (dh/dp) (dp/dσ_s)
                = (dh/dp) (M_k)_s
        grad_{σ_s} h(p) = ((M_k)_s)^* (grad_p h(p))
    2. Solve the adjoint system
        A(q)^* λ_s = -(dh/dσ_s)^* for s=1,2,...,N_s
                   = - grad_{σ_s} h(p)
    3. Output the gradient
        grad_q h(F_k[q]) = (dg/dq)^* λ
                      = k^2 [conj(G_k σ_s + u_s^{in}) (*) λ_s]_s  # (*) indicates element-wise multiplication
                      = k^2 [conj(u_s^{tot}) (*) λ_s]_s
    where u_s^{in} is the incident wave with source direction s evaluated at each point on the grid
    Note that this operation corresponds to applying the adjoint of the forward model's derivative:
        grad_q h(p(q)) = DF_k[q]^* grad_p h(p) # this is from the chain rule
    and
        DF_k[q]^* xi = k^2 sum_s [conj(u_s^{tot}) (*) (-(A(q)^*)^{-1} M_k^* (xi)_s )]

    Parameters:
        solver (HelmholtzSolverBicgstab): the solver instance whose forward
            solve produced q/Gk_sigma; supplies device/frequency/G_fft/etc.
        q (torch.Tensor, shape: (N_x, N_x) or (N_x**2)): a single scattering potential
        grad_p_h (torch.Tensor, shape: (N_r, N_s)): gradient of h(p) with respect to h
            This should be supplied by Pytorch's autograd
            Alternately, this could be some xi to apply DF_k[q]^* to
        Gk_sigma (torch.Tensor, shape: (N_s, N_x**2)): Gk applied to sigma from the forward pass
            Required by the adjoint state method
        batch_size (int): number of right hand sides to process at once
        source_dirs (np.ndarray, optional, shape: (N_s,)): the source direction angles to use
            defaults to the solver object's usual sources
        usin (torch.Tensor, optional, shape: (N_s, N_x**2)): incident wave for source direction s
            Will be recomputed if passed as None
        adj_linsys_solver (string): retained for call-site compatibility; must be "bicgstab" or None
            (gmres was dropped in this interface)
        rtol (float=DEFAULT_RTOL): relative tolerance for the residual of the linear system solve
        max_iter (int): maximum number of iterations for the linear system
        verbose (bool): whether to output logging information during the linear system solves
        error_unless_converged (bool): whether to throw an error if the system has not
            converged to the expected relative tolerance
        adj_use_half_grid (bool): whether to warm-start the adjoint system solve using the
            half-grid system's solution as an initialization point

    Returns:
        grad_q_h (torch.Tensor): gradient of h with respect to q
    """
    _check_linsys_solver(adj_linsys_solver)

    N_x = q.shape[-1]
    k = solver.frequency
    qflat = q.clone().flatten().unsqueeze(-1)
    if source_dirs is None:
        source_dirs = solver.source_dirs
    if usin is None:
        usin = solver._get_uin(torch.tensor(source_dirs, device=solver.device))
    report_status = kwargs.get("report_status", False)
    convergence_by_dir = kwargs.get("convergence_by_dir", False)

    # 1. Calculate grad_{dσ_s} h, which goes into the RHS of the adjoint system
    # Expected operation is dh/{dσ_s} = (dh/d{p_s}) (M_k)_s
    # Or, grad_{σ_s} h = -(M_k)_s^* grad_{p_s} h(p) for each source direction s
    # But, due to the way the arrays are shaped, the solver.exterior_greens_function
    # array is applied on the other side than would be expected
    grad_sig_h = grad_p_h @ solver.exterior_greens_function.conj()
    rhs = -grad_sig_h

    # 2. Set up and solve the adjoint linear system of equations
    # 2a. Prepare the operators for use in the linear systems
    def adjoint_matvec_from_torch(x: torch.Tensor) -> torch.Tensor:
        """Apply (I-diag(q)k^2 G_k)* (adjoint) to vector x"""
        in_shape = x.shape
        x_shaped = x.reshape(N_x*N_x, -1)
        g_out  = solver._G_apply(torch.multiply(qflat, x_shaped), adj=True)
        term2 = (k**2) * g_out
        y = x_shaped + to_cfloat(term2)
        return y.reshape(in_shape)
    # Half-grid versions
    rhs_hg = rhs.reshape(-1, N_x, N_x)[..., ::2, ::2].reshape(-1, (N_x//2)**2).to(solver.device)
    if adj_use_half_grid:
        half_grid_tol_ratio = kwargs.get("half_grid_tol_ratio", 0.5)
        rtol_hg = rtol * half_grid_tol_ratio
        qflat_hg = qflat.reshape(N_x, N_x, -1)[::2, ::2].reshape((N_x//2)**2, -1).to(solver.device)

    # 2b. Solve the adjoint system using these adjoint matvec operators
    lam_list = []
    for j in range(0, len(source_dirs), batch_size):
        j_upper = min(j + batch_size, N_x)
        j_range = j_upper - j
        js = slice(j,j_upper)
        directions = source_dirs[js]

        # Solve the system using our custom bicgstab_batch function
        lam_init = None
        if adj_use_half_grid:
            lam_init = _half_grid_lambda_init(
                solver, qflat_hg, rhs_hg[js], N_x, k, rtol_hg, max_iter, restart,
                verbose, convergence_by_dir, report_status,
                debug_arrays=dict(
                    q=qflat.reshape(N_x, N_x), q_flat_fg=qflat, rhs_fg=rhs, grad_output=grad_p_h,
                ),
                log_resid_norm=kwargs.get("log_resid_norm", True),
            )

        # After the multi-grid initialization (if used), proceed with the full grid
        lam_chunk, out_info = bicgstab_batch(
            adjoint_matvec_from_torch,
            rhs[js].T.unsqueeze(0), # this is the right way to feed the RHS into bicgstab_batch
            X0=lam_init,
            rtol=rtol,
            maxiter=max_iter,
            restart=restart,
            verbose=verbose,
            convergence_by_dir=convergence_by_dir,
            log_resid_norm=kwargs.get("log_resid_norm", True),
        )

        lam_chunk = lam_chunk.squeeze(0).T
        if report_status:
            logging.info(f"bicgstab exited after {out_info['niter']} iterations with "
                         f"status {'optimal' if out_info['optimal'] else 'not optimal'}")
            print(f"bicgstab exited after {out_info['niter']} iterations with "
                  f"status {'optimal' if out_info['optimal'] else 'not optimal'}")
        if error_unless_converged and out_info["optimal"] != True:
            raise RuntimeError(f"BiCGSTAB failed to converge. Run info: {out_info}")

        lam_list.append(lam_chunk)

    # 2c. Collect the outputs into a single lambda object
    lam_full = torch.concatenate(lam_list, dim=0)

    # 3. Apply dg/dq
    # Note: casts to real because q and h(p(q)) are assumed to both be constrained to real values
    # so the gradient should also be real
    grad_q_h_tmp = k**2 * torch.multiply((Gk_sigma + usin).conj(), lam_full).sum(0).real
    grad_q_h = grad_q_h_tmp.reshape(N_x, N_x).detach().clone()

    return grad_q_h


class PytorchPDESolver(torch.autograd.Function):
    """Wrapper for the PDE solver that enables both forward and backwards passes
    Use with PytorchPDESolver.apply(...)
    Example code:
        # apply
        d_rs_fwd = PytorchPDESolver.apply(q, solver_obj, config={...})
        loss_val_module = loss_fn(d_rs_fwd)
        loss_val_module.backward()
        q_grad_asm = q.grad.clone().detach()
    Parameters:
        ctx: a context object managed internally by PyTorch's autodiff system
        q (torch.Tensor): a single scattering potential
        so (HelmholtzSolverBicgstab): the solver object
        config (dict): a dictionary containing the configuration information; see the fields next:
    Config settings (within the config dictionary)
        batch_size (int): number of right-hand-sides for the solver to process at once
        rtol (float): relative tolerance for the forward system
        max_iter (int): maximum number of iterations for the linear system solves
        verbose (bool): whether to output logging information
        error_unless_converged (bool): whether to throw an error if a linear system solve
            does not converge to the requested tolerance
        restart (int): interval for when to restart the linear system solvers
        convergence_by_dir (bool): option for bicgstab for whether to stop working on each
            direction based on their convergence criteria independently
        use_half_grid (bool): whether to choose the initializer by solving the problem on a smaller grid first
        half_grid_tol_ratio (float): choose the tolerance for the half-grid solve relative to the full-grid solve
        # adjoint versions; if not specified, they default to the forward versions
        adj_batch_size (int, optional): number of right-hand-sides for the solver to process at once
        adj_rtol (float, optional): relative tolerance for the adjoint system
        adj_max_iter (int, optional): maximum number of iterations for the adjoint linear system solves
        adj_use_half_grid (bool): whether to choose the initializer by solving the problem on a smaller grid first
        adj_half_grid_tol_ratio (float): choose the tolerance for the half-grid solve relative to the full-grid solve
    """
    @staticmethod
    def forward(ctx, q: torch.Tensor, so: HelmholtzSolverBicgstab, config: Dict=None) -> torch.Tensor:
        """Apply the solver's forward direction
        See the class's doc-string for information on the arguments
        """
        config = config if config is not None else dict()
        # Solve the PDE for d_rs (the far-field solution) and sigma (the intermediate value)
        fwd_args = {
            # Default values
            "batch_size": config.get("batch_size", 100),
            "rtol": config.get("rtol", DEFAULT_RTOL),
            "max_iter": config.get("max_iter", 100),
            "error_unless_converged": config.get("error_unless_converged", False),
            "restart": config.get("restart", 50),
            "convergence_by_dir": config.get("convergence_by_dir", True),
            # Config from arguments; can override these values (excluding
            # keys that only make sense for the adjoint solve)
            **{k: v for k, v in config.items()
               if k not in _FORWARD_ONLY_CONFIG_KEYS and k not in _ADJOINT_ONLY_CONFIG_KEYS},
        }

        d_rs, sigma = so.Helmholtz_solve_exterior_batched(
            q,
            return_sigma=True,
            return_as_torch=True,
            **fwd_args,
        )
        # Also prepare Gk(sigma) for the next step
        Gk_sigma = so._G_apply(sigma.T).T

        if torch.any(torch.isnan(sigma)) or torch.any(torch.isnan(d_rs)):
            uin = so._get_uin(torch.tensor(so.source_dirs, device=so.device))
            rhs = so._get_b(uin, q)
            debug_fp = _dump_nan_debug(
                "debug_fwd_solver_nan",
                q=q,
                rhs=rhs,
                sigma=sigma,
                Gk_sigma=sigma,
                d_rs=d_rs,
            )
            msg = (
                f"NaN encountered in the output of the forward solver. "
                f"See {debug_fp} file for the inputs/outputs causing this error."
            )
            print(msg)
            logging.info(msg)
            raise RuntimeError(msg)

        # Save the relevant quantities to the context object
        ctx.my_config = {**config}
        ctx.so_frequency = so.frequency
        ctx.save_for_backward(q, Gk_sigma)
        ctx.so = so
        return d_rs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply the solver's backward/adjoint solve to find the gradient
        See the class's doc-string for information on the arguments
        """
        # Load the relevant values from the context object
        q, Gk_sigma = ctx.saved_tensors
        so = ctx.so
        config = ctx.my_config

        adj_args = {
            # Config from the forward call, minus keys that don't apply to
            # the adjoint solve (or that are superseded by the adj_* values
            # computed just below)
            **{k: v for k, v in config.items() if k not in _FORWARD_ONLY_CONFIG_KEYS
               and k not in _ADJOINT_ONLY_CONFIG_KEYS},
            # Adjoint settings, each falling back to its forward counterpart
            # if not explicitly overridden (e.g. adj_rtol defaults to rtol)
            "batch_size": _cfg(config, "adj_batch_size", "batch_size", 100),
            "rtol": _cfg(config, "adj_rtol", "rtol", DEFAULT_RTOL),
            "max_iter": _cfg(config, "adj_max_iter", "max_iter", 100),
            "adj_use_half_grid": _cfg(config, "adj_use_half_grid", "use_half_grid", False),
            "adj_half_grid_tol_ratio": _cfg(config, "adj_half_grid_tol_ratio", "half_grid_tol_ratio", False),
            "restart": config.get("restart", 50),
            "verbose": config.get("verbose", False),
            "error_unless_converged": config.get("error_unless_converged", False),
            "convergence_by_dir": config.get("convergence_by_dir", True),
        }

        # Use the adjoint state method to calculate the gradient for this layer
        grad_q = helmholtz_gradient_exterior(
            so,
            q,
            grad_p_h=grad_output,
            Gk_sigma=Gk_sigma,
            **adj_args,
        )
        if torch.any(torch.isnan(grad_q)):
            debug_fp = _dump_nan_debug(
                "debug_adj_solver_nan",
                q=q,
                Gk_sigma=Gk_sigma,
                grad_output=grad_output,
                grad_q=grad_q,
            )
            msg = (
                f"NaN encountered in the output of the adjoint state method. "
                f"See {debug_fp} file for the inputs/outputs causing this error."
            )
            print(msg)
            logging.info(msg)
            raise RuntimeError(msg)

        return grad_q, None, None

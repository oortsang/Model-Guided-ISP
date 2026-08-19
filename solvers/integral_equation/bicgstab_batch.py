# bicgstab_batch.py: contains a pytorch implementation of bicgstab that operates on
# multiple right-hand-sides at the same time. In principle it can also handle multiple
# operators, though some of the features have not been tested with this setting.



from typing import Callable, Tuple, Dict, Any
import torch
import time
import logging
import numpy as np

def bicgstab_batch(
    A_bmm: Callable,
    B: torch.Tensor,
    # K_1_inv_bmm: Callable = None,
    # M_bmm: torch.Tensor = None,
    X0: torch.Tensor = None,
    rtol: float = 1e-03,
    atol: float = 0.0,
    maxiter: int = None,
    verbose: bool = False,
    convergence_by_dir: bool = False,
    restart: int = None,
    restart_tol: float = 0,
    breakpoint_on_nan: bool = False,
    log_resid_norm: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solves a batch of PD linear systems using the BiCGSTAB algorithm.
    This function solves a batch of linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    So this is batched in 2 different ways: it can be batched for m different RHS vectors for a given A matrix,
    or it can be batched for K different A matrices.

    Notes on bicgstab_batch's behavior:
        - This code is designed to work with autodiff, though in practice the gradients appear to be a bit unstable.
        Better is to calculate the gradients via adjoint state method.
        - In case the residual hits zero, it will get restarted to avoid a div-by-zero error
        - There are still a couple very rare cases when BiCGSTAB encounters a NaN.
        - If convergence_by_dir is set to True, then the code will stop working on right-hand-sides or "directions"
        once they have reached the convergence threshold

    Args:
        A_bmm (Callable): A callable that performs a batch matrix multiply of A and a (K x n x m) matrix.
        B (torch.Tensor): A (K x n x m) matrix representing the right hand sides.
        # K_1_inv_bmm (Callable, optional): A callable that performs a batch matrix multiply for
        #     left preconditioning. Defaults to None.
        X0 (torch.Tensor, optional): Initial guess for X, defaults to M_bmm(B). Defaults to None.
        rtol (float, optional): Relative tolerance for norm of residual. Defaults to 1e-03.
        atol (float, optional): Absolute tolerance for norm of residual. Defaults to 0.0.
        maxiter (int, optional): Maximum number of iterations to perform. Defaults to None.
            If maxiter is set to 0 or None, it will effectively be set to 5*n.
        verbose (bool, optional): Whether or not to print status messages. Defaults to False.
        convergence_by_dir (bool, optional): Whether to stop working on each system independently
            (i.e., corresponding to different source directions)
            NOTE: only tested for a single operator A_i
            If the residual_norm is being logged each iteration, the residual array will still have the full shape,
            even if not all the systems are active
        restart (int): How often to restart all the search directions. If restart=None or 0, there will be no restarts.
        restart_tol (float, optional): the tolerance level under which the quantity
            <R_k, R_tilde> will trigger a reset in R_tilde and P_k
            Currently, the value is not well calibrated (i.e., even restarting at 1e-20 seems too high),
            so it appears to be best to leave it to 0
        breakpoint_on_nan (bool): helper value for debugging purposes. In case the code
            encounters a NaN, it can trigger a pdb breakpoint.
        log_resid_norm (bool): flag for whether to log the residual norms at each iteration
            if set to False, the residual of the final iteration will still be available in the info dictionary
            Time savings are minimal, but usually the residual norm is not required after the fact anyway.
        Extra keyword arguments:
            None so far, but provided to simplify the call interface
    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]: _description_
    """

    # Get shape information and assert the input shapes are correct.
    K, n, m = B.shape
    m_active = m
    # if K_1_inv_bmm is None:
    #     K_1_inv_bmm = lambda x: x

    if X0 is None:
        X0 = torch.zeros_like(B)
    if maxiter is None or (maxiter == 0):
        maxiter = 5 * n
    restart = restart if restart is not None else 0

    assert B.shape == (K, n, m), f"B.shape = {B.shape}, (K, n, m) = {(K, n, m)}"
    assert X0.shape == (K, n, m), f"X0.shape = {X0.shape}, (K, n, m) = {(K, n, m)}"
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    # Initialize the variables for the BiCGSTAB algorithm. I am using the variable names as given in
    # the Wikipedia article. Another reference for the algorithm is the book
    # Matrix Computations by Golub and Van Loan.

    X_k = X0
    R_k = B - A_bmm(X_k)
    R_tilde = R_k.clone()
    P_k = R_k.clone()
    X_full = X_k.clone() # full size; intended to hold the solutions that have reached the desired tolerance

    rho_k = _inner_prod(R_tilde, R_k).unsqueeze(1)

    B_norm = torch.linalg.norm(B, dim=1)
    full_stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

    # (2025-02-22: return instead of throwing an error in case of zero-valued residual)
    if torch.all(rho_k == torch.zeros_like(rho_k)):
        logging.info(f"rho_k is all zeroes. Possible that the initialization X0 is already at the optimum.")
        residual_norm = torch.linalg.norm(R_k, dim=1)
        info = {
            "niter": 0,
            "optimal": bool(torch.all(residual_norm <= full_stopping_matrix)),
            "resid_norm_lst": [residual_norm.detach().cpu().numpy()],
            "stopping_matrix": full_stopping_matrix.detach().cpu().numpy(),
        }
        return X_full, info

    # assert (
    #     rho_k != torch.zeros_like(rho_k)
    # ).any(), "Initializing R_tilde failed. May have initialized at the optimum."

    # Helper functions for later
    any_nan = lambda x: torch.any(torch.isnan(x))
    any_nan_list = lambda *xs: np.any([torch.any(torch.isnan(x)).item() for x in xs])

    full_residual_norm = torch.zeros_like(B_norm)
    stopping_matrix = full_stopping_matrix.clone().detach()
    resid_norm_lst = []

    if verbose:
        use_ratio = not torch.any(stopping_matrix == 0)
        if use_ratio:
            print("%03s | %010s %07s %03s" % ("it", "ratio", "it/s", "m"))
        else:
            print("%03s | %010s %07s %03s" % ("it", "dist", "it/s", "m"))
    optimal = False
    # Use these if convergence_by_dir
    # converged_idcs = torch.zeros((K, 1, m), dtype=bool, device=X0.device) # finished
    full_active_idcs = torch.ones((K, 1, m), dtype=bool, device=X0.device) # what we're working with

    # timer_list = np.zeros(20, dtype=np.double)

    total_loop_time = 0
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        loop_start = time.perf_counter()
        start_iter = time.perf_counter()

        # Update the variables for the BiCGSTAB algorithm
        # y = K_1_inv_bmm(P_k)
        nu_k = A_bmm(P_k)

        # Our objects have shape (K, n, m), so inner products are pointwise multiplication, then
        # summing over the n dimension.

        # alpha should have shape (K, m) but we expand it to (K, 1, m) to allow broadcasting
        # (2025-03-05: occasionally <R_tilde, nu_k> contains a zero value -- maybe consider handling this case?)
        alpha = rho_k / _inner_prod(R_tilde, nu_k).unsqueeze(1)
        H_k = X_k + alpha * P_k
        alpha_nu_k = alpha * nu_k
        S_k = R_k - alpha_nu_k

        # Test whether S_k is close enough to zero for an early exit
        S_norm = torch.norm(S_k, dim=1)
        if k == 1 and (S_norm <= stopping_matrix).all():
            optimal = True
            X_k = H_k
            break

        # Z_k = K_1_inv_bmm(S_k)
        T_k = A_bmm(S_k)
        # K_inv_t = K_1_inv_bmm(T_k)
        # Omega should have shape (K, m) but we expand it to (K, 1, m) to allow broadcasting
        omega = (_inner_prod(T_k, S_k) / _inner_prod(T_k, T_k)).unsqueeze(1)
        X_kp1 = H_k + omega * S_k
        R_kp1 = S_k - omega * T_k
        end_iter = time.perf_counter()

        # Calculate the stopping criterion and check if we are done.
        # Also save the residual norm to the logging info
        residual_norm = torch.linalg.norm(R_kp1, dim=1)
        if log_resid_norm:
            if (convergence_by_dir and m_active < m):
                # Expand from the active dimension to the full dimension
                expanded_residual_norm = full_residual_norm.clone()
                expanded_residual_norm[full_active_idcs.squeeze(1)] = residual_norm.flatten()
                resid_norm_lst.append(expanded_residual_norm.detach().cpu().numpy())
            elif log_resid_norm:
                resid_norm_lst.append(residual_norm.detach().cpu().numpy())

        if verbose:
            print(
                "%03d | %8.4e %7.2f %3d"
                % (
                    k,
                    # torch.max(residual_norm - stopping_matrix),
                    (torch.max(residual_norm / stopping_matrix)
                    if use_ratio else
                    torch.max(residual_norm - stopping_matrix)),
                    1.0 / (end_iter - start_iter),
                    m_active,
                )
            )

        done_idcs  = (residual_norm <= stopping_matrix)
        part_retire_idcs = done_idcs.unsqueeze(1)
        full_convergence_cond = done_idcs.all()

        if full_convergence_cond:
            optimal = True
            if convergence_by_dir:
                X_full[full_active_idcs.expand(-1, n, -1)] = X_kp1.flatten()
            else:
                X_k = X_kp1
            break

        # If not exiting, we need to update rho_k and P_k
        rho_kp1 = _inner_prod(R_tilde, R_kp1).unsqueeze(1)
        beta = (rho_kp1 / rho_k) * (alpha / omega)

        P_kp1 = R_kp1 + beta * (P_k - omega * nu_k)

        # (OOT 2025-02-03) Check rho_kp1 against the restart tolerance...
        # See: https://utminers.utep.edu/xzeng/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln07.pdf
        restart_idcs = (torch.abs(rho_kp1) <= restart_tol)
        if torch.any(restart_idcs):
            # Restart the indices whose rho_kp1 value is too small to avoid NaNs
            ri = restart_idcs.expand(-1, n, -1)
            R_tilde[ri] = R_kp1[ri]
            P_kp1[ri]   = R_kp1[ri]
            rho_kp1[restart_idcs] = _inner_prod(R_tilde, R_kp1).unsqueeze(1)[restart_idcs]
            beta[restart_idcs] = (
                (rho_kp1[restart_idcs] / rho_k[restart_idcs])
                * (alpha[restart_idcs] / omega[restart_idcs])
            )
            if verbose:
                logging.info(f"Notice: restarting indices ({len(torch.argwhere(restart_idcs))} set(s))")

        # Restart every direction if requested
        if (restart != 0 and k % restart == 0):
            R_tilde[:] = R_kp1
            P_kp1[:]   = R_kp1
            rho_kp1[:] = _inner_prod(R_tilde, R_kp1).unsqueeze(1)
            beta[:] = (rho_kp1 / rho_k) * (alpha / omega)
            # logging.info(f"Notice: restarting all indices (iteration k={k})")

        # Log information about any NaNs encountered; this will help track where the errors
        # are introduced to the system
        if any_nan_list(X_k, rho_k, R_k, P_k, nu_k, alpha, H_k, alpha_nu_k, \
            S_k, T_k, omega, rho_kp1, beta, X_kp1, R_kp1, P_kp1):
            _log_bicgstab_nan_debug(k, locals(), breakpoint_on_nan)
            break

        # Shrink the active set of right-hand-sides if necessary
        if convergence_by_dir and torch.any(part_retire_idcs):
            # build the full_retire_idcs object to let us to save the
            # appropriate entries in X_full
            m_prev = torch.sum(full_active_idcs) # previous number of directions
            m_next = m_prev - torch.sum(part_retire_idcs) # next number of directions

            if breakpoint_on_nan:
                import pdb; pdb.set_trace()
            # Expand the indices being "retired," and copy over the corresponding solutions
            # to the X_full tensor, which holds the final outputs
            full_retire_idcs = torch.zeros_like(full_active_idcs)
            full_retire_idcs[full_active_idcs] = part_retire_idcs
            X_full[full_retire_idcs.expand(-1, n, -1)] = X_kp1[part_retire_idcs.expand(-1, n, -1)]
            full_residual_norm[full_retire_idcs.squeeze(1)] = residual_norm[part_retire_idcs.squeeze(1)]

            # Update the other objects while shrinking them
            next_vec_shape = K, n, m_next
            part_save_idcs = torch.logical_not(part_retire_idcs)
            psi_expanded = part_save_idcs.expand(-1, n, -1)
            R_tilde = R_tilde[psi_expanded].reshape(next_vec_shape)
            X_k = X_kp1[psi_expanded].reshape(next_vec_shape)
            R_k = R_kp1[psi_expanded].reshape(next_vec_shape)
            P_k = P_kp1[psi_expanded].reshape(next_vec_shape)
            rho_k = rho_kp1[part_save_idcs].reshape(K, 1, m_next)
            stopping_matrix = stopping_matrix[part_save_idcs.squeeze(1)].reshape(K, m_next)

            # Update full_active_idcs and m_active to track the active set
            full_active_idcs = torch.logical_and(
                full_active_idcs,
                torch.logical_not(full_retire_idcs)
            )
            m_active = m_next
        else:
            # Simply perform the updates if nothing needs to be taken out of the system
            X_k = X_kp1
            R_k = R_kp1
            P_k = P_kp1
            rho_k = rho_kp1
        loop_end = time.perf_counter()
        total_loop_time += loop_end - loop_start
    if not convergence_by_dir:
        X_full = X_k.clone()

    # Add a single entry for the last iteration
    if not log_resid_norm:
        if (convergence_by_dir and m_active < m):
            # Expand from the active dimension to the full dimension
            expanded_residual_norm = full_residual_norm.clone()
            expanded_residual_norm[full_active_idcs.squeeze(1)] = residual_norm.flatten()
            resid_norm_lst.append(expanded_residual_norm.detach().cpu().numpy())
        elif log_resid_norm:
            resid_norm_lst.append(residual_norm.detach().cpu().numpy())

    end = time.perf_counter()
    if verbose:
        if optimal:
            print(
                "Terminated in %d steps (optimal). Took %.3f ms."
                % (k, (end - start) * 1000)
            )
        else:
            print(
                "Terminated in %d steps (reached maxiter). Took %.3f ms."
                % (k, (end - start) * 1000)
            )

    info = {
        "niter": k,
        "optimal": optimal,
        "resid_norm_lst": resid_norm_lst,
        "stopping_matrix": full_stopping_matrix.detach().cpu().numpy(),
    }
    return X_full, info

def _inner_prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Assumes x and y are of shape (K, n, m) and returns the inner product of x and y
    in the n dimension. The result is of shape (K, m)."""
    return torch.sum(x.conj() * y, dim=1)

# Names logged (in this order) by _log_bicgstab_nan_debug when a NaN is found
# mid-iteration -- these are exactly bicgstab_batch's per-iteration loop
# variables, at the point in the loop where the NaN check happens.
_NAN_DEBUG_VAR_NAMES = [
    "X_k", "rho_k", "R_k", "P_k", "nu_k", "alpha", "H_k", "alpha_nu_k",
    "S_k", "T_k", "omega", "rho_kp1", "beta", "X_kp1", "R_kp1", "P_kp1",
]

def _log_bicgstab_nan_debug(k: int, ns: dict, breakpoint_on_nan: bool = False) -> None:
    """Logs diagnostic info about a NaN found during a bicgstab_batch
    iteration, to help track down where it was introduced.

    `ns` is expected to be the caller's locals() dict at the point the NaN
    was detected, rather than each of the dozen-plus loop variables being
    passed in individually -- this keeps the (single) call site short, at
    the cost of making the helper's actual variable dependencies implicit:
    it will KeyError if a needed name isn't present in `ns` (e.g. if a
    variable inside bicgstab_batch's loop gets renamed) instead of that
    being caught by a type checker. Acceptable here since this is a
    narrowly-scoped, debug-only helper with one call site.
    """
    def any_nan(x):
        return torch.any(torch.isnan(x))

    logging.info(f"BiCGSTAB found a NaN in iteration {k}")
    for name in _NAN_DEBUG_VAR_NAMES:
        logging.info(f"{name + ':':<12}{any_nan(ns[name])}")

    # Extras... mostly collections of scalars
    alpha, beta, omega = ns["alpha"], ns["beta"], ns["omega"]
    S_k, T_k = ns["S_k"], ns["T_k"]
    logging.info("")
    logging.info(f"alpha contains 0? {torch.any(alpha==0)}")
    logging.info(f"alpha contains infinite? {torch.any(torch.isinf(alpha))}")
    logging.info(f"alpha = {alpha}")
    logging.info(f"beta  = {beta}")
    logging.info(f"omega = {omega}")
    logging.info(f"<S_k, T_k> = {_inner_prod(T_k, S_k)}")
    logging.info(f"<T_k, T_k> = {_inner_prod(T_k, T_k)}")
    logging.info(f"Convergence ratio array: {(ns['residual_norm'] / ns['stopping_matrix'])}")

    if breakpoint_on_nan:
        import pdb; pdb.set_trace()

# Could make a compiled version available (may be slow for the first run)
# _torch_compile_bicgstab_batch = torch.compile(_bicgstab_batch)

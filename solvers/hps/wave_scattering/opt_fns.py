# Optimization utilities
# e.g., filtered back-projection, Gauss-Newton, and recursive linearization
# Note that these objects are mostly built off of the SharedSolver object
# since it lets use re-use intermediate work without needing to manage
# everything manually here.

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg as jax_cg

import numpy as np

import time
from typing import List, Tuple
import logging

from  solvers.hps.wave_scattering import (
    ScatteringProblem,
    HPSScatteringSolver,
    SharedSolver,
)

class GaussNewtonOperator():
    """Operator from Gauss-Newton steps"""
    def __init__(
        self,
        q: jnp.array,
        hss: HPSScatteringSolver,
        eps: float=0,
        shared_solver: SharedSolver=None,
        verbosity: int=0,
    ):
        """Set up the Gauss-Newton operator
        (i.e., (DF[q]^* DF[q] + eps*I), as part of the normal equations)
        """
        self.q_shape  = q.shape
        if shared_solver is not None:
            self.q_solver = shared_solver
        else:
            self.q_solver = SharedSolver(
                hps_scattering_solver,
                q1_hpst,
                device=jax_device,
            )
            # At the moment, this is needed to complete the setup...
            self.q_solver.forward_interior()
        self.eps = eps
        self.verbosity = verbosity

    def apply(self, x):
        """Apply the Gauss-Newton operator (DF[q]^* DF[q] + eps*I)
        """
        if self.verbosity >= 4:
            logging.info(f"Apply.. starting")
        in_shape = x.shape
        x        = x.reshape(self.q_shape)
        with jax.disable_jit(False):
            DFx    = self.q_solver.jvp_exterior(x, rebuild_solver=False)
            DFhDFx = self.q_solver.vjp_exterior(DFx, rebuild_solver=False, verbosity=self.verbosity)
        out_hpst = DFhDFx + self.eps * x

        if self.verbosity >= 5:
            jax.debug.print(f"GN op called")
        return out_hpst.reshape(in_shape)

def _rel_err_fn(x, ref, **kwargs):
    return jnp.linalg.norm(x-ref, **kwargs) / jnp.linalg.norm(ref)

def gauss_newton_loop_single_sample(
    hss: HPSScatteringSolver,
    ref_dk: jax.Array,
    q_init: jax.Array,
    gn_iters: int=5,
    gn_step_size: float=1,
    gn_eps: float=0,
    cg_rtol: float=1e-4,
    cg_iters: int=10,
    verbosity: int=0,
    jax_device: jax.Device=None,
    allow_increase_error: bool=False,
) -> Tuple[jax.Array, List[jax.Array]]:
    """Perform a number of gauss-newton steps for a single sample (at a single frequency).
    CAUTION: works with scattering potentials in the HPS quadtree grid
    Solves the equation
        (DF[q]^*DF[q]+eps*I) dq = DF[q]^*(dk-F[q])
    at each iteration using JAX's implementation of conjugate gradient

    Parameters:
        hss (HPSScatteringSolver): hps solver object
        ref_dk (jax.Array): reference data measurement for the relevant
            frequency. Expected shape: (N_r, N_s)
        q_init (jax.Array): initialization for scattering potential q
        gn_iters (int): number of Gauss-Newton iterations to perform
        gn_step_size (float): relative step size to take after each iteration
            (sometimes 1.0 could overshoot the optimal value if the linear
            approximation is not valid for a large enough region)
        gn_eps (float): regularization parameter for the Gauss-Newton operator
        cg_rtol (float): relative tolerance for conjugate gradient
        cg_iters (int): maximum number of iterations allowed for conjugate gradient
        allow_increase_error (bool): whether to perform steps that would increase the
            measurement error. If such a step is encountered, the GN loop will terminate
            without any additional steps.
    Returns:
        q_out (jax.Array): final output for q
        q_list (List[jax.Array]): list of intermediate q values, including the
            initialization and final output
    """
    jax_device = jax_device if jax_device is not None else jax.devices()[0]
    q_loop = jnp.copy(q_init)
    q_list = [q_loop]

    # Track the measurement error before stepping
    q_init_solver = SharedSolver(
        hss,
        q_init,
        device=jax_device,
    )

    Fkt_q_init = q_init_solver.forward_exterior().T
    init_meas_err = (
        _rel_err_fn(Fkt_q_init, ref_dk)
        if not allow_increase_error
        else None
    )
    if verbosity >= 3:
        logging.info(f"Initial measurement error: {init_meas_err:.3e}")

    # Prepare for the loop
    q_loop        = q_init
    q_loop_solver = q_init_solver
    curr_meas_err = init_meas_err

    t0 = time.perf_counter()
    for t in range(1, 1+gn_iters):
        t1 = time.perf_counter()
        if verbosity >= 2:
            logging.info(f"({t1-t0:.3f}s) Gauss-Newton Iteration {t}")

        # At the moment, this is needed to complete the setup...
        # (at least, for use with the vjp and jvp)
        q_loop_solver.forward_interior()

        # Set up the operator and right-hand side
        gn_op = GaussNewtonOperator(
            q_loop,
            hss,
            eps=gn_eps,
            shared_solver=q_loop_solver,
            verbosity=verbosity,
        )

        rhs_loop = q_loop_solver.backproject_diff_exterior(
            dk=ref_dk,
            transpose_dk=True,
            recompute_interior_soln=False,
        )

        if verbosity >= 3:
            t_tmp = time.perf_counter()
            logging.info(f"({t_tmp-t0:.3f}s) Starting CG")

        dq, info = jax_cg(
            gn_op.apply,
            rhs_loop,
            x0=q_loop,
            tol=cg_rtol,
            maxiter=cg_iters,
        )
        if verbosity >= 3:
            t_tmp = time.perf_counter()
            logging.info(f"({t_tmp-t0:.3f}s) Finished CG")

        # Proposed next step
        q_prop = q_loop + gn_step_size * dq

        # If it's the last iteration and increasing error is allowed,
        # end without preparing the next solver
        if allow_increase_error and (t == gn_iters):
            q_loop = q_prop
            break

        # Set up solver for the next iteration
        q_prop_solver = SharedSolver(
            hss,
            q_prop,
            device=jax_device,
        )

        # If relevant, check the new measurement error
        if not allow_increase_error:
            Fkt_q_prop = q_prop_solver.forward_exterior().T
            prop_meas_err = (
                _rel_err_fn(Fkt_q_prop, ref_dk)
                if not allow_increase_error
                else None
            )
            if verbosity >= 3:
                logging.info(f"Proposed measurement error: {prop_meas_err:.3e}")

            # If the step would increase error, break out
            if np.isnan(prop_meas_err) or prop_meas_err > curr_meas_err:
                logging.info(
                    f"(iter={t}) Identified step that would increase measurement error "
                    f"and will end Gauss-Newton"
                )
                break

        # Apply proposed step
        q_loop        = q_prop
        q_loop_solver = q_prop_solver
        curr_meas_err = prop_meas_err

        q_list.append(q_loop)
    q_gn = q_loop
    t2 = time.perf_counter()
    if verbosity >= 1:
        logging.info(f"Gauss-Newton finished {t} iterations in {t2-t0:.3f}s")
    return q_gn, q_list

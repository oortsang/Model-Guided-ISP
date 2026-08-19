# Wrap the HPS solver for use with pytorch
# backprop is not supported at the moment :(
# TODO: add a pytorch.autograd.Function interface

import numpy as np
import jax
import jax.numpy as jnp
import jaxlib
import torch

import logging
from typing import Tuple
import time

from .interior_solver import forward_model_interior
from .exterior_solver import forward_model_exterior
from .hps_scattering_solver import HPSScatteringSolver
from .shared_solver import SharedSolver
from .derivative_solver import apply_vjp

# Double-precision in jax, single-precision in pytorch
HPS_JAX_RDTYPE = jnp.float64
HPS_JAX_CDTYPE = jnp.complex128
HPS_TORCH_RDTYPE = torch.float
HPS_TORCH_CDTYPE = torch.cfloat

# Helper functions that do the translation between jax and torch
# as well as the appropriate type casting
is_complex = lambda x: x in [jnp.complex64, jnp.complex128, torch.cfloat, torch.cdouble]
jax_to_torch = lambda jx: (
    torch.utils.dlpack.from_dlpack(jx)
    .to(HPS_TORCH_CDTYPE if is_complex(jx.dtype) else HPS_TORCH_RDTYPE)
)
torch_to_jax = lambda tx: (
    jax.dlpack.from_dlpack(tx.detach())
    .astype(HPS_JAX_CDTYPE if is_complex(tx.dtype) else HPS_JAX_RDTYPE)
)

class PytorchHPSSolver(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        hss: HPSScatteringSolver,
        config: dict = None,
    ) -> torch.Tensor:
        """Applies the forward solve and saves the relevant info to ctx
        Creates and saves a SharedSolver object if config["hps_save_tree"] is set to True.
        Otherwise, the HPS solver will be re-built during the backward pass
        """
        # Pre-pend variable names with j for jax or t for torch
        # Move to Jax
        # jq_unif = jax.dlpack.from_dlpack(q.detach()).astype(HPS_JAX_RDTYPE)
        jq_unif = torch_to_jax(q)
        jq_hpst = hss.UtQ.apply(jq_unif)
        jax_device = jq_hpst.device

        # Save to context depending on the config dictionary
        ctx.config = config
        save_tree  = config.get("hps_save_tree", False) # Default to correctness for safety
        ctx.hss = hss

        if save_tree:
            q_shared_solver = SharedSolver(
                hss,
                jq_hpst,
                device=jax_device,
            )
            jd_rs = q_shared_solver.forward_exterior().T
            ctx.q_shared_solver = q_shared_solver
        else:
            jd_rs = forward_model_exterior(
                hss.scat_problem,
                jq_hpst,
                rebuild_solver=True,
                device=jax_device,
            ).T
            ctx.q_shared_solver = None
            ctx.save_for_backward(*map(jax_to_torch, [jq_hpst]))

        # Move back to pytorch
        td_rs = torch.utils.dlpack.from_dlpack(jd_rs).to(HPS_TORCH_CDTYPE)
        return td_rs

    @staticmethod
    def backward(ctx, tgrad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply the vector-jacobian product code onto tgrad_output
        If ctx.config["hps_save_tree"] is set to True, then this pass will reuse the
        q_shared_solver object created during the forward pass.
        This is faster but would result in incorrect gradients if any other forward pass
        was used in the meantime (since it would overwrite the matrices in the HPS solver)

        To ensure correctness when computing multiple forward passes at a time, set
        ctx.config["hps_save_tree"] to False
        """
        hss = ctx.hss
        k   = hss.k
        jgrad_output = jax.dlpack.from_dlpack(tgrad_output).astype(HPS_JAX_CDTYPE)
        jax_device = jgrad_output.device

        # Apply vjp
        if ctx.q_shared_solver is not None:
            # If we still have the q_shared_solver this is quite straightforward
            # Collectively takes ~190 ms for L=4, p=16 (on a L40S)
            # By contrast, the forward pass takes ~80 ms
            ctx.q_shared_solver.forward_interior()
            jax_vjp_out_hpst = ctx.q_shared_solver.vjp_exterior(
                jgrad_output.T, rebuild_solver=False
            )
            ctx.q_shared_solver = None
        else:
            # jq_hpst, jusc_int = list(map(torch_to_jax, ctx.saved_tensors))
            jq_hpst, = list(map(torch_to_jax, ctx.saved_tensors))
            # Collectively takes ~280 ms for L=4, p=16 (on a L40S)
            # This is ~90 ms longer than with the q_shared_solver for L=4, p=16
            # Differences vs. the q_shared_solver version:
            # 1. Rebuilds the solver (probably the main thing)
            # 2. Recomputes usc_bdry and usc_dn_bdry
            # (might only need to store usc_dn_bdry+1j*k*usc_bdry)
            # Another option is to use a coarser discretization for the backward pass

            # I have a logic error somewhere so I have to explicitly prepare usc_int
            # (the tree isn't properly rebuilt in eval_beta_bdry_with_source,
            # so later get_utot_int complains that some matrices are missing)
            jusc_int = forward_model_interior(
                hss.scat_problem,
                jq_hpst,
                return_exterior_soln=False,
                rebuild_solver=True,
                device=jax_device,
            )
            jax_vjp_out_hpst = apply_vjp(
                hss.scat_problem,
                q=jq_hpst,
                vec=jgrad_output.T,
                Gk_ring_to_omega=hss.Gk_ring_to_hpst,
                usc_int=jusc_int,
                # T_ext_DtN=None,
                T_ext_DtN=hss.T_ext_DtN_adj,
                rebuild_solver=False,
                device=jax_device,
            )
        jax_vjp_out_unif = hss.QtU.apply(jax_vjp_out_hpst)

        # Convert back to pytorch
        torch_vjp_out = torch.utils.dlpack.from_dlpack(jax_vjp_out_unif).to(HPS_TORCH_RDTYPE)

        # Unlink the other objects... or try anyway
        ctx.hss = None
        ctx.config = None
        return torch_vjp_out, None, None

def pytorch_backproject_diff(
    q: torch.Tensor,
    dk: torch.Tensor,
    hss: HPSScatteringSolver,
) -> torch.Tensor:
    """Computes DF[q]^* (dk-F[q]) where DF[]* and F[] are supplied by hss
    Only operates on a single sample at a time (for now)
    """
    # Torch to Jax
    jdk = torch_to_jax(dk)
    jq  = torch_to_jax(q)
    q_hpst = hss.UtQ.apply(jq)
    jax_device = q_hpst.device

    # Compute the backprojection of the difference (is there a better name for this?)
    q_shared_solver = SharedSolver(hss, q_hpst, device=jax_device)
    jDFh_diff_hpst = q_shared_solver.backproject_diff_exterior(
        jdk, transpose_dk=True,
    )
    jDFh_diff_unif = hss.QtU.apply(jDFh_diff_hpst)

    # Map back to torch
    DFh_diff = jax_to_torch(jDFh_diff_unif)
    return DFh_diff

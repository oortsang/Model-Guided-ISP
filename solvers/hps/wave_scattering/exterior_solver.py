# wave_scattering/exterior_solver.py
# Contains the code to compute the solution
# to the scattering problem at exterior points

import jax.numpy as jnp
import jax
import jaxlib
from typing import Tuple
import logging

from jaxhps import Domain, PDEProblem, DiscretizationNode2D, build_solver

from .scattering_problem import ScatteringProblem
from .gen_SD_exterior import (
    gen_D_exterior,
    gen_S_exterior,
)

from .scattering_utils import (
    get_DtN_from_ItI,
    get_uin_and_normals,
    get_uin,
    setup_scattering_lin_system,
    get_uscat_and_dn,
)

def forward_model_exterior(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
    use_ItI: bool=True,
    T_int_DtN: jax.Array = None,
    rebuild_solver: bool = True,
    device: jax.Device = jax.devices()[0],
) -> jax.Array:
    """
    Implements the foward scattering model by finding uscat on the boundary of the computational domain and
    evaluating uscat at exterior sensor points.

    If scattering_problem does not have incident waves specified, this function will construct incident plane waves using
    scattering_problem.source_dirs and k=pde_problem.eta

    For simplicity, uses the same device for host/compute
    (The scattering problems tend to be smaller systems)
    """
    # logging.info(f"Starting exterior solve...")
    pde_problem = scattering_problem.pde_problem
    k = pde_problem.eta

    # Set up the parts of the PDE
    # i_term = pde_problem.eta**2 * (jnp.ones_like(q) + q)
    i_term = k**2 * (q+1)

    pde_problem.update_coefficients(I_coefficients=i_term)

    # Build the solver
    # logging.info(f"Building the solver...")
    if T_int_DtN is not None:
        T_DtN = T_int_DtN
    elif not rebuild_solver:
        T_DtN = scattering_problem.T_DtN
    else:
        T_matrix = build_solver(
            pde_problem=pde_problem,
            return_top_T=True,
            compute_device=device,
            host_device=device,
        )
        if use_ItI:
            T_DtN = get_DtN_from_ItI(T_matrix, k)
        else:
            T_DtN = T_matrix

    # If the incident waves are not specified, use plane waves with freq k and
    # directions indicated by scattering_problem.source_dirs
    # logging.info(f"Computing uin and uin_dn on the boundary")
    if scattering_problem.uin_bdry is None:
        uin_bdry, uin_dn_bdry = get_uin_and_normals(
            k=pde_problem.eta,
            bdry_pts=pde_problem.domain.boundary_points,
            source_directions=scattering_problem.source_dirs,
        )
        # Safe to save these since there is no q dependence
        scattering_problem.uin_bdry = uin_bdry
        scattering_problem.uin_dn_bdry = uin_dn_bdry
    else:
        uin_bdry = scattering_problem.uin_bdry
        uin_dn_bdry = scattering_problem.uin_dn_bdry

    # Scattered field and its outward normal derivative on the boundary
    # logging.info(f"Computing uscat values and normal derivatives on the boundary...")
    uscat_homog, uscat_dn_homog = get_uscat_and_dn(
        S=scattering_problem.S_int,
        D=scattering_problem.D_int,
        T=T_DtN,
        uin=uin_bdry,
        uin_dn=uin_dn_bdry,
    )

    # Now we need to compute the scattered field at the exterior points
    # logging.info(f"Mapping to exterior points...")
    out = (
        scattering_problem.D_ext @ uscat_homog
        - scattering_problem.S_ext @ uscat_dn_homog
    )
    # logging.info(f"Finished and returning!")
    return out

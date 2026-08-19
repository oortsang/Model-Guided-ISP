# wave_scattering/interior_solver.py
# Contains the code to compute the solution
# to the scattering problem at interior points

import jax.numpy as jnp
import jax
import os
from .scattering_utils import (
    get_uin,
    get_uin_and_normals,
    get_uscat_and_dn,
    get_DtN_from_ItI,
    get_ItD,
)
from .scattering_problem import ScatteringProblem
from jaxhps import build_solver, solve
from jaxhps.up_pass import up_pass_uniform_2D_ItI
import logging
import matplotlib.pyplot as plt

def forward_model_interior(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
    return_exterior_soln: bool = False,
    rebuild_solver: bool = True,
    device: jax.Device = None,
) -> jax.Array:
    pde_problem = scattering_problem.pde_problem
    device = device if device is not None else q.device

    # Set up the parts of the PDE
    eta = pde_problem.eta
    i_term = eta**2 * (q+1)
    pde_problem.update_coefficients(I_coefficients=i_term)

    # Fetch u^in
    if scattering_problem.uin_interior is None:
        uin_bdry, uin_dn_bdry = get_uin_and_normals(
            k=eta,
            bdry_pts=pde_problem.domain.boundary_points,
            source_directions=scattering_problem.source_dirs,
        )
        uin_interior = get_uin(
            eta,
            pde_problem.domain.interior_points,
            scattering_problem.source_dirs,
        )
        # Save uin
        scattering_problem.uin_bdry = uin_bdry
        scattering_problem.uin_dn_bdry = uin_dn_bdry
        scattering_problem.uin_interior = uin_interior
    else:
        uin_bdry = scattering_problem.uin_bdry
        uin_dn_bdry = scattering_problem.uin_dn_bdry
        uin_interior = scattering_problem.uin_interior

    # Build the solver and DtN/ItI matrices
    # print(f"(forward_model_interior) rebuild_solver={rebuild_solver}")
    if rebuild_solver:
        T_ItI = build_solver(
            pde_problem=pde_problem,
            return_top_T=True,
            compute_device=device,
            host_device=device,
        )
        T_DtN = get_DtN_from_ItI(T_ItI, eta)

        # Save the precomputin'
        scattering_problem.T_DtN = T_DtN
        scattering_problem.T_ItI = T_ItI
        # logging.info(f"(forward_model_interior) rebuilt the solver")
    else:
        T_ItI = scattering_problem.T_ItI
        T_DtN = scattering_problem.T_DtN
        # logging.info(f"(forward_model_interior) using pre-computed T_DtN")

    # Get u^sc on the boundary
    uscat, uscat_dn = get_uscat_and_dn(
        S=scattering_problem.S_int,
        D=scattering_problem.D_int,
        T=T_DtN,
        uin=uin_bdry,
        uin_dn=uin_dn_bdry,
    )

    # Prepare the incoming impedance and then perform a solve
    incoming_imp = (uscat_dn + uin_dn_bdry) + 1j * eta * (
        uscat + uin_bdry
    )
    zero_source = jnp.zeros_like(uin_interior)
    utot_int = solve(
        pde_problem=pde_problem,
        boundary_data=incoming_imp,
        source=zero_source,
        compute_device=device,
        host_device=device,
    )
    uscat_int = utot_int - uin_interior

    if return_exterior_soln:
        # Compute the scattered field on the exterior points
        uscat_ext = (
            scattering_problem.D_ext @ uscat
            - scattering_problem.S_ext @ uscat_dn
        )
        return uscat_int, uscat_ext

    return uscat_int

def get_utot_int(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
    usc_int: jax.Array = None,
    rebuild_solver: bool = True,
    device: jax.Device = None,
) -> jax.Array:
    """Helper function to get the total wave field inside the domain
    Computes uin and usc on the domain interior if they have
    not already been computed.
    """
    pde_problem = scattering_problem.pde_problem
    k = pde_problem.eta
    device = device if device is not None else q.device

    # Get u_sc
    if usc_int is None:
        usc_int = forward_model_interior(
            scattering_problem,
            q,
            return_exterior_soln=False,
            rebuild_solver=rebuild_solver,
            device=device,
        )

    # Get u_in
    if scattering_problem.uin_interior is None:
        # Construct it from the source directions
        uin_int = get_uin(
            k=pde_problem.eta,
            pts=pde_problem.domain.interior_points,
            source_directions=scattering_problem.source_dirs,
        )
    else:
        uin_int = scattering_problem.uin_interior

    utot_int = uin_int + usc_int
    return utot_int

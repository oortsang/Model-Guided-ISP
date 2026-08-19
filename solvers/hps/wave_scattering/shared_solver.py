# wave_scattering/shared_solver_prep.py
# Contains the code to perform the setup for values
# shared across forward and backward passes of the
# solver (which involves both interior and exterior solves)

import jax.numpy as jnp
import jax
import jaxlib
from typing import Tuple
import logging

from jaxhps import build_solver, solve
from jaxhps.local_solve import local_solve_stage_uniform_2D_ItI
from jaxhps.merge import merge_stage_uniform_2D_ItI
from jaxhps.down_pass import down_pass_uniform_2D_ItI


from .scattering_problem import ScatteringProblem
from .hps_scattering_solver import HPSScatteringSolver
from .gen_SD_exterior import (
    gen_D_exterior,
    gen_S_exterior,
)

from .scattering_utils import (
    get_DtN_from_ItI,
    get_uin_and_normals,
    get_uin,
)
from .exterior_solver import (
    setup_scattering_lin_system,
    get_uscat_and_dn,
)
from .derivative_solver import (
    apply_vjp,
    apply_jvp,
)


class SharedSolver:
    """A class to organize the data for a single scatterer
    Each object is only expected for a single scattering object,
    in contrast to the ScatteringProblem, PDEProblem, and
    HPSScatteringSolver objects, which are set up for a given
    problem setting (e.g., wavenumber, grids, sources/receivers)
    """
    def __init__(
        self,
        hps_scattering_solver: HPSScatteringSolver,
        # scattering_problem: ScatteringProblem,
        q_hpst: jax.Array,
        device: jax.Device = jax.devices()[0],
    ):
        """Set up the shared solver
        Note: q is expected to be in hps tree format
        """
        scattering_problem = hps_scattering_solver.scat_problem
        self.scat_problem = scattering_problem
        self.pde_problem  = scattering_problem.pde_problem
        self.hss = hps_scattering_solver
        self.q = q_hpst
        self.device = device

        # Setup
        self.eta = self.pde_problem.eta
        self.i_term = self.pde_problem.eta**2 * (q_hpst+1)

        prep_dict = shared_solver_prep(
            scattering_problem,
            q_hpst,
            device=device,
        )

        self.T_DtN = prep_dict["T_DtN"]
        self.T_ItI = prep_dict["T_ItI"]
        self.uin_bdry    = prep_dict["uin_bdry"]
        self.uin_dn_bdry = prep_dict["uin_dn_bdry"]
        self.usc_bdry    = prep_dict["usc_bdry"]
        self.usc_dn_bdry = prep_dict["usc_dn_bdry"]
        self.uin_int     = prep_dict["uin_int"]
        self.zero_source = jnp.zeros(
            (
                self.pde_problem.domain.interior_points.shape[0],
                self.pde_problem.domain.interior_points.shape[1],
                self.uin_bdry.shape[-1],
            ),
        )
        self.usc_int = None

    def forward_exterior(self) -> jax.Array:
        """Computes the solution on the exterior receiver ring
        Note: output has axes flipped vs. the d_rs convention
        """
        usc_meas = (
            self.scat_problem.D_ext @ self.usc_bdry
            - self.scat_problem.S_ext @ self.usc_dn_bdry
        )
        return usc_meas

    def forward_interior(self) -> jax.Array:
        """Computes the solution on the interior"""
        incoming_imp = (
            self.usc_dn_bdry + self.uin_dn_bdry
            + 1j * self.eta * (self.usc_bdry + self.uin_bdry)
        )

        utot_int = solve(
            pde_problem=self.pde_problem,
            boundary_data=incoming_imp,
            source=self.zero_source,
            compute_device=self.device,
            host_device=self.device,
        )
        usc_int = utot_int - self.uin_int
        # Save this...
        # self.utot_int = utot_int
        self.usc_int = usc_int
        return usc_int

    def vjp_exterior(self, vec: jax.Array, rebuild_solver: bool=False, verbosity: int=0) -> jax.Array:
        """Applies the adjoint of the forward model's derivative to a vector vec
        vec lives in the same space as the far-field scattering measurements
        Outputs vec J == vec DF[q] == (DF[q]^* vec)^*
        See Thm 3.2 in Borges et al. 2016,
            High Resolution Inverse Scattering In
            Two Dimensions Using Recursive Linearization
        """
        vjp_out = apply_vjp(
            scattering_problem=self.scat_problem,
            q=self.q,
            vec=vec,
            Gk_ring_to_omega=self.hss.Gk_ring_to_hpst,
            usc_int=self.usc_int,
            T_ext_DtN=self.hss.T_ext_DtN_adj,
            rebuild_solver=rebuild_solver,
            device=self.device,
            verbosity=verbosity,
        )
        return vjp_out

    def jvp_exterior(self, dq: jax.Array, rebuild_solver: bool=False, verbosity: int=0) -> jax.Array:
        """Applies the forward model's derivative to a vector dq
        dq lives in the same space as the scattering potential
        Outputs J dq == DF[q] dq, which is represented as the field u

        See Thm 3.1 in Borges et al. 2016,
            High Resolution Inverse Scattering In
            Two Dimensions Using Recursive Linearization
        """
        jvp_out = apply_jvp(
            scattering_problem=self.scat_problem,
            q=self.q,
            vec=dq,
            usc_int=self.usc_int,
            T_ext_DtN=self.hss.T_ext_DtN_std,
            rebuild_solver=rebuild_solver,
            device=self.device,
            verbosity=verbosity,
        )
        return jvp_out

    def backproject_diff_exterior(
        self,
        dk: jax.Array,
        forward_exterior_val: jax.Array=None,
        transpose_dk: bool=False,
        recompute_interior_soln: bool=True,
    ) -> jax.Array:
        """Compute the back-projection of difference (d-F[q]) as DF[q]^*(d-F[q])
        Expected shapes:
            dk: (N_r, N_s) (or transposed)
            forward_exterior_val (optional): (N_r, N_s)
        Returns an object on the HPS grid
        """
        if recompute_interior_soln:
            self.forward_interior()

        Fq_ext = (
            self.forward_exterior()
            if forward_exterior_val is None
            else forward_exterior_val
        )
        diff = (dk.T if transpose_dk else dk) - Fq_ext
        DFh_diff = self.vjp_exterior(diff, rebuild_solver=False)
        return DFh_diff

def shared_solver_prep(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
    device: jax.Device = jax.devices()[0],
) -> jax.Array:
    """Perform the shared work for the interior and exterior models
    Based on the ItI matrices
    """
    pde_problem = scattering_problem.pde_problem
    eta = pde_problem.eta
    i_term = eta**2 * (q+1)
    pde_problem.update_coefficients(I_coefficients=i_term)

    # Set up the uin fields
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
        # Write in uin since it does not depend on q
        scattering_problem.uin_bdry     = uin_bdry
        scattering_problem.uin_dn_bdry  = uin_dn_bdry
        scattering_problem.uin_interior = uin_interior
    else:
        uin_bdry = scattering_problem.uin_bdry
        uin_dn_bdry = scattering_problem.uin_dn_bdry
        uin_interior = scattering_problem.uin_interior

    # Build the T matrices
    T_ItI = build_solver(
        pde_problem=pde_problem,
        return_top_T=True,
        compute_device=device,
        host_device=device,
    )
    T_DtN = get_DtN_from_ItI(T_ItI, eta)

    # Also load up the scattering problem object
    scattering_problem.T_DtN = T_DtN
    scattering_problem.T_ItI = T_ItI

    usc_bdry, usc_dn_bdry = get_uscat_and_dn(
        S=scattering_problem.S_int,
        D=scattering_problem.D_int,
        T=T_DtN,
        uin=uin_bdry,
        uin_dn=uin_dn_bdry,
    )

    prep_dict = {
        "usc_bdry": usc_bdry,
        "usc_dn_bdry": usc_dn_bdry,
        "T_ItI": T_ItI,
        "T_DtN": T_DtN,
        "uin_bdry":     uin_bdry,
        "uin_dn_bdry":  uin_dn_bdry,
        "uin_int": uin_interior,
    }
    return prep_dict

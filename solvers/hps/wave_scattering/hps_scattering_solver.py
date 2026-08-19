from typing import Tuple, Iterable
import jax.numpy as jnp
import jax
import jaxlib
import argparse
import logging
import os
import h5py
import matplotlib.pyplot as plt
import scipy
from jaxhps import Domain, PDEProblem, DiscretizationNode2D, build_solver
from jaxhps._device_config import HOST_DEVICE
from .scattering_problem import ScatteringProblem
from .gen_SD_exterior import (
    get_ring_points,
    gen_D_exterior,
    gen_S_exterior,
)
from .scattering_utils import (
    # get_DtI_from_DtN,
    get_exterior_DtN,
    load_SD_matrices,
)
from .interp_utils import (
    prep_grids_unif_2d,
    prep_grids_cheb_2d,
    reorder_tree_cheb_for_hps,
)
from .interp_ops import (
    QuadtreeToUniform,
    UniformToQuadtree,
)
from .exterior_solver import (
    forward_model_exterior,
)
from .interior_solver import (
    forward_model_interior,
)

def _device_put_all_arrays(obj, device: jax.Device) -> None:
    """Moves every jax.Array-valued attribute of obj (and lists of
    jax.Array, e.g. PDEProblem's per-merge-level solution operator lists)
    to device in place. Attributes holding other kinds of objects (e.g.
    QtU/UtQ, which are shared across solvers with the same (grid_size, L, p)
    via SolverCache's interp op cache and must not be moved here; or nested
    Domain/PDEProblem/ScatteringProblem objects, handled via their own call
    to this function) are left untouched.
    """
    for name, val in vars(obj).items():
        if isinstance(val, jax.Array):
            setattr(obj, name, jax.device_put(val, device))
        elif isinstance(val, list) and val and isinstance(val[0], jax.Array):
            setattr(obj, name, [jax.device_put(v, device) for v in val])


class HPSScatteringSolver():
    def __init__(
        self,
        L: int, p: int, N_x: int,
        k: float,
        S_int: jax.Array,
        D_int: jax.Array,
        S_ext: jax.Array = None,
        D_ext: jax.Array = None,
        N_r: int = None,
        N_s: int = None,
        unif_domain_bounds: Iterable = (-0.5, 0.5, -0.5, 0.5),
        quad_domain_bounds: Iterable = (-0.5, 0.5, -0.5, 0.5),
        receiver_radius: float = 100,
        UtQ: UniformToQuadtree = None,
        source_dirs: jax.Array = None,
        receiver_dirs: jax.Array = None,
        QtU: QuadtreeToUniform = None,
        use_ItI: bool = True,
    ):
        """A wrapper for the HPS solver for the wave scattering problem (on the receiver ring)

        Parameters:
            L (int): number of levels in the HPS quadtree
            p (int): polynomial order of the chebyshev grids on the leaf-level of HPS grids
            N_x (int): number of points on the uniform grid (e.g., 192) corresponding to the
                domain given by unif_domain_bounds
            k (float): angular wavenumber used for the source waves
                i.e., already contains the factor of 2pi
            S_int (jax.Array): interior single-layer potential scattering matrix
                pre-computed in Matlab with ChunkIE and chebfun
            D_int (jax.Array): interior double-layer potential scattering matrix
                pre-computed in Matlab with ChunkIE and chebfun
            S_ext (jax.Array): exterior single-layer potential scattering matrix
                can be precomputed with gen_S_exterior or computed in this initialization phase
            D_ext (jax.Array): exterior double-layer potential scattering matrix
                can be precomputed with gen_D_exterior or computed in this initialization phase
            N_r (int): number of receivers
            N_s (int): number of sources
            unif_domain_bounds (jax.Array): the bounds of the uniformly-distributed grid
                meant for use as the space where the scattering potential lives
                Note: format is [xmin, xmax, ymin, max]
            quad_domain_bounds (jax.Array): the bounds of the HPS quadtree's computational grid
                meant for use computing the scattered wave u_scat
                Expanding the computational domain can prevent the solution from
                degrading when it would normally leave the computational domain
                Note: format is [xmin, xmax, ymin, max]
            receiver_radius (float): radius where the receiver locations are distributed
            UtQ (UniformToQuadtree): an interpolation object that helps map from the uniform grid
                to the quadtree discretization (and node ordering)
                Will be computed if not supplied
            source_dirs (jax.Array): the angles corresponding to source waves
                Note: should be set to np.pi/2-np.linspace(0,2*np.pi,N_s,endpoint=False)
                in order to match the Lippmann-Schwinger solver configuration
            receiver_dirs (jax.Array): the angles corresponding to receiver nodes on the receiver radius
                Note: should be set to np.pi/2-np.linspace(0,2*np.pi,N_s,endpoint=False)
                in order to match the Lippmann-Schwinger solver configuration

        Caution: the source_dirs and receiver_dirs should be different from the usual setup
        To agree with the LS solver with source_dirs=np.linspace(0,2*np.pi,N_s,endpoint=False)
        We instead need to take source_dirs=np.pi/2-np.linspace(0,2*np.pi,N_s,endpoint=False)
        I believe there is some sort of discrepancy in axis ordering that causes this.
        """
        self.L = L
        self.p = p
        self.N_x = N_x
        self.n_per_leaf = N_x // 2**L
        self.k = k
        self.receiver_radius = receiver_radius

        self.leaf_bounds = (-1., 1., -1., 1.)
        self.unif_domain_bounds = jnp.array(unif_domain_bounds)
        self.quad_domain_bounds = jnp.array(quad_domain_bounds)
        self.root = DiscretizationNode2D(*self.quad_domain_bounds)
        self.domain = Domain(p=p, q=p - 2, root=self.root, L=L)

        # Calculate the grids in case they'll be useful later...
        self.leaf_unif_x, self.leaf_unif_y, self.leaf_unif_xy = prep_grids_unif_2d(
            0, self.n_per_leaf, domain_bounds=self.leaf_bounds, rel_offset=0,
        )
        self.leaf_cheb_x, self.leaf_cheb_y, self.leaf_cheb_xy = prep_grids_cheb_2d(
            0, self.p, domain_bounds=self.leaf_bounds
        )
        self.tree_unif_x, self.tree_unif_y, self.tree_unif_xy = prep_grids_unif_2d(
            0, self.N_x, domain_bounds=self.unif_domain_bounds, rel_offset=0,
        )
        self.tree_cheb_x, self.tree_cheb_y, self.tree_cheb_xy = prep_grids_cheb_2d(
            self.L, self.p, domain_bounds=self.quad_domain_bounds
        )
        self.hps_tree_xy = reorder_tree_cheb_for_hps(self.tree_cheb_xy, L=L, p=p)

        # Expect N_r or receiver_dirs and N_s or source_dirs
        # If both are given but conflict, the values from the {source,receiver}_dirs will be used
        assert not (N_r is None and receiver_dirs is None)
        assert not (N_s is None and source_dirs is None)
        self.N_r = N_r if N_r is not None else receiver_dirs.shape[0]
        self.N_s = N_s if N_s is not None else source_dirs.shape[0]
        self.receiver_dirs = receiver_dirs if receiver_dirs is not None else \
            jnp.pi/2-jnp.linspace(0, 2*jnp.pi, self.N_r, endpoint=False)
        self.source_dirs = source_dirs if source_dirs is not None else \
            jnp.pi/2-jnp.linspace(0, 2*jnp.pi, self.N_s, endpoint=False)

        # Save the interior S, D matrices
        self.S_int = S_int
        self.D_int = D_int

        # Compute the exterior S, D matrices in case they were not already passed
        self.S_ext = S_ext if S_ext is not None else \
            gen_S_exterior(
                domain=self.domain,
                k=k,
                rad=receiver_radius,
                source_dirs=self.receiver_dirs
            )
        self.D_ext = D_ext if D_ext is not None else \
            gen_D_exterior(
                domain=self.domain,
                k=k,
                rad=receiver_radius,
                source_dirs=self.receiver_dirs
            )

        # Set up the PDE Problem object
        self.use_ItI = use_ItI
        self.d_xx_evals = jnp.ones_like(self.domain.interior_points[..., 0])
        self.d_yy_evals = jnp.ones_like(self.domain.interior_points[..., 0])
        self.pde_problem = PDEProblem(
            domain=self.domain,
            D_xx_coefficients=self.d_xx_evals,
            D_yy_coefficients=self.d_yy_evals,
            eta=self.k,
            use_ItI=use_ItI,
        )

        # Set up the Scattering Problem object
        self.scat_problem = ScatteringProblem(
            pde_problem=self.pde_problem,
            S_int=self.S_int,
            D_int=self.D_int,
            S_ext=self.S_ext,
            D_ext=self.D_ext,
            target_points_reg=None,
            source_dirs=self.source_dirs,
        )
        self.UtQ = UtQ if UtQ is not None else \
            UniformToQuadtree(
                self.L, self.p, self.N_x,
                self.unif_domain_bounds,
                self.quad_domain_bounds,
                rel_offset=0,
                interp_use_jax=True,
                use_sparse_ops=True,
            )
        # in case I need this later...
        self.QtU = QtU if QtU is not None else \
            QuadtreeToUniform(
                self.L, self.p, self.n_per_leaf, self.N_x,
                quad_domain_bounds=self.quad_domain_bounds,
                unif_domain_bounds=self.unif_domain_bounds,
                rel_offset=0,
            )

        # Prepare the Greens operator from receiver ring to the quadtree domain
        rec_pts = get_ring_points(self.receiver_radius, self.receiver_dirs)
        tgt_pts = self.hps_tree_xy.reshape(-1, 2)
        dists = jnp.linalg.norm(tgt_pts[:, None, :] - rec_pts[None, :, :], axis=-1)
        unif_domain_length = unif_domain_bounds[1] - unif_domain_bounds[0]
        scaling_factor = 1 / (self.N_r * self.N_s) * unif_domain_length # maybe??
        # Note that this is the Greens function satisfying the adjoint sommerfeld
        # radiation condition
        self.Gk_ring_to_hpst = jnp.array(
            -1j/4 * scipy.special.hankel2(0, k*dists) * scaling_factor
        )

        # Also prepare the T_ext operator for use in the backward pass
        # Prepare for both standard and adjoint Sommerfeld radiation conditions
        self.T_ext_DtN_std = get_exterior_DtN(self.S_int, self.D_int)
        self.T_ext_DtN_adj = get_exterior_DtN(self.S_int.conj(), self.D_int.conj())

        # Tracks where the big matrices live
        self._resident_device = jax.devices()[0]

    def to_device(self, device: jax.Device = None) -> None:
        """Moves this solver's arrays (and self.scat_problem's/
        self.pde_problem's) to device. Does not touch QtU/UtQ, which are
        shared across solvers with the same (grid_size, L, p) -- they
        aren't jax.Array instances themselves, so the generic sweep in
        _device_put_all_arrays skips them automatically.

        No-op if the solver is already resident on device.
        """
        device = device if device is not None else jax.devices()[0]
        if self._resident_device == device:
            return
        _device_put_all_arrays(self, device)
        _device_put_all_arrays(self.scat_problem, device)
        _device_put_all_arrays(self.pde_problem, device)
        self._resident_device = device

    def to_host(self) -> None:
        """Moves this solver's big arrays to host (CPU) RAM."""
        self.to_device(HOST_DEVICE)

    def reset(self) -> None:
        """Frees solution-operator/incident-field arrays that are cheap to
        regenerate on the next solve (uin_interior, Y, v, Phi, S_lst,
        g_tilde_lst, ...), rather than just moving them to host RAM.
        """
        self.scat_problem.reset()
        self.pde_problem.reset()

    def solve_exterior(
        self, q: jax.Array, quadtree_in: bool=False,
    ) -> Tuple[jax.Array]:
        """Solve for uscat on the exterior for the values at the receivers
        Note: returns usc_ext_hps and also d_rs, which has its axes flipped
        to match the format of the Lippmann-Schwinger solver's outputs
        Returns both to hopefully reduce the chance of confusion later on...
        """
        # logging.info(f"Converting q to quadtree if needed")
        q_quadtree = q if quadtree_in else self.UtQ.apply(q)
        # logging.info(f"Calling the forward model...")
        usc_ext_hps = forward_model_exterior(
            scattering_problem=self.scat_problem,
            q=q_quadtree,
            use_ItI=self.use_ItI,
        )
        d_rs_hps = usc_ext_hps.T
        # logging.info(f"Returning usc_ext_hps and d_rs_hps")
        return usc_ext_hps, d_rs_hps

    def solve_interior(
        self,
        q: jax.Array,
        quadtree_in: bool = False,
        return_exterior_soln: bool = False,
    ) -> Tuple[jax.Array]:
        """Solve for uscat on the interior of the domain
        """
        q_quadtree = q if quadtree_in else self.UtQ.apply(q)
        res = forward_model_interior(
            scattering_problem=self.scat_problem,
            q=q_quadtree,
            return_exterior_soln=return_exterior_soln,
        )
        return res

def setup_hps_scattering_solver(
    N_x: float,
    spatial_domain_max,
    k: float,
    receiver_radius: float,
    hps_sd_int_mat_dir,
    hps_l,
    hps_p,
    hps_comp_domain_factor: float=1.0,
    device: jax.Device = jax.devices()[0],
    N_s: int = None,
    kbar_str: str = None,
) -> HPSScatteringSolver:
    """Setup for the HPS Scattering solver, intended as the interface for use in pytorch
    kbar is k/2pi
    """
    # Set up bounds
    unif_bounds = jnp.array([
        -spatial_domain_max,
        +spatial_domain_max,
        -spatial_domain_max,
        +spatial_domain_max,
    ])
    quad_bounds = hps_comp_domain_factor * unif_bounds
    # Fetch the S and D matrices
    dom_hlen_str = "1" if hps_comp_domain_factor >= 1.5 else "0.5"
    kbar_0_decimal = jnp.round(k/2/jnp.pi, decimals=0)
    kbar_1_decimal = jnp.round(k/2/jnp.pi, decimals=1)
    # Use one decimal point if needed, otherwise use a string
    kbar_str = kbar_str if kbar_str is not None else \
        f"{kbar_1_decimal:.1f}" if not jnp.isclose(kbar_1_decimal, kbar_0_decimal) else \
        str(int(kbar_0_decimal))

    logging.info(f"Using kbar_str={kbar_str} to load the HPS SD matrices")
    SD_matrices_fp = os.path.join(
        hps_sd_int_mat_dir,
        f"SD_kbar{kbar_str}_L{hps_l}_n{hps_p-2}_dom{dom_hlen_str}.mat"
    )
    S_int, D_int = load_SD_matrices(SD_matrices_fp)
    N_s = N_s if N_s is not None else N_x
    N_r = N_s

    # Get the main solver object
    hps_scattering_solver = HPSScatteringSolver(
        hps_l,
        hps_p,
        N_x,
        k,
        S_int,
        D_int,
        N_r=N_r,
        N_s=N_s,
        unif_domain_bounds=unif_bounds,
        quad_domain_bounds=quad_bounds,
        receiver_radius=receiver_radius,
        use_ItI=True,
        # Use the default values
        source_dirs=None,
        receiver_dirs=None,
        UtQ=None,
        QtU=None,
    )
    return hps_scattering_solver

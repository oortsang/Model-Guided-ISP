# helmholtz_solver_bicgstab.py: a cleaned-up, bicgstab-only version of the
# differentiable Helmholtz solver based on the Lippmann-Schwinger equation.
#
# This is a behavior-preserving split of HelmholtzSolverDifferentiable.py's
# bicgstab code path: gmres support is dropped (bicgstab only), and the
# adjoint-state gradient / autograd wrapper are moved to
# helmholtz_solver_gradients.py. HelmholtzSolverDifferentiable.py is left
# unchanged and still supports gmres.
#
# This file contains:
# - HelmholtzSolverBicgstab (class definition)
#   Methods include:
#   * Helmholtz_solve_exterior_batched
#   * Helmholtz_solve_interior_batched
# - setup_bicgstab_solver


import logging
import time
import uuid
from typing import List, Tuple

import numpy as np
import torch

from solvers.integral_equation.Helmholtz_solver_utils import (
    greensfunction3,
    getGscat2circ,
    find_diag_correction,
    get_extended_grid,
    find_diag_correction_torch,
)
from solvers.integral_equation.bicgstab_batch import bicgstab_batch

from src.data.data_transformations import prep_conv_interp_2d, apply_interp_2d

DEFAULT_ATOL = 0  # 1e-2
DEFAULT_RTOL = 1e-4

# Start out with single-precision but with an option to bump up the precisions.
# NOTE: these globals are independent of (not shared with) the ones in
# HelmholtzSolverDifferentiable.py -- calling set_solver_types() here does not
# affect that module, and vice versa.
NP_CDTYPE    = np.complex64  # np.cfloat was an alias for this, removed in numpy 2.0
TORCH_CDTYPE = torch.cfloat
TORCH_RDTYPE = torch.float
def set_solver_types(np_cdtype=None, torch_cdtype=None, torch_rdtype=None):
    global NP_CDTYPE, TORCH_CDTYPE, TORCH_RDTYPE
    if np_cdtype is not None:
        NP_CDTYPE = np_cdtype
    if torch_cdtype is not None:
        TORCH_CDTYPE = torch_cdtype
    if torch_rdtype is not None:
        TORCH_RDTYPE = torch_rdtype

def get_solver_types():
    return (NP_CDTYPE, TORCH_CDTYPE, TORCH_RDTYPE)

@torch.compiler.disable(recursive=False)
def to_cfloat(x: torch.Tensor) -> torch.Tensor:
    """Helper function to perform casting
    that won't get disrupted by the torch.compile step
    """
    global TORCH_CDTYPE
    return x.to(TORCH_CDTYPE)


def _dump_nan_debug(tag_prefix: str, scratch_dir: str = "scratch_dir", **arrays) -> str:
    """Shared helper for the NaN-detect-and-save-to-disk pattern used across
    the forward and adjoint solves. Converts any torch.Tensor values in
    **arrays to numpy before saving. Returns the file path written.
    """
    date_str = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    tag = uuid.uuid4().hex[:12]
    debug_fp = f"{scratch_dir}/{date_str}_{tag_prefix}_{tag}.npz"

    debug_contents = {}
    for key, val in arrays.items():
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        debug_contents[key] = val

    np.savez(debug_fp, **debug_contents)
    return debug_fp


def _finalize_batched_output(
    main_list: List[np.ndarray | torch.Tensor],
    sigma_list: List[np.ndarray | torch.Tensor],
    return_as_torch: bool,
    return_sigma: bool,
) -> np.ndarray | torch.Tensor:
    """Shared tail end of Helmholtz_solve_{exterior,interior}_batched: stitches
    together the per-chunk outputs of the batching loop, in either torch or
    numpy form, optionally alongside the concatenated sigma values.
    """
    if return_as_torch:
        output = torch.concatenate(main_list)
        if return_sigma:
            output = (output, torch.concatenate(sigma_list))
        return output

    to_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    output = np.concatenate([to_numpy(x) for x in main_list])
    if return_sigma:
        output = (output, np.concatenate([to_numpy(x) for x in sigma_list]))
    return output


def _check_linsys_solver(linsys_solver: str) -> None:
    """Raise loudly if a non-bicgstab solver is requested. This interface
    intentionally dropped gmres support; use HelmholtzSolverDifferentiable
    if gmres is needed."""
    if linsys_solver is not None and linsys_solver.lower() != "bicgstab":
        raise ValueError(
            f"helmholtz_solver_bicgstab only supports linsys_solver='bicgstab' "
            f"(got {linsys_solver!r}); gmres was dropped in this interface. "
            f"Use HelmholtzSolverDifferentiable for gmres."
        )


class HelmholtzSolverBicgstab:
    def __init__(
        self,
        domain_points: np.ndarray,
        extended_domain_points: np.ndarray,
        G_fft: np.ndarray,
        frequency: float,
        exterior_greens_function: np.ndarray,
        N: int,
        source_dirs: np.ndarray,
        x_vals: np.ndarray,
        max_iter: int = 10,
        diag_correction: float = None,
        G_adjoint_fft: np.ndarray = None,
        device: torch.cuda.device = None,
        prepare_half_grid: bool = False,
        receiver_radius: float = 100,
    ) -> None:
        """Initialize the bicgstab-only differentiable Helmholtz equation
        solver. Solves the Lippmann-Schwinger equation.

        Parameters:
            domain_points (np.ndarray): the (2d) grid points collectively
                Expected shape is (N, N, 2)
            extended_domain_points (np.ndarray): Specify the extended domain
                For use padding the inputs when applying convolutions in the Fourier domain
                Expected shape is something like (3N, 3N, 2)
            G_fft (np.ndarray): the 2D Fourier transform of the interior greens function operator
                Expected shape: (3N, 3N) (or matching the level of padding used in the extended_domain_points)
            frequency (float): the angular frequency, i.e., k=2pi*nu
            exterior_greens_function (np.ndarray): the linear operator mapping the interior solution
                to the scattered wavefield at the exterior ring
            N (int): number of grid points per dimension
            x_vals (np.ndarray): the (1d) grid points for each dimension of the domain
                Expected shape: (N,)
            source_dirs (np.ndarray): source direction angles to use with the solver
                Expected shape: (N_s,)
            max_iter (int): maximum number of iterations for the linear system
                solver to use by default. Can be overridden.
            diag_correction (float): what correction, if any, is applied to the
                diagonal of the Greens function matrix
            G_adjoint_fft (np.ndarray): the 2D Fourier transform of the
                adjoint of the interior greens function operator
                Note that this is different from the adjoint of the G_fft
                object (it appears to be just complex conjugate) since
                we want the kernel corresponding to adjoint of a convolution matrix...
            device (torch.cuda.device): which device to load relevant torch data onto
                if device=None, defaults to a cuda device if available or CPU otherwise
            prepare_half_grid (bool): specify whether to prepare a solver
                for a grid with half as many domain points per side
                Intended to yield a multigrid-based initial estimate
                of the solution to the linear systems
            receiver_radius (float): convenience argument only used if prepare_half_grid=True
                so that we can just call the standard setup function
        """
        self.domain_points = torch.from_numpy(domain_points).to(torch.float)
        self.extended_domain_points = extended_domain_points
        self.frequency = frequency
        self.frequency_torch = torch.Tensor([frequency]).to(torch.float)

        self.N = N
        self.source_dirs = source_dirs
        self.x_vals = x_vals
        self.domain_points_arr = domain_points.reshape((N, N, 2))

        self.h = self.domain_points_arr[0, 1, 0] - self.domain_points_arr[0, 0, 0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device is None else device
        self.exterior_greens_function = (
            torch.from_numpy(exterior_greens_function).to(TORCH_CDTYPE).to(self.device)
        )
        self.G_fft = torch.from_numpy(G_fft).to(self.device)
        self.G_adjoint_fft = torch.from_numpy(G_adjoint_fft).to(self.device)

        self.max_iter = max_iter
        self.diag_correction = diag_correction

        self.prepared_half_grid = prepare_half_grid
        if prepare_half_grid:
            # Set up the half-grid version and also interpolation operators
            half_to_full_xy_x, half_to_full_xy_y = prep_conv_interp_2d(
                x_vals[::2], # x vals
                x_vals[::2], # y vals
                xi=np.array(np.meshgrid(x_vals, x_vals)).T.reshape(N**2, 2),
                bc_modes=("zero", "zero"),
            )
            self.torch_half_to_full_xy_x = torch.tensor(
                half_to_full_xy_x.todense(), dtype=TORCH_CDTYPE, requires_grad=False, device=self.device,
            )
            self.torch_half_to_full_xy_y = torch.tensor(
                half_to_full_xy_y.todense(), dtype=TORCH_CDTYPE, requires_grad=False, device=self.device,
            )

            def half_to_full_grid(x):
                """Expect x with shape (..., N_s//2, N_x//2)
                """
                N_x = self.N
                x_shape = x.shape
                x_3d = x.reshape(-1, N_x//2, N_x//2)
                x_upscaled_3d = torch.zeros(
                    x_3d.shape[0], N_x**2, dtype=x_3d.dtype, device=self.device
                )
                for i in range(x_3d.shape[0]):
                    x_upscaled_3d[i] = apply_interp_2d(
                        self.torch_half_to_full_xy_x,
                        self.torch_half_to_full_xy_y,
                        x_3d[i]
                    )
                out = x_upscaled_3d.reshape(*x_shape[:-2], N_x, N_x)
                return out
            self.half_to_full_grid = half_to_full_grid

            spatial_domain_max = np.abs(-self.x_vals[0])
            self.half_grid_solver = setup_bicgstab_solver(
                N//2, spatial_domain_max, frequency/(2*np.pi), receiver_radius, device=device,
                prepare_half_grid=False,
            )

    def _get_uin(self, source_directions: torch.Tensor) -> torch.Tensor:
        """Returns a plane wave e^{ik<x,s>} sampled at the points x listed in self.domain_points_arr.
        In this equation, k is the angular frequency = self.frequency, and s is the unit vector pointing in
        direction specified by source_directions, in radiana.

        Args:
            source_directions (torch.Tensor): Has shape (n_directions,)

        Returns:
            torch.Tensor: Has shape (n_directions, self.N**2)
        """
        inc = torch.stack(
            [torch.cos(source_directions), torch.sin(source_directions)]
        ).to(torch.float)
        inner_prods = self.domain_points.to(self.device) @ inc

        uin = (
            torch.exp(1j * self.frequency * inner_prods).to(TORCH_CDTYPE).permute(1, 0)
        )
        return uin

    def _get_b(
        self, uin: torch.Tensor, q: torch.Tensor, radially_symmetric: bool = False
    ) -> torch.Tensor:
        """Generates the right-hand side of the integral equation

        int_{x in \\Omega} (I + k^2 diag(q) G) sigma = -k^2 uin q


        If radially_symmetric is True, then we assume q() is radially symmetric
        and return -k^2 q J_0(k|x|) which is also radially symmetric. This is
        used for checking against a reference solution.

        Args:
            uin (torch.Tensor): Has shape (n_directions, N**2)
            q (torch.Tensor): Has shape (N, N)

        Returns:
            torch.Tensor: Has shape (n_directions, N**2)
        """
        if radially_symmetric:
            r_vals = torch.norm(self.domain_points, dim=-1)
            b = (
                -(self.frequency**2)
                * q.flatten()
                * torch.special.bessel_j0(self.frequency_torch * r_vals)
            )
        else:
            b = -(self.frequency**2) * q * uin.permute(1, 0)
        return b.to(TORCH_CDTYPE).cpu().numpy().flatten()


    def _zero_pad(self, v: torch.Tensor, n: int) -> torch.Tensor:
        """v has shape (n_small, n_small, n_dirs) and output has shape (n, n, n_dirs)"""
        o = torch.zeros((n, n, v.shape[2]), dtype=v.dtype, device=self.device)
        o[: v.shape[0], : v.shape[1]] = v
        return o

    def _G_apply(self, x: torch.Tensor, adj: bool=False) -> torch.Tensor:
        """
        Applies the Green's function by
        1. copying x to a larger grid padded with zeros
        2. computing the 2D Fourier transform of x
        3. pointwise multiplying with the 2D FT of the Green's function.
        4. inverting the Fourier transform and undoing the padding step.

        Args:
            x (torch.Tensor): Has shape (self.N**2, n_dirs)
            adj (bool): whether to apply the adjoint instead

        Returns:
            torch.Tensor: Has shape (self.N**2, n_dirs)
        """
        x_shape = x.shape
        batched = False
        if x.ndim == 3:
            # Put the first dimension last so the reshape folds the directions
            # and batches together.
            x = x.permute(1, 2, 0)
            x_shape = x.shape
            batched = True
        x_square = x.reshape(self.N, self.N, -1)
        x_pad = self._zero_pad(x_square, self.G_fft.shape[0])
        x_fft = torch.fft.fft2(x_pad, dim=(0, 1))

        if adj:
            prod = torch.einsum("ab,abc->abc", self.G_adjoint_fft, x_fft)
        else:
            prod = torch.einsum("ab,abc->abc", self.G_fft, x_fft)
        out_ifft = torch.fft.ifft2(prod, dim=(0, 1))
        out = out_ifft[:self.N, :self.N]
        o = out.reshape(x_shape)
        if batched:
            o = o.permute(2, 0, 1)

        return o


    def _solve_Helmholtz_inv(
        self,
        scattering_obj: np.ndarray | torch.Tensor,
        uin: np.ndarray,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        radially_symmetric: bool = False,
        linsys_solver: str = None,
        max_iter: int = None,
        restart: int = 50,
        error_unless_converged: bool = False,
        report_status: bool = False,
        sigma_init: np.ndarray | torch.Tensor = None,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Main code that interfaces with the linear system solves

        Generates a solution to the integral equation

        int_{x in \\Omega} (I + k^2 diag(q) G) sigma = -k^2 uin q

        Args:
            scattering_obj (np.ndarray): Has shape (N, N)
            uin (np.ndarray): Has shape (n_directions, self.N**2)
            rtol (float): relative tolerance for the linear system solves
            atol (float): absolute tolerance for the linear system solves
            radially_symmetric (bool): special option to use an alternate right-hand-side
            linsys_solver (str): retained for call-site compatibility; must be
                "bicgstab" or None (gmres was dropped in this interface)
            max_iter (int): maximum number of iterations to use
            restart (int): restart interval; make sure that this is relatively high for best results.
            error_unless_converged (bool): can optionally throw an error if the
                linear system solve fails to converge to the desired tolerance
            report_status (bool): logs (and prints to stdio) the number of iterations used
            sigma_init (np.ndarray or torch.Tensor): initial value of sigma to start with
            Miscellaneous keyword arguments:
                _solve_Helmholtz_inv_msg (bool): whether to log when this function is called
                verbose (bool): whether the linear solver should log status messages
                convergence_by_dir (bool): option for bicgstab_batch to stop working on the solution
                    for each right-hand-side based on their individual convergence statuses
                log_resid_norm (bool): whether bicgstab_batch should log the norms of the residuals
                    at every iteration (if True) or just the last iteration (if False)

        Returns:
            out (np.ndarray or torch.Tensor): Has shape (n_directions, self.N**2)
        """
        _check_linsys_solver(linsys_solver)
        max_iter = max_iter if max_iter is not None else self.max_iter

        if kwargs.get("_solve_Helmholtz_inv_msg", True):
            logging.debug(f"_solve_Helmholtz_inv: starting (using bicgstab)")

        n = scattering_obj.shape[0]
        q = scattering_obj.flatten().unsqueeze(-1)

        def _matvec_from_torch(x: torch.Tensor) -> torch.Tensor:
            gout = self._G_apply(x)
            term2 = (self.frequency**2) * q * gout
            y = x + to_cfloat(term2)
            return y
        b_bicgstab = -(self.frequency**2 * q * uin.permute(1, 0)).to(TORCH_CDTYPE).unsqueeze(0)
        n_src = b_bicgstab.shape[-1]
        # Convert sigma_init to torch if it is a np.ndarray
        if sigma_init is not None and isinstance(sigma_init, np.ndarray):
            sigma_init = torch.tensor(sigma_init, device=b_bicgstab.device, dtype=TORCH_CDTYPE)

        sigma, out_info = bicgstab_batch(
            _matvec_from_torch,
            b_bicgstab,
            X0=sigma_init.T.unsqueeze(0) if sigma_init is not None else None,
            atol=atol,
            rtol=rtol,
            maxiter=max_iter,
            restart=restart,
            verbose=kwargs.get("verbose", False),
            convergence_by_dir=kwargs.get("convergence_by_dir", False),
            log_resid_norm=kwargs.get("log_resid_norm", True),
        )
        if isinstance(sigma, torch.Tensor) and torch.any(torch.isnan(sigma)) \
           or isinstance(sigma, np.ndarray) and np.any(np.isnan(sigma)):
            logging.info(f"Located a NaN in BiCGSTAB's sigma output")
        if report_status:
            logging.debug(f"bicgstab exited after {out_info['niter']} iterations with "
                  f"status {'optimal' if out_info['optimal'] else 'not optimal'}")
            print(f"bicgstab exited after {out_info['niter']} iterations with "
                  f"status {'optimal' if out_info['optimal'] else 'not optimal'}")
        if out_info["optimal"] != True:
            resid_ratios = out_info['resid_norm_lst'][-1] / out_info['stopping_matrix']
            logging.info(f"Max residual ratio: {np.max(resid_ratios, axis=-1)} (entry {np.argmax(resid_ratios, axis=-1)})")
            logging.info(f"All residual ratios: {resid_ratios}")
        if error_unless_converged and out_info["optimal"] != True:
            raise RuntimeError(f"BiCGSTAB failed to converge. Run info: {out_info}")

        # Reshape sigma as needed to prepare to return the value
        out = sigma.reshape(-1, n_src).permute(1, 0).to(self.device)
        if kwargs.get("_solve_Helmholtz_inv_msg", True):
            logging.debug("_solve_Helmholtz_inv: returning")

        return out


    def _get_uin_sigma(
        self,
        source_directions: torch.Tensor,
        scattering_obj: torch.Tensor,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        radially_symmetric: bool = False,
        linsys_solver: str = None,
        max_iter: int = None,
        sigma_init: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For a batch of source directions, this function generates the incoming
        plane waves uin, and also generates solutions to the integral equation

        int_{x in \\Omega} (I + k^2 diag(q) G) sigma = -k^2 uin q

        Args:
            source_direction (torch.Tensor): Has shape (n_directions,)
            scattering_obj (torch.Tensor): shape (N, N)
            rtol (float, optional): Relative tolerance for the linear solve.
            atol (float, optional): Absolute tolerance for the linear solve.
            radially_symmetric (bool, optional): If True, incident wave field is J_0(k|x|).
            max_iter (int): maximum number of iterations (passed on to self._solve_Helmholtz_inv)
            sigma_init (np.ndarray or torch.Tensor): initial value of sigma to start with

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First output is uin which has shape
            (n_directions, N**2). Second output is sigma, which has shape (n_directions, N**2)
        """
        uin = self._get_uin(source_directions)

        _check_linsys_solver(linsys_solver)

        if torch.all(scattering_obj == torch.zeros_like(scattering_obj)):
            sigma = torch.zeros(
                (source_directions.shape[0], self.N**2),
                dtype=TORCH_CDTYPE, device=self.device,
            )

        else:
            sigma = self._solve_Helmholtz_inv(
                scattering_obj,
                uin,
                rtol=rtol,
                atol=atol,
                radially_symmetric=radially_symmetric,
                linsys_solver=linsys_solver,
                max_iter=max_iter,
                sigma_init=sigma_init,
                **kwargs
            )
        return uin, sigma

    def _half_grid_sigma_init(
        self,
        directions_torch: torch.Tensor,
        scattering_obj_torch: torch.Tensor,
        rtol: float,
        atol: float,
        linsys_solver: str,
        max_iter: int,
        **kwargs,
    ) -> torch.Tensor | None:
        """Multigrid warm-start helper: solve on the half-resolution grid
        (self.half_grid_solver) and interpolate the result up to a sigma
        initializer for the full-grid solve. Returns None -- instead of
        raising -- if the coarse solve itself produced a NaN, after dumping
        debug info to scratch_dir/ so the caller can fall back to a cold
        start.
        """
        scattering_obj_torch_hg = scattering_obj_torch[..., ::2, ::2]
        half_grid_tol_ratio = kwargs.get("half_grid_tol_ratio", 0.5)
        d_rs_half_grid, sigma_init_half_grid = self.half_grid_solver.Helmholtz_solve_exterior(
            directions_torch,
            scattering_obj_torch_hg, # downsampled by a factor of two
            rtol=rtol*half_grid_tol_ratio,
            atol=atol*half_grid_tol_ratio,
            linsys_solver=linsys_solver,
            max_iter=max_iter,
            return_as_torch=True,
            return_sigma=True,
            use_half_grid=False,
            **kwargs,
        )

        if torch.any(torch.isnan(sigma_init_half_grid)):
            debug_fp = _dump_nan_debug(
                "debug_fwd_hg_solver_nan",
                q=scattering_obj_torch_hg,
                sigma_init_hg=sigma_init_half_grid,
                d_rs_hg=d_rs_half_grid,
            )
            msg = (
                f"sigma_init_half_grid contains a NaN! Discarding "
                f"and saving debug info to {debug_fp} file for the inputs/outputs causing this error."
            )
            print(msg)
            logging.info(msg)
            return None

        N_x = self.N
        N_batch = len(directions_torch)
        return self.half_to_full_grid(
            sigma_init_half_grid.reshape(N_batch, N_x//2, N_x//2)
        ).reshape(N_batch, N_x**2).to(self.device)

    def _solve_sigma_chunk(
        self,
        source_directions: np.ndarray | torch.Tensor,
        scattering_obj: np.ndarray | torch.Tensor,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        linsys_solver: str = None,
        max_iter: int = None,
        radially_symmetric: bool = False,
        sigma_init: np.ndarray | torch.Tensor = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core per-chunk sigma solve, shared by _solve_exterior_chunk and
        _solve_interior_chunk. Converts inputs to torch tensors on this
        solver's device, optionally warm-starts from the half-resolution
        grid, and solves the Lippmann-Schwinger integral equation for sigma
        -- the one genuinely expensive step both the exterior (far-field) and
        interior (in-domain) solves build on top of: exterior applies
        exterior_greens_function to sigma to get the far-field d_rs, interior
        applies _G_apply to sigma and adds the incident wave.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (directions_torch, uin, sigma)
        """
        _check_linsys_solver(linsys_solver)

        if isinstance(source_directions, torch.Tensor):
            directions_torch = source_directions
        else:
            directions_torch = torch.from_numpy(source_directions)
        directions_torch = directions_torch.to(self.device)

        if isinstance(scattering_obj, torch.Tensor):
            scattering_obj_torch = scattering_obj
        else:
            scattering_obj_torch = torch.from_numpy(scattering_obj).to(self.device)

        # If requested, prepare sigma_init based off the half grid solver's solution
        if sigma_init is None and use_half_grid and self.prepared_half_grid:
            sigma_init = self._half_grid_sigma_init(
                directions_torch, scattering_obj_torch, rtol, atol, linsys_solver, max_iter, **kwargs,
            )

        uin, sigma = self._get_uin_sigma(
            directions_torch,
            scattering_obj_torch,
            rtol=rtol,
            atol=atol,
            radially_symmetric=radially_symmetric,
            linsys_solver=linsys_solver,
            max_iter=max_iter,
            sigma_init=sigma_init,
            **kwargs,
        )
        return directions_torch, uin, sigma

    def _solve_exterior_chunk(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        linsys_solver: str = None,
        max_iter: int = None,
        return_as_torch: bool = False,
        return_sigma: bool = False,
        sigma_init: np.ndarray = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Core per-chunk solve: solves the Helmholtz equation on the exterior
        ring for exactly the given source_directions, in a single linear
        solve (no internal batching). This is the shared implementation
        underneath both Helmholtz_solve_exterior_batched (which calls this
        once per chunk of batch_size directions) and Helmholtz_solve_exterior
        (the non-batched, backward-compatible entry point, which calls this
        once for the full set of requested directions).

        Returns the scattered wave field. See Helmholtz_solve_exterior_batched
        for the parameter and return documentation.
        """
        _, _, sigma = self._solve_sigma_chunk(
            source_directions, scattering_obj, rtol=rtol, atol=atol,
            linsys_solver=linsys_solver, max_iter=max_iter,
            sigma_init=sigma_init, use_half_grid=use_half_grid, **kwargs,
        )

        FP = self.exterior_greens_function @ sigma.permute(1, 0)

        res = FP.permute(1, 0)
        if not return_as_torch:
            res = res.cpu().numpy()
        if return_sigma:
            res = (res, sigma)
        return res

    def Helmholtz_solve_exterior(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        linsys_solver: str = None,
        max_iter: int = None,
        return_as_torch: bool = False,
        return_sigma: bool = False,
        sigma_init: np.ndarray = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Non-batched, backward-compatible entry point for solving the
        Helmholtz equation on the exterior ring for a given set of source
        directions and a given scattering object.

        Equivalent to calling Helmholtz_solve_exterior_batched with
        batch_size equal to the number of source directions requested here
        (i.e. everything in a single chunk, no internal batching) -- that
        method does the actual solving; see its docstring for the full
        parameter and return documentation.
        """
        n_dirs = source_directions.shape[0]
        return self.Helmholtz_solve_exterior_batched(
            scattering_obj,
            batch_size=n_dirs,
            rtol=rtol,
            atol=atol,
            linsys_solver=linsys_solver,
            max_iter=max_iter,
            return_as_torch=return_as_torch,
            return_sigma=return_sigma,
            source_directions=source_directions,
            sigma_init=sigma_init,
            use_half_grid=use_half_grid,
            **kwargs,
        )

    def Helmholtz_solve_exterior_batched(
        self,
        scattering_obj: np.ndarray | torch.Tensor,
        batch_size: int,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        linsys_solver: str = None,
        max_iter: int = None,
        return_as_torch: bool = False,
        return_sigma: bool = False,
        source_directions: np.ndarray = None, # can optionally override the default sources
        sigma_init: np.ndarray = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Helper function for Helmholtz_solve_exterior that handles the batching
        Intended as the main interface through which to call the PDE solver.
        Solves the Helmholtz Equation and returns the values of the solution on the
        exterior receiver ring

        Args:
            scattering_obj (np.ndarray or torch.Tensor): scattering object in question
                expected to have shape (N_x, N_x); currently code does not handle batching
                over different scattering objects
            batch_size (int): number of right-hand-sides to perform at once
            rtol (float): relative tolerance of the linear system solves
            atol (float): absolute tolerance of the linear system solves
            linsys_solver (str): retained for call-site compatibility; must be "bicgstab" or None
            max_iter (int): maximum number of iterations for the linear system solve
            return_as_torch (bool): whether to return the output as a torch.Tensor object or np.ndarray
            return_sigma (bool): whether to include sigma in the outputs (as the second entry in a tuple)
            source_directions (np.ndarray): optional array of source directions (as angles from 0 to 2pi)
                By default, all the source_directions from the solver setup are used, but this provides an
                option to override the defaults
            sigma_init (np.ndarray): can provide an initial value for sigma to use within the solves
                expected shape: (N_s, N_x**2), regardless of the batch_size value
            use_half_grid (bool): indicate whether to pick an initializer by a multigrid solution,
                i.e., by solving the problem on a grid with half as many points per dimension.
                the PDE solver should have been prepared with the prepare_half_grid=True flag when
                calling setup_bicgstab_solver
            Miscellaneous keyword arguments:
                _solve_Helmholtz_inv_msg (bool): whether to log when this function is called
                verbose (bool): whether the linear solver should log status messages
                convergence_by_dir (bool): option for bicgstab_batch to stop working on the solution
                    for each right-hand-side based on their individual convergence statuses
                log_resid_norm (bool): whether bicgstab_batch should log the norms of the residuals
                    at every iteration (if True) or just the last iteration (if False)

        Returns:
            u_scat_ext_full (np.ndarray or torch.Tensor): scattered wave field on the receiver ring
                for all the source directions; has shape (N_s, N_r)
            sigma (if return_sigma=True; np.ndarray or torch.Tensor): the value of sigma from the
                linear solves; has shape (N_s, N_x**2)
        """
        source_directions = source_directions if source_directions is not None \
            else self.source_dirs
        N_s = source_directions.shape[0]

        u_scat_ext_list = []
        sigma_list = []

        for j in range(0, N_s, batch_size):
            j_upper = min(j+batch_size, N_s)
            j_slice = slice(j, j_upper)
            directions = source_directions[j_slice]
            if sigma_init is not None:
                sigma_init_slice = sigma_init[j_slice]
            else:
                sigma_init_slice = None

            res = self._solve_exterior_chunk(
                directions,
                scattering_obj,
                rtol=rtol,
                atol=atol,
                linsys_solver=linsys_solver,
                max_iter = max_iter,
                return_as_torch=return_as_torch,
                return_sigma=return_sigma,
                sigma_init=sigma_init_slice,
                use_half_grid=use_half_grid,
                **kwargs,
            )
            if return_sigma:
                u_scat_ext, sigma_part = res
                sigma_list.append(sigma_part)
            else:
                u_scat_ext = res
            u_scat_ext_list.append(u_scat_ext)

        return _finalize_batched_output(u_scat_ext_list, sigma_list, return_as_torch, return_sigma)


    def _solve_interior_chunk(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        linsys_solver: str = None,
        max_iter: int = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        radially_symmetric: bool = False,
        return_as_torch: bool = False,
        return_sigma: bool = False,
        sigma_init: np.ndarray = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core per-chunk solve: solves the Helmholtz equation on the
        scattering domain for exactly the given source_directions, in a
        single linear solve (no internal batching). This is the shared
        implementation underneath both Helmholtz_solve_interior_batched
        (which calls this once per chunk of batch_size directions, keeping
        only the u_scat component for its own return value) and
        Helmholtz_solve_interior (the non-batched, backward-compatible entry
        point, which calls this once for the full set of requested
        directions and returns the full (u_tot, u_in, u_scat) triple).

        Returns the total wave field, the incident wave field, and the
        scattered wave field.

        Args:
            source_directions (np.ndarray): Angle in radians
            scattering_obj (np.ndarray or torch.Tensor): scattering object in question
                expected to have shape (N_x, N_x); currently code does not handle batching
                over different scattering objects
            rtol (float): relative tolerance of the linear system solves
            atol (float): absolute tolerance of the linear system solves
            linsys_solver (str): retained for call-site compatibility; must be "bicgstab" or None
            max_iter (int): maximum number of iterations for the linear system solve
            radially_symmetric (bool): special case that sets a different the right-hand-side
            return_as_torch (bool): whether to return the output as a torch.Tensor object or np.ndarray
            return_sigma (bool): whether to include sigma in the outputs (as the second entry in a tuple)
            sigma_init (np.ndarray): can provide an initial value for sigma to use within the solves
                expected shape: (N_s, N_x**2), regardless of the batch_size value
            use_half_grid (bool): indicate whether to pick an initializer by a multigrid solution,
                i.e., by solving the problem on a grid with half as many points per dimension.
                the PDE solver should have been prepared with the prepare_half_grid=True flag when
                calling setup_bicgstab_solver
            Miscellaneous keyword arguments
                _solve_Helmholtz_inv_msg (bool): whether to log when this function is called
                verbose (bool): whether the linear solver should log status messages
                convergence_by_dir (bool): option for bicgstab_batch to stop working on the solution
                    for each right-hand-side based on their individual convergence statuses
                log_resid_norm (bool): whether bicgstab_batch should log the norms of the residuals
                    at every iteration (if True) or just the last iteration (if False)
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (u_tot, u_in, u_scat).
            Each has shape (N, N)
        """
        n_directions = source_directions.shape[0]
        out_shape = (n_directions, self.N, self.N)
        _, uin, sigma = self._solve_sigma_chunk(
            source_directions, scattering_obj, rtol=rtol, atol=atol,
            linsys_solver=linsys_solver, max_iter=max_iter,
            radially_symmetric=radially_symmetric, sigma_init=sigma_init,
            use_half_grid=use_half_grid, **kwargs,
        )

        if radially_symmetric:
            r_vals = torch.norm(self.domain_points, dim=-1)
            uin = torch.special.bessel_j0(self.frequency_torch * r_vals)
            out_shape = (1, self.N, self.N)

        u_scat = self._G_apply(sigma.permute(1, 0)).permute(1, 0)
        u_tot  = u_scat + uin

        if return_as_torch:
            res = tuple([u.reshape(out_shape) for u in (u_tot, uin, u_scat)])
        else:
            res = tuple([u.reshape(out_shape).cpu().numpy() for u in (u_tot, uin, u_scat)])

        if return_sigma:
            res = (res, sigma)
        return res

    def Helmholtz_solve_interior(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        linsys_solver: str = None,
        max_iter: int = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        radially_symmetric: bool = False,
        return_as_torch: bool = False,
        return_sigma: bool = False,
        sigma_init: np.ndarray = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Non-batched, backward-compatible entry point for solving the
        Helmholtz equation on the scattering domain for a given set of source
        directions and a given scattering object. Returns the total wave
        field, the incident wave field, and the scattered wave field.

        Solves for exactly the given source_directions in a single chunk (no
        internal batching), via the same per-chunk core used by
        Helmholtz_solve_interior_batched -- but unlike that batched method
        (which, for backward compatibility, only returns the u_scat
        component when looping over multiple chunks), this returns the full
        (u_tot, u_in, u_scat) triple in one call. See _solve_interior_chunk
        for the full parameter documentation.
        """
        return self._solve_interior_chunk(
            source_directions, scattering_obj,
            linsys_solver=linsys_solver, max_iter=max_iter, rtol=rtol, atol=atol,
            radially_symmetric=radially_symmetric, return_as_torch=return_as_torch,
            return_sigma=return_sigma, sigma_init=sigma_init, use_half_grid=use_half_grid, **kwargs,
        )

    def Helmholtz_solve_interior_batched(
        self,
        scattering_obj: np.ndarray,
        batch_size: int,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        radially_symmetric: bool = False,
        linsys_solver: str = None,
        max_iter: int = None,
        return_as_torch: bool = False,
        return_sigma: bool = False,
        source_directions: np.ndarray = None,
        sigma_init: np.ndarray = None,
        use_half_grid: bool = False,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Helper function for Helmholtz_solve_interior that handles the batching.
        Note: for backward compatibility, only the u_scat component of each
        chunk's (u_tot, u_in, u_scat) triple is kept here -- call
        Helmholtz_solve_interior directly if you need u_tot/u_in as well.

        use_half_grid (bool): as in Helmholtz_solve_exterior_batched, warm-
            start each chunk's solve via a coarse solve on the half-resolution
            grid (requires prepare_half_grid=True at solver setup)."""
        source_directions = source_directions if source_directions is not None \
            else self.source_dirs
        N_s = source_directions.shape[0]

        u_scat_int_list = []
        sigma_list = []

        for j in range(0, N_s, batch_size):
            j_upper = min(j+batch_size, N_s)
            j_slice = slice(j, j_upper)
            directions = source_directions[j_slice]
            if sigma_init is not None:
                sigma_init_slice = sigma_init[j_slice]
            else:
                sigma_init_slice = None
            res = self._solve_interior_chunk(
                directions,
                scattering_obj,
                rtol=rtol,
                atol=atol,
                radially_symmetric=radially_symmetric,
                linsys_solver=linsys_solver,
                max_iter = max_iter,
                return_as_torch=return_as_torch,
                return_sigma=return_sigma,
                sigma_init=sigma_init_slice,
                use_half_grid=use_half_grid,
                **kwargs,
            )
            if return_sigma:
                u_scat_tup, sigma_part = res
                sigma_list.append(sigma_part)
            else:
                u_scat_tup = res
            u_scat_int = u_scat_tup[-1]
            u_scat_int_list.append(u_scat_int)

        return _finalize_batched_output(u_scat_int_list, sigma_list, return_as_torch, return_sigma)


def setup_bicgstab_solver(
    n_pixels: int,
    spatial_domain_max: float,
    wavenumber: float,
    receiver_radius: float,
    diag_correction: bool = True,
    max_iter: int = 1_000_000,
    n_dirs: int = None,
    device: torch.cuda.device = None,
    prepare_half_grid: bool = False,
    pad_fft_factor: float = None,
) -> HelmholtzSolverBicgstab:
    """Precomputes objects that are reused across different PDE solves.

    Args:
        n_pixels (int): The number of spatial points along each axis of the
            scattering domain. Also, the number of source/receiver
            directions.
        spatial_domain_max (float): the maximum grid value in the spatial domain
            the spatial domain is assumed to be symmetric about zero in each dimension,
            so [-spatial_domain_max, spatial_domain_max]^2
            Usually this value is 0.5.
        wavenumber (float): The non-angular wavenumber being used in the problem. This is
            the number of waves across the spatial domain.
        receiver_radius (float): the radius of the receiver ring at which the measurements are taken
        diag_correction (bool): whether to compute the diagonal correction while setting up the greens function
            (this is good to deal with the singularity that occurs at zero for the greens function)
        max_iter (int): default number of iterations for the linear solvers to use
        n_dirs (int): number of source/receiver directions to use
        device (torch.cuda.device): can specify a cuda device, in which case the setup will take place
            on that device rather than the CPU
        prepare_half_grid (bool): whether to set up the solver for a grid with half as many grid points per dimension
        pad_fft_factor (float): optionally override the default FFT padding, but it seems best to leave this alone

    Returns:
        HelmholtzSolverBicgstab object
    """

    frequency = 2 * np.pi * wavenumber

    if n_dirs is None:
        n_dirs = n_pixels

    # Set up the relevant grids
    source_receiver_directions = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    x = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    y = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y)
    domain_points_list = np.array([X.flatten(), Y.flatten()]).T

    extended_domain_points_grid = get_extended_grid(n_pixels, h, pad_fft_factor=pad_fft_factor)

    receiver_points = (
        receiver_radius
        * np.array(
            [np.cos(source_receiver_directions), np.sin(source_receiver_directions)]
        ).T
    )

    # Calculate the diagonal correction if requested (using a cuda device if available)
    if diag_correction:
        print(f"Calling find_diag_correction(h={h:.4e}, frequency={frequency:.2f})...")
        if device is None:
            diag_correction_val = find_diag_correction(h, frequency)
        else:
            diag_correction_val = find_diag_correction_torch(h, frequency, device)
    else:
        diag_correction_val = None

    # Prepare the Green's function in normal space
    G_int = greensfunction3(
        extended_domain_points_grid,
        frequency,
        diag_correction=diag_correction_val,
        dx=h,
    )

    # Convert to Fourier space to apply convolutions quickly
    G_int_fft = np.fft.fft2(G_int)
    G_int_adjoint_fft = G_int_fft.conj()

    # This takes ~20% of the runtime
    exterior_greens_function = getGscat2circ(
        domain_points_list, receiver_points, frequency, dx=h,
    )

    # This takes ~75% of the runtime, mostly dominated by the prep_conv_interp_2d step
    out = HelmholtzSolverBicgstab(
        domain_points_list,
        extended_domain_points_grid,
        G_int_fft,
        frequency,
        exterior_greens_function,
        n_pixels,
        source_receiver_directions,
        x,
        max_iter=max_iter,
        diag_correction=diag_correction_val,
        G_adjoint_fft=G_int_adjoint_fft,
        device=device,
        prepare_half_grid=prepare_half_grid,
        receiver_radius=receiver_radius,
    )
    return out

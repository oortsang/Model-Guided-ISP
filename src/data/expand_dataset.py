# expand_dataset.py
# Regenerates fields dropped by shrink_dataset.py (q_polar in scattering-
# object files; q_cart, q_polar, q_cart_lpf, q_polar_lpf, d_mh in
# measurement files), operating in place on individual HDF5 files.
#
# Reference math for these transforms lives in generate_scattering_files.py
# and generate_measurement_files.py; this module mirrors that math but
# batches the interpolation/filtering across all samples in a file at once
# instead of looping per-sample.

from dataclasses import dataclass
from typing import Callable, Literal
import logging

import numpy as np
import torch

from src.data.data_io import (
    load_field_in_hdf5,
    save_field_in_hdf5,
    get_fields_in_hdf5,
)
from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    Q_CART_LPF,
    Q_POLAR_LPF,
    D_RS,
    D_MH,
)
from src.data.data_transformations import (
    prep_conv_interp_2d,
    apply_interp_2d_batched,
    prep_rs_to_mh_interp,
    polar_to_euclidean,
    CONST_D_MH_SCALE_FACTOR,
)
from src.data.lowpass_filter import prep_lpf_from_wavenum, apply_filter_fourier_2d
from src.data.shrink_dataset import copy_dataset  # re-exported for convenience

Backend = Literal["torch", "jax"]


### Operator factories ###

def prepare_cart_to_polar_transform(
    x_vals: np.ndarray,
    theta_vals: np.ndarray,
    rho_vals: np.ndarray,
    backend: Backend = "torch",
    sparse_x: bool = False,
    device=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Builds a batched cart->polar interpolation closure.

    Mirrors the reference math in generate_scattering_files.py /
    generate_measurement_files.py (prep_conv_interp_2d over the cartesian
    grid, sampled at the polar grid points, bc_modes="extend").

    The returned closure accepts (N_x, N_x) or (N_batch, N_x, N_x) real
    arrays and returns (N_theta, N_rho) or (N_batch, N_theta, N_rho) arrays.
    """
    N_theta = theta_vals.shape[0]
    N_rho = rho_vals.shape[0]
    polar_grid = polar_to_euclidean(theta_vals, rho_vals)

    if backend == "torch":
        interp_op_x, interp_op_y = prep_conv_interp_2d(
            x_vals, x_vals, polar_grid, bc_modes=("extend", "extend"), a_neg_half=True,
        )
        interp_op_x = torch.tensor(
            interp_op_x.todense(), dtype=torch.float32, device=device, requires_grad=False,
        )
        interp_op_y = torch.tensor(
            interp_op_y.todense(), dtype=torch.float32, device=device, requires_grad=False,
        )

        def cart_to_polar(q_cart: np.ndarray) -> np.ndarray:
            batched = q_cart.ndim == 3
            arr = q_cart if batched else q_cart[np.newaxis]
            arr_t = torch.as_tensor(arr, dtype=torch.float32, device=device)
            out = apply_interp_2d_batched(interp_op_x, interp_op_y, arr_t)
            out = out.reshape(out.shape[0], N_theta, N_rho)
            out = out.cpu().numpy()
            return out if batched else out[0]

        return cart_to_polar

    elif backend == "jax":
        import jax.numpy as jnp
        from solvers.hps.wave_scattering.interp_utils import (
            prep_conv_interp_2d as iu_prep_conv_interp_2d,
            apply_conv_interp_2d as iu_apply_conv_interp_2d,
        )

        interp_op_x, interp_op_y = iu_prep_conv_interp_2d(
            x_vals, x_vals, polar_grid, bc_modes=("extend", "extend"), a_neg_half=True,
            use_jax=True, use_sparse_ops=sparse_x,
        )

        def cart_to_polar(q_cart: np.ndarray) -> np.ndarray:
            batched = q_cart.ndim == 3
            arr = q_cart if batched else q_cart[np.newaxis]
            # interp_utils.apply_conv_interp_2d expects grid dims first, batch trailing
            arr_t = jnp.asarray(arr).transpose(1, 2, 0)
            out = iu_apply_conv_interp_2d(interp_op_x, interp_op_y, arr_t)  # (N_theta*N_rho, N_batch)
            out = out.transpose(1, 0).reshape(-1, N_theta, N_rho)
            out = np.asarray(out)
            return out if batched else out[0]

        return cart_to_polar

    else:
        raise ValueError(f"Unrecognized backend {backend!r}; expected 'torch' or 'jax'")


def prepare_rs_to_mh_transform(
    theta_vals: np.ndarray,
    num_m: int,
    num_h: int,
    backend: Backend = "torch",
    sparse_x: bool = False,
    device=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Builds a batched rs->mh interpolation closure (includes the
    CONST_D_MH_SCALE_FACTOR geometric-spreading correction, matching
    generate_measurement_files.py).

    The returned closure accepts (N_theta, N_theta) or (N_batch, N_theta,
    N_theta) complex arrays and returns (num_m, num_h) or
    (N_batch, num_m, num_h) complex arrays.
    """
    if backend == "torch":
        conv_rs_to_m, conv_rs_to_h = prep_rs_to_mh_interp(
            theta_vals, theta_vals, num_m, num_h, a_neg_half=True,
        )
        op_m = torch.tensor(
            conv_rs_to_m.todense(), dtype=torch.complex64, device=device, requires_grad=False,
        )
        op_h = torch.tensor(
            conv_rs_to_h.todense(), dtype=torch.complex64, device=device, requires_grad=False,
        )

        def rs_to_mh(d_rs: np.ndarray) -> np.ndarray:
            batched = d_rs.ndim == 3
            arr = d_rs if batched else d_rs[np.newaxis]
            arr_t = torch.as_tensor(arr, dtype=torch.complex64, device=device)
            out = apply_interp_2d_batched(op_m, op_h, arr_t)
            out = out.reshape(out.shape[0], num_m, num_h) * CONST_D_MH_SCALE_FACTOR
            out = out.cpu().numpy()
            return out if batched else out[0]

        return rs_to_mh

    elif backend == "jax":
        import jax.numpy as jnp
        from solvers.hps.wave_scattering.interp_utils import (
            prep_conv_interp_2d as iu_prep_conv_interp_2d,
            apply_conv_interp_2d as iu_apply_conv_interp_2d,
        )

        # Replicates prep_rs_to_mh_interp's (m,h) -> (r,s) coordinate-grid
        # construction (src/data/data_transformations.py) directly, since
        # there's no jax-capable equivalent to call.
        grid_m = np.linspace(0, 2 * np.pi, num_m, endpoint=False)
        grid_h = np.linspace(-np.pi / 2, np.pi / 2, num_h, endpoint=False)
        grid_mh = np.array(np.meshgrid(grid_m, grid_h)).T.reshape(num_m * num_h, 2)
        grid_mh_in_rs_coords = np.array([
            np.mod(grid_mh[:, 0] + grid_mh[:, 1], 2 * np.pi),
            np.mod(grid_mh[:, 0] - grid_mh[:, 1], 2 * np.pi),
        ]).T

        conv_rs_to_m, conv_rs_to_h = iu_prep_conv_interp_2d(
            theta_vals, theta_vals, grid_mh_in_rs_coords,
            bc_modes=("periodic", "periodic"), a_neg_half=True,
            use_jax=True, use_sparse_ops=sparse_x,
        )

        def rs_to_mh(d_rs: np.ndarray) -> np.ndarray:
            batched = d_rs.ndim == 3
            arr = d_rs if batched else d_rs[np.newaxis]
            arr_t = jnp.asarray(arr).transpose(1, 2, 0)
            out = iu_apply_conv_interp_2d(conv_rs_to_m, conv_rs_to_h, arr_t)  # (num_m*num_h, N_batch)
            out = out.transpose(1, 0).reshape(-1, num_m, num_h) * CONST_D_MH_SCALE_FACTOR
            out = np.asarray(out)
            return out if batched else out[0]

        return rs_to_mh

    else:
        raise ValueError(f"Unrecognized backend {backend!r}; expected 'torch' or 'jax'")


def prepare_gaussian_lpf(
    nu_sf: float, num_x: int, pad_mode: str = "power-of-two",
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the (x, y) Gaussian low-pass filter pair for one frequency,
    matching generate_measurement_files.py's convention of nu_lpf = 2*nu_sf.
    """
    nu_lpf = 2 * nu_sf
    lpf_x, _, _ = prep_lpf_from_wavenum(nu_lpf, num_x, pad_mode=pad_mode)
    lpf_y = np.copy(lpf_x)
    return lpf_x, lpf_y


### Transform containers ###

@dataclass
class DatasetTransforms:
    """Frequency-independent operators, built once for the whole dataset."""
    cart_to_polar: Callable[[np.ndarray], np.ndarray]
    rs_to_mh: Callable[[np.ndarray], np.ndarray]
    backend: Backend = "torch"

    @classmethod
    def from_grids(
        cls,
        x_vals: np.ndarray,
        theta_vals: np.ndarray,
        rho_vals: np.ndarray,
        h_vals: np.ndarray,
        backend: Backend = "torch",
        sparse_x: bool = False,
        device=None,
    ) -> "DatasetTransforms":
        cart_to_polar = prepare_cart_to_polar_transform(
            x_vals, theta_vals, rho_vals, backend=backend, sparse_x=sparse_x, device=device,
        )
        rs_to_mh = prepare_rs_to_mh_transform(
            theta_vals, num_m=theta_vals.shape[0], num_h=h_vals.shape[0],
            backend=backend, sparse_x=sparse_x, device=device,
        )
        return cls(cart_to_polar=cart_to_polar, rs_to_mh=rs_to_mh, backend=backend)


@dataclass
class FrequencyTransforms:
    """The one operator that's per-frequency (nu-dependent): the Gaussian LPF."""
    nu_sf: float
    gaussian_lpf_x: np.ndarray
    gaussian_lpf_y: np.ndarray

    @classmethod
    def from_nu(
        cls, nu_sf: float, num_x: int, pad_mode: str = "power-of-two",
    ) -> "FrequencyTransforms":
        lpf_x, lpf_y = prepare_gaussian_lpf(nu_sf, num_x, pad_mode=pad_mode)
        return cls(nu_sf=nu_sf, gaussian_lpf_x=lpf_x, gaussian_lpf_y=lpf_y)

    def apply_lpf(self, q_cart: np.ndarray) -> np.ndarray:
        return apply_filter_fourier_2d(q_cart, self.gaussian_lpf_x, self.gaussian_lpf_y)


### Per-file, in-place expand functions ###

def expand_scobj_hdf5(
    fp: str,
    dataset_transforms: DatasetTransforms,
    exists_ok: bool = False,
) -> None:
    """Regenerates q_polar (from q_cart) in-place in a scattering-object file.
    Skips entirely (no load, no compute, no write) if q_polar is already
    present and exists_ok=False.
    """
    existing = set(get_fields_in_hdf5(fp))
    if Q_POLAR in existing and not exists_ok:
        logging.info(f"{fp}: {Q_POLAR} already present, skipping")
        return

    q_cart = load_field_in_hdf5(Q_CART, fp)
    q_polar = dataset_transforms.cart_to_polar(q_cart)
    save_field_in_hdf5(Q_POLAR, q_polar, fp, overwrite=exists_ok)


def expand_meas_hdf5(
    meas_fp: str,
    scobj_fp: str,
    dataset_transforms: DatasetTransforms,
    freq_transforms: FrequencyTransforms,
    exists_ok: bool = False,
) -> None:
    """Regenerates q_cart, q_polar, q_cart_lpf, q_polar_lpf, and d_mh
    in-place in a measurement file. q_cart and q_polar are just copied over
    from the paired scobj_fp (meas files don't carry them after a shrink
    with drop_scobj_from_meas=True) -- run expand_scobj_hdf5 on scobj_fp
    first if q_polar might also be missing there.

    Each field is only (re)computed/copied if it's missing, or if
    exists_ok=True forces a recompute. If q_polar_lpf needs recomputing but
    q_cart_lpf is already present (and not being forced), the existing
    q_cart_lpf is loaded rather than recomputed from q_cart -- avoids
    redundant LPF work on a dataset that wasn't maximally shrunk.
    """
    existing = set(get_fields_in_hdf5(meas_fp))

    need_cart = exists_ok or (Q_CART not in existing)
    need_polar = exists_ok or (Q_POLAR not in existing)
    need_cart_lpf = exists_ok or (Q_CART_LPF not in existing)
    need_polar_lpf = exists_ok or (Q_POLAR_LPF not in existing)

    q_cart = None
    if need_cart or need_cart_lpf:
        q_cart = load_field_in_hdf5(Q_CART, scobj_fp)
    if need_cart:
        save_field_in_hdf5(Q_CART, q_cart, meas_fp, overwrite=exists_ok)
    if need_polar:
        q_polar = load_field_in_hdf5(Q_POLAR, scobj_fp)
        save_field_in_hdf5(Q_POLAR, q_polar, meas_fp, overwrite=exists_ok)

    if need_cart_lpf or need_polar_lpf:
        if need_cart_lpf:
            q_cart_lpf = freq_transforms.apply_lpf(q_cart)
        else:
            q_cart_lpf = load_field_in_hdf5(Q_CART_LPF, meas_fp)

        if need_cart_lpf:
            save_field_in_hdf5(Q_CART_LPF, q_cart_lpf, meas_fp, overwrite=exists_ok)
        if need_polar_lpf:
            q_polar_lpf = dataset_transforms.cart_to_polar(q_cart_lpf)
            save_field_in_hdf5(Q_POLAR_LPF, q_polar_lpf, meas_fp, overwrite=exists_ok)

    need_d_mh = exists_ok or (D_MH not in existing)
    if need_d_mh:
        if D_RS not in existing:
            logging.warning(
                f"{meas_fp}: d_mh needs (re)computing but d_rs is not present; skipping"
            )
        else:
            d_rs = load_field_in_hdf5(D_RS, meas_fp)
            d_mh = dataset_transforms.rs_to_mh(d_rs)
            save_field_in_hdf5(D_MH, d_mh, meas_fp, overwrite=exists_ok)

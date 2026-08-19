"""
This file has helper functions used in our experiments with noisy inputs.
"""

import numpy as np
import torch
import logging

def get_norm(
    d: np.ndarray | torch.Tensor,
    axis: tuple=(-1, -2),
    norm_mode: str=None,
    keepdims: bool=True,
) -> np.ndarray | torch.Tensor:
    """Helper function that fetches the relevant norm
    Defaults to infinity norm to make it easier to re-run experiments
    """
    # norm_mode = norm_mode.lower() if norm_mode is not None else "l2"
    norm_mode = norm_mode.lower() if norm_mode is not None else "inf"

    # Decide whether to use pytorch or numpy
    if isinstance(d, torch.Tensor):
        norm_linf = lambda x: torch.amax(torch.abs(d), axis=axis, keepdim=keepdims)
        norm_l2   = lambda x: torch.linalg.norm(d,  axis=axis, keepdim=keepdims)
    else:
        # in case of numpy
        norm_linf = lambda x: np.amax(np.abs(d), axis=axis, keepdims=keepdims)
        norm_l2   = lambda x: np.linalg.norm(d,  axis=axis, keepdims=keepdims)

    # Choose the norm scaling
    if norm_mode == "inf" or norm_mode == "linf":
        norm_fn = norm_linf
    elif norm_mode == "l2" or norm_mode == 2 or norm_mode == "2":
        norm_fn = norm_l2
    else:
        raise ValueError(
            f"get_norm received norm_mode={norm_mode} but expects either 'inf' or 'l2'"
        )

    return norm_fn(d)

def get_noise_scaling(
    d: np.ndarray | torch.Tensor,
    noise_ratio: float,
    axis: tuple=(-1, -2),
    norm_mode: str=None,
    keepdims: bool=True,
) -> np.ndarray | torch.Tensor:
    """Computes the noise scaling; can handle noise ratios with respect to
    l_2 or l_infinity norms
    """
    norm_mode = norm_mode.lower() if norm_mode is not None else "inf"
    d_norm = get_norm(d, axis=axis, norm_mode=norm_mode, keepdims=keepdims)

    if norm_mode == "inf" or norm_mode == "linf":
        norm_factor = 1
    elif norm_mode == "l2" or norm_mode == 2 or norm_mode == "2":
        norm_factor = np.sqrt(np.prod(np.take(d.shape, axis)))
    else:
        raise ValueError(
            f"get_norm received norm_mode={norm_mode} but expects either 'inf' or 'l2'"
        )

    scaling_factor = noise_ratio * d_norm / norm_factor

    # Also reduce the scaling if d is complex
    d_is_complex = (
        torch.is_complex(d)
        if isinstance(d, torch.Tensor)
        else np.iscomplexobj(d)
    )
    if d_is_complex:
        scaling_factor *= np.sqrt(0.5)

    return scaling_factor

def add_noise_to_d(
    d: np.ndarray | torch.Tensor,
    noise_to_sig_ratio: float,
    noise_seed: int | list=None,
    seed_mode: str=None,
    norm_mode: str=None,
) -> np.ndarray | torch.Tensor:
    """Additive noise of the form
    d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
    where noise is standard normal.

    Assumes d has at least 3 dimensions. The first dimensions are batch dimensions, and the last 2 are the
    dimensions of the individual samples to which we are adding noise.

    This function checks the type of the input and calls the corresponding function to add noise.
    """
    assert (
        len(d.shape) > 2
    ), "The input must have at least 3 dimensions. It batches over everything but the final 2 dimensions."
    if isinstance(d, np.ndarray):
        if noise_seed is None or seed_mode not in ["shared", "sequential", None]:
            return _add_noise_numpy(
                d,
                noise_to_sig_ratio,
                norm_mode=norm_mode,
            )
        else:
            return _add_noise_numpy_seeded(
                d,
                noise_to_sig_ratio,
                noise_seed,
                seed_mode=seed_mode,
                norm_mode=norm_mode,
            )
    elif isinstance(d, torch.Tensor):
        if noise_seed is None or seed_mode not in ["shared", "sequential", None]:
            return _add_noise_torch(
                d,
                noise_to_sig_ratio,
                norm_mode=norm_mode,
            )
        else:
            # Torch does not offer the same sort of rng object
            # so for safety we simply sample the noise in numpy
            np_d_in    = d.detach().cpu().numpy()
            np_d_noisy = _add_noise_numpy_seeded(
                np_d_in,
                noise_to_sig_ratio,
                noise_seed,
                seed_mode=seed_mode,
            )
            torch_d_noisy = torch.tensor(np_d_noisy, dtype=d.dtype).to(d.device)
            return torch_d_noisy
    else:
        raise TypeError("d must be either a numpy array or a torch tensor.")

# # Original version of the code; does not support seeds
# def add_noise_to_d(
#     d: np.ndarray | torch.Tensor,
#     noise_to_sig_ratio: float,
# ) -> np.ndarray | torch.Tensor:
#     """Additive noise of the form
#     d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
#     where noise is standard normal.

#     Assumes d has at least 3 dimensions. The first dimensions are batch dimensions, and the last 2 are the
#     dimensions of the individual samples to which we are adding noise.

#     This function checks the type of the input and calls the corresponding function to add noise.
#     """
#     assert (
#         len(d.shape) > 2
#     ), "The input must have at least 3 dimensions. It batches over everything but the final 2 dimensions."
#     if isinstance(d, np.ndarray):
#         return _add_noise_numpy(d, noise_to_sig_ratio)
#     elif isinstance(d, torch.Tensor):
#         return _add_noise_torch(d, noise_to_sig_ratio)
#     else:
#         raise TypeError("d must be either a numpy array or a torch tensor.")


def _add_noise_numpy(
    d: np.ndarray,
    noise_to_sig_ratio: float,
    norm_mode: str=None
) -> np.ndarray:
    """Adds additive random normal noise to the input.
    If d_tilde is the noisy array, we want to have:
    d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
    where noise is standard normal.

    This function will check wheter the input is complex or not, and will add noise accordingly.
    """

    # Computes the norms of the matrices in the last 2 dimensions. Assumes the dimensions coming before that
    # are batch dimensions.
    # d_norm = np.linalg.norm(d, axis=(-2, -1), keepdims=True)
    # d_norm = get_norm(d, norm_mode=norm_mode, axis=(-2, -1))
    # norm_factor = np.sqrt(d.shape[-2] * d.shape[-1])

    noise = np.random.normal(size=d.shape).astype(d.dtype)
    scaling_factor = get_noise_scaling(
        d,
        noise_to_sig_ratio,
        axis=(-2, -1),
        keepdims=True,
        norm_mode=norm_mode
    )

    if np.iscomplexobj(d):
        logging.debug("Adding complex values to the noise array.")
        noise += 1j * np.random.normal(size=d.shape).astype(d.dtype)
        # scaling_factor /= np.sqrt(2)
    # return (d + noise_to_sig_ratio * scaling_factor * noise).astype(d.dtype)
    return (d + scaling_factor * noise).astype(d.dtype)


def _add_noise_torch(
    d: torch.Tensor,
    noise_to_sig_ratio: float,
    norm_mode: str=None
) -> torch.Tensor:
    """Adds additive random normal noise to the input.
    If d_tilde is the noisy array, we want to have:
    d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
    where noise is standard normal.

    This function will check wheter the input is complex or not, and will add noise accordingly.
    """
    # d_norm = torch.linalg.norm(d, dim=(-2, -1), keepdim=True)
    # d_norm = get_norm(d, norm_mode=norm_mode, axis=(-2, -1))
    scaling_factor = get_noise_scaling(
        d,
        noise_to_sig_ratio,
        axis=(-2, -1),
        keepdims=True,
        norm_mode=norm_mode
    )

    noise = torch.randn(d.shape).to(d.dtype)
    # norm_factor = torch.sqrt(
    #     torch.Tensor(
    #         [
    #             d.shape[-2] * d.shape[-1],
    #         ]
    #     )
    # )
    if torch.is_complex(d):
        logging.debug("Adding complex values to the noise array.")
        noise += 1j * torch.randn(d.shape)
        # norm_factor *= torch.sqrt(
        #     torch.Tensor(
        #         [
        #             2.0,
        #         ]
        #     )
        # )
    # return (d + noise_to_sig_ratio * d_norm / norm_factor * noise).to(d.dtype)
    return (d + scaling_factor * noise).to(d.dtype)


def _add_noise_numpy_seeded(
    d: np.ndarray,
    noise_to_sig_ratio: float,
    noise_seed: int | list,
    seed_mode: str = None,
    norm_mode: str = None,
) -> np.ndarray:
    """Specify a seed and use the numpy rng object for guaranteed consistent behavior

    There are two seed modes:
    1. "sequential," in which offsets of the seed are used based on the first index of d
        e.g., if noise_seed=2341 then sample 0 will use 2341, sample 1 uses 2342, etc.
        This should be a bit more reliable in case file shard sizes are ever changed
    2. "shared," in which a single seed is used to generate the noise instance
        for the entire d array at once
        This should be a bit faster, but it is likely a bit less reproducible

    If noise_seed is passed as a list, each entry in the list will be set the
    base noise seed for the second axis
    """
    seed_mode = seed_mode.lower() if seed_mode is not None else "sequential"

    # d_norm = np.linalg.norm(d, axis=(-2, -1), keepdims=True)
    # d_norm = get_norm(d, norm_mode=norm_mode)
    # norm_factor = np.sqrt(d.shape[-2] * d.shape[-1])
    scaling_factor = get_noise_scaling(
        d,
        noise_to_sig_ratio,
        axis=(-2, -1),
        keepdims=True,
        norm_mode=norm_mode
    )

    # Generate real/imaginary noise arrays from standard normals
    if seed_mode == "shared":
        # Use a single seed for everything
        rng = np.random.default_rng(noise_seed)
        # Generate real and imaginary objects at the same time,
        # even if we only use the real ones
        noise_real = rng.normal(size=d.shape) # .astype(d.dtype)
        noise_imag = rng.normal(size=d.shape) # .astype(d.dtype)
    elif seed_mode == "sequential":
        noise_real = np.zeros(d.shape)
        noise_imag = np.zeros(d.shape)
        # I think that adding noise frequency-wise should look something like this...
        if isinstance(noise_seed, list) and len(noise_seed) == d.shape[1]:
            logging.info(f"Received noise seed list {noise_seed}")
            entry_shape = d.shape[-2:]
            for j in range(d.shape[1]):
                fj_noise_base = noise_seed[j]
                for i in range(d.shape[0]):
                    rng = np.random.default_rng(fj_noise_base + i)
                    noise_real[i, j] = rng.normal(size=entry_shape)
                    noise_imag[i, j] = rng.normal(size=entry_shape)
        else:
            entry_shape = d.shape[1:]
            for i in range(d.shape[0]):
                # Use a new seed per sample, offset by i
                rng = np.random.default_rng(noise_seed+i)
                # Generate real and imaginary objects at the same time,
                # even if we only use the real ones
                noise_real[i] = rng.normal(size=entry_shape)
                noise_imag[i] = rng.normal(size=entry_shape)
    else:
        raise ValueError(
            f"_add_noise_numpy_seed received seed_mode='{seed_mode}' "
            "but expects either 'shared' or 'sequential'"
        )

    # Combine
    if np.iscomplexobj(d):
        noise = noise_real + 1j * noise_imag
        # norm_factor *= np.sqrt(2) # this gets divided out later
    else:
        noise = noise_real
    # scaled_noise = noise_to_sig_ratio * d_norm / norm_factor * noise

    return (d + scaling_factor * noise).astype(d.dtype)

### Alternate noise scaling: fix the PSNR of the noised images
def psnr_to_ratio(psnr_level: float):
    """Converts a PSNR level to a ratio of noise's RMSE to the l_infinity norm
    Here we define PSNR (in decibels) as
        PSNR(x,ref) = 10 * log10(MAX(ref)^2 / MSE(x, ref))
    where
        MAX(ref) = || ref ||_\infty
    Note that this is slightly different from taking MAX to be the range of outputs
    in the reference array because the range is not well-defined for complex values
    """
    rmse_to_linf_ratio = 10.0** (-psnr_level / 20)
    return rmse_to_linf_ratio

def ratio_to_psnr(rmse_to_linf_ratio: float):
    """Converts a ratio of RMSE to the (squared) l_infinity norm to a PSNR level
    Here we define PSNR (in decibels) as
        PSNR(x,ref) = 10 * log10(MAX(ref)^2 / MSE(x, ref))
    where
        MAX(ref) = || ref ||_\infty
    Note that this is slightly different from taking MAX to be the range of outputs
    in the reference array because the range is not well-defined for complex values
    """
    psnr_level = -20 * np.log10(rmse_to_linf_ratio)
    return psnr_level

def calc_mse(arr, ref, axis=None, start_axis: int = None) -> np.ndarray:
    over_axes = axis if start_axis is None \
        else tuple(np.arange(start_axis % arr.ndim, arr.ndim))
    return np.mean(np.abs(arr-ref)**2, axis=over_axes)

def calc_max_abs(arr, axis=None, start_axis: int = None) -> np.ndarray:
    """Calculates the max magnitude of any entry
    This is intended for use with
    """
    over_axes = axis if start_axis is None \
        else tuple(np.arange(start_axis % arr.ndim, arr.ndim))
    return np.amax(np.abs(arr), axis=over_axes)

def calc_psnr(
    x: np.ndarray,
    ref: np.ndarray,
    start_axis: int=2,
    over_axes: tuple=None,
) -> np.ndarray:
    """Computes the PSNR over x with respect to ref
    This appears to be slightly more standard as a definition
    """
    over_axes = tuple(np.arange(start_axis % ref.ndim, ref.ndim)) \
        if over_axes is None else over_axes
    ref_max   = calc_max_abs(ref, axis=over_axes)
    x_mse     = calc_mse(x, ref, axis=over_axes)
    x_psnr = 10 * np.log10(ref_max**2 / x_mse)
    return x_psnr

# def calc_range(arr, axis=None, start_axis: int = None) -> np.ndarray:
#     over_axes = axis if start_axis is None \
#         else tuple(np.arange(start_axis % arr.ndim, arr.ndim))
#     return np.amax(arr, axis=over_axes) - np.amin(arr, axis=over_axes)

# def calc_psnr_real(
#     x: np.ndarray,
#     ref: np.ndarray,
#     start_axis: int=2,
#     over_axes: tuple=None,
# ) -> np.ndarray:
#     """Computes the PSNR over x with respect to ref
#     if real/imaginary components are treated as separate pixels
#     """
#     over_axes = tuple(np.arange(start_axis % ref.ndim, ref.ndim)) \
#         if over_axes is None else over_axes
#     ref_range = calc_range(ref, axis=over_axes)
#     x_mse     = calc_mse(x, ref, axis=over_axes)
#     x_psnr = 10 * np.log10(ref_range**2 / x_mse)
#     return x_psnr

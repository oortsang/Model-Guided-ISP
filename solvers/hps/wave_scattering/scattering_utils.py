# wave_scattering/scattering_utils.py
# Utility functions used by the other scattering solver files in this directory.
#
# Several functions reference the Gillmann, Barnett, Martinsson paper from 2014:
# A spectrally accurate direct solution technique for frequency-domain scattering problems with variable media
# Arxiv preprint: https://arxiv.org/abs/1308.5998
# Springer paper: https://link.springer.com/article/10.1007/s10543-014-0499-8

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import os
from .scattering_problem import ScatteringProblem

def _float_formatter_helper(val: float, d: int) -> str:
    """Returns val as a string rounded to exactly d decimal points"""
    fmt_code = f".{d}f"
    out_str  = f"{{0:{fmt_code}}}".format(np.round(val, d))
    return out_str

def _float_formatter_decimals(val: float, force_decimals: int=None, max_decimals: int=4) -> str:
    """Formats floating point variables as strings with easy control over the number of decimal points.
    Chooses the minimum number of decimal points needed to describe the float with the same accuracy
    as using 4 additional decimal points. (this is meant to deal with concerns of small floating point errors)

    Intended for use with file names.

    force_decimals can be set to specify an exact number of decimal points
    max_decimals is the maximum number of decimal points the function will output
    """
    if force_decimals is not None:
        return _float_formatter_helper(val, force_decimals)
    for decimals in range(1+max_decimals):
        val_rounded  = np.round(val, decimals)
        val_extended = np.round(val, decimals+4)
        if val_rounded == val_extended:
            return _float_formatter_helper(val_rounded, decimals)
    # In case max_decimals was insufficient
    return _float_formatter_helper(val_rounded, max_decimals)

def get_SD_matrices_fp(
    kbar_str: str|int|float, L: int, p: int, domain_half_length: float,
    comp_domain_factor: float=1,
    SD_matrices_dir: str = None,
) -> str:
    """Helper function to prepare the filepath name for the S and D matrix files
    Note: does not include the directory by default
    """
    # 2026-08-04: convert numeric forms of kbar into strings
    if isinstance(kbar_str, int):
        kbar_str = str(kbar_str)
    elif isinstance(kbar_str, float):
        kbar_str = _float_formatter_decimals(kbar_str)

    dom_str = _float_formatter_decimals(domain_half_length * comp_domain_factor)

    filename = f"SD_kbar{kbar_str}_L{L}_n{p-2}_dom{dom_str}.mat"
    filename = os.path.join(SD_matrices_dir, filename) \
        if SD_matrices_dir is not None \
        else filename
    return filename

def load_SD_matrices(fp: str) -> Tuple[jnp.array, jnp.array]:
    """
    Load S and D matrices from a MATLAB v7.3 .mat file using h5py.

    Args:
        fp (str): File path to the .mat file

    Returns:
        Tuple[jnp.array, jnp.array]: A tuple containing (S, D) matrices

    Raises:
        ValueError: If the file doesn't contain required matrices
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If there's an error reading the file
    """
    # try:

    # Open the file in read mode
    with h5py.File(fp, "r") as f:
        # Check if required matrices exist
        if "S" not in f or "D" not in f:
            raise ValueError("File must contain both 'S' and 'D' matrices")

        # Helper function to convert structured array to complex array
        def to_complex(structured_array):
            # Convert structured array to complex numpy array
            complex_array = (
                structured_array["real"] + 1j * structured_array["imag"]
            )
            # Convert to jax array
            return jnp.array(complex_array)

        # Load matrices and convert to complex numbers
        # Note: transpose the arrays as MATLAB stores them in column-major order
        S = to_complex(f["S"][:].T)
        D = to_complex(f["D"][:].T)

        return S, D

@jax.jit
def get_DtN_from_ItI(R: jnp.array, eta: float) -> jnp.array:
    """
    Given an ItI matrix, generates the corresponding DtN matrix.

    equation 2.17 from the Gillman, Barnett, Martinsson paper.

    Implements the formula: T = -i eta (R - I)^{-1}(R + I)

    Args:
        R (jnp.array): Has shape (n, n)
        eta (float): Real number; parameter used in the ItI map creation.

    Returns:
        jnp.array: Has shape (n, n)
    """
    n = R.shape[0]
    I = jnp.eye(n)
    T = -1j * eta * jnp.linalg.solve(R - I, R + I)
    return T

@jax.jit
def get_ItD(T: jax.Array, R: jax.Array, eta: float) -> jax.Array:
    """
    Formula is (T + i \\eta I)^{-1} R^{-1}
    """
    a = R @ (T + 1j * eta * jnp.eye(T.shape[0]))
    return jnp.linalg.inv(a)

@jax.jit
def get_uin(
    k: float, pts: jnp.array, source_directions: jnp.array
) -> jnp.array:
    source_vecs = jnp.stack(
        [jnp.cos(source_directions), jnp.sin(source_directions)],
        axis=0
    )
    uin = jnp.exp(1j * k * jnp.dot(pts, source_vecs))
    return uin

@jax.jit
def get_uin_and_normals(
    k: float, bdry_pts: jnp.array, source_directions: jnp.array
) -> Tuple[jnp.array, jnp.array]:
    """
    Given the boundary points and the source directions, computes the incoming wave and the normal vectors.

    uin(x) = exp(i k <x,s>), where s = (cos(theta), sin(theta)) is the direction of the incoming plane wave.

    d uin(x) / dx = ik s_0 uin(x)
    d uin(x) / dy = ik s_1 uin(x)

    Args:
        k (float): Frequency of the incoming plane waves
        bdry_pts (jnp.array): Has shape (n, 2)
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: uin, normals. uin has shape (n,). normals has shape (n, 2).
    """
    n_per_side = bdry_pts.shape[0] // 4

    uin = get_uin(k, bdry_pts, source_directions)
    source_vecs = jnp.array(
        [jnp.cos(source_directions), jnp.sin(source_directions)]
    ).T

    normals = jnp.concatenate(
        [
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[:n_per_side],  # -1 duin/dy
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[n_per_side : 2 * n_per_side],  # duin/dx
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[2 * n_per_side : 3 * n_per_side],  # duin/dy
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[3 * n_per_side :],
        ]
    )

    return uin, normals

def setup_scattering_lin_system(
    S: jnp.array,
    D: jnp.array,
    T_int: jnp.array,
    uin: jax.Array,
    uin_dn: jax.Array,
) -> Tuple[jnp.array, jnp.array]:
    """
    Sets up the BIE system in eqn (3.4) of the Gillman, Barnett, Martinsson paper.

    Args:
        S (jnp.array): Single-layer potential matrix. Has shape (n,n)
        D (jnp.array): Double-layer potential matrix. Has shape (n,n)
        T_int (jnp.array): Dirichlet-to-Neumann matrix. Has shape (n,n)
        gauss_bdry_pts (jnp.array): Has shape (n, 2)
        k (float): Frequency of the incoming plane waves
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: A, which has shape (n, n) and b, which has shape (n, n_sources).
    """
    n_bdry_pts = S.shape[0]

    A = 0.5 * jnp.eye(n_bdry_pts) - D + S @ T_int
    b = S @ (uin_dn - T_int @ uin)

    return A, b

def get_uscat_and_dn(
    S: jnp.array,
    D: jnp.array,
    T: jnp.array,
    uin: jax.Array,
    uin_dn: jax.Array,
) -> jnp.array:
    """Get uscat on the boundary and its normal derivatives
    For use computing uscat outside the computational domain
    """
    A, b = setup_scattering_lin_system(
        S=S, D=D, T_int=T, uin=uin, uin_dn=uin_dn
    )

    # logging.info(f"Solving the system... (A.shape={A.shape}; b.shape={b.shape})")
    # Solve the lin system to get uscat on the boundary
    uscat = jnp.linalg.solve(A, b)

    # Eqn 1.12 from the Gillman, Barnett, Martinsson paper
    uscat_dn = T @ (uscat + uin) - uin_dn

    return uscat, uscat_dn


### Functions for computing the vjp and jvp ###

def get_exterior_DtN(S: jax.Array, D: jax.Array) -> jax.Array:
    """Prepare the Dirichlet-to-Neumann map for use with the solution on the exterior
    of the scattering domain.
    This operates on the u_scat object or beta (to be defined below).
    Based on the jump relations (see Gillman et al.), but can also be applied to
    the complex conjugate of the S and D matrices to satisfy the adjoint radiation
    condition rather than the standard one.
    """
    return jnp.linalg.solve(S, (D - 0.5 * jnp.eye(S.shape[0])))

def get_DtI_from_DtN(DtN: jax.Array, eta: jax.Array) -> jax.Array:
    """Prepare the Dirichlet-to-(incoming-)Imedance map using a Dirichlet-to-Neumann
    map and the eta value (typically taken as k)
    """
    return DtN + 1j * eta * jnp.eye(DtN.shape[0])

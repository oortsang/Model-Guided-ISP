import jax
import jax.numpy as jnp
from jaxhps import PDEProblem
from jaxhps.quadrature import chebyshev_weights
import numpy as np
import logging

class ScatteringProblem:
    def __init__(
        self,
        pde_problem: PDEProblem,
        S_int: jax.Array,
        D_int: jax.Array,
        S_ext: jax.Array,
        D_ext: jax.Array,
        target_points_reg: jax.Array,
        source_dirs: jax.Array,
        uin_bdry: jax.Array = None,
        uin_dn_bdry: jax.Array = None,
        uin_interior: jax.Array = None,
    ):
        """
        S_int and D_int are the S and D matrices created by the MATLAB code.
        S_ext and D_ext are created by the python functions in gen_SD_exterior.py

        uin_bdry, uin_dn_bdry, uin_interior are all optional. If they are not specified, the incident fields
        are created as plane waves using source_dirs and wavenumber k = pde_problem.eta.

        target_points_reg (Optional) is an array of size (nx, ny, 2) to keep track of the regular grid on which we want to recover q.
        """

        self.pde_problem: PDEProblem = pde_problem
        self.S_int: jax.Array = S_int
        self.D_int: jax.Array = D_int
        self.S_ext: jax.Array = S_ext
        self.D_ext: jax.Array = D_ext
        self.target_points_reg: jax.Array = target_points_reg
        self.source_dirs: jax.Array = source_dirs

        if target_points_reg is not None:
            dx = target_points_reg[1, 0, 0] - target_points_reg[0, 0, 0]
            dy = target_points_reg[0, 0, 1] - target_points_reg[0, 1, 1]
            assert dx > 0 and dy > 0, "dx and dy must be positive"
            self.quad_weights: jax.Array = (
                jnp.ones_like(target_points_reg[..., 0]) * dx * dy
            )
        else:
            self.quad_weights: jax.Array = None

        self.uin_bdry: jax.Array = uin_bdry
        self.uin_dn_bdry: jax.Array = uin_dn_bdry
        self.uin_interior: jax.Array = uin_interior

        # These next few lines compute quadrature weights for the HPS grid.
        # Side length of each panel is 1 / 2**L * (xmin, xmax)
        side_len = (
            pde_problem.domain.root.xmax - pde_problem.domain.root.xmin
        ) / (2**pde_problem.domain.L)
        bounds = jnp.array([0.0, side_len])
        cheby_weights_1d = chebyshev_weights(pde_problem.domain.p, bounds)
        # Generate weights for 2D panel by taking the outer product of the 1D weights
        cheby_weights_2d = jnp.outer(cheby_weights_1d, cheby_weights_1d)
        r = rearrange_indices_ext_int(pde_problem.domain.p)
        cheby_weights_2D_rearranged = cheby_weights_2d.flatten()[r]
        cheby_weights_2D_rearranged = cheby_weights_2D_rearranged[None, :]
        self.quad_weights_hps = jnp.repeat(
            cheby_weights_2D_rearranged,
            pde_problem.domain.n_leaves,
            axis=0,
        )

        self.T_DtN: jax.Array = None
        self.T_ItI: jax.Array = None
        self.T_ItD: jax.Array = None

    def reset(self) -> None:
        """Resets stored incident-field/solution data that can be
        regenerated on the next solve.
        """
        self.uin_bdry = None
        self.uin_dn_bdry = None
        self.uin_interior = None
        self.T_DtN = None
        self.T_ItI = None
        self.T_ItD = None

def rearrange_indices_ext_int(n: int) -> jnp.ndarray:
    """This function gives the array indices to rearrange the 2D Cheby grid so that the
    4(p-1) boundary points are listed first, starting at the SW corner and going clockwise around the
    boundary. The interior points are listed after.
    """

    idxes = np.zeros(n**2, dtype=int)
    # S border
    for i, j in enumerate(range(n - 1, n**2, n)):
        idxes[i] = j
    # W border
    for i, j in enumerate(range(n**2 - 2, n**2 - n - 1, -1)):
        idxes[n + i] = j
    # N border
    for i, j in enumerate(range(n**2 - 2 * n, 0, -n)):
        idxes[2 * n - 1 + i] = j
    # S border
    for i, j in enumerate(range(1, n - 1)):
        idxes[3 * n - 2 + i] = j
    # Loop through the indices in column-rasterized form and fill in the ones from the interior.
    current_idx = 4 * n - 4
    nums = np.arange(n**2)
    for i in nums:
        if i not in idxes:
            idxes[current_idx] = i
            current_idx += 1
        else:
            continue

    return jnp.array(idxes)

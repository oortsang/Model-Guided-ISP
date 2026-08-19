# wave_scattering/interp_ops.py
# Interpolation operator objects
# Classes:
# 1. QuadtreeToUniformFixedDomain
# 2. QuadtreeToUniform
# 3. UniformToQuadtree

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Iterable

from .interp_utils import (
    prep_quadtree_to_unrolled_indices,
    _rearrange_indices_ext_int_2D,
    prep_grids_cheb_2d,
    prep_grids_unif_2d,
    prep_conv_interp_2d,
    apply_conv_interp_2d,
)
from jaxhps.quadrature import (
    barycentric_lagrange_interpolation_matrix_2D,
)


# 1. Quadtree-to-uniform object
########################################

# Quadtree (Chebyshev) -> Uniform on a fixed domain
class QuadtreeToUniformFixedDomain:
    """Class to interpolate data from quadtree form with chebyshev-grid leaves
    to a contiguous form on a uniform grid.
    Simplifying assumptions:
    1. Assume each leaf contains an integer number of points from the uniform grid,
    and that these points are always in the same positions relative to the leaf boxes
    2. Assume that the domain size does not change
    """
    def __init__(
            self,
            L: int, p: int, n_per_leaf: int,
            # quad_domain_bounds: Iterable = (-1., 1., -1., 1.),
            # unif_domain_bounds: Iterable = (-1., 1., -1., 1.),
            rel_offset: float = 0):
        """Set up the reusable objects
        Parameters:
            L (int): number of levels in the tree
            p (int): polynomial order of the leaf-level chebyshev grids
            n_per_leaf (int): number of points in the uniform grid per leaf
            quad_domain_bounds (Iterable): iterable containing
                [xmin, xmax, ymin, ymax]
                as the domain boundaries for the quadtree grid (input space)
            unif_domain_bounds (Iterable): domain boundaries for the uniform grid
                (output space)
            rel_offset (float): relative offset of where to sample the uniform grid
                If a leaf's chebyshev grid is sampled on the interval [-1, 1],
                the rel_offset values correspond to the following behavior:
                0:   default behavior, sample at    [0, 1, 2, ..., n_per_leaf-1]*2/n_per_leaf-1
                0.5: cell-centered, sample at       [0.5, 1.5, 2.5, ..., n_per_leaf-0.5]*2/n_per_leaf-1
                1:   sample on the opposite end, at [1, 2, ..., n_per_leaf]*2/n_per_leaf-1
        """
        self.L = L
        self.p = p
        self.n_per_leaf = n_per_leaf
        # self.domain_bounds = domain_bounds
        self.leaf_bounds = (-1., 1., -1., 1.) # internal representation

        self.quadtree_to_unrolled_idcs = prep_quadtree_to_unrolled_indices(L)

        leaf_cheb_grids  = prep_grids_cheb_2d(
            0, p, self.leaf_bounds
        )
        leaf_unif_grids  = prep_grids_unif_2d(
            0, n_per_leaf, self.leaf_bounds, rel_offset=rel_offset
        )
        self.leaf_cheb_x = leaf_cheb_grids[0]
        self.leaf_cheb_y = leaf_cheb_grids[1]
        self.leaf_unif_x = leaf_unif_grids[0]
        self.leaf_unif_y = leaf_unif_grids[1]

        tmp_interp_leaf_cheb_to_unif = barycentric_lagrange_interpolation_matrix_2D(
            from_pts_x=self.leaf_cheb_x,
            from_pts_y=self.leaf_cheb_y,
            to_pts_x=self.leaf_unif_x,
            to_pts_y=self.leaf_unif_y,
        )
        self.rearrange_leaf_idcs = _rearrange_indices_ext_int_2D(p)
        self.interp_leaf_cheb_to_unif = (
            tmp_interp_leaf_cheb_to_unif
            [:, self.rearrange_leaf_idcs]
        )

    def apply(self, data_quadtree_cheb: jax.Array) -> jax.Array:
        """Applies the quadtree chebyshev-to-uniform operation
        Parameters:
            data_quadtree_cheb (jax.Array, shape: (4**L, p**2, ...)):
                Data from the HPS quadtree, sampled on Chebyshev grids at the leaf level
        Output:
            data_unif (jax.Array, shape (2**L * n_per_leaf, 2**L * n_per_leaf), ...):
                Data on the uniform grid without the quadtree structure
        """
        L = self.L
        p = self.p
        n_per_leaf = self.n_per_leaf
        leftover_shape = data_quadtree_cheb.shape[2:]
        leftover_idcs = 4 + jnp.arange(0, data_quadtree_cheb.ndim-2)

        # 1. Reorder the leaf nodes to a more standard ordering (~block-wise row-major-order)
        data_leaves_cheb = jnp.take(
            data_quadtree_cheb,
            self.quadtree_to_unrolled_idcs,
            axis = 0,
        )

        # 2. Map values from Chebyshev to Uniform grids on the leaf-level
        data_leaves_unif = jnp.einsum(
            "jl,il...->ij...",
            self.interp_leaf_cheb_to_unif,
            data_leaves_cheb,
        )

        # 3. Reshape leaf-level data into square matrices, then rearrange
        data_unif = (
            data_leaves_unif
            .reshape(2**L, 2**L, n_per_leaf, n_per_leaf, *leftover_shape)
            .transpose(0,3,1,2, *leftover_idcs)
            .reshape(2**L * n_per_leaf, 2**L * n_per_leaf, *leftover_shape)

            # Not sure why the outputs need to be transposed here...
            .transpose(1,0, *(leftover_idcs-2))
        )
        return data_unif

# Now for a somewhat more flexible version...
class QuadtreeToUniform:
    def __init__(
        self,
        L: int, p: int, n_per_leaf: int, unif_n: int,
        quad_domain_bounds: Iterable = (-1., 1., -1., 1.),
        unif_domain_bounds: Iterable = (-1., 1., -1., 1.),
        rel_offset: float = 0,
    ):
        """QuadtreeToUniform with potentially different input/output domains
        Expected usage: the HPS quadtree operates on a larger domain than what you need,
        so provide an option to also interpolate to the clipped domain.
        Parameters:
            L (int): number of levels in the tree
            p (int): polynomial order of the leaf-level chebyshev grids
            n_per_leaf (int): number of points in the uniform grid per leaf (in the quadtree domain)
            unif_n (int): number of points in the final output grid, when cropping is requested
            quad_domain_bounds (Iterable): domain boundaries for the quadtree grid (input)
            clip_domain_bounds (Iterable): iterable containing
                [xmin, xmax, ymin, ymax]
                as the domain boundaries for the clipped output uniform grid
        """
        self.qtu_fixed = QuadtreeToUniformFixedDomain(
            L, p, n_per_leaf, rel_offset=rel_offset,
        )
        self.L = L
        self.p = p
        self.n_per_leaf = n_per_leaf
        self.quad_n = 2**L * n_per_leaf
        self.unif_n = unif_n
        self.quad_domain_bounds = quad_domain_bounds
        self.unif_domain_bounds = unif_domain_bounds

        quad_cheb_grids   = prep_grids_cheb_2d(L, p, quad_domain_bounds)
        quad_unif_grids   = prep_grids_unif_2d(L, n_per_leaf, quad_domain_bounds, rel_offset)
        clip_unif_grids = prep_grids_unif_2d(0, unif_n, unif_domain_bounds, rel_offset)
        self.quad_cheb_x  = quad_cheb_grids[0]
        self.quad_cheb_y  = quad_cheb_grids[1]
        self.quad_cheb_xy = quad_cheb_grids[2]
        self.quad_unif_x  = quad_unif_grids[0]
        self.quad_unif_y  = quad_unif_grids[1]
        self.quad_unif_xy = quad_unif_grids[2]
        self.clip_unif_x  = clip_unif_grids[0]
        self.clip_unif_y  = clip_unif_grids[1]
        self.clip_unif_xy = clip_unif_grids[2]

        # compare quad_unif_x and clip_unif_x (and ditto for *_y)
        quad_unif_dx = self.quad_unif_x[1] - self.quad_unif_x[0]
        quad_unif_dy = self.quad_unif_y[1] - self.quad_unif_y[0]
        clip_unif_dx = self.clip_unif_x[1] - self.clip_unif_x[0]
        clip_unif_dy = self.clip_unif_y[1] - self.clip_unif_y[0]
        can_slice_for_x = (self.clip_unif_x[0] in self.quad_unif_x) \
            and jnp.isclose(quad_unif_dx, clip_unif_dx, rtol=1e-6)
        can_slice_for_y = (self.clip_unif_y[0] in self.quad_unif_y) \
            and jnp.isclose(quad_unif_dy, clip_unif_dy, rtol=1e-6)
        self.use_slice = can_slice_for_x and can_slice_for_y
        if self.use_slice:
            start_idx_x = jnp.where(self.quad_unif_x == self.clip_unif_x[0])[0].item()
            self.slice_x_idcs = jnp.arange(
                start_idx_x,
                start_idx_x + self.clip_unif_x.shape[0]
            )

            start_idx_y = jnp.where(self.quad_unif_y == self.clip_unif_y[0])[0].item()
            self.slice_y_idcs = jnp.arange(
                start_idx_y,
                start_idx_y + self.clip_unif_y.shape[0]
            )
            self.slice_idcs = jnp.ix_(self.slice_x_idcs, self.slice_y_idcs)
        else:
            quad_to_clip_x, quad_to_clip_y = prep_conv_interp_2d(
                self.quad_unif_x,
                self.quad_unif_y,
                self.clip_unif_xy,
            )
            self.quad_to_clip_x = jnp.array(quad_to_clip_x.todense())
            self.quad_to_clip_y = jnp.array(quad_to_clip_y.todense())

    def apply(self, data_quadtree_cheb: jax.Array, output_mode: str = "clip") -> Tuple[jax.Array]:
        """Apply the Quadtree-to-Uniform operation; can return the outputs on the original uniform
        grid (with the same bounds as the quadtree) or the clipped uniform grid
        Parameters:
            data_quadtree (jax.Array, shape (4^2L, p^2)): data on the quadtree
            output_mode (string): specify whether to return the data interpolated onto
                the "quad" domain, "clip" domain, or "both" domains
        Returns:
            data_quad_unif (jax.Array): data interpolated onto a uniform grid on the quadtree's domain
            data_clip_unif (jax.Array): data interpolated onto a uniform grid on the clipped domain
        """
        clip_output = output_mode.lower() in ["both", "clip"]
        data_quad_unif = self.qtu_fixed.apply(data_quadtree_cheb)
        # extra_shape = data_quadtree_cheb.shape[2:]
        if clip_output:
            if self.use_slice:
                data_clip_unif = data_quad_unif[self.slice_idcs]
            else:
                if data_quad_unif.ndim == 2:
                    in_reshape  = (self.quad_n, self.quad_n)
                    out_reshape = (self.unif_n, self.unif_n)
                else:
                    in_reshape  = (self.quad_n, self.quad_n, -1)
                    out_reshape = (self.unif_n, self.unif_n, *data_quad_unif.shape[2:])
                data_clip_unif = apply_conv_interp_2d(
                    self.quad_to_clip_x,
                    self.quad_to_clip_y,
                    data_quad_unif.reshape(*in_reshape),
                ).reshape(*out_reshape)

        output = (data_quad_unif, data_clip_unif) if output_mode.lower() == "both" \
            else (data_quad_unif) if output_mode.lower() == "quad" \
            else (data_clip_unif)
        return output


# 2. Uniform-to-Quadtree (Chebyshev)
########################################
class UniformToQuadtree:
    """Class to interpolate data from a contiguous uniform grid to
    a quadtree form with chebyshev-grid leaves. Uses bicubic interpolation.
    Simplifying assumptions:
    1. Assume each leaf contains an integer number of points from the uniform grid,
    and that these points are always in the same positions relative to the leaf boxes
    2. Assume that the domain size does not change
    For now, fix it at [-1, 1, -1, 1] for both input/output since the true scaling does
    not really matter, other than if the domain were non-square.
    """
    def __init__(
        self,
        L: int, p: int, n_unif: int,
        unif_domain_bounds: Iterable = (-1., 1., -1., 1.),
        quad_domain_bounds: Iterable = (-1., 1., -1., 1.),
        domain_bounds: Iterable = None,
        rel_offset: float = 0,
        interp_use_jax: bool = True,
        use_sparse_ops: bool = False,
    ):
        """Set up the reusable objects
        Parameters:
            L (int): number of levels in the tree
            p (int): polynomial order of the leaf-level chebyshev grids
            n_unif (int): total number of points in the uniform grid
            unif_domain_bounds (Iterable): iterable containing
                [xmin, xmax, ymin, ymax]
                as the domain boundaries for the uniform grid (input)
            quad_domain_bounds (Iterable): domain boundaries for the quadtree grid (output)
            domain_bounds (Iterable): can be used to set the input/output domains at once
            rel_offset (float): relative offset of where to sample the uniform grid
                If a leaf's chebyshev grid is sampled on the interval [-1, 1],
                the rel_offset values correspond to the following behavior:
                0:   default behavior, sample at    [0, 1, 2, ..., n_per_leaf-1]*2/n_per_leaf-1
                0.5: cell-centered, sample at       [0.5, 1.5, 2.5, ..., n_per_leaf-0.5]*2/n_per_leaf-1
                1:   sample on the opposite end, at [1, 2, ..., n_per_leaf]*2/n_per_leaf-1
            interp_use_jax (bool): whether to prepare the interpolation operators as
                dense jax arrays
            use_sparse_ops (bool): if using jax arrays, indicate whether to use
                the jax.experimental.sparse arrays
                If this is chosen, tree_unif_to_cheb_x will be BCOO and
                tree_unif_to_cheb_y will be of type BCSR
        """
        self.L = L
        self.p = p
        self.n = n_unif
        self.n_per_leaf = n_unif // 2**L
        if domain_bounds is not None:
            self.unif_domain_bounds = domain_bounds
            self.quad_domain_bounds = domain_bounds
        else:
            self.unif_domain_bounds = unif_domain_bounds
            self.quad_domain_bounds = quad_domain_bounds

        # Permutation maps for ordering leaves and chebyshev points within the leaves
        new_order = (0, 1, 3, 2)
        self.quadtree_to_unrolled_idcs = prep_quadtree_to_unrolled_indices(
            L, s=1, new_order=new_order
        )
        self.unrolled_to_quadtree_idcs = jnp.argsort(self.quadtree_to_unrolled_idcs) # invert
        self.rearrange_leaf_idcs       = _rearrange_indices_ext_int_2D(p)
        self.inv_rearrange_leaf_idcs   = jnp.argsort(self.rearrange_leaf_idcs) # invert

        # Tree-level grids, scaled on domain_bounds
        tree_cheb_grids = prep_grids_cheb_2d(L, p, quad_domain_bounds)
        tree_unif_grids = prep_grids_unif_2d(0, n_unif, unif_domain_bounds, rel_offset)
        self.tree_cheb_x  = tree_cheb_grids[0]
        self.tree_cheb_y  = tree_cheb_grids[1]
        self.tree_cheb_xy = tree_cheb_grids[2]
        self.tree_unif_x  = tree_unif_grids[0]
        self.tree_unif_y  = tree_unif_grids[1]
        self.tree_unif_xy = tree_unif_grids[2]

        # Get the chebyshev grid with hps tree ordering
        self.hps_tree_cheb_xy = (
            self.tree_cheb_xy
            .reshape(2**L, p, 2**L, p, 2) # Get leaf/chebyshev structure
            .transpose(2, 0, 1, 3, 4)     # Put leaf ordering together
            .reshape(4**L, p**2, 2)       # Combine leaf ordering/content
            [self.unrolled_to_quadtree_idcs, :, :] # Rearrange leaf nodes
            [:, self.rearrange_leaf_idcs, :] # Rearrange Chebyshev nodes
            .reshape(-1, 2) # Unroll
        )

        # Interpolation operation, tree-wide
        tree_unif_to_cheb_x, tree_unif_to_cheb_y = prep_conv_interp_2d(
            self.tree_unif_x,
            self.tree_unif_y,
            self.hps_tree_cheb_xy,
            # use a zero-valued boundary condition to mimic empty neighboring leaves
            bc_modes=("zero", "zero"),
            use_jax=interp_use_jax,
            use_sparse_ops=use_sparse_ops,
        )
        # Should/can I move the interpolation operators to the GPU?
        self.tree_unif_to_cheb_x = tree_unif_to_cheb_x
        self.tree_unif_to_cheb_y = tree_unif_to_cheb_y

    def apply(self, data_unif: jax.Array) -> jax.Array:
        """Take the data on the uniform grid, then map
        to the HPS quadtree format with leaf-level chebyshev grids
        Assume data_unif has shape (n, n)
        Parameters:
            data_unif (jax.Array, shape (n, n)): data on the uniform grid
        Output:
            data_tree_cheb (jax.Array, shape (4**L, p**2)): data in the
                quadtree/chebyshev-grid format
        """
        L = self.L
        p = self.p
        n = self.n

        # Perform in batches if necessary
        if data_unif.ndim > 2:
            # Assumes the shape is (n, n, ...)
            extra_shape = data_unif.shape[2:]
            data_unif_flat = data_unif.reshape(*data_unif.shape[:2], *extra_shape)
            data_tree_cheb = jnp.stack(
                [
                    apply_conv_interp_2d(
                        self.tree_unif_to_cheb_x,
                        self.tree_unif_to_cheb_y,
                        data_unif_flat[..., i].reshape(n, n),
                    ).reshape(4**L, p**2)
                    for i in range(data_unif_flat.shape[-1])
                ],
                axis=2,
            ).reshape(4**L, p**2, *extra_shape)
        else:
            data_tree_cheb = apply_conv_interp_2d(
                self.tree_unif_to_cheb_x,
                self.tree_unif_to_cheb_y,
                data_unif.reshape(n, n),
            ).reshape(4**L, p**2)
        return data_tree_cheb

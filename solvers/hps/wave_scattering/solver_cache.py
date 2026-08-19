# Interface to help manage a cache of solvers
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

import logging

from solvers.hps.wave_scattering import (
    QuadtreeToUniform,
    UniformToQuadtree,
    get_SD_matrices_fp,
    load_SD_matrices,
    HPSScatteringSolver,
)

def get_quadtree_interp_ops(
    N_x: int,
    hps_l: int,
    hps_p: int,
    domain_hlen: float = 0.5,
    hps_cdf: float = 1.1,
):
    """Helper to set up the quadtree interpolation operators"""
    unif_bounds = domain_hlen * np.array([-1., 1., -1., 1.])
    hpst_bounds = hps_cdf * unif_bounds

    QtU = QuadtreeToUniform(
        L=hps_l,
        p=hps_p,
        n_per_leaf=N_x//(2**hps_l),
        unif_n=N_x,
        unif_domain_bounds=unif_bounds,
        quad_domain_bounds=hpst_bounds,
        rel_offset=0,
    )
    UtQ = UniformToQuadtree(
        L=hps_l,
        p=hps_p,
        n_unif=N_x,
        unif_domain_bounds=unif_bounds,
        quad_domain_bounds=hpst_bounds,
        rel_offset=0,
    )
    return (QtU, UtQ)

class SolverCache:
    def __init__(
        self,
        kbar_str_list: list,
        grid_size: int,
        default_l: int = 3,
        default_p: int = 16,
        domain_hlen: float = 0.5,
        num_sources: int=1,
        num_receivers: int=1,
        source_dirs: Optional[jnp.ndarray]=None,
        receiver_dirs: Optional[jnp.ndarray]=None,
        comp_domain_factor: float = 1.1,
        hps_sd_mat_dir: str = "HPS_SD_matrices",
        cache_size: Optional[int] = None,
    ):
        self.kbar_str_list = [*kbar_str_list]
        self.kbar_list = list(map(int, kbar_str_list))
        self.grid_size = grid_size
        self.default_l = default_l
        self.default_p = default_p
        self.domain_hlen = domain_hlen
        self.num_sources = num_sources
        self.num_receivers = num_receivers
        self.source_dirs = source_dirs
        self.receiver_dirs = source_dirs
        self.comp_domain_factor = comp_domain_factor
        self.hps_sd_mat_dir = hps_sd_mat_dir

        self.unif_bounds = domain_hlen * np.array([-1., 1., -1., 1.])
        self.hpst_bounds = self.comp_domain_factor * self.unif_bounds
        (default_QtU, default_UtQ) = get_quadtree_interp_ops(
            N_x=grid_size,
            hps_l=self.default_l,
            hps_p=self.default_p,
            domain_hlen=self.domain_hlen,
            hps_cdf=self.comp_domain_factor,
        )
        interp_key = (grid_size, self.default_l, self.default_p)
        self.interp_cache = {
            interp_key: (default_QtU, default_UtQ)
        }

        self.solver_cache = dict()

        self.cache_size = cache_size
        self.cache_lru = []

    def touch_solver(self, solver_key: tuple) -> None:
        """Marks solver_key as the most-recently-used entry.
        If the cache is at capacity and solver_key is a new entry,
        evicts the least-recently-used solver by moving its arrays to
        host RAM (self.solver_cache keeps holding a reference to it, so
        it isn't rebuilt from scratch next time -- just moved back).
        """
        if solver_key in self.cache_lru:
            self.cache_lru.remove(solver_key)
        elif self.cache_size is not None and len(self.cache_lru) >= self.cache_size:
            lru_key = self.cache_lru.pop(0)
            logging.info(f"SolverCache: evicting {lru_key} to host RAM")
            self.solver_cache[lru_key].to_host()
        self.cache_lru.append(solver_key)

    def get_interp_ops(
        self,
        grid_size: int = None,
        hps_l: int=3,
        hps_p: int=16,
    ):
        interp_key = (grid_size, hps_l, hps_p)
        # Set up interpolation operators if not available
        if interp_key not in self.interp_cache.keys():
            # Generate new operators (QtU and UtQ)
            interp_ops = get_quadtree_interp_ops(
                N_x=grid_size,
                hps_l=hps_l,
                hps_p=hps_p,
                domain_hlen=self.domain_hlen,
                hps_cdf=self.comp_domain_factor,
            )
            # Update cache
            self.interp_cache[interp_key] = interp_ops
        return self.interp_cache[interp_key]

    def get_solver(
        self,
        kbar_str: str,
        grid_size: int = None,
        hps_l: int=None,
        hps_p: int=None,
    ) -> HPSScatteringSolver:
        """Attempts to retrieve solver from cache; sets it up if not available
        """
        hps_l = hps_l if hps_l is not None else self.default_l
        hps_p = hps_p if hps_p is not None else self.default_p

        solver_key = (kbar_str,  hps_l, hps_p)
        # Set up solver and add to cache if not yet available
        if solver_key in self.solver_cache.keys():
            # Cache hit -- may have been parked on host RAM by a previous
            # eviction, so make sure it's back on the compute device
            self.solver_cache[solver_key].to_device()
            self.touch_solver(solver_key)
        else:
            # Cache miss -- evict the LRU entry (if the cache is full) to
            # free up device memory before building the new solver
            self.touch_solver(solver_key)

            # Fetch interior S and D matrices
            hps_sd_fp = get_SD_matrices_fp(
                kbar_str=kbar_str,
                L=hps_l,
                p=hps_p,
                domain_half_length=self.domain_hlen,
                comp_domain_factor=self.comp_domain_factor,
                SD_matrices_dir=self.hps_sd_mat_dir,
            )
            S_int, D_int = load_SD_matrices(hps_sd_fp)

            # Fetch interp ops
            grid_size = grid_size if grid_size is not None else self.grid_size
            QtU, UtQ = self.get_interp_ops(grid_size, hps_l, hps_p)

            # Set up the solver object
            kbar = float(kbar_str)
            k = 2 * np.pi * kbar
            hps_solver = HPSScatteringSolver(
                L=hps_l, p=hps_p, N_x=grid_size, k=k,
                S_int=S_int,
                D_int=D_int,
                N_s=self.num_sources,
                N_r=self.num_receivers,
                source_dirs=self.source_dirs,
                receiver_dirs=self.receiver_dirs,
                unif_domain_bounds=self.unif_bounds,
                quad_domain_bounds=self.hpst_bounds,
                use_ItI=True,
                QtU=QtU,
                UtQ=UtQ,
            )
            # Save to cache
            self.solver_cache[solver_key] = hps_solver
        return self.solver_cache[solver_key]

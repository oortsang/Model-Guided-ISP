# Wave scattering solver with HPS
This directory contains the HPS solver used for the wave scattering problem, as well as notes in `hps_gradients_notes.pdf` documenting the equations being solved in the Jacobian-vector and vector-Jacobian products of `derivative_solver.py`.

Additionally, because HPS discretizes the computational domain on a quadtree, with each leaf corresponding to a Cartesian product of Chebyshev grid points whereas most of our code outside this directory expects discretization on a uniformly-sampled grid, this directory also includes interpolation utilities that quickly go between each of these representations.


Directory contents:

- Primary solver files
  - `scattering_problem.py`
  - `hps_scattering_solver.py`
  - `interior_solver.py`
  - `exterior_solver.py`
  - `derivative_solver.py`
  - `opt_fns.py`
- Setup and interpolation
  - `gen_SD_exterior`
  - `scattering_utils.py`
  - `interp_utils.py`
  - `interp_ops.py`
- Alternate interfaces
  - `shared_solver.py`
  - `solver_cache.py`
  - `pytorch_wrapper.py`
- Notes for the derivative/gradient operators
  - `hps_gradients_notes.pdf`

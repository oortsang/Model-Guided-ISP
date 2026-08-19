# Directory contents:

 - `helmholtz_solver_utils.py`: utility functions for `helmholtz_solver_bicgstab.py`.
 - `helmholtz_solver_bicgstab.py`: Defines the PDE solver object used in data generation.
 - `helmholtz_solver_gradients.py`: calculates gradients; also offers an autodiff-ready PyTorch interface for the PDE solver
 - `bicgstab_batch.py`: PyTorch implementation of BiCGSTAB for multiple right-hand-sides
 - `data_generation_new.py`: functions defining our distribution of scattering potentials.
 - `random_shape_generation.py`: Helper functions for `data_generation_new.py`.

# __init__.py file for solvers/hps/wave_scattering
# Make most of the useful functions accessible from here...

from .interp_utils import (
    prep_grids_cheb_2d,
    prep_grids_unif_2d,
    reorder_tree_cheb_for_hps,
    prep_conv_interp_1d,
    prep_conv_interp_2d,
    apply_conv_interp_1d,
    apply_conv_interp_2d,
)

from .interp_ops import (
    QuadtreeToUniform,
    UniformToQuadtree,
)

from .gen_SD_exterior import (
    gen_D_exterior,
    gen_S_exterior,
)

from .scattering_utils import (
    get_SD_matrices_fp,
    load_SD_matrices,
    get_DtN_from_ItI,
    get_uin,
    get_uin_and_normals,
    get_uscat_and_dn,
)

from .exterior_solver import (
    forward_model_exterior,
)
from .interior_solver import (
    forward_model_interior,
    get_utot_int,
)
from .derivative_solver import (
    eval_beta_bdry_with_source,
    apply_vjp,
    apply_jvp,
)

from .shared_solver import (
    SharedSolver,
    shared_solver_prep,
)
from .scattering_problem import ScatteringProblem
from .hps_scattering_solver import (
    HPSScatteringSolver,
    setup_hps_scattering_solver,
)
from .pytorch_wrapper import (
    PytorchHPSSolver,
    pytorch_backproject_diff,    
)

# Optimization stuff for recursive linearization
from .opt_fns import (
    GaussNewtonOperator,
    gauss_newton_loop_single_sample,
)

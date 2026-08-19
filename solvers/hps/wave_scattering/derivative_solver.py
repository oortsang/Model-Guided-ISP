# wave_scattering/derivative_solver.py
# Contains the code to apply the vector-jacobian
# and jacobian-vector products of the forward model

import jax.numpy as jnp
import jax
import jaxlib
from typing import Tuple
import logging

from jaxhps import Domain, PDEProblem, DiscretizationNode2D, build_solver
from jaxhps.up_pass import up_pass_uniform_2D_ItI
from jaxhps.down_pass import down_pass_uniform_2D_ItI

from .scattering_problem import ScatteringProblem
from .gen_SD_exterior import (
    gen_D_exterior,
    gen_S_exterior,
)

from .scattering_utils import (
    get_DtN_from_ItI,
    get_uin_and_normals,
    get_uin,
    setup_scattering_lin_system,
    get_uscat_and_dn,
    get_exterior_DtN,
    get_DtI_from_DtN,
)

from .exterior_solver import forward_model_exterior
from .interior_solver import forward_model_interior, get_utot_int

def eval_beta_bdry_with_source(
    scat_problem: ScatteringProblem,
    q: jax.Array,
    source: jax.Array,
    adjoint_radiation_condition: bool = False,
    return_v_g_tilde: bool = False,
    rebuild_solver: bool = True,
    precondition_with_S: bool = False,
    T_ext_DtN: jax.Array = None,
    device: jax.Device = jax.devices()[0],
    verbosity: int=0,
) -> jax.Array:
    """Evaluate the beta(x) function on the domain boundary, given a source term f(x).
    Note that this can also be used for u_scat(x) just as well.

    The PDE in question obeys the standard or adjoint sommerfeld radiation condition
        (laplacian) beta(x) + k^2 (q(x)+1) beta(x) = f(x), x in Omega
        sqrt(r) * ( d( beta(x) )/dr (-/+) ik * beta(x)) -> 0 as r=||x|| -> infinity
    where d()/dr indicates the derivative in the direction away from the origin,
    and the radiation condition uses (-) for the standard version
    and (+) for the adjoint version.

    ~~~ PDE solution method ~~~
    The solution proceeds by splitting the PDE into two sub-problems, one particular
    and one homogeneous:
        beta(x) = beta_part(x) + beta_homog(x).
    1. Particular solution
    The particular portion of the problem is:
        (laplacian) beta_part(x) + k^2 (q(x)+1) beta_part(x) = f(x), x in Omega
        d_n beta_part(x) + ik beta_part(x) = 0, x on the boundary of Omega
    where d_n indicates the derivative relative to the outward-pointing normal vector
    on the boundary of Omega. The code solves for both beta_part(x) and its
    corresponding outgoing impedance data on the boundary.

    2. Homogeneous solution
    The homogeneous problem is something like
        (laplacian) beta_homog(x) + k^2 (q(x)+1) beta_homog(x) = 0, x in Omega
        beta_homog(x) = g(x), x on the boundary of Omega
        sqrt(r) * ( d( beta_homog(x) )/dr (-/+) ik * beta_homog(x)) -> 0 as r=||x|| -> infinity
    This is solved using a Boundary Integral Equation (BIE) that comes from the
    jump relations and continuity conditions across the Omega boundary. However,
    since we are only interested in d_n beta_homog and beta_homog on the boundary,
    this function does not deal with solving for beta_homog on the domain interior.
    Consequently, we just need to find g(x).

    The BIE comes from enforcing continuity in the dirichlet and neumann values in
    beta from the inside and outside.
    For clarity, consider beta(x) the PDE solution inside Omega and gamma(x) the solution
    outside Omega. We need
        gamma|∂Ω = beta|∂Ω = beta_part|∂Ω + beta_homog|∂Ω (dirichlet)
        d_n gamma|∂Ω = d_n beta|∂Ω = d_n beta_part|∂Ω + d_n beta_homog|∂Ω (neumann)
        d_n gamma|∂Ω + ik gamma|∂Ω = d_n beta_homog|∂Ω + ik beta_homog|∂Ω (robin/impedance)
    In the last line we can drop beta_part, since by construction is incoming impedance
    value at ∂Ω is zero.

    Suppose we have G_int_DtI and G_ext_DtI, which are dirichlet-to-(incoming-)impedance maps
    that operate on beta and gamma, respectively. Then we can set up the BIE as follows:
        d_n gamma|∂Ω + ik gamma|∂Ω = G_ext_DtI @ gamma|∂Ω
                                   = G_ext_DtI @ beta|∂Ω by continuity in dirichlet condition
        G_ext_DtI @ gamma|∂Ω = d_n beta_homog|∂Ω + ik beta_homog|∂Ω
                             = G_int_DtI @ beta_homog|∂Ω
    So:
        G_ext_DtI @ (beta_part|∂Ω + beta_homog|∂Ω) = G_int_DtI @ beta_homog|∂Ω
    or
        (G_int_DtI - G_ext_DtI) beta_homog|∂Ω = G_ext_DtI @ beta_part|∂Ω
    (which can be paired with d_n beta_homog|∂Ω = T_int_DtI @ beta_homog|∂Ω).

    To compute the matrices involved:
    The T_int_DtN map is derived using the jump relations (see Gillman et al.
    or Colton/Kress, Integral Equation Methods in Scattering Theory for more info)
    and maps beta_homog -> d_n beta_homog
    From there, the DtI maps are simply G_{int,ext}_DtI = T_{int,ext}_DtN + i*k * I.
    Also note that this implies (G_int_DtI - G_ext_DtI)=(T_int_DtN - T_ext_DtN).

    Once we have beta_part, d_n beta_part, beta_homog, and d_n beta_homog, we can sum
    the particular and homogeneous components to get beta and d_n beta on the domain
    boundary, which is what this function returns.

    ~~~ PDE solution recap / overview ~~~
    1. (Particular solution) compute beta_part, d_n beta_part values on the
        domain boundary using an upward pass of the HPS tree.
    2. (Homogeneous solution) compute beta_homog, d_n beta_homog using a BIE
        based on DtI maps and continuity conditions across the domain boundary.
    3. (Full solution) add together the two separate portions to
        find beta, d_n beta on the domain boundary.
        Return these values, which can be used to find beta inside the domain later.

    ~~~ Function usage ~~~
    Parameters:
        problem (ScatteringProblem): scattering problem object, which holds the HPS tree
            and relevant matrices
        source (jax.Array): source term f(x) evaluated on the HPS tree grid
        q (jax.Array): scattering potential q(x) evaluated on the HPS tree grid
        adjoint_radiation_condition (bool): whether to use the adjoint radiation condition
        return_v_g_tilde (bool): whether to return the v_part and g_tilde_lst objects
            from the upward pass of the HPS tree while computing impedance data associated
            with the particular solution
        rebuild_solver (bool): whether to rebuild the HPS tree.
            Not needed if the most recent solve involved the same q(x) object,
            but it may be needed if running the forward model on multiple q(x) objects
            before performing the backward passes.
        precondition_with_S (bool): whether to precondition the BIE system by multiplying
            by the S matrix, inspired by the ItI case from Gillman et al..
        T_ext_DtN (jax.Array): exterior DtN operator, which can be passed
            to avoid recomputing it, since there is no q(x) or source f(x) dependence.
    Returns:
        beta_bdry, beta_bdry_dn (jax.Array objects):
            beta and its normal derivative evaluated on the domain boundary
        Optionally, v_part (jax.Array), g_tilde_lst (list of jax.Array):
            additional data from the HPS upward pass that can be re-used
            during the subsequent downward pass to find beta throughout the
            interior of the domain.
    """
    pde_problem = scat_problem.pde_problem
    k = pde_problem.eta

    # 0a. Compute T_int_DtN; rebuild the solver if requested
    if rebuild_solver:
        I_term = k**2 * (q + 1)
        pde_problem.update_coefficients(I_coefficients=I_term)
        R_int = build_solver(
            pde_problem=pde_problem,
            return_top_T=True,
            compute_device=device,
            host_device=device,
        )
        T_int_DtN = get_DtN_from_ItI(
            R=R_int,
            eta=k,
        )
        scat_problem.T_ItI = R_int
        scat_problem.T_DtN = T_int_DtN
    else:
        T_int_DtN = scat_problem.T_DtN

    # 0b. Fetch S, D, T_ext_DtN
    if adjoint_radiation_condition:
        S = scat_problem.S_int.conj()
        D = scat_problem.D_int.conj()
    else:
        S = scat_problem.S_int
        D = scat_problem.D_int

    T_ext_DtN = T_ext_DtN if T_ext_DtN is not None else get_exterior_DtN(S=S, D=D)

    # 1. Compute the impedance boundary conditions associated with the
    # particular solution of beta. By construction, the incoming impedance g_part is zero;
    # we compute the outgoing impedance h_part with an upward pass on the HPS tree.
    # Convert these into dirichlet/neumann boundary data as well.
    # (note: v_part is the particular solution throughout the HPS tree domain)
    v_part, g_tilde_lst, h_part = up_pass_uniform_2D_ItI(
        source=source, pde_problem=pde_problem, return_h_last=True
    )

    g_part = 0
    # These are the boundary evaluations of the particular solution and its normal derivative.
    beta_part_dn = (h_part + g_part) / 2
    beta_part    = (h_part - g_part) / (-2j * k)

    # 2. Set up and solve the boundary integral equation for the homogeneous portion
    # of the beta PDE.
    # The G matrices are DtI operators
    # G_ext = get_DtI_from_DtN(DtN=T_ext_DtN, eta=k)
    # G_int = get_DtI_from_DtN(DtN=T_int_DtN, eta=k)
    # lhs = G_int - G_ext
    # rhs = G_ext @ beta_part

    # Avoid forming unnecessary matrices to hopefully reduce overhead
    # Comes from continuity
    A_bie = T_int_DtN - T_ext_DtN # note: equivalent to G_int - G_ext
    b_bie = (T_ext_DtN @ beta_part) + 1j*k * beta_part
    if precondition_with_S:
        # Does not necessarily help conditioning
        A_bie = S @ A_bie
        b_bie = S @ b_bie
    # Dirichlet and neumann data for beta_homog
    beta_homog    = jnp.linalg.solve(A_bie, b_bie)
    beta_homog_dn = T_int_DtN @ beta_homog

    # 3. Combine the particular and homogeneous solutions to get the dirichlet/neumann
    # boundary data for the full beta(x) solution
    # All the variables live on the boundary but these variables have "bdry" in the names
    # since they are returned; this is intended to help with readability.
    beta_bdry    = beta_homog    + beta_part
    beta_bdry_dn = beta_homog_dn + beta_part_dn

    if return_v_g_tilde:
        return beta_bdry, beta_bdry_dn, v_part, g_tilde_lst, T_ext_DtN
    else:
        return beta_bdry, beta_bdry_dn

def apply_vjp(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
    vec: jax.Array,
    Gk_ring_to_omega: jax.Array,
    usc_int: jax.Array = None,
    T_ext_DtN: jax.Array = None,
    T_int_DtN: jax.Array = None,
    rebuild_solver: bool = True,
    device: jax.Device = None,
    verbosity: int=0,
) -> jax.Array:
    """Apply the vector-jacobian product to vec
    ~~~ PDE solution method ~~~
    Borges et al. suggest solving the following PDE:
        (laplacian) w(x) + k^2 (q(x)+1) w(x) = -k^2 χ(ξ, ∂B)(x), x in Ω
        adjoint radiation condition for w(x)
    where ξ represents vec for a single source direction, and χ(ξ, ∂B)(x) embeds the
    ξ values, originally only on the receiver ring, into R^2.
    Once w is solved, the vjp is given by:
        (DF[q]^* vec)(x) = sum_s Re[conj(utot)(x) * w(x)]

    The w(x) PDE can be split into w(x)=α(x)+β(x), where alpha(x) is
    the solution to a constant-coefficient problem, and beta(x) is the solution
    to a variable-coefficient problem whose source term's support lies within
    the scattering domain Ω. Both α(x) and β(x) satisfy the adjoint Sommerfeld
    radiation condition.

    1. alpha(x): the alpha PDE is simple and can be solved by applying a Greens operator:
        (laplacian + k^2) α(x) = -k^2 χ(ξ, ∂B)(x), x in Ω
        adjoint Sommerfeld radiation condition
    with solution given by
        α(x) = k^2 int_Ω G_k(x-x') ξ(x') dx'
    where the Greens function G_k satisfies the adjoint Sommerfeld radiation condition
    rather than the standard one. It can be computed as the complex conjugate of the
    Greens function satisfying the standard Sommerfeld radiation condition.

    2. beta(x): the beta PDE is given by
        (laplacian) β(x) + k^2 (q(x)+1) β(x) = -k^2 q(x) α(x)
        adjoint Sommerfeld radiation condition
    where the source term f(x)=-k^2 q(x) α(x) has support within Ω.
    Here we call a helper function that computes β(x) on ∂Ω, then we can compute
    it within the entire domain using a downward pass of the HPS tree.


    ~~~ Function usage ~~~
    Parameters
        scattering_problem (ScatteringProblem): object holding the HPS tree as well
            as other important operators
        q (jax.Array): scattering potential on the HPS tree grid
        vec (jax.Array): vector on which to apply the adjoint jacobian
            Expected shape: (N_r, N_s)
            Note: may need to transpose this if coming from d_rs, which has the
            axes flipped and shape (N_s, N_r)
        Gk_ring_to_omega (jax.Array): Green's operator satisfying the adjoint Sommerfeld
            radiation condition. matmul performs integration over the receiver ring and
            outputs values on the interior of dthe domain.
            Expected shape: (4**L * p**2, N_r)
            Requires re-shaping the outputs to match the shape (4**L * p**2, N_s)
        T_ext_DtN (jax.Array): exterior DtN map
            Will be recomputed if not passed.
        usc_int (jax.Array): scattered wave field on the interior domain
            Will be recomputed if not passed.
        rebuild_solver (bool): rebuild the HPS tree/solver
            Intended if the scattering potential in question may have changed since the
            most recent HPS build.
    Returns:
        out (jax.Array): an array corresponding to the vector-Jacobian product
            has shape (4**L, p**2)
            (I think you do not need to conjugate the outputs...)
    """
    pde_problem = scattering_problem.pde_problem
    k = pde_problem.eta
    device = device if device is not None else q.device

    # 1. Compute alpha(x) and the source term
    # alpha_int  = +k**2 * (Gk_ring_to_omega @ vec.T).reshape(*q_shape, -1)
    alpha_int  = +k**2 * (Gk_ring_to_omega @ vec).reshape(*q.shape, -1)
    source_int = -k**2 * q[..., None] * alpha_int

    # 2. Compute beta(x)
    # Use the helper function to get the boundary values
    if verbosity >= 4:
        logging.info(f"vjp: about to call eval_beta_bdry_with_source")
    with jax.disable_jit(True):
        outputs = eval_beta_bdry_with_source(
            scat_problem=scattering_problem,
            q=q,
            source=source_int,
            adjoint_radiation_condition=True,
            return_v_g_tilde=True,
            rebuild_solver=rebuild_solver,
            precondition_with_S=False,
            T_ext_DtN=T_ext_DtN,
            device=device,
            verbosity=verbosity,
        )
    beta_bdry, beta_bdry_dn, v_part, g_tilde, T_ext_DtN = outputs
    # Solve for beta(x) on the interior
    if verbosity >= 4:
        logging.info(f"vjp: about to call down_pass_uniform_2D_ItI")

    incoming_imp = beta_bdry_dn + 1j*k * beta_bdry
    with jax.disable_jit(True):
        beta_int = down_pass_uniform_2D_ItI(
            boundary_data=incoming_imp,
            S_lst=pde_problem.S_lst,
            g_tilde_lst=g_tilde,
            Y_arr=pde_problem.Y,
            v_arr=v_part,
            device=device,
            host_device=device,
        )
    w_int = alpha_int + beta_int

    # 3. Compute w(x) and get the final output
    if verbosity >= 4:
        logging.info(f"vjp: about to get utot_int")
    with jax.disable_jit(True):
        utot_int = get_utot_int(
            scattering_problem,
            q,
            usc_int=usc_int,
            rebuild_solver=False,
            device=device,
        )

    out_complex = (utot_int.conj() * w_int).sum(-1)
    out = out_complex.real
    if verbosity >= 4:
        logging.info(f"vjp: done!")
    return out


def apply_jvp(
    scattering_problem: ScatteringProblem,
    q: jax.Array,
    vec: jax.Array,
    usc_int: jax.Array = None,
    T_ext_DtN: jax.Array = None,
    rebuild_solver: bool = True,
    device: jax.Device = None,
    verbosity: int=0,
) -> jax.Array:
    """Apply the jacobian-vector product to vec (sometimes referred to as dq)
    See Borges et al. for details
    """
    pde_problem = scattering_problem.pde_problem
    k = pde_problem.eta
    device = device if device is not None else q.device

    with jax.disable_jit(True):
        utot_int = get_utot_int(
            scattering_problem,
            q=q,
            usc_int=usc_int,
            rebuild_solver=rebuild_solver,
            device=device,
        )
    source = -k**2 * vec[..., None] * utot_int

    with jax.disable_jit(True):
        u, un = eval_beta_bdry_with_source(
            scat_problem=scattering_problem,
            source=source,
            q=q,
            adjoint_radiation_condition=False,
            T_ext_DtN=T_ext_DtN,
            rebuild_solver=False, # already would be rebuilt by get_utot_int
            # rebuild_solver=rebuild_solver,
            device=device,
            verbosity=verbosity,
        )
    u_ext = scattering_problem.D_ext @ u - scattering_problem.S_ext @ un
    return u_ext

# Loading helper functions for logging and jax
# to streamline notebooks/scripts

import os
import datetime
import logging

from typing import Optional, Any

def get_date_str():
    """Mini helper to get date string"""
    today = datetime.date.today().strftime("%Y-%m-%d")
    return today

def jax_setup(
    vram_fraction: float=0.8,
    cpu_only: bool=False,
    jax_async_dispatch: Optional[str]=None,
    jax_enable_x64: bool=True,
    max_num_cpus: int=None,
    env_vars: Optional[dict]=None,
):
    if cpu_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(vram_fraction)
    if jax_async_dispatch is not None:
        # jax.config.update("jax_async_dispatch", jax_async_dispatch)
        os.environ["JAX_ASYNCHRONOUS_BATCHING"]  = jax_async_dispatch
    import jax
    jax_device = jax.devices(jax.default_backend())[0]
    jax.config.update("jax_default_device", jax_device)
    jax.config.update("jax_enable_x64", jax_enable_x64)

    # Avoid crashing when calling active_contour
    try:
        num_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    except:
        num_cpus = "1"
    if max_num_cpus is not None:
        # Optionally cap the number of cpus
        num_cpus = max(num_cpus, max_num_cpus)
        os.environ["NUMEXPR_MAX_THREADS"]  = num_cpus
        os.environ["OMP_NUM_THREADS"]      = num_cpus
        os.environ["MKL_NUM_THREADS"]      = num_cpus # "1"
        os.environ["OPENBLAS_NUM_THREADS"] = num_cpus # "1"

    if env_vars is not None:
        for env_var, val in env_vars.items():
            os.environ[env_var] = str(val)

    return jax_device

def logging_setup(module_name: str=None, level=logging.INFO):
    module_name = module_name if module_name is not None else "invsc"
    logging.basicConfig(
        format=(
            f"%(asctime)s.%(msecs)03d:{module_name}: "
            f"%(levelname)s - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

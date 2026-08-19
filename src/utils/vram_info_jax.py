# Like vram_info but for use with jax

import psutil
import jax


def get_memory_info_jax(device=None, print_msg: bool=True):
    """Helper function that tells the RAM and VRAM usage"""
    msg_1  = f"RAM Used (MB): {psutil.virtual_memory().used >> 20}"
    device = (
        device if device is not None else
        jax.config.jax_default_device
    )
    if device is None or device.device_kind == "cpu":
        if not print_msg:
            print(msg_1)
        return msg_1
    stats = device.memory_stats()
    # First values are in bytes
    t = stats["bytes_limit"] # total (within preallocation)
    u = stats["bytes_in_use"]
    p = stats["peak_bytes_in_use"]
    f = t-u # free (within preallocation)
    msg_2 = (
        f"VRAM (MB): {f>>20} free of {t>>20} total (within preallocation); "
        f"{u>>20}; peaked at {p>>20}"
    )
    msg_full = msg_1 + "\n" + msg_2
    if print_msg:
        print(msg_full)
    return msg_full

def get_vram_total_mb_jax(device=None):
    """Helper function that gets total VRAM amount"""
    device = (
        device if device is not None else
        jax.config.jax_default_device
    )
    tot_mb = device.memory_stats["bytes_in_use"] >> 20
    return tot_mb

def vram_mb_to_frac_jax(block_mb: float, device=None) -> float:
    """Helper function that takes an amount of vram in megabytes
    and returns what fraction of total VRAM that would be
    """
    tot_mb = get_vram_total_mb_jax(device)
    return (block_mb / tot_mb) if tot_mb != 0 else 0


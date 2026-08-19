# RAM/VRAM info helper functions

import psutil
import torch

def get_memory_info(device=None, print_msg: bool = True):
    """Helper function that tells the RAM and VRAM usage"""
    msg_1  = f"RAM Used (MB): {psutil.virtual_memory().used >> 20}"
    device = (
        device if device is not None else
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device == "cpu":
        if not print_msg:
            print(msg_1)
        return msg_1
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a # free inside reserved
    msg_2 = f"VRAM (MB): {f>>20} free of {r>>20} reserved; {a>>20} allocated out of {t>>20} total"
    msg_full = msg_1 + "\n" + msg_2
    if print_msg:
        print(msg_full)
    return msg_full

def free_vram(device=None):
    """Helper function to call free up VRAM, but sometimes you just need to restart
    from https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch
    """
    import gc
    import inspect
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_delete = [device]
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()

def get_vram_total_mb(device=None) -> int:
    """Helper function that gets the total amount of VRAM of a given pytorch device in megabytes
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            return 0
    t = torch.cuda.get_device_properties(device).total_memory >> 20
    return t

def vram_mb_to_frac(block_mb: float, device=None) -> float:
    """Helper function that takes an amount of vram in megabytes
    and returns what fraction of total VRAM that would be
    """
    tot_mb = get_vram_total_mb(device)
    return (block_mb / tot_mb) if tot_mb != 0 else 0

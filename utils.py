import gc
import torch
import os
import random
import numpy as np

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.empty_cache()


def set_seed(seed=42):
    # For Python random numbers
    random.seed(seed)
    
    # For NumPy random numbers
    np.random.seed(seed)

    # For PyTorch CPU random numbers
    torch.manual_seed(seed)
    
    # For PyTorch GPU random numbers (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.

    # For cudnn to avoid nondeterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optionally, fix the hash seed for the Python runtime (useful for reproducibility in some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)

def number_to_cpu(item):
    # If item is a PyTorch tensor, ensure it's on the CPU and extract its value.
    if isinstance(item, torch.Tensor):
        # If this is a scalar tensor, you can simply use .item()
        return item.cpu().detach().item()
    # If it's not a tensor, just return the item as-is.
    return item
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import gc
import torch

from model.data_loader import MRIData


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.empty_cache()




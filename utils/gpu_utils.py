import random
from typing import Optional, List

from .paths import model_names as paths_model_names


def gpu_filter(
    gpu_index: int, num_gpu: int, model_names: Optional[List[str]] = None
):
    if gpu_index is None or num_gpu is None:
        gpu_index = 0
        num_gpu = 1
    
    if model_names is None:
        model_names = paths_model_names

    mod_eq = lambda a, b: (a % num_gpu) == (b % num_gpu)

    return [name for i, name in enumerate(model_names) if mod_eq(i, gpu_index)]




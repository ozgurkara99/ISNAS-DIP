from functools import reduce

from . import funcs as fn
from .common_types import *



def np_to_torch(arr: NumpyArray) -> Tensor:
    '''Convert a numpy array to a torch tensor.'''
    
    return torch.from_numpy(arr)

def torch_to_np(arr: Tensor) -> NumpyArray:
    '''Convert a torch tensor to a numpy array.'''
    
    arr = arr.detach().cpu().numpy()
    
    return arr


def exponential_average(arrays: List[Array], exp_weight: float = 0.99) -> Array:
    '''Calculate the exponential average of the arrays in arrays with the \
        given weight. For example, if arrays = [a, b] then the results is \
        b * exp_weight + a * (1 - exp_weight).'''
    
    avg = reduce(
        lambda val, arr: fn.average(val, arr, exp_weight, 1-exp_weight), 
        reversed(arrays[:-1]), arrays[-1]
    )
    
    return avg
    
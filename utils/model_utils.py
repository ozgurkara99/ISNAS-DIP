from typing import Optional, Tuple

import torch

from models.cross_skip import skip
from . import basic_utils as bu
from . import denoising_utils as du
from . import image_utils as imu
from . import array_utils as au
from .keywords import *
from .common_types import *


def get_device(model: Model) -> torch.device:
    return next(model.parameters()).device

def get_dtype(model: Model) -> torch.dtype:
    return next(model.parameters()).dtype

def to(
    model: Model, 
    device: Optional[torch.device] = None, 
    dtype: Optional[torch.dtype] = None    
) -> Model:

    if device is not None:
        model = model.to(device=device)
    
    if dtype is not None:
        model = model.to(dtype=dtype)
        
    return model

def to_same(model1: Model, model2: Model) -> Model:
    return to(model1, device=model2.device, dtype=model2.dtype)

def from_name_to_index_skip(model_name: str) -> Tuple[int, NumpyArray]:
    tmp = model_name.split('_')
    index, skip = tmp[1], tmp[-1]
    
    index = int(index)
    
    skip = map(int, skip)
    skip = list(skip)
    skip = np.array(skip)
    skip = skip.reshape((5, 5))
    
    return index, skip

def from_index_skip_to_name(index: int, skip: NumpyArray) -> str:
    
    skip = skip.reshape((-1,))
    skip = list(skip)
    skip = map(str, skip)
    skip = ''.join(skip)
    
    return f'0_{index}_iteration_4000_sigma_25_skip_{skip}'

def create_model_(
    index: int, skip_connections: np.ndarray, 
    in_channels: int = 1, out_channels: int = 1
) -> Model:
    
    skip_connections = skip_connections.reshape((-1,))
    skip_connections = list(skip_connections)
    skip_connections = [str(elm) for elm in skip_connections]
    skip_connections = ''.join(skip_connections)
    
    model_name = f'0_{index}_iteration_4000_sigma_25_skip_{skip_connections}'
    
    return create_model(
        model_name,
        in_channels=in_channels, out_channels=out_channels
    )

def create_model(
    model_name: str, in_channels: int = 1, out_channels: int = 1
) -> Model:
    '''Creates the model from its name.
    
    Args:
        model_name: The name of the model. The name should follow the \
            following similar pattern: \
            0_101_iteration_4000_sigma_25_skip_0001000000011100100011100. \
            Here 101 is the model index and 0001000000011100100011100 is the \
            skip connections matrix.
    Returns:
        model: Pytorch model corresponding to the given name.'''
    
    index = int(model_name.split('_')[1])

    skip_connect = np.array(list(map(int, model_name.split('_')[-1])))
    skip_connect = np.reshape(skip_connect, (5, 5))

    model = skip(
        model_index=index,
        skip_index=skip_connect,
        num_input_channels=in_channels,
        num_output_channels=out_channels,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[4] * 5,
        upsample_mode='bilinear',
        downsample_mode='stride',
        need_sigmoid=True,
        need_bias=True,
        pad='reflection',
        act_fun='LeakyReLU'
    )

    model.model_index = index
    model.skip_connect = skip_connect
    model.name = model_name

    return model

def get_random_output(
    model: Model,
    input_noise: Optional[Tensor] = None,
    input_size: Optional[Union[Tuple[int, int], int]] = None, 
    in_channel: Optional[int] = None,
    noise_fmt: Optional[du.NoiseFormat] = None,
    sigma: Optional[float] = None,
    fmt: ArrayFormat = 'np'
) -> Array:
    
    if input_noise is None:
        assert input_size is not None
        assert in_channel is not None
        assert noise_fmt is not None
        assert sigma is not None
        
        device = get_device(model)
        dtype = get_dtype(model)
        
        input_noise = du.get_noise(
            shape=(1, in_channel, input_size[0], input_size[1]),
            sigma=sigma,
            noise_fmt=noise_fmt
        )
        
        input_noise = au.np_to_torch(input_noise)
        input_noise = imu.to_device(input_noise, device, dtype)
    
    with torch.no_grad():
        random_output = model(input_noise)
    
    if fmt == 'np':
        random_output = imu.torch_to_np(random_output)
    
    return random_output

def get_random_model_name() -> str:
    
    index = np.random.randint(1, 300)
    skip = np.random.randint(0, 2, size=(5, 5))
    
    return from_index_skip_to_name(index, skip)


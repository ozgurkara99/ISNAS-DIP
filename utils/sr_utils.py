from typing import Dict, List, Optional, TypedDict

import numpy as np
from PIL import Image
from torch import optim
from torch import nn
from tqdm import tqdm

from . import image_utils as imu
from . import basic_utils as bu
from . import funcs as fn
from . import model_utils as mu
from . import denoising_utils as du
from . import array_utils as au
import models.downsampler as ds
from .keywords import *
from .common_types import *



def get_lr_img(img: Img, zoom: int, fmt: ImageFormat = 'np') -> ImageFormat:
    '''Return a low resolution image.'''
    
    if isinstance(img, torch.Tensor):
        img = bu.torch_to_pil(img)
        
    if isinstance(img, np.ndarray):
        img = imu.np_to_pil(img)
        
    img = img.resize((img.size[0] // zoom, img.size[1] // zoom), Image.ANTIALIAS)
    
    if fmt == 'np':
        img = imu.pil_to_np(img)
    if fmt == 'torch':
        img = imu.pil_to_torch(img)
    
    return img

def get_downsampler(zoom: int, in_channels: int = 3) -> Model:
    '''Return a fixed downsampler pytorch model.'''
    
    downsampler = ds.Downsampler(
        n_planes=in_channels, 
        factor=zoom, 
        kernel_type='lanczos2', 
        phase=0.5, 
        preserve_size=True    
    )
    
    return downsampler


def sr(
    model: Model,
    optimizer: Optional[optim.Optimizer],
    
    img_true_np: NumpyArray,
    img_noisy_torch: Optional[Tensor],
    input_noise: Optional[Tensor],
    
    downsampler: Model = None,
    
    num_iter: int = 4000,
    atleast: Optional[int] = None,
    exp_weight: float = 0.99,
    exp_window: Optional[int] = None,
    reg_noise_std: float = 1/30,
    
    show_every: Optional[int] = None,
    save_outputs_every: Optional[int] = None,
    
    get_outputs_at: Optional[Union[int, List[int]]] = None
):
    '''Super-resolution for the given image using DIP and return all the \
        information related to the performance of the given model.
        
    Args:
        model: Pytorch model.
        optimizer: Optimizer for the model.

        img_true_np: numpy.ndarray version of the ground truth image of shape \
            (C, H, W).
        img_noisy_torch: torch.Tensor version of the noisy image with shape \
            (1, C, H, W).
        input_noise: This is the input to the model throught the DIP process. \
            Its shape is (1, C, H, W).
            
        downsampler: Downsampler.

        num_iter: Number of iterations.
        atleast: After atleast many iterations, DIP process will stop as soon \
            as PSNR of the output of the model falls below of the highest PSNR \
            achieved by 3 dB.
        exp_weight: The exponential average weight to calculate the \
            exponential average of the output of the model during the DIP \
            process. The formula for the exponential average is like: \
            out_avg = out_avg * exp_weight + out * (1 - exp_weight).
        exp_window: Size of the exponential average window.
        reg_noise_std: Std of the regularization noise. This regularization \
            noise is added to the input_noise and fed to the model at each \
            iteration for a regularization effect.
            
        show_every: If verbose is True, some statistics about the DIP process \
            is printed at one iteration in every show_every iterations.
        save_outputs_every: Save the outputs of the model at \
            every save_outputs_every iterations.
        
        get_outputs_at: A list of integers representing number of iterations \
            at which the output of the model is saved 

    Returns:
        htr: A dictionary.'''

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    if img_noisy_torch is None:
        dtype = bu.get_dtype(model)
        
        img_noisy_np = get_lr_img(img_true_np, zoom=4, fmt='np')
        img_noisy_torch = bu.np_to_torch(img_noisy_np).type(dtype)
    
    if input_noise is None:
        input_noise = du.get_noise_like(
            img_noisy_torch, sigma=1/10, noise_fmt='uniform'
        )
        
    if downsampler is None:
        dtype = mu.get_dtype(model)
        zoom = img_true_np.shape[-1] // img_noisy_torch.shape[-1]
        downsampler = get_downsampler(zoom).type(dtype)
    
    if atleast is None:
        atleast = num_iter * 2 # just to be on the safe side, mult. by 2
        
    if isinstance(get_outputs_at, int):
        get_outputs_at = [get_outputs_at]
        
    
    htr: HtrDict = {
        'loss': [],
        'psnr_gt': [],
        'psnr_gt_sm': [],

        'best_psnr_gt': 0,
        'best_psnr_gt_sm': 0,
        
        'last_out': None,
        'last_out_sm': None,

        'best_out': None,
        'best_out_sm': None, 
        
        'best_iter': None,
        'best_iter_sm': None,
        
        'outs': {},
        'outs_sm': {}
    }
    out_sm_np: NumpyArray = None
    if exp_window is not None:
        outs = ds.Queue(exp_window)
    
    mse = nn.MSELoss()
    iterable = bu.optimize(
        model=model, 
        optimizer=optimizer,
        input_noise=input_noise,
        loss_fn=lambda out: mse(downsampler(out), img_noisy_torch),
        num_iter=num_iter,
        reg_noise_std=reg_noise_std
    )
    for i, (out_np, loss) in tqdm(enumerate(iterable), total=num_iter):
        out_np: NumpyArray
        loss: float
        
        if exp_window is not None:
            outs.push(out_np)
            
            out_sm_np = au.exponential_average(outs, exp_weight)
        else:
            if out_sm_np is None:
                out_sm_np = out_np
            else:
                out_sm_np = out_sm_np * exp_weight + out_np * (1 - exp_weight)
            
        psnr_gt = fn.psnr(img_true_np, out_np)
        psnr_gt_sm = fn.psnr(img_true_np, out_sm_np)
        
        htr['loss'].append(loss)
        htr['psnr_gt'].append(psnr_gt)
        htr['psnr_gt_sm'].append(psnr_gt_sm)
        
        if psnr_gt > htr['best_psnr_gt']:
            htr['best_psnr_gt'] = psnr_gt
            htr['best_out'] = out_np
            htr['best_iter'] = i

        if psnr_gt_sm > htr['best_psnr_gt_sm']:
            htr['best_psnr_gt_sm'] = psnr_gt_sm
            htr['best_out_sm'] = out_sm_np
            htr['best_iter_sm'] = i
        
        if get_outputs_at is not None and i in get_outputs_at:
            htr['outs'][i] = out_np
            htr['outs_sm'][i] = out_sm_np
            
        if save_outputs_every is not None and i % save_outputs_every == 0:
            htr['outs'][i] = out_np
            htr['outs_sm'][i] = out_sm_np
            
        if show_every is not None and i % show_every == 0:
            best_psnr_gt = htr['best_psnr_gt']
            best_psnr_gt_sm = htr['best_psnr_gt_sm']

            print(
                f'{i}/{num_iter}: {loss = :.3e}   {psnr_gt = :.2f}   ' +
                f'{psnr_gt_sm = :.2f}   {best_psnr_gt = :.2f}   '+
                f'{best_psnr_gt_sm = :.2f}'
            )
        
        if i >= atleast and psnr_gt + 3 < htr['best_psnr_gt']:
            break
    htr['last_out'] = out_np
    htr['last_out_sm'] = out_sm_np

    return htr
    
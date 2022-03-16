import os
from typing import Callable, Dict, Iterable, Optional, Tuple, List

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from . import funcs as fn
from . import image_utils as imu
from . import denoising_utils as du
from . import inpainting_utils as iu
from . import sr_utils as su
from .paths import ROOT, IMG_EXT 
from .keywords import *
from .common_types import *


def optimize(
    model: Model, optimizer: optim.Optimizer, 
    loss_fn: Callable[[Tensor], Tensor],
    
    input_noise: Tensor,
    
    num_iter: int,
    reg_noise_std: float = 1/30
) -> Iterable[Tuple[NumpyArray, float]]:
    '''
    Optimization loop for DIP.
    Args:
        model: Pytorch model.
        loss_fn: A callable taking a tensor and outputting a pytorch tensor \
            representing a float to calculate the loss and do a backward pass.
        input_noise: Input to the model.
        num_iter: Number of iterations.
        reg_noise_std: Standard devition of the regularization noise to be \
            added on te the input_noise at each iteration.
    
    Returns:
        An iterable consisting of output of the model and the loss at each \
            iteration.'''
    
    for _ in range(num_iter):
        inp = input_noise + du.get_noise_like(
            input_noise, sigma=reg_noise_std, noise_fmt='normal'
        )
        out = model(inp)
        
        loss = loss_fn(out)
        
        out_np = imu.torch_to_np(out)
        
        yield out_np, loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def get_process(
    sigma: Optional[int] = None,
    p: Optional[int] = None,
    zoom: Optional[int] = None
) -> Process:
    '''Return the process name represented by the variables sigma, p and zoom.'''
    
    # decide the process according to the given parameters
    if sigma is not None:
        if not isinstance(sigma, int):
            raise TypeError('sigma should be an integer. It is the \
            std of the noise for the images of range [0, 255]')
            
        process = DENOISING
    elif p is not None:
        if not isinstance(p, int):
            raise TypeError('p should be an integer. It is the percent \
            probability of the black pixels of the noisy images.')
            
        process = INPAINTING
    elif zoom is not None:
        if not isinstance(zoom, int):
            raise TypeError('zoom should be an integer. It is the \
            ratio of the dimensions of ground-trueth image and noisy image.')
        
        process = SR
    else:
        raise TypeError('Either sigma, p or zoom must be given.')
    
    return process


def stem_from_name(name: str) -> str:
    '''Return the stem of a file name. For example, it will return new_file, \
        if the input is new_file.txt.'''
        
    return os.path.splitext(name)[0]

def name_from_stem(stem: str) -> str:
    '''Add .png extension to the and of the given string.'''
    
    return stem + IMG_EXT

def read_img_stems(
    process: Process
) -> List[str]:
    '''Returns the stem of all images inside the ROOT/IMAGES/PROCESS/TRUE \
        directory.
    Args:
        process: DENOISING, INPAINTING, SR.
    
    Returns:
        img_names: A list of stems of the images.'''
    
    img_names = read_img_names(process)
    img_stems = [stem_from_name(name) for name in img_names]
    
    return img_stems

def read_img_names(
    process: Process
) -> List[str]:
    '''Returns the name of all images inside the ROOT/IMAGES/PROCESS/TRUE \
        directory.
    Args:
        process: DENOISING, INPAINTING, SR.
    
    Returns:
        img_names: A list of names of the images.'''
    
    img_names = ROOT[IMAGES][process][TRUE].physical_children()
    
    return img_names

def read_true_image(process: str, stem: str) -> NumpyArray:
    '''Returns the wanted ground-truth image inside the ROOT/IMAGES/PROCESS/TRUE \
        directory.
        
    Args:
        process: DENOISING, INPAINTING, SR.
    
    Returns:
        true_img: The ground-truth image.'''
        
    name = name_from_stem(stem)
    return ROOT[IMAGES][process][TRUE][name].load()

def read_noisy_image(
    stem: str, 
    sigma: Optional[int] = None, p: Optional[int] = None, zoom: Optional[int] = None,
    ret_noise: bool = False
) -> Tuple[NumpyArray, Optional[NumpyArray]]:
    '''Returns the wanted noisy image inside the ROOT/IMAGES/PROCESS \
        directory.
    
    Args:
        sigma: If sigma is given, the process assumed to be denoising. \
            sigma is the std of the noise in the noisy image (for the images \
            of range [0, 255]).
        p: If p is given, the process assumed to be inpainting. p is the \
            percent probability of black pixels in the  noisy image.
        zoom: If zoom is given, the process assumed to be \
            super-resolution. zoom is the ratio of the dimensions of \
            ground-truth image and noisy image.
        ret_noise: If true, returns also a the noise. No effect for \
            super-resolution.
    
    Returns:
        noisy_img: Noisy image.
        noises: Noise. Always None for super-resolution.'''
        
    process = get_process(sigma=sigma, p=p, zoom=zoom)
    
    if process == DENOISING:    
        try:
            noisy_img = ROOT[IMAGES][process][NOISY][sigma][f'{stem}.png'].load()
            noise = ROOT[IMAGES][process][NOISE][sigma][f'{stem}.npy'].load()
        except FileNotFoundError:
            # if there is no noisy image yet, create it
            print(f'Noisy version of the {stem}.png was not exists so it is created.')

            true_img = read_true_image(process, stem)
            noisy_img, noise = du.get_noisy_img(true_img, sigma=sigma, ret_noise=True)

            # save the noise and the noisy_img to their corresponding places
            ROOT[IMAGES][process][NOISY][sigma][f'{stem}.png'].save(noisy_img)
            ROOT[IMAGES][process][NOISE][sigma][f'{stem}.npy'].save(noise)
            
        if ret_noise:
            return noisy_img, noise
        else:
            return noisy_img, None
            
    if process == INPAINTING:            
        try:
            noisy_img = ROOT[IMAGES][process][NOISY][p][f'{stem}.png'].load()
            noise = ROOT[IMAGES][process][NOISE][p][f'{stem}.npy'].load()
        except FileNotFoundError:
            # if there is no noisy image yet, create it
            print(f'Noisy version of the {stem}.png was not exists but it is created.')
            
            true_img = read_true_image(process, stem)
            noisy_img, noise = iu.get_masked_img(true_img, p=p, ret_mask=True)
            
            # save the noise and the noisy_img to their corresponding places
            ROOT[IMAGES][process][NOISY][p][f'{stem}.png'].save(noisy_img)
            ROOT[IMAGES][process][NOISE][p][f'{stem}.npy'].save(noise)
            
        if ret_noise:
            return noisy_img, noise
        else:
            return noisy_img, None

    if process == SR:            
        try:
            noisy_img = ROOT[IMAGES][process][NOISY][zoom][f'{stem}.png'].load()
        except FileNotFoundError:
            # if there is no noisy image yet, create it
            print(f'Noisy version of the {stem}.png was not exists but it is created.')
            
            true_img = read_true_image(process, stem)
            noisy_img = su.get_lr_img(true_img, zoom, fmt='np')
            
            # save the noise and the noisy_img to their corresponding places
            ROOT[IMAGES][process][NOISY][zoom][f'{stem}.png'].save(noisy_img)
            
        return noisy_img, None
    
def read_true_images(
    process: Process
) -> Dict[str, NumpyArray]:
    '''Returns all the ground-truth images inside the ROOT/IMAGES/PROCESS/TRUE \
        directory.
        
    Args:
        process: DENOISING, INPAINTING, SR.
    
    Returns:
        true_imgs: A dictionary of ground-truth images. Keys are the stems of \
            the images and the values are the images themselves.'''
    
    img_stems = read_img_stems(process)
    
    true_imgs = [read_true_image(process, stem) for stem in img_stems]
    true_imgs = {stem: img for stem, img in zip(img_stems, true_imgs)}
    
    return true_imgs

def read_noisy_images(
    sigma: Optional[int] = None, p: Optional[int] = None, zoom: Optional[int] = None,
    ret_noise: bool = False
) -> Tuple[Dict[str, NumpyArray], Optional[Dict[str, NumpyArray]]]:
    '''Returns all the noisy images inside the ROOT/IMAGES/PROCESS \
        directory.
    
    Args:
        sigma: If sigma is given, the process assumed to be denoising. \
            sigma is the std of the noise in the noisy image (for the images \
            of range [0, 255]).
        p: If p is given, the process assumed to be inpainting. p is the \
            percent probability of black pixels in the  noisy image.
        zoom: If zoom is given, the process assumed to be \
            super-resolution. zoom is the ratio of the dimensions of \
            ground-truth image and noisy image.
        ret_noise: If true, returns also a the noise. No effect for \
            super-resolution.
    
    Returns:
        noisy_imgs: A dictionary of noisy images. Keys are the stems of the \
            images and the values are the images themselves.
        noises: A dictionary of noises. Keys are the stems of the images and \
            the values are the noises themselves. Always None for SR.'''
    
    process = get_process(sigma, p, zoom)
    
    img_stems = read_img_stems(process)
    
    if ret_noise and process in (DENOISING, INPAINTING):
        noisy_imgs: Dict[str, NumpyArray] = {}
        noises: Dict[str, NumpyArray] = {}
        for stem in img_stems:
            noisy_img, noise = read_noisy_image(process, stem, sigma, p, zoom, ret_noise)
            noisy_imgs[stem] = noisy_img
            noises[stem] = noise
        return noisy_imgs, noises
    else:
        noisy_imgs: Dict[str, NumpyArray] = {}
        for stem in img_stems:
            noisy_img= read_noisy_image(process, stem, stem, sigma, p, zoom, ret_noise)
            noisy_imgs[stem] = noisy_img
        return noisy_imgs, None

def read_images(
    process: Process, **kwargs
) -> Tuple[Dict[str, NumpyArray], Dict[str, NumpyArray], Dict[str, NumpyArray]]:
    '''A wrapper for the functions read_true_images and read_noisy_images. \
        Returns all the true images, noisy images and the corresponding noises \
        inside the ROOT/IMAGES directory.
    
    Args:
        process: DENOISING, INPAINTING, SR.
        sigma: If the process is DENOISING, sigma is the std of the \
            noise in the noisy image (for the images of range [0, 255]).
        p: If the process is INPAINTING, p is the percent probability of \
            black pixels in the  noisy image.
        ret_noise: If true, returns also a dictionary of the noises \
            called noises.
    
    Returns:
        true_imgs: A dictionary of ground-truth images. Keys are the stems of \
            the images and the values are the images themselves.'
        noisy_imgs: A dictionary of noisy images. Keys are the stems of the \
            images and the values are the images themselves.
        noises: A dictionary of noises. Keys are the stems of the images and \
            the values are the noises themselves.'''
    
    if 'ret_noise' in kwargs:
        ret_noise = kwargs['ret_noise']
    else:
        ret_noise = False
    
    true_imgs = read_true_images(process)
    
    if ret_noise:
        noisy_imgs, noises = read_noisy_images(process, **kwargs)
    else:
        noisy_imgs = read_noisy_images(process, **kwargs)
    
    if ret_noise:
        return true_imgs, noisy_imgs, noises
    else:
        return true_imgs, noisy_imgs

    

def imshow(
    img: NumpyArray, normalize: bool = True, fname: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> None:
    '''Displays a numpy.ndarray as an image.
    
    Args:
        img: An array of shape (1, H, W), (H, W) or (3, H, W).
        normalize: If it is true, img is normalized between 0 and 1 before \
            plotting.
        fname: A path. If it is not None, the plot is saved to fname.
        ax: Matplotlib axes object. If it is not None, the image is plotted on \
            this axes.'''
    
    if ax is None:
        fig, ax = plt.subplots()
    
    img = np.squeeze(img)

    if normalize:
        img = fn.norm(img)
        
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        if img.shape[0] == 1:
            img = img.squeeze()
        else:
            img = np.moveaxis(img, 0, -1)
            
        ax.imshow(img, vmin=0, vmax=1)

    ax.set_axis_off()
    
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', pad_inches=0)



def scatter(x, y, s=1):
    plt.scatter(x, y, s=s)
    plt.show()

def completed_img_stems(
    sigma: Optional[int] = None,
    p: Optional[int] = None,
    zoom: Optional[int] = None
) -> List[str]:
    '''Return the name of files under the directory \
        ROOT/BENCHMARK/PROCESS/[sigma, p, zoom]'''
    
    
    PROCESS = get_process(sigma, p, zoom)
    
    if PROCESS == DENOISING:
        img_stems = ROOT[BENCHMARK][PROCESS][sigma].physical_children()
        return img_stems
    if PROCESS == INPAINTING:
        img_stems = ROOT[BENCHMARK][PROCESS][p].physical_children()
        return img_stems
    if PROCESS == SR:
        img_stems = ROOT[BENCHMARK][PROCESS][zoom].physical_children()
        return img_stems
from typing import Tuple

import numpy as np

from utils.common_types import *


def wedge_mask(img_dim: Tuple[int, int], wedge_width_ratio: float = 0.2) -> NumpyArray:
    
    
    img_dimy, img_dimx = img_dim
    
    xx, yy = np.meshgrid(range(img_dimx), range(img_dimy))

    xx = xx / img_dimx
    yy = yy / img_dimy

    xx = xx - 1/2
    yy = yy - 1/2

    mask = np.logical_or(
        np.abs(xx) < wedge_width_ratio/2, np.abs(yy) < wedge_width_ratio/2
    )

    return mask[None, :, :]

def circle_mask(img_dim: Tuple[int, int], diameter_ratio: float = 0.5) -> NumpyArray:
    
    
    img_dimy, img_dimx = img_dim
    
    xx, yy = np.meshgrid(range(img_dimx), range(img_dimy))

    xx = xx / img_dimx
    yy = yy / img_dimy

    xx = xx - 1/2
    yy = yy - 1/2

    mask = xx**2 + yy**2 < (diameter_ratio/2) ** 2

    return mask[None, :, :]

def circular_strip_mask(
    img_dim: Tuple[int, int], 
    outer_diameter_ratio: float = 0.2, inner_diameter_ratio: float = 0.1
) -> NumpyArray:

    
    outer_circle = circle_mask(img_dim, diameter_ratio=outer_diameter_ratio)
    inner_circle = circle_mask(img_dim, diameter_ratio=inner_diameter_ratio)
    
    strip = np.logical_xor(outer_circle, inner_circle)

    return strip

def rectangle_mask(
    img_dim: Tuple[int, int], height_ratio: float = 0.3, width_ratio: float = 0.7
) -> NumpyArray:
    
    
    img_dimx, img_dimy = img_dim
    
    xx, yy = np.meshgrid(range(img_dimx), range(img_dimy))

    xx = xx / img_dimx
    yy = yy / img_dimy

    xx = xx - 1/2
    yy = yy - 1/2

    mask = np.logical_and(
        np.abs(xx) < width_ratio/2, np.abs(yy) < height_ratio/2
    )

    return mask[None, :, :]

def square_mask(img_dim: Tuple[int, int], width_ratio: float = 0.5) -> NumpyArray:
    return rectangle_mask(img_dim, width_ratio, width_ratio)


def apply_mask(arr: NumpyArray, mask: NumpyArray) -> NumpyArray:
    
    
    channels = arr.shape[-3]
    
    if channels != 1:
        mask = np.concatenate([mask] * channels, axis=0)
    
    if len(arr.shape) == 3:
        return arr[mask]

    if len(arr.shape) == 4:
        return arr[:, mask]
    
    assert False
    
    
    
    
    
    
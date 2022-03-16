from typing import Dict, List, TypedDict, Union, Literal

import torch
import numpy as np
import pandas as pd
import PIL.Image as Image

from .keywords import *


DataFrame = pd.DataFrame
NumpyArray = np.ndarray
Tensor = torch.Tensor
Model = torch.nn.Module
Array = Union[NumpyArray, Tensor]
PILImg = Image.Image
Img = Union[PILImg, NumpyArray, Tensor]
Process = Literal[DENOISING, INPAINTING, SR]
ArrayFormat = Literal['np', 'torch']
ImageFormat = Literal['np', 'torch', 'pil']

class HtrDict(TypedDict):
    loss: List[float]
    psnr_gt: List[float]
    psnr_gt_sm: List[float]

    best_psnr_gt: float
    best_psnr_gt_sm: float
    
    last_out: NumpyArray
    last_out_sm: NumpyArray

    best_out: NumpyArray
    best_out_sm: NumpyArray
    
    best_iter: int
    best_iter_sm: int
    
    outs: Dict[int, NumpyArray]
    outs_sm: Dict[int, NumpyArray]
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import cv2

from . import basic_utils as bu
from . import array_utils as au
from .common_types import *


def is_np(img: Img) -> bool:
    '''Check whether the given image is in numpy array format.'''
    
    if not isinstance(img, NumpyArray):
        return False
    
    if len(img.shape) != 3:
        return False
    
    if img.shape[0] not in (1, 3):
        return False
    
    if not img.shape[1] > 3:
        return False
    
    if not img.shape[2] > 3:
        return False
    
    return True

def is_torch(img: Img) -> bool:
    '''Check whether the given image is in torch tensor format.'''
    
    if not isinstance(img, Tensor):
        return False
    
    if len(img.shape) != 4:
        return False
    
    if img.shape[0] != 1:
        return False
    
    if img.shape[1] not in (1, 3):
        return False
    
    if not img.shape[2] > 3:
        return False
    
    if not img.shape[3] > 3:
        return False
    
    return True

def is_pil(img: Img) -> bool:
    '''Check whether the given image is in PIL image format.'''
    
    if not isinstance(img, PILImg):
        return False
    
    return True

def np_to_torch(img_np: NumpyArray) -> Tensor:
    '''Converts a numpy.ndarray of shape (C, H, W) to a torch.tensor of shape \
        (1, C, H, W).'''
        
    assert is_np(img_np)
    
    img_torch = au.np_to_torch(img_np)
    img_torch = torch.unsqueeze(img_torch, 0)
    return img_torch

def torch_to_np(img_torch: Tensor) -> NumpyArray:
    '''Converts a torch.tensor of shape (1, C, H, W) to a numpy.ndarray of \
        shape (C, H, W).'''
        
    assert is_torch(img_torch)
    
    img_np = au.torch_to_np(img_torch)
    img_np = np.squeeze(img_np, 0)
    return img_np

def np_to_pil(img_np: NumpyArray) -> PILImg:
    '''Converts a numpy array of shape (C, H, W) with values in [0, 1], to a \
        PIL image.'''
        
    assert is_np(img_np)
    
    if img_np.shape[0] == 1:
        mode = 'L'
    elif img_np.shape[0] == 3:
        mode = 'RGB'
        
    img_np = np.moveaxis(img_np, 0, -1)
    img_np = img_np * 255
    img_np = img_np.clip(0, 255)
    img_np = img_np.astype(np.uint8)
    
    if mode == 'L':
        img_np = img_np.squeeze(-1)
    
    img_pil = Image.fromarray(img_np, mode=mode)
    return img_pil

def pil_to_np(img_pil: PILImg) -> NumpyArray:
    '''Converts a PIL image to a numpy array of shape (C, H, W) with values in \
        [0, 1].'''
        
    assert is_pil(img_pil)
        
    img_np = np.asarray(img_pil)
    img_np = img_np.astype(np.float32)
    img_np = img_np / 255
    img_np = img_np.clip(0, 1)
    
    if len(img_np.shape) == 2:
        img_np = img_np[None, :, :]
    else:
        img_np = np.moveaxis(img_np, -1, 0)
        
    return img_np

def torch_to_pil(img_torch: Tensor) -> PILImg:
    '''Converts a torch tensor of shape (1, C, H, W) with values in [0, 1], to \
        a PIL image.'''
        
    assert is_torch(img_torch)
    
    img_np = torch_to_np(img_torch)
    img_pil = np_to_pil(img_np)
    return img_pil

def pil_to_torch(img_pil: PILImg) -> PILImg:
    '''Converts a PIL image to a torch tensor of shape (1, C, H, W) with \
        values in [0, 1].'''
        
    assert is_pil(img_pil)
        
    img_np = pil_to_np(img_pil)
    img_torch = np_to_torch(img_np)
    return img_torch

def to_pil(img: Img) -> PILImg:
    '''Convert img to pil img.'''
    
    if is_pil(img):
        return img
    
    if is_np(img):
        return np_to_pil(img)
    
    if is_torch(img):
        return torch_to_pil(img)
    
    assert False

def to_np(img: Img) -> NumpyArray:
    '''Convert img to np img.'''
    
    if is_np(img):
        return img
    
    if is_pil(img):
        return pil_to_np(img)
    
    if is_torch(img):
        return torch_to_np(img)
    
    assert False

def to_torch(img: Img) -> Tensor:
    '''Convert img to torch img.'''
    
    if is_torch(img):
        return img
    
    if is_pil(img):
        return pil_to_torch(img)
    
    if is_np(img):
        return np_to_torch(img)
    
    assert False

def get_img_fmt(img: Img) -> Literal['pil', 'np', 'torch']:
    '''Determine the format of the image: pil, np or torch.'''
    
    if is_pil(img):
        return 'pil'
    elif is_np(img):
        return 'np'
    elif is_torch(img):
        return 'torch'
    
    assert False

def to(img: Img, fmt: Literal['pil', 'np', 'torch']) -> Img:
    '''Convert img to the wanted format.'''
    
    if fmt == 'pil':
        return to_pil(img)
    
    if fmt == 'np':
        return to_np(img)
    
    if fmt == 'torch':
        return to_torch(img)
    
    assert False

def resize(img: Img, new_size: Union[Tuple[int], int]) -> Img:
    '''Resize the image. The format of the output image is the same as the \
        format of the input image.'''
    
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    h, w = new_size
    
    fmt = get_img_fmt(img)
    
    img_pil = to_pil(img)
    img_pil = img_pil.resize((w, h))
    img = to(img_pil, fmt)
    return img

def crop_box(
    img: Img, 
    top_left: Tuple[float, float], 
    size: Union[float, Tuple[float, float]],
    copy: bool = False
) -> Img:
    '''Crops out a box from the given image. The format of the output is the \
        same as the format of the input image.
    
    Args:
        img: The image.
        top_left: Coordinate of the top left corner of the box. The coordinate \
            is the ratio of the indices of the top left corner of the box to \
            the dimensions of the given image.
        size: Size of the box. The size is the ratio of the dimensions of the \
            box to the dimensions of the image.
        copy: If it is true, a deep copy is returned.
    
    Returns:
        The box with same format of the input image.
        '''
    
    if isinstance(size, int):
        size = (size, size)
    
    img_np = to_np(img)
    img_size = img_np.shape[1], img_np.shape[2]
        
    boxh = int(size[0] * img_size[0])
    boxw = int(size[1] * img_size[1])
    
    boxi = int(top_left[0] * img_size[0])
    boxj = int(top_left[1] * img_size[1])
    
    fmt = get_img_fmt(img)
    
    img_np = to_np(img)
    box_np = img[:, boxi: boxi + boxh, boxj: boxj + boxw]
    
    if copy:
        box_np = np.array(box_np)
    
    box = to(box_np, fmt)
    return box

def image_and_box(
    img: Img,
    top_left1: Tuple[float, float], 
    top_left2: Tuple[float, float], 
    size: Union[float, Tuple[float, float]],
    sep_px: int = 7,
    sep_color: str = 'black',
    location: Literal['top', 'bottom', 'left', 'right'] = 'bottom'
) -> Img:
    '''Crops two boxes from the image and concatenate them to the given location.
    
    Args:
        img: The image.
        top_left1, top_left2: The coordinates of the top left corners of the \
            boxes. The coordinates are the ratios of the indices of the top \
            left corners to the dimensions of the image.
        size: Size of the boxes. The size is the ratio of the dimensions of \
            the box to the dimensions of the image.
        sep_px: The number pixels to be added at the output between the \
            originial image the boxes.
        sep_color: The color pixels to be added at the output between the \
            originial image the boxes.
        location: The location to be concatenate the cropped boxes.
        
    Returns:
        An image consisting of the original image and the two cropped boxes. \
            The boxes are scaled appropiately.'''
    
    fmt = get_img_fmt(img)
    img_np = to_np(img)
    img_np = np.array(img_np)
    
    # crop the boxes
    box1_np = crop_box(img_np, top_left1, size, copy=True)
    box2_np = crop_box(img_np, top_left2, size, copy=True)
    
    # resize the boxes
    c, h, w = img_np.shape
    hbox, wbox = int(size[0] * h), int(size[1] * w)
    if location == 'bottom':
        new_hbox = int(hbox * ((w/2) / wbox))

        box1_np = resize(
            box1_np, 
            (new_hbox, w//2)
        )
        box2_np = resize(
            box2_np, 
            (new_hbox, w - w//2)
        )
    elif location == 'right':
        new_wbox = int(wbox * ((h/2) / hbox))
        
        box1_np = resize(
            box1_np, 
            (h//2, new_wbox)
        )
        box2_np = resize(
            box2_np, 
            (h - h//2, new_wbox)
        )
        
    
    if location == 'bottom':
        box_np = np.concatenate((box1_np, box2_np), axis=-1)
    elif location == 'right':
        box_np = np.concatenate((box1_np, box2_np), axis=-2)
        
    
    # mark the boxes on img for boxes1
    tck_x = 0.005
    tck_y = 0.005
    
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    xx = np.logical_and(top_left1[1] <= xx, xx <= top_left1[1] + size[1])
    yy = np.logical_and(top_left1[0] <= yy, yy <= top_left1[0] + size[0])
    mask1 = np.logical_and(xx, yy)
    
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    xx = np.logical_and(top_left1[1] + tck_x <= xx, xx <= top_left1[1] + size[1] - tck_x)
    yy = np.logical_and(top_left1[0] + tck_y <= yy, yy <= top_left1[0] + size[0] - tck_y)
    mask2 = np.logical_and(xx, yy)
    
    mask = np.logical_xor(mask1, mask2)
    img_np[:, mask] = 0
    
    # mark the boxes on img for boxes2
    tck_x = 0.005
    tck_y = 0.005
    
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    xx = np.logical_and(top_left2[1] <= xx, xx <= top_left2[1] + size[1])
    yy = np.logical_and(top_left2[0] <= yy, yy <= top_left2[0] + size[0])
    mask1 = np.logical_and(xx, yy)
    
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    xx = np.logical_and(top_left2[1] + tck_x <= xx, xx <= top_left2[1] + size[1] - tck_x)
    yy = np.logical_and(top_left2[0] + tck_y <= yy, yy <= top_left2[0] + size[0] - tck_y)
    mask2 = np.logical_and(xx, yy)
    
    mask = np.logical_xor(mask1, mask2)
    img_np[:, mask] = 0
    
    
    # create the total image
    if location == 'bottom':
        total_img_np = np.zeros((c, h + new_hbox + sep_px, w))
        
        total_img_np[:, :h, :] = img_np
        
        if sep_color == 'black':
            pass
        elif sep_color == 'red':
            total_img_np[0, h:(h + sep_px), :] = 1
        elif sep_color == 'green':
            total_img_np[1, h:(h + sep_px), :] = 1
        elif sep_color == 'blue':
            total_img_np[2, h:(h + sep_px), :] = 1
        elif sep_color == 'white':
            total_img_np[:, h:(h + sep_px), :] = 1
            
        total_img_np[:, (h + sep_px):, :] = box_np
        
    elif location == 'right':
        total_img_np = np.zeros((c, h, w + new_wbox + sep_px))
        
        total_img_np[:, :, :w] = img_np
        
        if sep_color == 'black':
            pass
        elif sep_color == 'red':
            total_img_np[0, :, w:(w + sep_px)] = 1
        elif sep_color == 'green':
            total_img_np[1, :, w:(w + sep_px)] = 1
        elif sep_color == 'blue':
            total_img_np[2, :, w:(w + sep_px)] = 1
        elif sep_color == 'white':
            total_img_np[:, :, w:(w + sep_px)] = 1
            
        total_img_np[:, :, (w + sep_px):] = box_np
    
    total_img = to(total_img_np, fmt)
    return total_img

def qualitative_comparison(
    images: List[Img],
    top_left1: Tuple[float, float], 
    top_left2: Tuple[float, float], 
    size: Union[float, Tuple[float, float]],
    titles: List[str] = None,
    axs: Optional[List[plt.Axes]] = None,
    fontsize: Optional[int] = None,
    sep_px: int = 7,
    sep_color: str = 'black',
    location: Literal['top', 'bottom', 'left', 'right'] = 'bottom'
) -> None:
    '''Using the image_and_box function, creates a qualitative comparison plot.
    
    Args:
        images: A list of images.
        top_left1, top_left2: The coordinates of the top left corners of the \
            boxes. The coordinates are the ratios of the indices of the top \
            left corners to the dimensions of the image.
        size: Size of the boxes. The size is the ratio of the dimensions of \
            the box to the dimensions of the image.
        titles: Titles for the images. If it is None, no titles are added.
        axs: A list of Matplotlib axes objects to draw the plots. If it is \
            None, axes objects are created.
        sep_px: The number pixels to be added at the output between the \
            originial image the boxes.
        sep_color: The color pixels to be added at the output between the \
            originial image the boxes.
        location: The location to be concatenate the cropped boxes.'''
    
    if axs is None:
        fig, axs = plt.subplot(1, len(images))
    
    images_with_boxes = []
    for img in images:
        images_with_boxes.append(
            image_and_box(
                img, top_left1, top_left2, size, 
                sep_px=sep_px, sep_color=sep_color, location=location
            )
        )
    
    for i, img_with_boxes in enumerate(images_with_boxes):
        bu.imshow(img_with_boxes, normalize=False, ax=axs[i])
        
        if titles is not None and titles[i] is not None:
            axs[i].set_title(titles[i])
        
        if fontsize is not None:
            axs[i].title.set_fontsize(fontsize)

def crop_image(img: Img, d: int = 32) -> Img:
    '''Make dimensions divisible by d.'''
    
    fmt = get_img_fmt(img)
    img = to_np(img)
    _, h, w = img.shape
    hnew, wnew = h - h%d, w - w%d
    img = img[:, (h - hnew)//2:(h + hnew)//2, (w - wnew)//2:(w + wnew)//2]
    img = to(img, fmt)
    return img

# def crop_image(img, d=32):
    # '''Make dimensions divisible by d.'''
    # x = Image.open('u')

    # new_size = (img.size[0] - img.size[0] % d, 
    #             img.size[1] - img.size[1] % d)

    # bbox = [
    #         int((img.size[0] - new_size[0])/2), 
    #         int((img.size[1] - new_size[1])/2),
    #         int((img.size[0] + new_size[0])/2),
    #         int((img.size[1] + new_size[1])/2),
    # ]

    # img_cropped = img.crop(bbox)
    # return img_cropped

def rgb2ycbcr(im_rgb):
    '''Convert an image in RGB format to YCbCr format.'''
    
    im_rgb = np.moveaxis(im_rgb, 0, -1)
    
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    
    im_ycbcr = np.moveaxis(im_ycbcr, -1, 0)
    
    return im_ycbcr
    
def get_device(tensor: Tensor) -> torch.device:
    return tensor.device

def get_dtype(tensor: Tensor) -> torch.dtype:
    return tensor.dtype

def to_device(
    tensor: Tensor, 
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Tensor:
    
    if device is not None:
        tensor = tensor.to(device=device)
    
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
        
    return tensor

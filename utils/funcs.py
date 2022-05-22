import math
from typing import Any, Optional, Dict, Callable, Tuple

import numpy as np
from numpy.fft import fft2, fftshift

from scipy.stats import wasserstein_distance

from . import common_utils as cu
from . import basic_utils as bu
from . import image_utils as imu
from .data_structures import Tree, ModDefaultDict
from utils.masks import apply_mask
from utils.masks import circular_strip_mask, circle_mask
from .common_types import *


# to be used in metric calculations
HIST_BINS = 100
HIST_DENSITY = True
HIST_RANGE = None


GrayModes = Literal['YCbCr', 'Avg']
def to_gray(image: Array, mode: GrayModes) -> Array:
    '''Convert the given image to gray scale using.
    
    Args: 
        image: The image.
        mode: If it isYCbCr, the image is first converted to YCbCr and the Y \
            channel is returned as the gray scale image. If it is Avg, the \
            gray scale image is computed as the average of the RGB channels.
    
    Returns:
        The gray scale image.'''
    
    if isinstance(image, torch.Tensor):
        arr_fmt = 'torch'
    if isinstance(image, np.ndarray):
        arr_fmt = 'np'
    
    channels = image.shape[-3]
    if channels == 1:
        return image
    
    if len(image.shape) == 3:
        if mode == 'YCbCr':
            if arr_fmt == 'torch':
                image = bu.torch_to_np(image)
            
            image = cu.np_to_pil(image)
            image = image.convert('YCbCr')
            image = cu.pil_to_np(image)
            image = image[0]
            image = image[None, :, :]
            
            if arr_fmt == 'torch':
                image = bu.np_to_torch(image)
            
            return image
        
        if mode == 'Avg':
            image = image.mean(-3, keepdims=1)
            return image
        
    elif len(image.shape) == 4:
        gray_images = []
        for imag in image:
            gray_images.append(to_gray(imag, mode=mode))
        images = np.stack(gray_images, axis=0)
        return images
    
    assert False
    

def psd(image: NumpyArray) -> NumpyArray:
    '''Power spectral density of the given image.'''
    
    image_f = fft2(image, norm='forward')

    image_psd = np.abs(image_f)**2

    return fftshift(image_psd)

def db(arr: NumpyArray) -> NumpyArray:
    '''Calculate the dB of the given array element wise.'''
    
    arr_db = 10*np.log(arr)

    return arr_db

def norm(arr: Array, min=0, max=1) -> Array:
    '''Normalize the given array between min and max.'''
    
    arr = arr - arr.min()
    arr = arr / arr.max()
    arr = arr * (max - min)
    arr = arr + min
    return arr

def psd_db(image: NumpyArray) -> NumpyArray:
    '''Applie first psd and then db functions.'''
    
    image_psd = psd(image)
    return db(image_psd)

def psd_db_norm(image: NumpyArray) -> NumpyArray:
    '''Applie psd, db and norm functions.'''
    
    return norm(psd_db(image))

def nodc(arr: NumpyArray) -> NumpyArray:
    '''Remove the DC component.'''
    
    if len(arr.shape) in (1, 3):
        arr = arr - arr.mean()
        return arr
    
    if len(arr.shape) == 2:
        arr = arr - arr.mean(-1, keepdims=True)
        return arr
    
    if len(arr.shape) == 4:
        arr = arr - arr.mean((-1, -2, -3), keepdims=True)
        return arr
    
    assert False

def flatten(psd: NumpyArray, masks: NumpyArray) -> NumpyArray:
    '''Flattens a PSD using an array of masks. Calculates the average energy \
        of the given PSD for each masks and put it in an array.
    Args:
        psd: An array of shape (C, H, W) or (B1, C, H, W), preferably a power \
            spectral density.
        masks: An array of shape (B2, 1, H, W), preferably a boolean array or \
            an array of entries 0 and 1.
    Returns:
        avg_energy: An array of shape (B2,) or (B1, B2) if the shape of psd is \
            (C, H, W) or (B1, C, H, W) respectively.'''
            
    psd = to_gray(psd, mode='Avg')
    
    if len(psd.shape) == 3:
        masked = psd * masks

        tot_energy = masked.sum((1, 2, 3))
        num_pixels = masks.sum((1, 2, 3))
        avg_energy = tot_energy / num_pixels
        
        return avg_energy

    if len(psd.shape) == 4:
        
        avg_energy = np.zeros((psd.shape[0], masks.shape[0]))
        
        for i, ps in enumerate(psd):
            avg_energy[i, :] = flatten(ps, masks)
        
        return avg_energy


def histogram(
    arr: NumpyArray, bins=100, density=False, range=None, 
    threshold: Optional[float] = None, discard_zeros=True
):
    '''Creates an histogram of the arr. For more information see \
        numpy.histogram function.
    
    Args:
        arr: An array of shape (_,), (B, _), (_, _, _), (B, _, _, _). If the \
            shape of arr is (_, _, _) or (B, _, _, _), first it is converted \
            to the shape (_, ) or (B, _) respectively.
    
    Returns:
        hist: An array of sahpe (_,) or (B, _).'''
    
    if len(arr.shape) == 3:
        arr = arr.reshape((-1,))
    
    if len(arr.shape) == 1:
        if discard_zeros:
            arr = arr[arr != 0]
        
        if threshold is not None:
            arr = arr[arr >= threshold]
        
        hist, _ = np.histogram(
            arr, bins=bins, density=density, range=range
        )
        
        if not density:
            hist = hist / arr.size
        
        return hist

    if len(arr.shape) == 4:
        arr = arr.reshape((arr.shape[0], -1))
    
    if len(arr.shape) == 2:
        
        hists = []
        for ar in arr:
            hists.append(
                histogram(
                    ar, bins=bins, density=density, range=range, 
                    threshold=threshold, discard_zeros=discard_zeros
                )
            )
        
        hist = np.stack(hists)
        
        return hist


def mse(arr1: NumpyArray, arr2: NumpyArray):
    '''Mean square error between arr1 and arr2.'''
    
    if len(arr1.shape) in (1, 2) and len(arr2.shape) in (1, 2):
        dist = ((arr1 - arr2)**2).mean(-1)
        return dist
    
    if len(arr1.shape) in (3, 4) and len(arr2.shape) in (3, 4):
        dist = ((arr1 - arr2)**2).mean((-1, -2, -3))
        return dist

    assert False

def emd(arr1: NumpyArray, arr2: NumpyArray):
    '''Earth mover's distance between arr1 and arr2.'''
    
    if len(arr1.shape) == 1 and len(arr2.shape) == 1:
        dist = np.float64(wasserstein_distance(arr1, arr2))
        return dist
    
    if len(arr1.shape) == 2 and len(arr2.shape) == 1:
        arr1, arr2 = arr2, arr1
    
    if len(arr1.shape) == 1 and len(arr2.shape) == 2:
        
        dist = np.zeros((arr2.shape[0],))
        for i, ar2 in enumerate(arr2):
            dist[i] = np.float64(wasserstein_distance(arr1, ar2))
        
        return dist
    
    if len(arr1.shape) == 2 and len(arr2.shape) == 2:
        assert arr1.shape[0] == arr2.shape[1]
        
        dist = np.zeros((arr2.shape[0],))
        for i, (ar1, ar2) in enumerate(zip(arr1, arr2)):
            dist[i] = np.float64(wasserstein_distance(ar1, ar2))
        
        return dist
    
    assert False

def per_bw(psd: NumpyArray, masks : NumpyArray, p):
    '''p percent bandwidth of the given PSD. An array of masks is given. It is \
        assumed that these are circles of increasing diameter. The energy \
        contained in each mask is calculated and the diameter ratio of the \
        mask whose energy is the p percent of the total energy of the PSD.'''
    
    psd = to_gray(psd, mode='Avg')
    
    if len(psd.shape) == 3:
        
        dim = psd.shape[-1]
        total_energy = psd.sum()
        
        prev_energy = 0
        for i, mask in enumerate(masks):
            # print(i)
            energy = psd[mask].sum()
            
            if energy >= p * total_energy:
                break
            
            prev_energy = energy
        
        slope = (energy - prev_energy) / 1
        i = (p * total_energy - prev_energy) / slope + i
        
        return np.float64(i / masks.shape[0])
    
    if len(psd.shape) == 4:
        
        result = np.zeros((psd.shape[0],))
        for i, ps in enumerate(psd):
            result[i] = per_bw(ps, masks, p=p)
        
        return result

def db_bw(psd_db: NumpyArray, cut_off : float = 75):
    '''Calculates the bandwidth of the array using a cut off value. The ratio \
        of the number of pixels with value greater the cut_off value to the \
        total number of pixels is returned.'''
    
    psd_db = to_gray(psd_db, mode='Avg')
    
    c, h, w = psd_db.shape[-3], psd_db.shape[-2], psd_db.shape[-1]

    max = np.max(psd_db, axis=(-1, -2, -3), keepdims=True)

    pixels = psd_db > (max - cut_off)

    num = np.sum(pixels, axis=(-1, -2, -3))

    ratio = num / (c * h * w)

    return ratio


def psnr(img_true: Array, img_test: Array, ycbcr: bool = False) -> float:
    '''Calculates the PSNR between given images. If the ycbcr flag is set to \
        true, then the images are converted to YCbCr format and PSNR is \
        calculated between the Y channels.'''
    
    if ycbcr:
        result = psnr(imu.rgb2ycbcr(img_true)[0], imu.rgb2ycbcr(img_test)[0], ycbcr = False)
        return result
    
    if isinstance(img_true, np.ndarray):
        tmp = (img_true - img_test)**2
        mse = np.mean(tmp)

        result = -10*np.log10(mse)
        result = float(result)

        return result

    if isinstance(img_true, torch.Tensor):
        tmp = (img_true - img_test)**2
        mse = torch.mean()

        result = -10*torch.log10(mse)
        result = float(result)

        return result

def average(obj1: Any, obj2: Any, weight1: float = 0.5, weight2: float = 0.5) -> Any:
    '''Weighted average of the two objects.'''
    
    return (obj1*weight1 + obj2*weight2) / (weight1 + weight2)
    
class UsefullMaps:
    '''The commonly used maps in the experiments.'''
    
    def __init__(self, img_size: Tuple[int, int]):
    
        NUM_MASKS = 100
        diameters = np.linspace(0, 1, NUM_MASKS, endpoint=True)

        masks = []
        for i in range(1, NUM_MASKS):
            masks.append(
                circular_strip_mask(img_size, diameters[i], diameters[i-1])
            )
        masks = np.stack(masks, axis=0)
        
        strip = circular_strip_mask(img_size, 0.2, 0.1)
        circle = circle_mask(img_size, diameter_ratio=0.5)
        
        transformation_map : Dict[str, Callable[[NumpyArray], NumpyArray]] = {
            'psd': psd,
            'db': db,
            'norm': norm,
            'nodc': nodc,
            'flatten': lambda arr: flatten(arr, masks),
            'hist': lambda arr: histogram(
                arr, bins=HIST_BINS, density=HIST_DENSITY, range=HIST_RANGE, 
                discard_zeros=False
            ),
            'strip': lambda arr: apply_mask(arr, strip),
            'circle': lambda arr: apply_mask(arr, circle),
            
            'identity': lambda arr: arr,
            '': lambda arr: arr,
            
            'random': lambda arr: np.random.randn(*arr.shape)
        }
        
        #####
    
        NUM_MASKS = 100
        diameters = np.linspace(0, math.sqrt(2), NUM_MASKS, endpoint=True)

        masks = []
        for i in range(1, NUM_MASKS):
            masks.append(
                circle_mask(img_size, diameters[i])
            )
        masks = np.stack(masks, axis=0)
        
        def Factory(transformation: str) -> Callable[[NumpyArray], float]:
            tokens = transformation.split('_')
            assert len(tokens) == 3
            
            num = float(tokens[0])
            name = tokens[1] + '_' + tokens[2]
            
            if name == 'per_bw':
                return lambda arr: per_bw(arr, masks, p=num/100).mean()

            if name == 'db_bw':
                return lambda arr: db_bw(arr, cut_off=num).mean()
            
            assert False
    
        self.transformation_map : Dict[str, Callable[[NumpyArray], NumpyArray]] = \
            ModDefaultDict(
                Factory,
                transformation_map
            )
        
        self.loss_map : Dict[str, Callable[[NumpyArray, NumpyArray], float]] = {
            'mse': mse,
            'emd': emd,
        }

class Cache:
    '''Implements a cache object for fast calculation of the metrics. To use, \
        create a cache object. Register an array to this cache object using \
        the register method. After all the calculations are done, unregister \
        the array from the cache object using the unregister method.'''
    
    def __init__(self):
        self._cache: Dict[int, Tree] = {}
        
    def register(self, arr: NumpyArray):
        
        class Proxy:
            def __init__(proxy_self, arr: NumpyArray, cache: 'Cache') -> None:
                proxy_self.arr = arr
                proxy_self.cache = cache
                cache._cache[id(arr)] = Tree('', arr)
                
            def __enter__(proxy_self) -> None:
                pass
            
            def __exit__(proxy_self, *args, **kwargs) -> None:
                proxy_self.cache.unregister(proxy_self.arr)
                
        return Proxy(arr, self)
        
    def unregister(self, arr: NumpyArray) -> None:
        '''Unregisters the given array from the cache.'''
        
        del self._cache[id(arr)]

class Transformation:
    '''A utility class for the calculations of the transformations easily. \
        With this class, we define a straightforward language for the \
        transformations. For example, a transformation in this language is \
        "psd db". An instance of this class, created using this \
        transformation, will take two np.ndarray's as arguments, will compute \
        their psd's, and then their db's. This class also provides cache for \
        fast calculations.
        
    Attributes:
    -----------
    transformation: str
        This is the transformation to be calculated by this object.'''
    
    def __init__(
        self, transformation: str, 
        transformation_map: Dict[str, Callable[[NumpyArray], NumpyArray]],
        cache: Optional['Cache']
    ) -> None:
        
        self.cache = cache
        self.transformation_map = transformation_map
        self.transformation = transformation
        self.transformations = transformation.split()
        
    def __call__(self, arr: NumpyArray) -> NumpyArray:
        '''Computes the transformation self.transformation using cache.'''
        
        if self.cache is None:
            return self._apply_transformation_no_cache(arr)
        
        current_node = self.cache._cache[id(arr)]
        for transformation in self.transformations:
            
            # if this intermediate step is already calculated, just read the 
            # result from the cache
            if transformation in current_node: 
                current_node = current_node[transformation]
            else:
            # otherwise compute and store the result in the cache
                foo = self.transformation_map[transformation]
                
                new_arr = foo(current_node.data)
                new_node = Tree(transformation, new_arr)
                
                current_node.add_child(new_node)
                
                current_node = new_node
                
        return current_node.data
    
    # this is a function used mostly for debugging
    def _apply_transformation_no_cache(self, arr: NumpyArray) -> NumpyArray:
        '''Computes the transformation self.transformatin without using cache.'''
        
        for transformation in self.transformations:
            foo = self.transformation_map[transformation]
            arr = foo(arr)
        
        return arr

class Metric:
    '''A utility class for the calculations of the metrics easily. With this \
        class, we define a small language for the metrics. For example, a \
        valid metric in this language is "psd db mse". The last one is a loss \
        function, other ones are transformations. An instance of this class, \
        created by this metric, will take two np.ndarray's as arguments, will \
        compute their psd's, and then their db's, and finally the mse between \
        them. This class also provides cache for fast calculations.
        
    Attributes:
    -----------
    metric: str
        This is the metric to be calculated by this object.'''
    
    
    def __init__(
        self, metric: str,
        transformation_map: Dict[str, Callable[[NumpyArray], NumpyArray]],
        loss_map: Dict[str, Callable[[NumpyArray, NumpyArray], float]], 
        cache: Optional['Cache']
    ) -> None:
        
        self.cache = cache
        self.metric = metric
        
        tokens = metric.split()
        transformation = ' '.join(tokens[:-1])
        loss = tokens[-1]
        
        self.transformation = Transformation(
            transformation, transformation_map, cache
        )
        self.loss = loss_map[loss]
    
    def __call__(self, img: NumpyArray, out: NumpyArray) -> float:
        '''Computes the self.metric for img and out using cache.'''
        
        img_transformed = self.transformation(img)
        out_transformed = self.transformation(out)
        
        loss = self.loss(img_transformed, out_transformed).mean()
        
        return loss
    
    
    
    
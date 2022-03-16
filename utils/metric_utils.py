from typing import List, Optional

from . import funcs as fn
from . import metric_utils as metu
from .keywords import *
from .common_types import *
from .paths import ROOT


def calculate(
    metric: str, random_output: NumpyArray, 
    img_noisy_np: Optional[NumpyArray] = None
) -> float:
    '''Calculates the given metrics using the random_output and img_noisy_np.'''
    
    lowpass_metrics = metu.get_lowpass_metrics()
    similarity_metrics = metu.get_similarity_metrics()
    
    if metric in similarity_metrics:
        metric_type = 'sim'
    elif metric in lowpass_metrics:
        metric_type = 'low'
    else:
        assert TypeError('metric not found.')
        
    if metric_type == 'sim' and img_noisy_np is None:
        assert False, 'Noisy image is required for similarity metrics.' 
    
    if img_noisy_np is not None:
        assert random_output.shape == img_noisy_np.shape, 'Shapes of the \
            random output and the noisy image must be the same.'
    
    random_output_size = (random_output.shape[1], random_output.shape[2])
    
    # create metric maps
    maps = fn.UsefullMaps(img_size=random_output_size)
    class Transformation(fn.Transformation):
        def __init__(self, transformation: str, cache: fn.Cache) -> None:
            super().__init__(
                transformation, maps.transformation_map, cache
            )

    class Metric(fn.Metric):
        def __init__(self, metric: str, cache: fn.Cache) -> None:
            super().__init__(
                metric, 
                maps.transformation_map,
                maps.loss_map,
                cache
            )
            
    if metric_type == 'sim':
        metric_func = Metric(metric, None)
        result = metric_func(img_noisy_np, random_output)
        return result
    
    if metric_type == 'low':
        func = Transformation(metric, None)
        result = func(random_output)
        return result

def get_metrics(name: str) -> List[str]:
    
    metrics = ROOT[name].load()
    return metrics

def get_similarity_metrics(name: str = SIMILARITY_METRICS_LST) -> List[str]:
    
    return get_metrics(name)

def get_lowpass_metrics(name: str = LOWPASS_METRICS_LST) -> List[str]:
    
    return get_metrics(name)

def get_other_metrics(name: str = RANDOM_METRICS_LST) -> List[str]:
    
    return get_metrics(name)



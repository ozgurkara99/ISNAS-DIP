from typing import Dict, List

import utils.funcs as fn
import utils.metric_utils as metu
from utils.common_types import *

def average_output(
    model_outputs: Dict[str, NumpyArray], model_metrics: Dict[str, float]
) -> NumpyArray:
    
    # normalize metrics
    metrics = np.array(list(model_metrics.values()))
    metrics_min = metrics.min()
    metrics_max = metrics.max()
    
    model_coefs = {
        name: (-model_metrics[name] + metrics_max) / (-metrics_min + metrics_max) for name in model_metrics
    }
    
    # calculate average output
    normalizer = sum(model_coefs[name] for name in model_outputs)
    outputs = [model_outputs[name] * model_coefs[name] for name in model_outputs]
    outputs = np.stack(outputs, axis=0)
    average_output = outputs.sum(axis=0) / normalizer
    
    return average_output

def psnr_wrt(
    model_outputs: Dict[str, NumpyArray],
    img: NumpyArray,
    ycbcr: bool = False
) -> DataFrame:
    '''Calculates the PSNR of each output of the model with respect to the \
    given img.
    
    Args:
        model_outputs: A dictionary of with keys as the model names and values \
        as the numpy arrays of shape (C, H, W).
        img: A numpy array of shape (C, H, W).
    
    Returns:
        df: A pandas dataframe. Its index is 'model name' and only column is \
        'psnr'.'''
    
    models = list(model_outputs.keys())
    
    # calculate the psnr of the models
    model_psnr = {name: fn.psnr(model_outputs[name], img, ycbcr=ycbcr) for name in model_outputs}
    
    df = {
        'model name': [],
        'psnr': []
    }
    for name in models:
        df['model name'].append(name)
        df['psnr'].append(model_psnr[name])
    
    df = pd.DataFrame.from_dict(df)
    df = df.set_index('model name')
    df.sort_values(by='psnr', ascending=False)
    return df

def metric_wrt(
    model_outputs: Dict[str, NumpyArray],
    img: NumpyArray,
    metric: str
) -> DataFrame:
    '''Calculates the given metric of each output of the model with \
        respect to the given img.
    
    Args:
        model_outputs: A dictionary of with keys as the model names and values \
        as the numpy arrays of shape (C, H, W).
        img: A numpy array of shape (C, H, W).
        metric: Name of a metric.
    
    Returns:
        df: A pandas dataframe. Its index is 'model name' and only column is \
        metric.'''
    
    models = list(model_outputs.keys())
    
    # calculate the scores of the models
    model_metric = {name: metu.calculate(metric, model_outputs[name], img) for name in model_outputs}
    
    df = {
        'model name': [],
        metric: []
    }
    for name in models:
        df['model name'].append(name)
        df[metric].append(model_metric[name])
    df = pd.DataFrame.from_dict(df)
    df.set_index('model name')
    df.sort_values(by=metric, ascending=True)
    
    return df

def closest_to(
    model_outputs: Dict[str, NumpyArray],
    img: NumpyArray,
    ycbcr: bool = False
) -> str:
    '''Returns the name of the model whose output is closest (in the mse \
        sense) to the given img.
    
    Args:
        model_outputs: A dictionary of with keys as the model names and values \
        as the numpy arrays of shape (C, H, W).
        img: A numpy array of shape (C, H, W).
    
    Returns:
        model_name: Name of the selected model.'''
    
    df = psnr_wrt(model_outputs, img, ycbcr=ycbcr)

    selected_model = df.idxmax()['psnr']
    return selected_model

def furthest_to(
    model_outputs: Dict[str, NumpyArray],
    img: NumpyArray,
    ycbcr: bool = False
) -> str:
    
    df = psnr_wrt(model_outputs, img, ycbcr=ycbcr)

    selected_model = df.iloc[-1].index
    return selected_model

def similar_to(
    model_outputs: Dict[str, NumpyArray],
    img: NumpyArray,
    metric: str,
    ycbcr: bool = False
) -> str:
    
    metu.calculate(metric, )
        
    df = psnr_wrt(model_outputs, img, ycbcr=ycbcr)

    selected_model = df.iloc[0].index
    return selected_model

def closest_to_average(
    model_outputs: Dict[str, NumpyArray], model_metrics: Dict[str, float],
    ycbcr: bool = False
) -> str:
    
    avg_out = average_output(model_outputs, model_metrics)
    selected_model = closest_to(model_outputs, avg_out, ycbcr=ycbcr)
    return selected_model

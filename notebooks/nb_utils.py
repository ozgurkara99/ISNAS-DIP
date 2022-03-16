import os
from dataclasses import dataclass
from bisect import bisect_left
import itertools
from typing import Set, Tuple, List, Dict, Literal, Optional

from denoising import get_noisy_img
from utils.paths import root
from utils.keywords import * 
from utils.common_types import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModDataFrame(DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def union(*dfs: DataFrame, index: str = 'model name') -> DataFrame:
        
        if len(dfs) == 1:
            return dfs[0]
        
        df = dfs[0]
        for i in range(1, len(dfs)):
            df = df.merge(dfs[i], on=index)
            
        return ModDataFrame(df)
    
    def __add__(
        self, other: 'ModDataFrame', index: str = 'model name'
    ) -> 'ModDataFrame':
        
        return ModDataFrame.union(self, other, index=index)

@dataclass
class Data:
    name: str
    wrt: Optional[Literal['true', 'noisy']] = 'true'
    lr: Optional[Literal['high', 'low']] = 'low'
    process: Literal['denoising', 'inpainting'] = 'denoising'
    
    def __call__(self, img: Optional[str] = None) -> ModDataFrame:
        
        if self.name in ('psnr increase', 'psnr increase smooth'):
            df = root[BENCHMARK][self.process][img][SIMILARITY_METRICS_TRUE_IMG_CSV].load()
            df = df[self.name]
            df = ModDataFrame(df)
            return df
        
        if self.name in get_metrics('low'):
            if self.lr == 'low':
                df = root[BENCHMARK][LOWPASS_METRICS_CSV].load()
                df = df[self.name]
                df = ModDataFrame(df)
                return df

            if self.lr == 'high':
                df = root[BENCHMARK][self.process][img][LOWPASS_METRICS_HIGH_LR_CSV].load()
                df = df[self.name]
                df = ModDataFrame(df)
                return df
            
            assert False
        
        if self.name in get_metrics('sim'):
            if self.lr == 'low' and self.wrt == 'true':
                df = root[BENCHMARK][self.process][img][SIMILARITY_METRICS_TRUE_IMG_CSV].load()
                df = df[self.name]
                df = ModDataFrame(df)
                return df
            
            if self.lr == 'low' and self.wrt == 'noisy':
                df = root[BENCHMARK][self.process][img][SIMILARITY_METRICS_NOISY_IMG_CSV].load()
                df = df[self.name]
                df = ModDataFrame(df)
                return df

            if self.lr == 'high' and self.wrt == 'true':
                df = root[BENCHMARK][self.process][img][SIMILARITY_METRICS_HIGH_LR_TRUE_IMG_CSV].load()
                df = df[self.name]
                df = ModDataFrame(df)
                return df
            
            if self.lr == 'high' and self.wrt == 'noisy':
                df = root[BENCHMARK][self.process][img][SIMILARITY_METRICS_HIGH_LR_NOISY_IMG_CSV].load()
                df = df[self.name]
                df = ModDataFrame(df)
                return df
            
            assert False
            
        assert False

def get_data(
    data_objects: Union[List[Data], Data], img: Optional[str] = None
) -> ModDataFrame:
    
    if isinstance(data_objects, Data):
        data_objects = [data_objects]
    
    dfs = []
    for data in data_objects:
        df = data(img=img)
        dfs.append(df)
    
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df + dfs[i]
    
    return df

def get_all_data(
    img: Optional[str] = None, 
    process: Literal['denoising', 'inpainting'] = 'denoising',
    wrt: Literal['true', 'noisy'] = 'true', 
    lr: Literal['high', 'low'] = 'high'
) -> ModDataFrame:
    
    if img is None:
        df = root[BENCHMARK][LOWPASS_METRICS_CSV].load()
        df = ModDataFrame(df)
        return df
        
    if wrt == 'true' and lr == 'high':
        df1 = root[BENCHMARK][process][img][SIMILARITY_METRICS_HIGH_LR_TRUE_IMG_CSV]
        df1 = df1.load()
        df1 = ModDataFrame(df1)
        
        df2 = root[BENCHMARK][process][img][LOWPASS_METRICS_HIGH_LR_CSV]
        df2 = df2.load()
        df2 = ModDataFrame(df2)
        
        df = df1 + df2
        return df
    
    if wrt == 'noisy' and lr == 'high':
        df1 = root[BENCHMARK][process][img][SIMILARITY_METRICS_HIGH_LR_NOISY_IMG_CSV]
        df1 = df1.load()
        df1 = ModDataFrame(df1)
        
        df2 = root[BENCHMARK][process][img][LOWPASS_METRICS_HIGH_LR_CSV]
        df2 = df2.load()
        df2 = ModDataFrame(df2)
        
        df = df1 + df2
        return df

    if wrt == 'true' and lr == 'low':
        df1 = root[BENCHMARK][process][img][SIMILARITY_METRICS_TRUE_IMG_CSV]
        df1 = df1.load()
        df1 = ModDataFrame(df1)
        
        df2 = root[BENCHMARK][LOWPASS_METRICS_CSV]
        df2 = df2.load()
        df2 = ModDataFrame(df2)
        
        df = df1 + df2
        return df
    
    if wrt == 'noisy' and lr == 'low':
        df1 = root[BENCHMARK][process][img][SIMILARITY_METRICS_NOISY_IMG_CSV]
        df1 = df1.load()
        df1 = ModDataFrame(df1)
        
        df2 = root[BENCHMARK][LOWPASS_METRICS_CSV]
        df2 = df2.load()
        df2 = ModDataFrame(df2)
        
        df = df1 + df2
        return df

    assert False


def get_metrics(type: Literal['sim', 'low'] = 'sim') -> Set[str]:
    
    d1 = root[SIMILARITY_METRICS_YAML].load()
    d2 = root[LOWPASS_METRICS_YAML].load()
    metrics = set()
    
    if type == 'sim':
        metrics.update(d1['true image']['low lr'])
        metrics.update(d1['true image']['high lr'])
        metrics.update(d1['noisy image']['low lr'])
        metrics.update(d1['noisy image']['high lr'])
        return metrics
    
    if type == 'low':
        metrics.update(d2['low lr'])
        metrics.update(d2['high lr'])
        return metrics

def read_dataframes(
    process: Process,
    columns: List[str]
) -> Dict:
    
    img_stems, _, _, _ = read_images(DENOISING)
    
    Dict[str, Dict[str, Dict[str, DataFrame]]]
    
    dfs = {}
    for img_stem in img_stems:
        
        dfs[img_stem] = {}
        dfs[img_stem]['noisy'] = {}
        dfs[img_stem]['true'] = {}
        
        for wrt, lr in itertools.product(('noisy', 'true'), ('low', 'high')):
            
            df = get_all_data(img=img_stem, process=process, wrt=wrt, lr=lr)
            
            dfs[img_stem][wrt][lr] = df[columns]
        
    return dfs
    
    # df_lowpass = root[BENCHMARK][LOWPASS_METRICS_CSV].load()
    
    
    
    # dfs_true: Dict[str, DataFrame]  = {}
    # dfs_noisy: Dict[str, DataFrame]  = {}
    # dfs_high_true: Dict[str, DataFrame]  = {}
    # dfs_high_noisy: Dict[str, DataFrame]  = {}
    
    # dfs_high_lowpass: Dict[str, DataFrame] = {}

    # for img_stem in img_stems:
    #     tmp = root[BENCHMARK][process][img_stem]
    #     df_sim_true = tmp[SIMILARITY_METRICS_TRUE_IMG_CSV].load()
    #     df_sim_noisy = tmp[SIMILARITY_METRICS_NOISY_IMG_CSV].load()
    #     df_sim_high_true = tmp[SIMILARITY_METRICS_HIGH_LR_TRUE_IMG_CSV].load()
    #     df_sim_high_noisy = tmp[SIMILARITY_METRICS_HIGH_LR_NOISY_IMG_CSV].load()
        
    #     df_low_high = tmp[LOWPASS_METRICS_HIGH_LR_CSV].load()
        
    #     # df_sim_true = merge_dataframes(df_sim_true, df_lowpass)
    #     # df_sim_noisy = merge_dataframes(df_sim_noisy, df_lowpass)
    #     # df_sim_high_true = merge_dataframes(df_sim_high_true, df_lowpass)
    #     # df_sim_high_noisy = merge_dataframes(df_sim_high_noisy, df_lowpass)
        
    #     dfs_true[img_stem] = df_sim_true
    #     dfs_noisy[img_stem] = df_sim_noisy
    #     dfs_high_true[img_stem] = df_sim_high_true
    #     dfs_high_noisy[img_stem] = df_sim_high_noisy
    
        
    # return df_lowpass, dfs_true, dfs_noisy, dfs_high_true, dfs_high_noisy

def histogram(
    df: DataFrame, num_bins: int = 75, 
    keys: Tuple[PSNRKey, ...] = ('psnr increase', 'psnr increase smooth'), 
    model_names: Optional[Tuple[str, ...]] = None, 
    model_name_color: str = 'red', mode_color: str = 'green',
    suptitle: str = None
) -> None:
    
    if isinstance(keys, str):
        keys = (keys,)
    
    assert model_names is None or len(model_names) == len(keys)
    
    for i, key in enumerate(keys, start=1):
        values = list(df[key])
        max_val = max(values)
        min_val = min(values)
        bin_size = (max_val - min_val) / num_bins
        
        plt.subplot(1, len(keys), i)
        N, bins, patches = plt.hist(values, bins=num_bins)
        
        if len(keys) != 1:
            plt.title(key)
        
        mode_index = np.argmax(N)
        patches[mode_index].set_facecolor(mode_color)
        
        mode = bins[mode_index] + bin_size / 2
        x = mode - 0.3 * (max_val - min_val)
        plt.text(x, N[mode_index], f'{mode:.02f}dB')
        
        if model_names is None:
            continue
        
        model_name = model_names[i - 1]
        model_val = df.loc[model_name][key]
        model_index = bisect_left(bins, model_val)
        
        if model_index == len(patches):
            model_index -= 1
        
        patches[model_index].set_facecolor(model_name_color)
        
        x = model_val + 0.005*(max_val - min_val)
        plt.text(x, N[model_index], f'{model_val:.02f} dB')


            
    if suptitle is not None:
        plt.suptitle(suptitle)

def algorithm(
    metric: str, df: DataFrame, N: int, key: PSNRKey, ascending: bool = True
):
    tmp = df.sort_values(metric, ascending=ascending)
    scores = tmp.iloc[:N][key]
    
    model_name = scores.index[scores.argmax()]
    score = scores.max()
    
    return model_name, score
    
def create_algorithm_table(
    metrics: List[str], dfs: Dict[str, DataFrame], N: int, key: PSNRKey, 
    ascending: bool = True
) -> DataFrame:
    
    img_stems = list(dfs.keys())
    
    psnr_table = {'metric': metrics, **{img_stem: [] for img_stem in img_stems}}
    model_name_table = {'metric': metrics, **{img_stem: [] for img_stem in img_stems}}
    
    for img_stem in img_stems:
        for metric in metrics:
            
            model_name, score = algorithm(
                metric, dfs[img_stem], N=N, key=key, ascending=ascending
            )
            
            psnr_table[img_stem].append(score)
            model_name_table[img_stem].append(model_name)
    
    psnr_table = pd.DataFrame.from_dict(psnr_table)
    psnr_table = psnr_table.set_index('metric')
    
    model_name_table = pd.DataFrame.from_dict(model_name_table)
    model_name_table = model_name_table.set_index('metric')
    
    return model_name_table, psnr_table  
        
        
        
        
        
        
        
        
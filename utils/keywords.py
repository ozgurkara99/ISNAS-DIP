from typing import Final


BENCHMARK: Final = 'benchmark'
RANDOM_OUTPUTS: Final = 'random_outputs'
MODELS_CSV: Final = 'models.csv'
DENOISING: Final = 'denoising'
INPAINTING: Final = 'inpainting'
SR: Final = 'sr'
DATA: Final = 'data'
DIP: Final = 'dip'
NASDIP: Final = 'nasdip'

HTR_PKL = 'htr.pkl'
STOP_TXT = 'stop.txt'
BEST = 'best'
METRICS_CSV = 'metrics.csv'
GRID_PNG = 'grid.png'

IMAGES: Final = 'images'

HIGH_LR_OUTPUT_NPY: Final = 'high_lr_output_009.npy'

SIMILARITY_METRICS_TRUE_IMG_CSV: Final = 'similarity_metrics_true_img.csv'
SIMILARITY_METRICS_NOISY_IMG_CSV: Final = 'similarity_metrics_noisy_img.csv'
SIMILARITY_METRICS_HIGH_LR_TRUE_IMG_CSV: Final = 'similarity_metrics_high_lr_true_img.csv'
SIMILARITY_METRICS_HIGH_LR_NOISY_IMG_CSV: Final = 'similarity_metrics_high_lr_noisy_img.csv'

LOWPASS_METRICS_CSV: Final = 'lowpass_metrics.csv'
LOWPASS_METRICS_HIGH_LR_CSV: Final = 'lowpass_metrics_high_lr.csv'

SIMILARITY_METRICS_LST: Final = 'similarity_metrics.lst'
LOWPASS_METRICS_LST: Final = 'lowpass_metrics.lst'
RANDOM_METRICS_LST: Final = 'random_metrics.lst'

RESULTS: Final = 'results'

MODELS_GENERATED_LST: Final = 'models_generated.lst'

RANDOM_SEARCH: Final = 'random_search'

TRUE: Final = 'true'
NOISY: Final = 'noisy'
NOISE: Final = 'noise'

RANDOM_OUTPUT_NPY = 'random_output.npy'

SET12= [
    "cman_256", 
    "house_256", 
    "peppers_256", 
    "starfish_256", 
    "monar_256", 
    "f16_256",     
    "parrot_256", 
    "lena_512", 
    "barbara_512", 
    "boat_512", 
    "man_512", 
    "couple_512"
]

SET5 = [
    "woman_rgb", 
    "head_rgb", 
    "butterfly_rgb", 
    "bird_rgb", 
    "baby_rgb",
]

SET14 = [
    "set14-1_rgb", 
    "set14-2_rgb",
    "set14-3",
    "set14-4_rgb",
    "set14-5_rgb",
    "set14-6_rgb",
    "set14-7_rgb",
    "set14-8_rgb",
    "set14-9_rgb",
    "set14-10_rgb",
    "set14-11_rgb",
    "set14-12_rgb",
    "set14-13_rgb",
    "set14-14_rgb"
]

BM3D = [
    "cman_256", 
    "house_256", 
    "peppers_256", 
    "montage_256", 
    "lena_512",
    "barbara_512", 
    "boat_512", 
    "fprint_512", 
    "man_512", 
    "couple_512",
    "hill_512"
]

CBM3D = [
    "house_256_rgb", 
    "peppers_512_rgb", 
    "lena_512_rgb", 
    "baboon_512_rgb", 
    "f16_512_rgb", 
    "kodim01_rgb", 
    "kodim02_rgb", 
    "kodim03_rgb", 
    "kodim12_rgb"
]

CHEST = ['chest_512']
BLOOD = ['blood_512']

DATASETS = {
    'SET12': SET12,
    'SET5': SET5,
    'SET14': SET14,
    'BM3D': BM3D,
    'CBM3D': CBM3D,
    'CHEST': CHEST,
    'BLOOD': BLOOD
}

DENOISING_DATASETS = {
    'BM3D': BM3D,
    'SET12': SET12,
    'CBM3D': CBM3D,
    'CHEST': CHEST,
    'BLOOD': BLOOD
}

INPAINTING_DATASETS = {
    'SET12': SET12,
    'BM3D': BM3D,
}

SR_DATASETS = {
    'SET5': SET5,
    'SET14': SET14,
}



BENCHMARK_IMAGES = [
    '8_512',
    'lena_512',
    'barbara_512',
    'couple_512',
    'chest_512',
    'man_512',
    '4_512',
    'blood_512',
    'f16_512',
    'boat_512', 
]
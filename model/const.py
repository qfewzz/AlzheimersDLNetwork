import os

MAX_NUM_IMAGES = 6
CACHE_PATH = 'cache'
os.makedirs(CACHE_PATH, exist_ok=True)

CACHE_SINGLE_PATH = 'cache/single'
os.makedirs(CACHE_SINGLE_PATH, exist_ok=True)

TEMP_PATH = 'temp'
os.makedirs(TEMP_PATH, exist_ok=True)

STANDARD_DIM1 = int(200 * 1)
STANDARD_DIM2 = int(200 * 1)
STANDARD_DIM3 = int(200 * 1)
DIMESIONS = (STANDARD_DIM1, STANDARD_DIM2, STANDARD_DIM3)
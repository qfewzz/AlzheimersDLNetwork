from dataclasses import dataclass
import os


@dataclass
class Config:
    MAX_NUM_IMAGES: int = 6
    CACHE_PATH: str = 'cache'
    CACHE_SINGLE_PATH_READ: str = 'cache/single'
    CACHE_SINGLE_PATH_WRITE: str = 'cache/single'
    TEMP_PATH: str = 'temp'

    STANDARD_DIM1: int = int(200 * 1)
    STANDARD_DIM2: int = int(200 * 1)
    STANDARD_DIM3: int = int(200 * 1)
    DIMESIONS: tuple[int, int, int] = (STANDARD_DIM1, STANDARD_DIM2, STANDARD_DIM3)

    @classmethod
    def refresh(cls):
        os.makedirs(cls.CACHE_PATH, exist_ok=True)
        os.makedirs(cls.CACHE_SINGLE_PATH_WRITE, exist_ok=True)
        os.makedirs(cls.TEMP_PATH, exist_ok=True)


config_instance = Config()

import pickle
dumps = pickle.dumps(config_instance)
loads = pickle.loads(dumps)
print(loads)


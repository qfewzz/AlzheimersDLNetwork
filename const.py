from dataclasses import dataclass
import os
import random
from threading import Lock
from . import utils

# a = random.randint(1,10000)


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()  # this ensures thread-safe instantiation

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass
class Config(metaclass=SingletonMeta):
    SEED: int | None = 42
    CHECK_CACHE: bool = True
    VERBOSE: bool = True
    COMPRESSION: str = 'gzip'
    USE_VT: bool = False
    EPOCHES: int = 30
    LEARNING_RATE: float = 0.1
    BATCH_SIZE: int = 4
    MAX_NUM_IMAGES: int = 6
    CACHE_PATH: str = 'cache'
    CACHE_SINGLE_PATH_READ: str = 'cache/single'
    CACHE_SINGLE_PATH_WRITE: str = 'cache/single'
    TEMP_PATH: str = 'temp'

    STANDARD_DIM1: int = int(200 * 1)
    STANDARD_DIM2: int = int(200 * 1)
    STANDARD_DIM3: int = int(200 * 1)
    DIMESIONS: tuple[int, int, int] = (STANDARD_DIM1, STANDARD_DIM2, STANDARD_DIM3)

    def __post_init__(self):
        if self.SEED is None:
            self.SEED = random.Random().randint(1, 2**31 - 1)

    def refresh(self):
        self.__post_init__()
        os.makedirs(self.CACHE_PATH, exist_ok=True)
        os.makedirs(self.CACHE_SINGLE_PATH_WRITE, exist_ok=True)
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        utils.set_seed(self.SEED)


# import jsonpickle
# import dill
# Config.CACHE_PATH = 'aaaaaaaa'
# a = dill.dumps(Config)
# Config.CACHE_PATH = 'bbbbbbbb'
# b = dill.loads(a)

# print(b.CACHE_PATH)

import tensorflow as tf
import numpy as np
import os
import pickle as pkl
from collections import deque
import h5py as h5
import typing as tp
import enum as en

DEBUG_NONE = 0
DEBUG_BASIC = 1
DEBUG_DETAILED = 2

class Reward(en.Enum):
    NEUTRAL = 0
    WIN = 1
    LOSE = 2
    Intermediate = 3

class Model:
    def __init__(self, path_name, debug_level: int = 2) -> None:
        self.FILE_PATH = os.path.join(os.path.dirname(__file__), path_name)
        # if folder does not exist, create it
        if not os.path.exists(self.FILE_PATH):
            os.makedirs(self.FILE_PATH)
        self.debug_level = debug_level
        self.MEMORY_SIZE = 4
        self.state_memory = deque(maxlen=self.MEMORY_SIZE + 1)
        self.LEARNING_RATE = 0.001
        self.DISCOUNT_FACTOR = 0.99
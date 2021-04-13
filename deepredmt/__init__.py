import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# Nucleotide to integer
global _NT2ID
_NT2ID = {
    'N':0,
    'A':1,
    'C':2,
    'G':3,
    'T':4,
    'E':5,
    'e':6
}

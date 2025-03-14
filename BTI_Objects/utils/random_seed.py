import random
import numpy as np

"""
18apr2023, TT: handle setting random seeds
"""

def set_random_seed(seed):
    """ handle random seed """
    random.seed(seed)
    np.random.seed(seed)
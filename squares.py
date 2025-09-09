import numpy as np
import torch

def numpy_squares ( k ) :
    return np.arange(1, k+1) ** 2

def torch_squares ( k ) :
    return torch.arange(1, k+1) ** 2
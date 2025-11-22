import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.datasets import load_digits
from os import listdir
import re
from PIL import Image
from torchvision import transforms

''' Start SVM helpers '''
def poly_implementation(x, y, degree):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (1 + (x * y).sum()).pow(degree)

def poly(degree):
    return lambda x, y: poly_implementation(x, y, degree)

def rbf_implementation(x, y, sigma):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (-(x - y).norm().pow(2) / 2 / sigma / sigma).exp()

def rbf(sigma):
    return lambda x, y: rbf_implementation(x, y, sigma)

def xor_data():
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y

''' End SVM Helpers '''
import numpy as np
import torch
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



def gaussian_dataset(split, prefix="gaussian"):
    '''
    Arguments:
        split (str): "train" or "test"

    Returns:
        X (S x N LongTensor): features of each object, X[i][j] = 0/1
        y (S LongTensor): label of each object, y[i] = 0/1
    
    '''
    return torch.load(f"{prefix}_{split}.pth")

def gaussian_eval(prefix="gaussian"):
    import hw2_q2
    X, y = gaussian_dataset("train", prefix=prefix)
    mu, sigma2 = hw2_q2.gaussian_theta(X, y)
    p = hw2_q2.gaussian_p(y)
    Xtest, ytest = gaussian_dataset("test", prefix=prefix)
    ypred = hw2_q2.gaussian_classify(mu, sigma2, p, Xtest)
    return ypred, ytest

def setup_data():
    X = torch.tensor([[0.0, 3.0], [1.0, 3.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([[1], [1], [-1], [-1]])
    return X, y

def setup_more_data(w=np.array([-3., -2]), margin=1.5, size=100, bounds=[-5., 5.], trans=0.0):
    in_margin = lambda x: np.abs(w.reshape(-1).dot(x.reshape(-1))) / np.linalg.norm(w) < margin
    X = []
    y = []
    for i in range(size):
        x = np.random.uniform(bounds[0], bounds[1], 2) + trans
        while in_margin(x):
            x = np.random.uniform(bounds[0], bounds[1], 2) + trans
        if w.reshape(-1).dot(x.reshape(-1)) + trans > 0:
            y.append(np.array([1.]))
        else:
            y.append(np.array([-1.]))
        X.append(x)
    X = np.stack(X)
    y = np.stack(y).reshape(-1, 1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def get_line_width():
    return 2

def get_font_size():
    return 16







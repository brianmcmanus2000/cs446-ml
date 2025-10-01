import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from hw2_utils import gaussian_dataset


def gaussian_theta(X, y):
    '''
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''
    X = X.float()
    y = y.long()
    device = X.device

    mu_list = []
    var_list = []
    for cls in (0, 1):
        mask = (y == cls)
        Xc = X[mask]
        mu_c = Xc.mean(dim=0)
        var_c = Xc.var(dim=0, unbiased=False)
        mu_list.append(mu_c)
        var_list.append(var_c)

    mu = torch.stack(mu_list, dim=0).to(device)
    sigma2 = torch.stack(var_list, dim=0).to(device)
    return mu, sigma2

def gaussian_p(y):
    '''
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    y = y.view(-1).long()
    N = y.numel()
    p0 = (y == 0).sum().to(dtype=torch.float32) / float(N)
    return p0 

def gaussian_classify(mu,sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    X = X.float()
    mu = mu.float()
    sigma2 = sigma2.float()

    eps = 1e-9
    sigma2 = sigma2.clamp_min(eps)

    if not torch.is_tensor(p):
        p = torch.tensor(p, dtype=X.dtype, device=X.device)
    else:
        p = p.to(dtype=X.dtype, device=X.device)
    p = p.clamp(eps, 1 - eps)

    log_prior = torch.stack([torch.log(p), torch.log1p(-p)])
    Xe = X.unsqueeze(1)
    mu_e = mu.unsqueeze(0)
    var_e = sigma2.unsqueeze(0)

    loglik = -0.5 * (torch.log(2 * torch.pi * var_e) + (Xe - mu_e) ** 2 / var_e)
    scores = loglik.sum(dim=2) + log_prior

    yhat = scores.argmax(dim=1).long()
    return yhat
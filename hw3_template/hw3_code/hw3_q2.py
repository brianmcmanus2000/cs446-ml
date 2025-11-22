import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    # TODO
    x = x_train
    y = y_train.to(dtype=x.dtype)

    N = x.shape[0]
    K = torch.empty((N, N), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        for i in range(N):
            xi = x[i]
            for j in range(N):
                K[i, j] = kernel(xi, x[j])
    Q = K * torch.outer(y, y)
    alpha = torch.zeros(N, dtype=x.dtype, device=x.device, requires_grad=True)

    for _ in range(num_iters):
        Qa = Q @ alpha
        loss = 0.5 * (alpha @ Qa) - alpha.sum()

        loss.backward()

        with torch.no_grad():
            alpha -= lr * alpha.grad
            if c is None:
                alpha.clamp_(min=0)
            else:
                alpha.clamp_(min=0, max=c)
            alpha.grad.zero_()

    return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    # TODO
    device = x_train.device
    dtype = x_train.dtype
    alpha = alpha.to(device=device, dtype=dtype)
    y = y_train.to(device=device, dtype=dtype)
    X = x_train

    N = X.shape[0]
    M = x_test.shape[0]

    eps = 1e-12 
    sv_mask = alpha > eps
    if not torch.any(sv_mask):
        b = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        sv_idxs = torch.where(sv_mask)[0]
        min_idx_in_sv = torch.argmin(alpha[sv_idxs])
        i0 = sv_idxs[min_idx_in_sv].item()

        s = torch.tensor(0.0, device=device, dtype=dtype)
        xi = X[i0]
        for j in range(N):
            if alpha[j] > eps:
                s += alpha[j] * y[j] * kernel(X[j], xi)
        b = y[i0] - s 

    out = torch.empty(M, device=device, dtype=dtype)
    for m in range(M):
        acc = torch.tensor(0.0, device=device, dtype=dtype)
        xm = x_test[m]
        for j in range(N):
            if alpha[j] > eps:
                acc += alpha[j] * y[j] * kernel(X[j], xm)
        out[m] = acc + b

    return out
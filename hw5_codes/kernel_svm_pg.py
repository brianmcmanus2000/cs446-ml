# kernel_svm_pg.py
import torch
import numpy as np
import matplotlib.pyplot as plt

import hw5_utils as U  # uses svm_contour, poly, rbf, xor_data

# ---------- Core solver: dual PGD with equality and box projections ----------

def project_hyperplane(alpha, y):
    """
    Project alpha onto the hyperplane { a : y^T a = 0 } (Euclidean).
    """
    # If y is all +1/-1, ||y||^2 = N; general formula:
    denom = (y @ y).item()
    if denom == 0:
        return alpha
    return alpha - y * ((y @ alpha) / denom)

def project_box(alpha, C=None):
    """
    Box projection: [0, C] if C is given, else [0, +inf).
    """
    if C is None:
        return torch.clamp(alpha, min=0.0)
    return torch.clamp(alpha, min=0.0, max=C)

def build_gram(x, kernel):
    """
    Build Gram matrix K_ij = k(x_i, x_j).
    kernel: function taking two 1D tensors and returning scalar (torch.float).
    """
    n = x.shape[0]
    K = torch.zeros(n, n, dtype=torch.float)
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(x[i], x[j])
    return K

def solve_dual_pgd(x, y, kernel, steps=10_000, lr=0.1, C=None, verbose=False):
    """
    Minimize f(alpha) = 0.5*alpha^T Q alpha - 1^T alpha
    subject to y^T alpha = 0 and 0 <= alpha <= C (or C=None for hard margin).

    Returns alpha, support indices, b.
    """
    x = x.clone().float()
    y = y.clone().float()
    n = x.shape[0]

    # Build Gram and Q
    K = build_gram(x, kernel)             # n x n
    Y = y.view(-1, 1)
    Q = (Y @ Y.t()) * K                   # (y y^T) ∘ K

    # PGD
    alpha = torch.zeros(n, dtype=torch.float)
    one = torch.ones(n, dtype=torch.float)

    for t in range(steps):
        # gradient of f = Q alpha - 1
        g = Q @ alpha - one

        # gradient step
        alpha = alpha - lr * g

        # project onto equality constraint
        alpha = project_hyperplane(alpha, y)

        # project onto box
        alpha = project_box(alpha, C=C)

        if verbose and (t % 1000 == 0 or t == steps-1):
            fval = 0.5 * alpha @ (Q @ alpha) - one @ alpha
            print(f"iter {t:5d}  f={fval.item():.6f}  yTa={float(y@alpha):+.3e}")

    # Compute b using KKT:
    # Choose "true" support vectors: 0 < alpha_i < C (soft) or alpha_i > 1e-8 (hard)
    if C is None:
        sv_mask = alpha > 1e-6
    else:
        sv_mask = (alpha > 1e-6) & (alpha < C - 1e-6)

    if not torch.any(sv_mask):
        # fallback: use the largest alphas
        topk = min(5, n)
        sv_idx = torch.topk(alpha, k=topk).indices
    else:
        sv_idx = torch.where(sv_mask)[0]

    # b = y_i - sum_j alpha_j y_j k(x_j, x_i); average across chosen SVs
    bys = []
    for i in sv_idx:
        s = 0.0
        for j in range(n):
            s += alpha[j] * y[j] * K[j, i]
        bys.append(float(y[i] - s))
    b = float(np.mean(bys)) if len(bys) > 0 else 0.0

    return alpha, b, K

def make_predictor(x_train, y_train, alpha, b, kernel):
    """
    Returns a function pred(Xtest) that outputs the signed score f(x).
    Xtest: n_test x d torch tensor.
    """
    x_tr = x_train.clone().float()
    y_tr = y_train.clone().float()
    a = alpha.clone().float()
    def pred(Xtest):
        Xtest = Xtest.float()
        scores = torch.zeros(Xtest.shape[0], dtype=torch.float)
        with torch.no_grad():
            for i in range(Xtest.shape[0]):
                s = 0.0
                for j in range(x_tr.shape[0]):
                    s += a[j] * y_tr[j] * kernel(x_tr[j], Xtest[i])
                scores[i] = s + b
        return scores
    return pred

# ---------- Run the four experiments on XOR ----------

def run_and_plot(kernel, name, steps=10_000, lr=0.1, C=None):
    x, y = U.xor_data()  # 4 points, labels ±1
    alpha, b, K = solve_dual_pgd(x, y, kernel, steps=steps, lr=lr, C=C, verbose=False)
    pred = make_predictor(x, y, alpha, b, kernel)
    print(f"{name}:  #SV (alpha>1e-6) = {(alpha>1e-6).sum().item()},  b = {b:.4f}")
    U.svm_contour(pred, xmin=-8, xmax=8, ymin=-8, ymax=8, ngrid=101)

if __name__ == "__main__":
    # Polynomial kernel degree 3
    run_and_plot(U.poly(degree=3), "Poly deg=3", steps=10000, lr=0.02)

    # RBF kernels with sigma=1,2,5
    for sig in [1.0, 2.0, 5.0]:
        run_and_plot(U.rbf(sigma=sig), f"RBF sigma={sig}")

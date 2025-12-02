import random

import numpy as np
import torch

from hw6_q3 import hw6_q3_autograd as ad
from hw6_q3 import hw6_q3_nn as nn

torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_num_params(parameters):
    if isinstance(parameters, list) and isinstance(parameters[0], ad.Scalar):
        return len(parameters)
    return sum(param.numel() for param in parameters)


def compute_batch(mlp, X):
    # torch can automatically handle batched inputs
    # while we will have to do it manually
    return (
        [mlp(Xi) for Xi in X]  # our implementation
        if isinstance(mlp, nn.Module)
        else mlp(X).squeeze(1)  # torch's implementation
    )


def accuracy(mlp, X, y):
    logits = compute_batch(mlp, X)

    def is_correct(logit, label):
        return (logit.item() >= 0) == (label == 1)

    correct = sum(is_correct(logit, label) for logit, label in zip(logits, y))
    return correct / len(logits)


def train(
    mlp,
    loss_fn,
    optimizer,
    X,
    y,
    batch_size=8,
    num_steps=100,
):
    """Train the model

    Args:
        mlp: OurMLP or TorchMLP
        loss_fn: Our loss function or torch loss function
        optimizer (_type_): Our optimizer or torch optimizer
        X: Training data as a numpy array or torch tensor
        y (_type_): Training label as a numpy array or torch tensor
        batch_size (int, optional): Training batch size
        num_steps (int, optional): Number of training steps
        compute_acc (bool, optional): Whether to compute accuracy for the training step
    """
    set_seed()
    losses, accs = [], []
    for i in range(1, num_steps + 1):
        """
        Steps:
        - Sample a batch of X, y from ids using `random.sample`
        - Compute batch outputs
        - Compute batch loss
        - Zero gradient for all parameters
        - Backpropagate gradient
        - Perform one optimization steps
        """
        batch_ids = random.sample(range(len(X)), k=batch_size)
        batch_X = X[batch_ids]
        batch_y = y[batch_ids]

        ### YOUR IMPLEMENTATION START ###
        if isinstance(mlp, nn.Module):
            # ----- Our scalar-based autograd implementation -----
            # Forward: compute logits for each example in the batch
            logits = compute_batch(mlp, batch_X)  # list[ad.Scalar]

            # Make sure labels are 1-D and wrap them as Scalars
            batch_y_flat = np.asarray(batch_y).reshape(-1)
            labels = [ad.Scalar(float(lbl)) for lbl in batch_y_flat]

            # Compute scalar loss
            loss = loss_fn(logits, labels)

            # Zero gradients on all parameters
            mlp.zero_grad()

            # Backward: start from dL/dL = 1
            loss.grad = 1.0
            loss.backward()

            # Parameter update
            optimizer.step()
        else:
            # ----- PyTorch implementation -----
            logits = compute_batch(mlp, batch_X)  # tensor of shape [batch]
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ### YOUR IMPLEMENTATION END ###

        acc = accuracy(mlp, X, y)
        print(f"Step {i:#3d} \t Loss {loss.item():#.4f} \t Accuracy {acc}")

        losses.append(loss.item())
        accs.append(acc)

    return losses, accs

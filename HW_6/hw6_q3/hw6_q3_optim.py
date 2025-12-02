from numbers import Number

from .hw6_q3_autograd import Scalar


class SGD:
    def __init__(self, params: list[Scalar], lr: Number):
        self.params = params
        self.lr = lr

    def step(self):
        ### YOUR IMPLEMENTATION START ###
        # Iterate over all parameters and update using gradient
        for p in self.params:
            p.data -= self.lr * p.grad
        ### YOUR IMPLEMENTATION END ###
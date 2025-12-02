import textwrap
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain

from .hw6_q3_autograd import BCEWithLogitsLossFn, ReLUFn, Scalar, SigmoidFn


class Module(ABC):
    """Base class for all neural network modules."""

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def parameters(self) -> list[Scalar]:
        modules = [module for module in vars(self).values() if isinstance(module, Module)]
        return list(chain.from_iterable(module.parameters() for module in modules))

    def __repr__(self) -> str:
        indent = partial(textwrap.indent, prefix=" " * 4)

        def format(item):
            key, value = item
            repr_str = "[\n{}]".format(indent("".join(f"{i},\n" for i in value))) if isinstance(value, list) else value
            return f"{key}={repr_str},\n"

        modules = [module for module in vars(self).values() if isinstance(module, Module)]
        repr_strs = [format(module) for module in modules]
        return f"{self.__class__.__name__}(\n{indent(''.join(repr_strs))})"


class FunctionalModule(Module):
    def parameters(self) -> list[Scalar]:
        return []

    def __repr__(self):
        return self.__class__.__name__


class Neuron(Module):
    def __init__(self, dim: int):
        self.dim = dim
        self.weight = [Scalar(0.1) for _ in range(dim)]
        self.bias = Scalar(0.1)

    def __call__(self, args: list[Scalar]):
        if len(args) != self.dim:
            raise ValueError(f"Expect `arg` to have dimension {self.dim}")
        ### YOUR IMPLEMENTATION START ###
        # Compute the linear combination of inputs and weights, add bias
        out = self.bias
        for w, x in zip(self.weight, args):
            out = out + w * x
        return out
        ### YOUR IMPLEMENTATION END ###

    def parameters(self) -> list[Scalar]:
        return self.weight + [self.bias]


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def __call__(self, args: list[Scalar]):
        outputs = [neuron(args) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Scalar]:
        return list(chain.from_iterable(neuron.parameters() for neuron in self.neurons))


class ReLU(FunctionalModule):
    def __call__(self, args: list[Scalar]):
        return [ReLUFn().forward(arg) for arg in args]


class Sigmoid(FunctionalModule):
    def __call__(self, args: list[Scalar]):
        return [SigmoidFn().forward(arg) for arg in args]


class BCEWithLogitsLoss(FunctionalModule):
    def __call__(self, args: list[Scalar], labels: list[Scalar]):
        losses = [BCEWithLogitsLossFn().forward(arg, label) for arg, label in zip(args, labels)]
        return sum(losses, start=Scalar(0)) / len(losses)

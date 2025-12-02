import math
from abc import ABC, abstractmethod
from numbers import Number


def sigmoid(value: Number) -> Number:
    # This is the only place that use external library. You are not allowed to use any external library elsewhere in this file.
    from scipy.special import expit as _sigmoid

    return _sigmoid(value).item()


def log_sigmoid(value: Number) -> Number:
    # This is the only place that use external library. You are not allowed to use any external library elsewhere in this file.
    from scipy.special import log_expit as _log_sigmoid

    return _log_sigmoid(value).item()


class Function(ABC):
    """Base class for all autograd functions. You will need to implement the `.forward` and `.backward` methods."""

    def __init__(self):
        self.args: tuple["Scalar", ...] | None = None  # should be populated in forward
        self.out: "Scalar" | None = None  # should be populated in forward

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __repr__(self) -> str:
        class_name = self.__class__.__name__.removesuffix("Fn")
        return "".join(["_" + i.lower() if i.isupper() else i for i in class_name]).lstrip("_")

    def reset(self):
        self.args = None
        self.out = None


class IdentityFn(Function):
    def forward(self, arg: "Scalar"):
        return arg

    def backward(self):
        pass


class AdditionFn(Function):
    """Implement the `.forward` and `.backward` method for the addition (+) operation."""

    def forward(self, arg1: "ScalarLike", arg2: "ScalarLike"):
        arg1, arg2 = Scalar.ensure(arg1), Scalar.ensure(arg2)

        # Save the input arguments for use in backward pass
        self.args = (arg1, arg2)
        # Compute the output Scalar
        self.out = Scalar(arg1.data + arg2.data, self)
        return self.out

    def backward(self):
        assert self.args is not None and self.out is not None
        # Retrieve the saved input arguments
        arg1, arg2 = self.args
        # dL/da += dL/dz * dz/da = out.grad * 1
        arg1.grad += 1.0 * self.out.grad
        # dL/db += dL/dz * dz/db = out.grad * 1
        arg2.grad += 1.0 * self.out.grad
        self.reset()

class MultiplicationFn(Function):
    """Implement the `.forward` and `.backward` method for the multiplication (x) operation."""

    def forward(self, arg1: "ScalarLike", arg2: "ScalarLike"):
        arg1, arg2 = Scalar.ensure(arg1), Scalar.ensure(arg2)

        # Save the input arguments for use in backward pass
        self.args = (arg1, arg2)
        # Compute the output Scalar
        self.out = Scalar(arg1.data * arg2.data, self)
        return self.out

    def backward(self):
        assert self.args is not None and self.out is not None
        # Retrieve the saved input arguments
        arg1, arg2 = self.args
        # dL/da += dL/dz * dz/da = out.grad * b
        arg1.grad += arg2.data * self.out.grad
        # dL/db += dL/dz * dz/db = out.grad * a
        arg2.grad += arg1.data * self.out.grad
        self.reset()


class PowerFn(Function):
    def forward(self, arg1: "ScalarLike", arg2: "ScalarLike"):
        arg1, arg2 = Scalar.ensure(arg1), Scalar.ensure(arg2)

        self.args = (arg1, arg2)
        self.out = Scalar(arg1.data**arg2.data, self)
        return self.out

    def backward(self):
        assert self.args is not None and self.out is not None
        arg1, arg2 = self.args
        arg1.grad += (arg2.data * arg1.data ** (arg2.data - 1)) * self.out.grad

        if arg1.data > 0:
            arg2.grad += (arg1.data**arg2.data) * math.log(arg1.data) * self.out.grad
        elif arg1.data < 0:
            arg2.grad += float("nan")
        elif arg2.data > 0:
            arg2.grad += 0
        elif arg2.data < 0:
            arg2.grad += float("-inf")
        else:
            arg2.grad += float("nan")
        self.reset()


class SubtractionFn(AdditionFn):
    def forward(self, arg1: "ScalarLike", arg2: "ScalarLike"):
        return super().forward(arg1, MultiplicationFn().forward(arg2, -1.0))


class DivisionFn(MultiplicationFn):
    def forward(self, arg1: "ScalarLike", arg2: "ScalarLike"):
        return super().forward(arg1, PowerFn().forward(arg2, -1.0))


class ReLUFn(Function):
    def forward(self, arg: "ScalarLike"):
        arg = Scalar.ensure(arg)

        # Save the input arguments for use in backward pass
        self.args = (arg,)
        # Compute the output Scalar
        out_value = arg.data if arg.data > 0 else 0.0
        self.out = Scalar(out_value, self)
        return self.out

    def backward(self):
        assert self.args is not None and self.out is not None
        # Retrieve the saved input arguments
        (arg,) = self.args
        # Gradient of ReLU: 1 for x>0, 0 otherwise (we choose 0 at x=0)
        grad_input = 1.0 if arg.data > 0 else 0.0
        arg.grad += grad_input * self.out.grad
        self.reset()

class SigmoidFn(Function):
    def forward(self, arg: "ScalarLike"):
        arg = Scalar.ensure(arg)

        # Save the input arguments for use in backward pass
        self.args = (arg,)
        # Compute the output Scalar
        out_value = sigmoid(arg.data)
        self.out = Scalar(out_value, self)
        return self.out

    def backward(self):
        assert self.args is not None and self.out is not None
        # Retrieve the saved input arguments
        (arg,) = self.args
        # Derivative of sigmoid: s * (1 - s), where s is the sigmoid output
        s = self.out.data
        arg.grad += s * (1.0 - s) * self.out.grad
        self.reset()


class BCEWithLogitsLossFn(Function):
    def forward(self, logit: "ScalarLike", label: "ScalarLike"):
        logit, label = Scalar.ensure(logit), Scalar.ensure(label)

        if label.data not in [0, 1]:
            raise ValueError("Only accept binary label of 0, 1.")
        # Save the input arguments for use in backward pass
        self.args = (logit, label)

        y = float(label.data)
        x = float(logit.data)
        # Binary cross-entropy with logits, numerically stable
        loss_value = -(y * log_sigmoid(x) + (1.0 - y) * log_sigmoid(-x))
        self.out = Scalar(loss_value, self)
        return self.out

    def backward(self):
            assert self.args is not None and self.out is not None
            # Retrieve the saved input arguments
            logit, label = self.args
            y = float(label.data)
            x = float(logit.data)
            # Gradient w.r.t. logit
            logit.grad += (sigmoid(x) - y) * self.out.grad
            # Optional gradient w.r.t. label
            label.grad += (-log_sigmoid(x) + log_sigmoid(-x)) * self.out.grad
            self.reset()


class Scalar:
    def __init__(self, data: Number, out_fn: Function | None = None):
        self.data = data
        self.grad = 0.0
        self.out_fn = out_fn if out_fn else IdentityFn()

    @classmethod
    def ensure(cls, data: "ScalarLike") -> "Scalar":
        return data if isinstance(data, Scalar) else Scalar(data)

    def item(self) -> Number:
        return self.data

    def __repr__(self):
        return f"{self.__class__.__name__}[value={self.data}, grad={self.grad}, out_fn={self.out_fn}]"

    def backward(self):
        fns: list[Function] = []
        visited: list[Scalar] = []

        def dfs(node):
            if not isinstance(node, Scalar):
                return
            if node in visited:
                return
            if node.out_fn.args is None:
                return
            visited.append(node)
            for child in node.out_fn.args:
                dfs(child)

            fns.append(node.out_fn)

        dfs(self)
        self.grad = 1.0
        for fn in reversed(fns):
            fn.backward()

    def __add__(self, other: "ScalarLike"):
        return AdditionFn().forward(self, other)

    def __radd__(self, other: "ScalarLike"):
        return AdditionFn().forward(other, self)

    def __mul__(self, other: "ScalarLike"):
        return MultiplicationFn().forward(self, other)

    def __rmul__(self, other: "ScalarLike"):
        return MultiplicationFn().forward(other, self)

    def __pow__(self, other: "ScalarLike"):
        return PowerFn().forward(self, other)

    def __rpow__(self, other: "ScalarLike"):
        return PowerFn().forward(other, self)

    def __sub__(self, other: "ScalarLike"):
        return SubtractionFn().forward(self, other)

    def __rsub__(self, other: "ScalarLike"):
        return SubtractionFn().forward(other, self)

    def __truediv__(self, other: "ScalarLike"):
        return DivisionFn().forward(self, other)

    def __rtruediv__(self, other: "ScalarLike"):
        return DivisionFn().forward(other, self)


ScalarLike = Scalar | Number

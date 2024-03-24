"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.init_initializers.kaiming_uniform(in_features, out_features, dtype=dtype, device=device, requires_grad=True))
        if bias:
            self.bias = Parameter(init.init_initializers.kaiming_uniform(out_features, 1, dtype=dtype, device=device, requires_grad=True).reshape((1, out_features)))
        else: 
            self.bias = Parameter(init.init_basic.constant(1,out_features, c=0, device=device, dtype=dtype, requires_grad=False))
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        out_dim = 1 
        for dim in X.shape[1:]: 
            out_dim *= dim 
        return X.reshape((-1, out_dim))
        # raise NotImplementedError() 
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        output = x
        for module in self.modules: 
            output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError() 
        label_num = logits.shape[-1]
        y_int = Tensor(y, dtype=int, requires_grad=False)
        y_one_hot = init.one_hot(label_num, y_int) 
        Z_y = logits * y_one_hot
        Z_y = ops.summation(Z_y, axes=(1,)) # get a array (batch, 1), each element is the yth element of Z (the logits of y)
        losses = ops.ops_logarithmic.logsumexp(logits, axes=(1,)) - Z_y
        return ops.summation(losses) / losses.shape[0]

        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        batch_dims = x.shape[:-1]
        batch_size = 1
        for dim in batch_dims: 
            batch_size = batch_size * dim 
        batch_dims = tuple([i for i, _ in enumerate(batch_dims)])
        x_mean = ops.summation(x, (0,)) / batch_size
        # calculate var
        x_minus_mean = x - ops.broadcast_to(x_mean, x.shape)
        x_var = ops.summation(x_minus_mean ** 2, (0,)) / batch_size
        # 这里假设 batch dimensions 有很多维（从0到-1全都是batch dims）
        if self.training:
            # update running mean 
            # 注意，这里不需要存储计算图，因此要用.data，否则会爆内存
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.data
            denominator = ops.broadcast_to((x_var + self.eps) ** 0.5, x.shape)
        else: 
            x_minus_mean = x - ops.broadcast_to(self.running_mean, x.shape)
            x_var = self.running_var
            denominator = ops.broadcast_to((x_var + self.eps) ** 0.5, x.shape)
        w = ops.broadcast_to(self.weight, x.shape) 
        b = ops.broadcast_to(self.bias, x.shape)
        return w * (x_minus_mean / denominator) + b 
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # x mean should take part in the gradient calculation!
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        # calculate x mean
        x_sum = ops.summation(x, (1,)).reshape((batch_size, 1))
        x_mean = x_sum / feature_size
        x_mean = ops.broadcast_to(x_mean, x.shape)
        # calculate var
        x_minus_mean = x - x_mean
        x_var = ops.summation(x_minus_mean ** 2, (1,)) / feature_size
        denominator = x_var.reshape((batch_size, 1)) + self.eps 
        denominator = denominator ** 0.5
        denominator = ops.broadcast_to(denominator, x.shape)
        
        w = ops.broadcast_to(self.weight, x.shape) 
        b = ops.broadcast_to(self.bias, x.shape)
        return w * ((x - x_mean) / denominator) + b 
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        mask = init.randb(*(x.shape), p=self.p, device=x.device)
        mask = mask / (1 - self.p)
        return x * mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.fn(x) + x
        ### END YOUR SOLUTION

from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_sub_max = array_api.exp(Z - max_z)
        exp_sum = array_api.sum(exp_sub_max, axis=self.axes, keepdims=True) 
        output = array_api.log(exp_sum) + max_z
        return array_api.squeeze(output)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inputs = node.inputs[0]
        z = inputs.cached_data
        max_z = array_api.max(z, axis=self.axes, keepdims=True)
        z = z - max_z
        exp_z = array_api.exp(z)
        sum_z = array_api.sum(exp_z, axis=self.axes, keepdims=True) 
        grad = Tensor(exp_z / sum_z)
        
        # out = out_grad * grad 
        # print("out grad shape: ", out.shape)
        if self.axes: 
            keep_dim_shape = list(inputs.shape)
            for i, shape in enumerate(inputs.shape):
                if self.axes == None or i in self.axes:
                    keep_dim_shape[i] = 1  
            keep_dim_out_grad = reshape(out_grad, keep_dim_shape) 
        else: 
            keep_dim_out_grad = out_grad
        
        keep_dim_out_grad = broadcast_to(keep_dim_out_grad, inputs.shape)
        # 先把 outgrad 恢复成原来的形状（类似 summation 的 gradient），然后再乘上 logsumexp的 gradient（就是 softmax！）
        return keep_dim_out_grad * grad 
        raise NotImplementedError()
        
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


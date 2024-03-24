"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError() 
        for idx, p in enumerate(self.params): 
            if p.grad is None: 
                continue  
            # 注意数据类型转换
            grad_data = ndl.Tensor(p.grad.numpy(), dtype="float32", requires_grad=False) + self.weight_decay * p.data
            if idx in self.u:
                self.u[idx] = self.momentum * self.u[idx] + (1 - self.momentum) * grad_data
            else: 
                self.u[idx] = (1 - self.momentum) * grad_data
            p.data = p.data - self.lr * self.u[idx]
        
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.t = self.t + 1
        for idx, p in enumerate(self.params): 
      
            if p.grad is None: 
                continue  
            # 注意数据类型转换
            grad_data = ndl.Tensor(p.grad.numpy(), dtype="float32", requires_grad=False) + self.weight_decay * p.data
            print("grad data require grad: ", grad_data.requires_grad)
            if idx in self.m:
                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad_data
            else: 
                self.m[idx] = (1 - self.beta1) * grad_data
            
            if idx in self.v:
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad_data ** 2)
            else: 
                self.v[idx] = (1 - self.beta2) * (grad_data ** 2)
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
            
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION

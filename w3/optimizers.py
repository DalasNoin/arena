import torch as t
from typing import Iterable
import utils

class SGD:
    params: list

    def __init__(self, params: Iterable[t.nn.parameter.Parameter], lr: float, momentum: float, weight_decay: float=0):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        self.params = list(params)

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {weight_decay}")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.timestep = 0

        # same as self.gs on streamlit
        self.gradient_updates = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        """Set param grads to None
        """
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        with t.inference_mode():
            for i, (gradient_update, param) in enumerate(zip(self.gradient_updates, self.params)):
                grads = param.grad
                if self.weight_decay != 0:
                    grads += self.weight_decay*param
                if self.momentum != 0 and self.timestep > 0:
                    # I wonder if this is correct, i thought it should be (1-momentum)*grads
                    grads += self.momentum * gradient_update
                self.params[i] -= self.lr * grads
                self.gradient_updates[i]=grads
            self.timestep += 1
            
            


    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return f"SGD lr={self.lr} momentum={self.momentum} weight_decay={self.weight_decay}"

utils.test_sgd(SGD)
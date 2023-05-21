from typing import List

from common.np import np


class BaseOptimizer:
    def __init__(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SGD(BaseOptimizer):

    """
    copy and paste from [1]

    確率的勾配降下法（Stochastic Gradient Descent）

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/optimizer.py#L7
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam(BaseOptimizer):
    """
    copy and paste from [2]

    References
    ----------
    [1] (http://arxiv.org/abs/1412.6980v8)
    [2] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/optimizer.py#L101
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

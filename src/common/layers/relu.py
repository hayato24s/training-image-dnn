from common.np import np


class Relu:
    """
    Relu Layer

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch/blob/f549a1886b4c03a252ac7549311e29c1ed78a8a1/common/layers.py#L7
    """

    def __init__(self) -> None:
        self.params = []
        self.grads = []

        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = x < 0
        self.cache = [mask]

        out = x.copy()
        out[mask] = 0

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        (mask,) = self.cache

        dx = dout.copy()
        dx[mask] = 0

        return dx

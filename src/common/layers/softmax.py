from common.functions import softmax
from common.np import np


class Softmax:
    """
    Softmax Layer

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/layers.py#L50
    """

    def __init__(self) -> None:
        self.params = []
        self.grads = []

        self.cache = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, D)

        Returns
        -------
        out : np.ndarray
            the shape of array is (N, D)
        """

        out = softmax(x)

        self.cache = [out]

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, D)

        Returns
        -------
        dx : np.ndarray
           the shape of array is (N, D)
        """

        (out,) = self.cache

        # (N, D)
        prod = out * dout

        # (N, D)
        dx = prod - np.sum(prod, axis=1, keepdims=True) * out

        return dx

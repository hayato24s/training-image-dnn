from typing import List

from common.functions import cross_entropy_error, softmax
from common.np import np


class SoftmaxWithLoss:
    """
    SoftmaxWithLoss Layer

    Calculate loss using softmax and cross entropy error functions.

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/layers.py#L66
    """

    def __init__(self, ignore_indices: List[int] = []):
        """
        Parameters
        ----------
        ignore_indices : list of int, optional
            by default []
        """

        self.params = []
        self.grads = []

        self.ignore_indices = ignore_indices

        self.cache = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, D)
        t : np.ndarray
            the shape of array is (N, )

        Returns
        -------
        float
            loss
        """

        # (N, D)
        y = softmax(x)

        # (N, )
        mask = np.ones_like(t, dtype="bool")

        for i in self.ignore_indices:
            mask &= t != i

        masked_y = y[mask]
        masked_t = t[mask]

        loss = cross_entropy_error(masked_y, masked_t)

        self.cache = [y, t, mask]

        return loss

    def backward(self, dout: float = 1) -> np.ndarray:
        """
        Parameters
        ----------
        dout : float, optional
            dout, by default 1

        Returns
        -------
        np.ndarray
            the shape of array is (N, D)
        """

        y, t, mask = self.cache
        N, _ = y.shape

        # (N, D)
        dx = y.copy()
        dx[np.arange(N), t] -= 1
        dx *= dout
        dx /= N
        dx[~mask] = 0

        return dx

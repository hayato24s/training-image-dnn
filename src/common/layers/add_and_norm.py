from typing import Tuple

from common.layers.layer_normalization import LayerNormalization
from common.np import np


class AddAndNorm:
    """
    Add and Layer Normalization
    """

    def __init__(self, D: int):
        """
        Parameters
        ----------
        D : int
            the dimension of vector
        """

        norm_g = np.ones(D, dtype="f")
        norm_b = np.zeros(D, dtype="f")

        self.norm = LayerNormalization(norm_g, norm_b)

        self.params = self.norm.params
        self.grads = self.norm.grads

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x1 : np.ndarray
            the shape of array is (N, D)
        x2 : np.ndarray
            the shape of array is (N, D)

        Returns
        -------
        np.ndarray
            the shape of array is (N, D)
        """

        # Add
        # (N, D) = (N, D) + (N, D)
        x = x1 + x2

        # Layer Normalization
        # (N, D) -> (N, D)
        x = self.norm.forward(x)

        return x

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, D)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the shape of first array is (N, D)
            the shape of second array is (N, D)
        """

        # Layer Normalization
        # (N, D) -> (N, D)
        dout = self.norm.backward(dout)

        # Repeat
        dx1, dx2 = dout.copy(), dout.copy()

        return dx1, dx2

from typing import List

import common.config as config
from common.np import np
from common.utils.create_positional_encoding import create_positional_encoding


class Embedding:
    """
    Embedding Layer

    N means batch size.
    T means sequence size.
    V means vocabulary size.
    D means dimention of vector.

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/layers.py#L151
    """

    def __init__(self, W: np.ndarray, ignore_indices: List[int] = [], positional_encoding: bool = False) -> None:
        """
        Parameters
        ----------
        W : np.ndarray
            the shape of array is (V, D)
        ignore_indices : list of int, optional
            by default []
        positional_encoding : bool
            If you want to apply positional encoding, you should set True, by default False
        """

        self.params = [W]
        self.grads = [np.zeros_like(W)]

        self.ignore_indices = ignore_indices
        self.positional_encoding = positional_encoding

        self.cache = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        idx : np.ndarray
            the shape of array is (N, T)

        Returns
        -------
        out : np.ndarray
            the shape of array is (N, T, D)
        """

        (W,) = self.params

        # (N, T, D)
        out = W[idx]

        _, T, D = out.shape

        if self.positional_encoding:
            out += create_positional_encoding(T, D)

        # (N, T)
        mask = np.zeros_like(idx, dtype="bool")

        for i in self.ignore_indices:
            mask |= idx == i

        out[mask] = 0

        self.cache = [idx, mask]

        return out

    def backward(self, dout: np.ndarray) -> None:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, T, D)

        Returns
        -------
        None
        """

        (dW,) = self.grads
        (idx, mask) = self.cache

        dout = dout.copy()
        dout[mask] = 0

        dW[...] = 0

        if config.GPU:
            import cupyx

            cupyx.scatter_add(dW, idx, dout)
        else:
            np.add.at(dW, idx, dout)

        return None

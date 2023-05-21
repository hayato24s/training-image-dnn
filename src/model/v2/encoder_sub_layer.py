from typing import List

from common.layers import AveragePooling, Convolution
from common.np import np

from .residual import Residual


class EncoderSubLayer:
    def __init__(self, PC: int, C: int, H: int, W: int, block_num: int, residual_num: int) -> None:
        """
        Parameters
        ----------
        PC : int
            previous channel
        C : int
            channel
        H : int
            height
        W : int
            width
        block_num : int
            number of blocks
        residual_num : int
            number of residuals
        """

        conv_K = (np.random.randn(C, PC, 1, 1) / np.sqrt(1 * 1 * C / 2)).astype("f")
        conv_b = np.zeros(C, dtype="f")
        conv_stride = 1
        conv_pad = 0

        pool_h = 2
        pool_w = 2
        pool_stride = 2
        pool_pad = 0

        self.conv = Convolution(conv_K, conv_b, conv_stride, conv_pad)
        self.pool = AveragePooling(pool_h, pool_w, pool_stride, pool_pad)

        self.params, self.grads = [], []

        self.params += self.conv.params
        self.grads += self.conv.grads

        self.residuals: List[Residual] = []

        for _ in range(residual_num):
            residual = Residual(C, H, W, block_num)
            self.residuals.append(residual)
            self.params += residual.params
            self.grads += residual.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, PC, 2H, 2W)

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, H, W)
        """

        # Convolution
        # (N, PC, 2H, 2W) -> (N, C, 2H, 2W)
        x = self.conv.forward(x)

        # Pooling
        # (N, C, 2H, 2W) -> (N, C, H, W)
        x = self.pool.forward(x)

        # Residuals
        # (N, C, H, W) -> (N, C, H, W)
        for residual in self.residuals:
            x = residual.forward(x)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, C, H, W)

        Returns
        -------
        np.ndarray
            the shape of array is (N, PC, 2H, 2W)
        """

        # Residuals
        # (N, C, H, W) -> (N, C, H, W)
        for residual in reversed(self.residuals):
            dout = residual.backward(dout)

        # Pooling
        # (N, C, H, W) -> (N, C, 2H, 2W)
        dout = self.pool.backward(dout)

        # Convolution
        # (N, C, 2H, 2W)-> (N, PC, 2H, 2W)
        dout = self.conv.backward(dout)

        return dout

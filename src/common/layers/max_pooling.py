from common.np import np
from common.utils import col2im, im2col


class MaxPooling:
    """
    Max Pooling Layer

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch/blob/f549a1886b4c03a252ac7549311e29c1ed78a8a1/common/layers.py#L246
    """

    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, pad: int = 0) -> None:
        """
        Parameters
        ----------
        pool_h : int
        pool_w : int
        stride : int, optional
            by default 1
        pad : int, optional
            by default 0
        """

        self.params = []
        self.grads = []

        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, C, H, W)

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, OH, OW)
        """

        N, C, H, W = x.shape
        FH, FW = self.pool_h, self.pool_w
        OH = (H + 2 * self.pad - FH) // self.stride + 1
        OW = (W + 2 * self.pad - FW) // self.stride + 1

        # (N * OH * OW, C * FH * FW)
        col = im2col(x, FH, FW, self.stride, self.pad)

        # (N * OH * OW * C, FH * FW)
        col = col.reshape(-1, FH * FW)

        # (N * OH * OW * C, )
        arg_max = np.argmax(col, axis=1)

        # (N * OH * OW * C, )
        out: np.ndarray = np.max(col, axis=1)

        # (N, C, OH, OW)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        self.cache = [x.shape, arg_max]

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, C, OH, OW)

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, H, W)
        """

        x_shape, arg_max = self.cache

        N, C, OH, OW = dout.shape
        FH, FW = self.pool_h, self.pool_w

        # (N * OH * OW * C, FH * FW)
        dcol = np.zeros((N * OH * OW * C, FH * FW), dtype="f")

        # (N * OH * OW * C, )
        dout = dout.transpose(0, 2, 3, 1).flatten()

        dcol[np.arange(N * OH * OW * C), arg_max] = dout

        # (N * OH * OW, C * FH * FW)
        dcol = dcol.reshape(-1, C * FH * FW)

        # (N, C, H, W)
        dx = col2im(dcol, x_shape, FH, FW, self.stride, self.pad)

        return dx

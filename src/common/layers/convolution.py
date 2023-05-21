from common.np import np
from common.utils import col2im, im2col


class Convolution:
    """
    Convolution Layer

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch/blob/f549a1886b4c03a252ac7549311e29c1ed78a8a1/common/layers.py#L198
    """

    def __init__(self, K: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0) -> None:
        """
        Parameters
        ----------
        K : np.ndarray
            kernel, the shape of array is (FN, C, FH, FW)
        b : np.ndarray
            bias, the shape of array is (FN, )
        stride : int, optional
            by default 1
        pad : int, optional
            by default 0
        """

        self.params = [K, b]
        self.grads = [np.zeros_like(K), np.zeros_like(b)]

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
        out : np.ndarray
            the shape of array is (N, FN, OH, OW)
        """

        K, b = self.params
        FN, C, FH, FW = K.shape
        N, C, H, W = x.shape
        OH = (H + 2 * self.pad - FH) // self.stride + 1
        OW = (W + 2 * self.pad - FW) // self.stride + 1

        # (N * OH * OW, C * FH * FW)
        col = im2col(x, FH, FW, self.stride, self.pad)

        # (C * FH * FW, FN)
        col_K = K.reshape(FN, C * FH * FW).T

        # (N * OH * OW, FN)
        out = np.dot(col, col_K) + b

        # (N, FN, OH, OW)
        out = out.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)

        self.cache = [x.shape, col, col_K]

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, FN, OH, OW)

        Returns
        -------
        dx : np.ndarray
            the shape of array is (N, C, H, W)
        """

        K, _ = self.params
        x_shape, col, col_K = self.cache

        N, FN, OH, OW = dout.shape
        FN, C, FH, FW = K.shape

        # (N * OH * OW, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(N * OH * OW, FN)

        # (N * OH * OW, C * FH * FW)
        dcol = np.dot(dout, col_K.T)

        # (C * FH * FW, FN)
        dcol_K = np.dot(col.T, dout)

        # (FN, )
        db = np.sum(dout, axis=0)

        # (N, C, H, W)
        dx = col2im(dcol, x_shape, FH, FW, self.stride, self.pad)

        # (FN, C, FH, FW)
        dK = dcol_K.transpose(0, 1).reshape(FN, C, FH, FW)

        self.grads[0][...] = dK
        self.grads[1][...] = db

        return dx

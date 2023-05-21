from common.layers import Convolution, LayerNormalization, Relu
from common.np import np


class Block:
    """
    Architecture
    ------------
    - Instance Normalization
    - Relu
    - Convolution
    """

    def __init__(self, C: int, H: int, W: int) -> None:
        norm_g = np.ones(H * W, dtype="f")
        norm_b = np.zeros(H * W, dtype="f")

        conv_K = (np.random.randn(C, C, 3, 3) / np.sqrt(3 * 3 * C / 2)).astype("f")
        conv_b = np.zeros(C, dtype="f")
        conv_stride = 1
        conv_pad = 1

        self.norm = LayerNormalization(norm_g, norm_b)
        self.relu = Relu()
        self.conv = Convolution(conv_K, conv_b, conv_stride, conv_pad)

        self.params, self.grads = [], []
        for layer in [self.norm, self.relu, self.conv]:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, C, H, W)

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, H, W)
        """

        N, C, H, W = x.shape

        # Change shape of data
        # (N, C, H, W) -> (N * C, H * W)
        x = x.reshape(N * C, H * W)

        # Instance Normalization
        # (N * C, H * W) -> (N * C, H * W)
        x = self.norm.forward(x)

        # Change shape of data
        # (N * C, H * W) -> (N, C, H, W)
        x = x.reshape(N, C, H, W)

        # Relu
        # (N, C, H, W) -> (N, C, H, W)
        x = self.relu.forward(x)

        # Convolution
        # (N, C, H, W) -> (N, C, H, W)
        x = self.conv.forward(x)

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
            the shape of array is (N, C, H, W)
        """

        N, C, H, W = dout.shape

        # Convolution
        # (N, C, H, W) -> (N, C, H, W)
        dout = self.conv.backward(dout)

        # Relu
        # (N, C, H, W) -> (N, C, H, W)
        dout = self.relu.backward(dout)

        # Change shape of data
        # (N, C, H, W) -> (N * C, H * W)
        dout = dout.reshape(N * C, H * W)

        # Instance Normalization
        # (N * C, H * W) -> (N * C, H * W)
        dout = self.norm.backward(dout)

        # Change shape of data
        # (N * C, H * W) -> (N, C, H, W)
        dout = dout.reshape(N, C, H, W)

        return dout

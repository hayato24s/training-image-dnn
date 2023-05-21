from common.layers.convolution import Convolution
from common.layers.layer_normalization import LayerNormalization
from common.layers.linear import Linear
from common.layers.max_pooling import MaxPooling
from common.layers.relu import Relu
from common.np import np


class Encoder:
    """
    Extract feature maps from image data.

    N means batch size.

    Layer Architecture
    - Convolution
    - Instance Normalization
    - Relu
    - Max Pooling
    - Convolution
    - Instance Normalization
    - Relu
    - Linear
    - Layer Normalization
    - Relu
    """

    def __init__(self) -> None:
        randn = np.random.randn

        conv1_K = (randn(64, 3, 11, 11) / np.sqrt(256 * 256 / 2)).astype("f")
        conv1_b = np.zeros(64, dtype="f")
        conv1_stride = 5
        conv1_pad = 0

        norm1_g = np.ones(50 * 50, dtype="f")
        norm1_b = np.zeros(50 * 50, dtype="f")

        pool_h = 2
        pool_w = 2
        pool_stride = 2
        pool_pad = 0

        conv2_K = (randn(128, 64, 4, 4) / np.sqrt(4 * 4 / 2)).astype("f")
        conv2_b = np.zeros(128, dtype="f")
        conv2_stride = 3
        conv2_pad = 0

        norm2_g = np.ones(8 * 8, dtype="f")
        norm2_b = np.zeros(8 * 8, dtype="f")

        linear_W = (randn(64, 8) / np.sqrt(64 / 2)).astype("f")
        linear_b = np.zeros(8, dtype="f")

        norm3_g = np.ones(8, dtype="f")
        norm3_b = np.zeros(8, dtype="f")

        self.conv1 = Convolution(conv1_K, conv1_b, conv1_stride, conv1_pad)
        self.norm1 = LayerNormalization(norm1_g, norm1_b)
        self.relu1 = Relu()
        self.pool = MaxPooling(pool_h, pool_w, pool_stride, pool_pad)

        self.conv2 = Convolution(conv2_K, conv2_b, conv2_stride, conv2_pad)
        self.norm2 = LayerNormalization(norm2_g, norm2_b)
        self.relu2 = Relu()

        self.linear = Linear(linear_W, linear_b)
        self.norm3 = LayerNormalization(norm3_g, norm3_b)
        self.relu3 = Relu()

        layers = [
            self.conv1,
            self.norm1,
            self.relu1,
            self.pool,
            self.conv2,
            self.norm2,
            self.relu2,
            self.linear,
            self.norm3,
            self.relu3,
        ]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 3, 256, 256)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 128, 8)
        """

        N = x.shape[0]

        # (N, 3, 256, 256) -> (N, 64, 50, 50)
        x = self.conv1.forward(x)

        # (N, 64, 50, 50) -> (N * 64, 50 * 50)
        x = x.reshape(N * 64, -1)

        # Instance Normalization
        # (N * 64, 50 * 50) -> (N * 64, 50 * 50)
        x = self.norm1.forward(x)

        # (N * 64, 50 * 50) -> (N, 64, 50, 50)
        x = x.reshape(N, 64, 50, 50)

        # (N, 64, 50, 50) -> (N, 64, 50, 50)
        x = self.relu1.forward(x)

        # (N, 64, 50, 50) -> (N, 64, 25, 25)
        x = self.pool.forward(x)

        # (N, 64, 25, 25) -> (N, 128, 8, 8)
        x = self.conv2.forward(x)

        # (N, 128, 8, 8) -> (N * 128, 8 * 8)
        x = x.reshape(N * 128, -1)

        # Instance Normalization
        # (N * 128, 8 * 8) -> (N * 128, 8 * 8)
        x = self.norm2.forward(x)

        # (N * 128, 8 * 8) -> (N * 128, 8 * 8)
        x = self.relu2.forward(x)

        # (N * 128, 64) -> (N * 128, 8)
        x = self.linear.forward(x)

        # Layer Normalization
        # (N * 128, 8) -> (N * 128, 8)
        x = self.norm3.forward(x)

        # (N * 128, 8) -> (N, 128, 8)
        x = x.reshape(N, 128, 8)

        # (N, 128, 8) -> (N, 128, 8)
        x = self.relu3.forward(x)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, 128, 8)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 3, 256, 256)
        """

        N = dout.shape[0]

        # (N, 128, 8) -> (N, 128, 8)
        dx = self.relu3.backward(dout)

        # (N, 128, 8) -> (N * 128, 8)
        dx = dx.reshape(N * 128, 8)

        # (N * 128, 8) -> (N * 128, 8)
        dx = self.norm3.backward(dx)

        # (N * 128, 8) -> (N * 128, 64)
        dx = self.linear.backward(dx)

        # (N * 128, 8 * 8) -> (N * 128, 8 * 8)
        dx = self.relu2.backward(dx)

        # (N * 128, 8 * 8) -> (N * 128, 8 * 8)
        dx = self.norm2.backward(dx)

        # (N * 128, 8 * 8) -> (N, 128, 8, 8)
        dx = dx.reshape(N, 128, 8, 8)

        # (N, 128, 8, 8) -> (N, 64, 25, 25)
        dx = self.conv2.backward(dx)

        # (N, 64, 25, 25) -> (N, 64, 50, 50)
        dx = self.pool.backward(dx)

        # (N, 64, 50, 50) -> (N, 64, 50, 50)
        dx = self.relu1.backward(dx)

        # (N, 64, 50, 50) -> (N * 64, 50 * 50)
        dx = dx.reshape(N * 64, 50 * 50)

        # (N * 64, 50 * 50) -> (N * 64, 50 * 50)
        dx = self.norm1.backward(dx)

        # (N * 64, 50 * 50) -> (N, 64, 50, 50)
        dx = dx.reshape(N, 64, 50, 50)

        # (N, 64, 50, 50) -> (N, 3, 256, 256)
        dx = self.conv1.backward(dx)

        return dx

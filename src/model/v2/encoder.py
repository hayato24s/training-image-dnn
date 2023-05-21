from common.layers import Linear
from common.np import np
from model.v2.base_encoder import BaseEncoder


class Encoder:
    def __init__(self):

        linear_W = (np.random.randn(16 * 16, 8 * 8) / np.sqrt(16 * 16 / 2)).astype("f")
        linear_b = np.zeros(8 * 8, dtype="f")

        self.base_encoder = BaseEncoder()
        self.linear = Linear(linear_W, linear_b)

        self.params = self.base_encoder.params + self.linear.params
        self.grads = self.base_encoder.grads + self.linear.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 3, 256, 256)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 512, 8 * 8)
        """

        # (N, 3, 256, 256) -> (N, 512, 16, 16)
        x = self.base_encoder.forward(x)

        # Change shape
        # (N, 512, 16, 16) -> (N * 512, 16 * 16)
        x = x.reshape(-1, 16 * 16)

        # Fully Connected
        # (N * 512, 16 * 16) -> (N * 512, 8 * 8)
        x = self.linear.forward(x)

        # Change shape
        # (N * 512, 8 * 8) -> (N, 512, 8 * 8)
        x = x.reshape(-1, 512, 8 * 8)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, 512, 8 * 8)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 3, 256, 256)
        """

        # Change shape
        # (N, 512, 8 * 8) -> (N * 512, 8 * 8)
        dout = dout.reshape(-1, 8 * 8)

        # Fully Connected
        # (N * 512, 8 * 8) -> (N * 512, 16 * 16)
        dout = self.linear.backward(dout)

        # Change shape
        # (N * 512, 16 * 16) -> (N, 512, 16, 16)
        dout = dout.reshape(-1, 512, 16, 16)

        # (N, 512, 16, 16) -> (N, 3, 256, 256)
        dout = self.base_encoder.backward(dout)

        return dout

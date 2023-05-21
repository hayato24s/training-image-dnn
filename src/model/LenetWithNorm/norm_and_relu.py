from common.layers import LayerNormalization, Relu
from common.np import np


class NormAndRelu:
    def __init__(self, D: int) -> None:
        """
        Parameters
        ----------
        D : int
            the dimension of vector
        """

        g = np.ones(D)
        b = np.zeros(D)

        self.norm = LayerNormalization(g, b)
        self.relu = Relu()

        self.params = self.norm.params
        self.grads = self.norm.grads

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

        # (N, C, H, W) -> (N, C * H * W)
        x = x.reshape(N, -1)

        x = self.norm.forward(x)

        x = self.relu.forward(x)

        # (N, C * H * W) -> (N, C, H, W)
        x = x.reshape(N, C, H, W)

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

        # (N, C, H, W) -> (N, C * H * W)
        dout = dout.reshape(N, -1)

        dout = self.relu.backward(dout)

        dout = self.norm.backward(dout)

        # (N, C * H * W) -> (N, C, H, W)
        dout = dout.reshape(N, C, H, W)

        return dout

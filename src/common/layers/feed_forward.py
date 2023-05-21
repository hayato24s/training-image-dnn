from common.layers.linear import Linear
from common.layers.relu import Relu
from common.np import np


class FeedForward:
    """
    FeedForward

    Architecture
    ------------
    - Fully Connected
    - Relu
    - Fully Connected
    """

    def __init__(self, D_in: int, H: int, D_out: int) -> None:
        """
        Parameters
        ----------
        D_in : int
            dimension of input vector
        H : int
            dimension of hidden vector
        D_out : int
            dimension of output vector
        """

        linear1_W = (np.sqrt(2 / D_in) * np.random.randn(D_in, H)).astype("f")
        linear1_b = np.zeros(H, dtype="f")

        linear2_W = (np.sqrt(2 / H) * np.random.randn(H, D_out)).astype("f")
        linear2_b = np.zeros(D_out, dtype="f")

        self.linear1 = Linear(linear1_W, linear1_b)
        self.relu = Relu()
        self.linear2 = Linear(linear2_W, linear2_b)

        self.params = self.linear1.params + self.linear2.params
        self.grads = self.linear1.grads + self.linear2.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, D_in)

        Returns
        -------
        np.ndarray
            the shape of array is (N, D_out)
        """

        # Fully Connected 1
        # (N, D_in) -> (N, H)
        x = self.linear1.forward(x)

        # Relu
        x = self.relu.forward(x)

        # Fully Connected 2
        # (N, H) -> (N, D_out)
        x = self.linear2.forward(x)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, D_out)

        Returns
        -------
        np.ndarray
            the shape of array is (N, D_in)
        """

        # Fully Connected 2
        # (N, D_out) -> (N, H)
        dout = self.linear2.backward(dout)

        # Relu
        dout = self.relu.backward(dout)

        # Fully Connected 1
        # (N, H) -> (N, D_in)
        dout = self.linear1.backward(dout)

        return dout

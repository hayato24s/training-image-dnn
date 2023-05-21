from common.np import np


class Linear:
    """
    Fully Connected Layer

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/layers.py#L27
    """

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        """
        Parameters
        ----------
        W : np.ndarray
            the shape of array is (D_in, D_out)
        b : np.ndarray
            the shape of array is (D_out, )
        """

        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]

        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, D_in)

        Returns
        -------
        out : np.ndarray
            the shape of array is (N, D_out)
        """

        W, b = self.params

        # (N, D_out)
        out = np.dot(x, W) + b

        self.cache = [x]

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, D_out)

        Returns
        -------
        dx : np.ndarray
            the shape of array is (N, D_in)
        """

        W, _ = self.params
        (x,) = self.cache

        # (N, D_out)
        dx = np.dot(dout, W.T)

        # (D_in, D_out)
        dW = np.dot(x.T, dout)

        # (D_out, )
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

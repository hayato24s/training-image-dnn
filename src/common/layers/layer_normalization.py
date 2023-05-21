from common.np import np
from common.utils import bmm


class LayerNormalization:
    """
    Layer Normalization

    N means batch size.
    H means hidden size.

    References
    ----------
    [1] https://arxiv.org/abs/1607.06450
    [2] https://medium.com/@neuralthreads/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
    [3] https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """

    def __init__(self, g: np.ndarray, b: np.ndarray) -> None:
        """
        Parameters
        ----------
        g : np.ndarray
            gain, the shape of array is (H, )
        b : np.ndarray
            bias, the shape of array is (H, )
        """

        self.params = [g, b]
        self.grads = [np.zeros_like(g), np.zeros_like(b)]

        self.cache = [None]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, H)

        Returns
        -------
        np.ndarray
            the shape of array is (N, H)
        """

        g, b = self.params
        _, H = x.shape

        # (N, 1)
        mu = np.sum(x, axis=1, keepdims=True) / H

        # (N, H)
        dev = x - mu

        # (N, 1)
        sigma = np.sqrt(np.sum(dev**2, axis=1, keepdims=True) / H)

        # (N, H)
        y = dev / (sigma + 1e-7)

        # (N, H)
        out = g * y + b

        self.cache = [sigma, y]

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, H)

        Returns
        -------
        np.ndarray
            the shape of array is (N, H)
        """

        g, _ = self.params
        sigma, y = self.cache

        _, H = dout.shape

        # (H, )
        dg = np.sum(dout * y, axis=0)

        # (H, )
        db = np.sum(dout, axis=0)

        # (N, H)
        dy = dout * g

        # (N, H)
        dx = H * dy - np.sum(dy, axis=1, keepdims=True) - y * np.sum(y * dy, axis=1, keepdims=True)
        dx /= H * sigma + 1e-7

        self.grads[0][...] = dg
        self.grads[1][...] = db

        return dx


class LayerNormalizationWithJacobian(LayerNormalization):
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, H)

        Returns
        -------
        np.ndarray
            the shape of array is (N, H)
        """

        sigma, y = self.cache

        N, H = dout.shape

        # (H, )
        dg = np.sum(dout * y, axis=0)

        # (H, )
        db = np.sum(dout, axis=0)

        # (N, H, 1)
        y = y.reshape(N, H, 1)

        # (1, H, H)
        I = np.eye(H, dtype="f").reshape(1, H, H)

        # (N, H, H)
        J = (H * I - 1 - (y * y.transpose(0, 2, 1))) / (H * sigma.reshape(N, 1, 1))

        # (N, H)
        dx = bmm(J, dout.reshape(N, H, 1)).reshape(N, H)

        self.grads[0][...] = dg
        self.grads[1][...] = db

        return dx

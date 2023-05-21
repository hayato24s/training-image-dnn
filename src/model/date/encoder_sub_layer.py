from common.layers import AddAndNorm, FeedForward, ScaledDotProductAttention
from common.np import np


class EncoderSubLayer:
    """
    Encoder sub layer

    Architecture
    ------------
    - Skip Connection
        - Self Attention
        - Add and Layer Normalization
    - Skip Connection
        - Feed Forward
        - Add and Layer Normalization

    References
    ----------
    [1] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [2] [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
    """

    def __init__(self, D_enc: int, H: int) -> None:
        """
        Parameters
        ----------
        D_enc : int
            the dimension of encoder vector
        H : int
            the dimension of hidden vector in feed forward
        """

        randn = np.random.randn

        atten_Wq = (randn(D_enc, D_enc) / np.sqrt(D_enc / 2)).astype("f")
        atten_Wk = (randn(D_enc, D_enc) / np.sqrt(D_enc / 2)).astype("f")
        atten_Wv = (randn(D_enc, D_enc) / np.sqrt(D_enc / 2)).astype("f")

        self.atten = ScaledDotProductAttention(atten_Wq, atten_Wk, atten_Wv, masked=False)
        self.add_and_norm1 = AddAndNorm(D_enc)

        self.ff = FeedForward(D_enc, H, D_enc)
        self.add_and_norm2 = AddAndNorm(D_enc)

        self.params, self.grads = [], []
        for layer in [self.atten, self.add_and_norm1, self.ff, self.add_and_norm2]:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs_enc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        xs_enc : np.ndarray
            the shape of array is (N, T_enc, D_enc)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_enc, D_enc)
        """

        N, T_enc, D_enc = xs_enc.shape
        xs = xs_enc

        # For skip connection
        # (N, T_enc, D_enc)
        skip_xs = xs.copy()

        # Self Attention
        # (N, T_enc, D_enc) and (N, T_enc, D_enc) -> (N, T_enc, D_enc)
        xs = self.atten.forward(xs, xs)

        # Change shape
        # (N, T_enc, D_enc) -> (N * T_enc, D_enc)
        skip_xs = skip_xs.reshape(N * T_enc, D_enc)
        xs = xs.reshape(N * T_enc, D_enc)

        # Add And Layer Normalization
        # (N * T_enc, D_enc) -> (N * T_enc, D_enc)
        xs = self.add_and_norm1.forward(skip_xs, xs)

        # For skip connection
        # (N * T_enc, D_enc)
        skip_xs = xs.copy()

        # Feed Forward
        # (N * T_enc, D_enc) -> (N * T_enc, D_enc)
        xs = self.ff.forward(xs)

        # Add And Layer Normalization
        # (N * T_enc, D_enc) -> (N * T_enc, D_enc)
        xs = self.add_and_norm2.forward(skip_xs, xs)

        # Change shape
        # (N * T_enc, D_enc) -> (N, T_enc, D_enc)
        xs = xs.reshape(N, T_enc, D_enc)

        return xs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, T_enc, D_enc)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_enc, D_enc)
        """

        N, T_enc, D_enc = dout.shape

        # Change shape
        # (N, T_enc, D_enc) -> (N * T_enc, D_enc)
        dxs = dout.reshape(N * T_enc, D_enc)

        # Add And Layer Normalization
        # (N * T_enc, D_enc) -> (N * T_enc, D_enc), (N * T_enc, D_enc)
        (dskip_xs, dxs) = self.add_and_norm2.backward(dxs)

        # Feed Forward
        # (N * T_enc, D_enc) -> (N * T_enc, D_enc)
        dxs = self.ff.backward(dxs)

        # For skip connection
        # (N * T_enc, D_enc)
        dxs = dskip_xs + dxs

        # Add And Layer Normalization
        # (N * T_enc, D_enc) -> (N * T_enc, D_enc), (N * T_enc, D_enc)
        (dskip_xs, dxs) = self.add_and_norm1.backward(dxs)

        # Change shape
        # (N * T_enc, D_enc) -> (N, T_enc, D_enc)
        dskip_xs = dskip_xs.reshape(N, T_enc, D_enc)
        dxs = dxs.reshape(N, T_enc, D_enc)

        # Self Attention
        # (N, T_enc, D_enc) -> (N, T_enc, D_enc) and (N, T_enc, D_enc)
        (dxs_1, dxs_2) = self.atten.backward(dxs)
        dxs = dxs_1 + dxs_2

        # For skip connection
        # (N, T_enc, D_enc)
        dxs_enc = dskip_xs + dxs

        return dxs_enc

from typing import Tuple

from common.layers import AddAndNorm, FeedForward
from common.layers.attention import ScaledDotProductAttention
from common.np import np


class DecoderSubLayer:
    """
    Decoder sub layer

    Architecture
    ------------
    - Skip Connection
        - Self Attention
        - Add and Layer Normalization
    - Skip Connection
        - Encoder Decoder Attention
        - Add and Layer Normalization
    - Skip Connection
        - Feed Forward
        - Add and Layer Normalization

    References
    ----------
    [1] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    [2] [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
    """

    def __init__(self, D_dec: int, D_enc: int, H: int) -> None:
        """
        Parameters
        ----------
        D_dec : int
            the dimension of encoder vector
        D_enc : int
            the dimension of decoder vector
        H : int
            the dimension of hidden vector in feed forward
        """

        randn = np.random.randn

        atten1_Wq = (randn(D_dec, D_dec) / np.sqrt(D_dec / 2)).astype("f")
        atten1_Wk = (randn(D_dec, D_dec) / np.sqrt(D_dec / 2)).astype("f")
        atten1_Wv = (randn(D_dec, D_dec) / np.sqrt(D_dec / 2)).astype("f")

        atten2_Wq = (randn(D_dec, D_dec) / np.sqrt(D_dec / 2)).astype("f")
        atten2_Wk = (randn(D_enc, D_dec) / np.sqrt(D_enc / 2)).astype("f")
        atten2_Wv = (randn(D_enc, D_dec) / np.sqrt(D_enc / 2)).astype("f")

        self.atten1 = ScaledDotProductAttention(atten1_Wq, atten1_Wk, atten1_Wv, masked=True)
        self.add_and_norm1 = AddAndNorm(D_dec)

        self.atten2 = ScaledDotProductAttention(atten2_Wq, atten2_Wk, atten2_Wv, masked=False)
        self.add_and_norm2 = AddAndNorm(D_dec)

        self.ff = FeedForward(D_dec, H, D_dec)
        self.add_and_norm3 = AddAndNorm(D_dec)

        self.params, self.grads = [], []
        for layer in [self.atten1, self.add_and_norm1, self.atten2, self.add_and_norm2, self.ff, self.add_and_norm3]:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs_dec: np.ndarray, xs_enc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        xs_dec : np.ndarray
            the shape of array is (N, T_dec, D_dec)
        xs_enc : np.ndarray
            the shape of array is (N, T_enc, D_enc)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_dec, D_dec)
        """

        N, T_dec, D_dec = xs_dec.shape
        xs = xs_dec

        # For skip connection
        # (N, T_dec, D_dec)
        skip_xs = xs.copy()

        # Self Attention
        # (N, T_dec, D_dec) and (N, T_dec, D_dec) -> (N, T_dec, D_dec)
        xs = self.atten1.forward(xs, xs)

        # Change shape
        # (N, T_dec, D_dec) -> (N * T_dec, D_dec)
        skip_xs = skip_xs.reshape(N * T_dec, D_dec)
        xs = xs.reshape(N * T_dec, D_dec)

        # Add And Layer Normalization
        # (N * T_dec, D_dec) -> (N * T_dec, D_dec)
        xs = self.add_and_norm1.forward(skip_xs, xs)

        # Change shape
        # (N * T_dec, D_dec) -> (N, T_dec, D_dec)
        xs = xs.reshape(N, T_dec, D_dec)

        # For skip connection
        # (N, T_dec, D_dec)
        skip_xs = xs.copy()

        # (N, T_dec, D_dec) and (N, T_enc, T_enc) -> (N, T_dec, D_dec)
        xs = self.atten2.forward(xs, xs_enc)

        # Change shape
        # (N, T_dec, D_dec) -> (N * T_dec, D_dec)
        skip_xs = skip_xs.reshape(N * T_dec, D_dec)
        xs = xs.reshape(N * T_dec, D_dec)

        # Add And Layer Normalization
        # (N * T_dec, D_dec) -> (N * T_dec, D_dec)
        xs = self.add_and_norm2.forward(skip_xs, xs)

        # For skip connection
        # (N * T_dec, D_dec)
        skip_xs = xs.copy()

        # Feed Forward
        # (N * T_dec, D_dec)
        xs = self.ff.forward(xs)

        # Add And Layer Normalization
        # (N * T_dec, D_dec) -> (N * T_dec, D_dec)
        xs = self.add_and_norm3.forward(skip_xs, xs)

        # Change shape
        # (N * T_dec, D_dec) -> (N, T_dec, D_dec)
        xs = xs.reshape(N, T_dec, D_dec)

        return xs

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, T_dec, D_dec)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - first array is dxs_dec with shape (N, T_dec, D_dec)
            - second array is dxs_dec with shape (N, T_enc, D_enc)
        """

        N, T_dec, D_dec = dout.shape

        # Change shape
        # (N, T_dec, D_dec) -> (N * T_dec, D_dec)
        dxs = dout.reshape(N * T_dec, D_dec)

        # Add And Layer Normalization
        # (N * T_dec, D_dec) -> (N * T_dec, D_dec), (N * T_dec, D_dec)
        (dskip_xs, dxs) = self.add_and_norm3.backward(dxs)

        # Feed Forward
        # (N * T_dec, D_dec)
        dxs = self.ff.backward(dxs)

        # For skip connection
        # (N * T_dec, D_dec)
        dxs = dskip_xs + dxs

        # Add And Layer Normalization
        # (N * T_dec, D_dec) -> (N * T_dec, D_dec), (N * T_dec, D_dec)
        (dskip_xs, dxs) = self.add_and_norm2.backward(dxs)

        # Change shape
        # (N * T_dec, D_dec) -> (N, T_dec, D_dec)
        dskip_xs = dskip_xs.reshape(N, T_dec, D_dec)
        dxs = dxs.reshape(N, T_dec, D_dec)

        # (N, T_dec, D_dec) -> (N, T_dec, D_dec) and (N, T_enc, T_enc)
        (dxs, dxs_enc) = self.atten2.backward(dxs)

        # For skip connection
        # (N, T_dec, D_dec)
        dxs = dskip_xs + dxs

        # Change shape
        # (N, T_dec, D_dec) -> (N * T_dec, D_dec)
        dxs = dxs.reshape(N * T_dec, D_dec)

        # Add And Layer Normalization
        # (N * T_dec, D_dec) -> (N * T_dec, D_dec), (N * T_dec, D_dec)
        (dskip_xs, dxs) = self.add_and_norm1.backward(dxs)

        # Change shape
        # (N * T_dec, D_dec) -> (N, T_dec, D_dec)
        dskip_xs = dskip_xs.reshape(N, T_dec, D_dec)
        dxs = dxs.reshape(N, T_dec, D_dec)

        # Self Attention
        # (N, T_dec, D_dec) -> (N, T_dec, D_dec) and (N, T_dec, D_dec)
        (dxs_1, dxs_2) = self.atten1.backward(dxs)
        dxs = dxs_1 + dxs_2

        # For skip connection
        # (N, T_dec, D_dec)
        dxs_dec = dskip_xs + dxs

        return dxs_dec, dxs_enc

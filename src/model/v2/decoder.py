from typing import List

from common.layers import Embedding, Linear
from common.np import np

from .decoder_sub_layer import DecoderSubLayer


class Decoder:
    def __init__(self, V: int, D_dec: int, D_enc: int, H: int, sub_layer_num: int, embed_ignore_indices: List[int]):

        embed_W = (np.random.randn(V, D_dec) / 100).astype("f")

        linear_W = (np.random.randn(D_dec, V) / np.sqrt(D_dec / 2)).astype("f")
        linear_b = np.zeros(V, dtype="f")

        self.embed = Embedding(embed_W, embed_ignore_indices)

        self.sub_layers: List[DecoderSubLayer] = []
        for _ in range(sub_layer_num):
            sub_layer = DecoderSubLayer(D_dec, D_enc, H)
            self.sub_layers.append(sub_layer)

        self.linear = Linear(linear_W, linear_b)

        self.params, self.grads = [], []

        self.params += self.embed.params
        self.grads += self.embed.grads

        for sub_layer in self.sub_layers:
            self.params += sub_layer.params
            self.grads += sub_layer.grads

        self.params += self.linear.params
        self.grads += self.linear.grads

    def forward(self, xs_dec: np.ndarray, xs_enc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        xs_dec : np.ndarray
            the shape of array is (N, T_dec)
        xs_enc : np.ndarray
            the shape of array is (N, T_enc, D_enc)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_dec, V)
        """

        N, T_dec = xs_dec.shape

        # Embedding
        # (N, T_dec) -> (N, T_dec, D_dec)
        xs = self.embed.forward(xs_dec)

        # Decoder Sub Layer
        for sub_layer in self.sub_layers:
            # (N, T_dec, D_dec) and (N, T_enc, D_enc) -> (N, T_dec, D_dec)
            xs = sub_layer.forward(xs, xs_enc)

        # Change shape
        # (N, T_dec, D_dec) -> (N * T_dec, D_dec)
        xs = xs.reshape(N * T_dec, -1)

        # Fully Connected
        # (N * T_dec, D_dec) -> (N * T_dec, V)
        xs = self.linear.forward(xs)

        # Change shape
        # (N * T_dec, V) -> (N, T_dec, V)
        xs = xs.reshape(N, T_dec, -1)

        return xs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, T_dec, V)

        Returns
        -------
        np.ndarray
            dxs_enc with shape (N, T_enc, D_enc)
        """

        N, T_dec, _ = dout.shape

        # Change shape
        # (N, T_dec, V) -> (N * T_dec, V)
        dxs = dout.reshape(N * T_dec, -1)

        # Fully Connected
        # (N * T_dec, V) -> (N * T_dec, D_dec)
        dxs = self.linear.backward(dxs)

        # Change shape
        # (N * T_dec, D_dec) -> (N, T_dec, D_dec)
        dxs = dxs.reshape(N, T_dec, -1)

        dxs_enc = 0

        # Decoder Sub Layer
        for sub_layer in reversed(self.sub_layers):
            # (N, T_dec, D_dec) -> (N, T_dec, D_dec) and (N, T_enc, D_enc)
            dxs, dxs_enc_tmp = sub_layer.backward(dxs)
            dxs_enc += dxs_enc_tmp

        # Embedding
        # (N, T_dec, D_dec) -> None
        self.embed.backward(dxs)

        return dxs_enc

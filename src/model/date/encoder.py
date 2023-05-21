from typing import List

from common.layers import Embedding
from common.np import np

from .encoder_sub_layer import EncoderSubLayer


class Encoder:
    """
    - Embedding
    - Sub Layer
        - Skip Connection
            - Self Attention
            - Add and Layer Normalization
        - Skip Connection
            - Feed Forward
            - Add and Layer Normalization
    """

    def __init__(self, V: int, D_enc: int, H: int, embed_ignore_indices: List[int]):

        embed_W = (np.random.randn(V, D_enc) / 100).astype("f")

        self.embed = Embedding(embed_W, embed_ignore_indices, positional_encoding=True)
        self.sub_layer = EncoderSubLayer(D_enc, H)

        self.params, self.grads = [], []
        for layer in [self.embed, self.sub_layer]:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs_enc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        xs_enc : np.ndarray
            the shape of array is (N, T_enc)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_enc, D_enc)
        """

        # Embedding
        # (N, T_enc) -> (N, T_enc, D_enc)
        xs_enc = self.embed.forward(xs_enc)

        # Encoder Sub Layer
        # (N, T_enc, D_enc) -> (N, T_enc, D_enc)
        xs_enc = self.sub_layer.forward(xs_enc)

        return xs_enc

    def backward(self, dxs_enc: np.ndarray) -> None:
        """
        Parameters
        ----------
        dxs_enc : np.ndarray
            the shape of array is (N, T_enc, D_enc)
        """

        # Encoder Sub Layer
        # (N, T_enc, D_enc) -> (N, T_enc, D_enc)
        dxs_enc = self.sub_layer.backward(dxs_enc)

        # Embedding
        # (N, T_enc, D_enc) -> None
        self.embed.backward(dxs_enc)

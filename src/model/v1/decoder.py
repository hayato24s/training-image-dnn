from typing import List, Tuple

from common.layers.attention import ScaledDotProductAttention
from common.layers.embedding import Embedding
from common.layers.layer_normalization import LayerNormalization
from common.layers.linear import Linear
from common.layers.relu import Relu
from common.np import np


class Decoder:
    """
    Creat caption from image features.

    N means batch size.
    T means sequential size of x.
    T_enc means sequential size of x_enc.
    D_enc means the dimension of x_enc's last axis.
    V means vocabulary size.

    Layer Architecture
    - Embedding
    - Self Attention
    - Layer Normalization
    - Encoder Decoder Attention
    - Layer Normalization
    - Linear
    - Relu
    - Linear
    - Layer Normalization
    - Linear
    """

    def __init__(self, vocab_size: int, ignore_indices: List[int] = []) -> None:
        randn = np.random.randn
        V = vocab_size

        embed_W = (randn(V, 64) / 100).astype("f")

        atten1_Wq = (randn(64, 64) / np.sqrt(64 / 2)).astype("f")
        atten1_Wk = (randn(64, 64) / np.sqrt(64 / 2)).astype("f")
        atten1_Wv = (randn(64, 64) / np.sqrt(64 / 2)).astype("f")

        norm1_g = np.ones(64, dtype="f")
        norm1_b = np.zeros(64, dtype="f")

        atten2_Wq = (randn(64, 64) / np.sqrt(64 / 2)).astype("f")
        atten2_Wk = (randn(8, 64) / np.sqrt(8 / 2)).astype("f")
        atten2_Wv = (randn(8, 64) / np.sqrt(8 / 2)).astype("f")

        norm2_g = np.ones(64, dtype="f")
        norm2_b = np.zeros(64, dtype="f")

        linear1_W = (randn(64, 256) / np.sqrt(64 / 2)).astype("f")
        linear1_b = np.zeros(256, dtype="f")

        linear2_W = (randn(256, 64) / np.sqrt(256 / 2)).astype("f")
        linear2_b = np.zeros(64, dtype="f")

        norm3_g = np.ones(64, dtype="f")
        norm3_b = np.zeros(64, dtype="f")

        linear3_W = embed_W.T  # sharing weight
        linear3_b = np.zeros(V, dtype="f")

        self.embed = Embedding(embed_W, ignore_indices)
        self.atten1 = ScaledDotProductAttention(atten1_Wq, atten1_Wk, atten1_Wv, masked=True)
        self.norm1 = LayerNormalization(norm1_g, norm1_b)
        self.atten2 = ScaledDotProductAttention(atten2_Wq, atten2_Wk, atten2_Wv, masked=False)
        self.norm2 = LayerNormalization(norm2_g, norm2_b)
        self.linear1 = Linear(linear1_W, linear1_b)
        self.relu = Relu()
        self.linear2 = Linear(linear2_W, linear2_b)
        self.norm3 = LayerNormalization(norm3_g, norm3_b)
        self.linear3 = Linear(linear3_W, linear3_b)

        layers = [
            self.embed,
            self.atten1,
            self.norm1,
            self.atten2,
            self.norm2,
            self.linear1,
            self.relu,
            self.linear2,
            self.norm3,
            self.linear3,
        ]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x: np.ndarray, x_enc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, T)
        x_enc : np.ndarray
            the shape of array is (N, T_enc, 8)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T, V)
        """

        N, T = x.shape

        # (N, T) -> (N, T, 64)
        x = self.embed.forward(x)

        # For residual connection
        # (N, T, 64)
        skip_x = x

        # (N, T, 64) and (N, T, 64) -> (N, T, 64)
        x = self.atten1.forward(x, x)

        # Residual connection
        # (N, T, 64) = (N, T, 64) + (N, T, 64)
        x = x + skip_x

        # (N, T, 64) -> (N * T, 64)
        x = x.reshape(N * T, 64)

        # (N * T, 64) -> (N * T, 64)
        x = self.norm1.forward(x)

        # (N * T, 64) -> (N, T, 64)
        x = x.reshape(N, T, 64)

        # For residual connection
        # (N, T, 64)
        skip_x = x

        # (N, T, 64) and (N, T_enc, 8) -> (N, T, 64)
        x = self.atten2.forward(x, x_enc)

        # Residual connection
        # (N, T, 64) = (N, T, 64) + (N, T, 64)
        x = x + skip_x

        # (N, T, 64) -> (N * T, 64)
        x = x.reshape(N * T, 64)

        # (N * T, 64) -> (N * T, 64)
        x = self.norm2.forward(x)

        # For residual connection
        # (N * T, 64)
        skip_x = x

        # (N * T, 64) -> (N * T, 256)
        x = self.linear1.forward(x)

        # (N * T, 256) -> (N * T, 256)
        x = self.relu.forward(x)

        # (N * T, 256) -> (N * T, 64)
        x = self.linear2.forward(x)

        # Residual connection
        # (N * T, 64) = (N * T, 64) + (N * T, 64)
        x = x + skip_x

        # (N * T, 64) -> (N * T, 64)
        x = self.norm3.forward(x)

        # (N * T, 64) -> (N * T, V)
        x = self.linear3.forward(x)

        # (N * T, V) -> (N, T, V)
        x = x.reshape(N, T, -1)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, T, V)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_enc, 8)
        """

        N, T, V = dout.shape

        # (N, T, V) -> (N * T, V)
        dx = dout.reshape(N * T, V)

        # (N * T, V) -> (N * T, 64)
        dx = self.linear3.backward(dx)

        # (N * T, 64) -> (N * T, 64)
        dx = self.norm3.backward(dx)

        # (N * T, 64)
        dskip_x = dx

        # (N * T, 64) -> (N * T, 256)
        dx = self.linear2.backward(dx)

        # (N * T, 256) -> (N * T, 256)
        dx = self.relu.backward(dx)

        # (N * T, 256) -> (N * T, 64)
        dx = self.linear1.backward(dx)

        # (N * T, 64) = (N * T, 64) + (N * T, 64)
        dx = dskip_x + dx

        # (N * T, 64) -> (N * T, 64)
        dx = self.norm2.backward(dx)

        # (N * T, 64) -> (N, T, 64)
        dx = dx.reshape(N, T, 64)

        # (N, T, 64)
        dskip_x = dx

        # (N, T, 64) -> (N, T, 64) and (N, T_enc, 8)
        dx, dx_enc = self.atten2.backward(dx)

        # (N, T, 64) = (N, T, 64) + (N, T, 64)
        dx = dx + dskip_x

        # (N, T, 64) -> (N * T, 64)
        dx = dx.reshape(N * T, 64)

        # (N * T, 64) -> (N * T, 64)
        dx = self.norm1.backward(dx)

        # (N * T, 64) -> (N, T, 64)
        dx = dx.reshape(N, T, 64)

        # (N, T, 64)
        dskip_x = dx

        # (N, T, 64) -> (N, T, 64) and (N, T, 64)
        dx1, dx2 = self.atten1.backward(dx)

        # (N, T, 64) = (N, T, 64) + (N, T, 64) + (N, T, 64)
        dx = dx1 + dx2 + dskip_x

        # (N, T, 64) -> None
        self.embed.backward(dx)

        return dx_enc

    def generage(
        self, feature_map: np.ndarray, start_id: int, end_id: int, max_length: int
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Parameters
        ----------
        feature_map : np.ndarray
            the shape of array is (1, 128, 8)
        start_id : int
        end_id : int
        max_length : int

        Returns
        -------
        Tuple[List[int], List[List[int]]]
            samples and scores
        """

        sample_id = start_id
        samples = [start_id]
        scores = []

        while len(samples) < max_length:
            # (1, 1)
            x = np.array([sample_id]).reshape(1, -1)

            # (1, 1) and (1, 128, 8) -> (1, 1, V)
            score = self.forward(x, feature_map)

            sample_id = int(np.argmax(score))
            samples.append(sample_id)
            scores.append(score.flatten().tolist())

            if sample_id == end_id:
                break

        return samples, scores

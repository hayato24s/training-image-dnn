from typing import List, Tuple

from common.base_model import BaseModel
from common.functions import softmax
from common.layers import SoftmaxWithLoss
from common.np import np

from .decoder import Decoder
from .encoder import Encoder


class Model(BaseModel):
    """
    Architecture
    ------------
    - Encoder
        - Embedding
        - Sub Layer
            - Skip Connection
                - Self Attention
                - Add and Layer Normalization
            - Skip Connection
                - Feed Forward
                - Add and Layer Normalization
    - Decoder
        - Embedding
        - Sub Layer
            - Skip Connection
                - Self Attention
                - Add and Layer Normalization
            - Skip Connection
                - Encoder Decoder Attention
                - Add and Layer Normalization
            - Skip Connection
                - Feed Forward
                - Add and Layer Normalization
        - Linear
    - Softmax With Loss
    """

    def __init__(self, vocab_size: int, ignore_indices: List[int] = []):
        V, D_dec, D_enc, H = vocab_size, 64, 64, 512

        self.encoder = Encoder(V, D_enc, H, ignore_indices)
        self.decoder = Decoder(V, D_dec, D_enc, H, ignore_indices)
        self.loss = SoftmaxWithLoss(ignore_indices)

        self.params = self.encoder.params + self.decoder.params + self.loss.params
        self.grads = self.encoder.grads + self.decoder.grads + self.loss.grads

    def predict(self, xs: np.ndarray, ts_input: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        xs : np.ndarray
            the shape of array is (N, T_enc)
        ts : np.ndarray
            caption with shape (N, T_dec).

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_dec, V)
        """

        # (N, T_enc) -> (N, T_enc, D_enc)
        feature_maps = self.encoder.forward(xs)

        # (N, T_dec) and (N, T_enc, D_enc) -> (N, T_dec, V)
        score = self.decoder.forward(ts_input, feature_maps)

        return score

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Parameters
        ----------
        xs : np.ndarray
            the shape of array is (N, T_enc)
        ts : np.ndarray
            caption with shape (N, T_dec + 1).

        Returns
        -------
        float
            Loss
        """

        # (N, T_dec)
        ts_input = ts[:, :-1]

        # (N, T_dec)
        ts_ans = ts[:, 1:]

        # (N, T_enc) and (N, T_dec) -> (N, T_dec, V)
        score = self.predict(xs, ts_input)

        N, T_dec, V = score.shape

        # (N, T_dec, V) -> (N * T_dec, V)
        score = score.reshape(-1, V)

        # (N, T_dec) -> (N * T_dec, )
        ts_ans = ts_ans.flatten()

        # (N * T_dec, V) and (N * T_dec, ) -> float
        loss = self.loss.forward(score, ts_ans)

        self.cache = [N, T_dec, V]

        return loss

    def backward(self, dout: float = 1) -> None:
        """
        Parameters
        ----------
        dout : float, optional
            by default 1
        """

        N, T_dec, V = self.cache

        # float -> (N * T_dec, V)
        dscore = self.loss.backward(dout)

        # (N * T_dec, V) -> (N, T_dec, V)
        dscore = dscore.reshape(N, T_dec, V)

        # (N, T_dec, V) -> (N, T_enc, D_enc)
        dfeature_maps = self.decoder.backward(dscore)

        # (N, T_enc, D_enc) -> None
        self.encoder.backward(dfeature_maps)

    def generate(self, xs: np.ndarray, start_id: int, max_length: int) -> Tuple[List[int], List[List[int]]]:
        """
        Parameters
        ----------
        xs : np.ndarray
            the shape of array is (1, T_enc)
        start_id : int
        max_length : int

        Returns
        -------
        Tuple[List[int], List[List[float]]]
            samples and scores
        """

        # (1, T_enc) -> (1, T_enc, D_enc)
        feature_map = self.encoder.forward(xs)

        sample_id = start_id
        samples = [start_id]
        scores = []

        while len(samples) < max_length:
            # (1, len(samples))
            xs = np.array(samples).reshape(1, -1)

            # (1, len(samples)) and (1, T_enc, D_enc) -> (1, len(samples), V)
            score = self.decoder.forward(xs, feature_map)

            # (1, len(samples), V) -> (V, )
            score = softmax(score[:, -1, :]).flatten()

            sample_id = int(np.argmax(score))
            samples.append(sample_id)
            scores.append(score.tolist())

        return samples, scores

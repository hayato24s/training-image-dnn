from typing import List, Tuple

from common.base_model import BaseModel
from common.functions import softmax
from common.layers import SoftmaxWithLoss
from common.np import np

from .decoder import Decoder
from .encoder import Encoder


class Model(BaseModel):
    def __init__(self, vocab_size: int, embed_ignore_indices: List[int] = [], loss_ignore_indices: List[int] = []):
        V, D_dec, D_enc, H = vocab_size, 64, 64, 512
        decoder_sub_layer_num = 3

        self.encoder = Encoder()
        self.decoder = Decoder(V, D_dec, D_enc, H, decoder_sub_layer_num, embed_ignore_indices)
        self.loss = SoftmaxWithLoss(loss_ignore_indices)

        self.params = self.encoder.params + self.decoder.params + self.loss.params
        self.grads = self.encoder.grads + self.decoder.grads + self.loss.grads

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Parameters
        ----------
        x : np.ndarray
            image data with shape (N, C, H, W).
            Now, the shape (N, C, H, W) must be (N, 3, 256, 256).
        t : np.ndarray
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

        # (N, C, H, W) -> (N, T_enc, D_enc)
        feature_maps = self.encoder.forward(xs)

        # (N, T_dec) and (N, T_enc, D_enc) -> (N, T_dec, V)
        score = self.decoder.forward(ts_input, feature_maps)

        N, T_dec, V = score.shape

        # (N, T_dec, V) -> (N * T_dec, V)
        score = score.reshape(-1, V)

        # (N, T_dec) -> (N * T_dec, )
        ts_ans = ts_ans.flatten()

        # (N * T_dec, V) and (N * T_dec, ) -> float
        loss = self.loss.forward(score, ts_ans)

        self.cache = [N, T_dec, V]

        return loss

    def backward(self, dout: float = 1) -> np.ndarray:
        """
        Parameters
        ----------
        dout : float, optional
            by default 1

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, H, W)
            Now, the shape (N, C, H, W) will be (N, 3, 256, 256).
        """

        N, T_dec, V = self.cache

        # float -> (N * T_dec, V)
        dscore = self.loss.backward(dout)

        # (N * T_dec, V) -> (N, T_dec, V)
        dscore = dscore.reshape(N, T_dec, V)

        # (N, T_dec, V) -> (N, T_enc, D_enc)
        dfeature_maps = self.decoder.backward(dscore)

        # (N, T_enc, D_enc) -> (N, C, H, W)
        dxs = self.encoder.backward(dfeature_maps)

        return dxs

    def generate(
        self, image_array: np.ndarray, start_id: int, end_id: int, max_length: int
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Generate captions from image array.

        Parameters
        ----------
        image_array : np.ndarray
            image array with shape (C, H, W).
            Now, the shape (C, H, W) must be (3, 256, 256).
        start_id : int
        end_id : int
        max_length : int

        Returns
        -------
        Tuple[List[int], List[List[float]]]
            samples and scores
        """

        # (C, H, W) -> (1, C, H, W)
        x = np.array(image_array).reshape(1, *image_array.shape)

        # (1, C, H, W) -> (1, T_enc, D_enc)
        feature_map = self.encoder.forward(x)

        sample_id = start_id
        samples = [start_id]
        scores = []

        while len(samples) < max_length:
            # (1, len(samples))
            x = np.array(samples).reshape(1, -1)

            # (1, len(samples)) and (1, T_enc, D_enc) -> (1, len(samples), V)
            score = self.decoder.forward(x, feature_map)

            # (1, len(samples), V) -> (V, )
            score = softmax(score[:, -1, :]).flatten()

            sample_id = int(np.argmax(score))
            samples.append(sample_id)
            scores.append(score.tolist())

            if sample_id == end_id:
                break

        return samples, scores

from typing import List, Tuple

from common.base_model import BaseModel
from common.layers.softmax_with_loss import SoftmaxWithLoss
from common.np import np

from .decoder import Decoder
from .encoder import Encoder


class Model(BaseModel):
    """
    Neural network for image captioning

    N means batch size.
    C means channel size of image.
    H means height of image.
    W means width of image.
    T means max length of caption - 1.
    """

    def __init__(self, vocab_size, embed_ignore_indices: List[int] = [], loss_ignore_indices: List[int] = []):
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size, embed_ignore_indices)
        self.loss = SoftmaxWithLoss(loss_ignore_indices)

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

        self.cache = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Parameters
        ----------
        x : np.ndarray
            image data with shape (N, C, H, W).
            Now, the shape (N, C, H, W) must be (N, 3, 256, 256).
        t : np.ndarray
            caption with shape (N, T+1).

        Returns
        -------
        float
            Loss
        """

        # (N, T)
        t_input = t[:, :-1]

        # (N, T)
        t_ans = t[:, 1:]

        # (N, 3, 256, 256) -> (N, 128, 8)
        feature_maps = self.encoder.forward(x)

        # (N, T) and (N, 128, 8) -> (N, T, V)
        score = self.decoder.forward(t_input, feature_maps)

        N, _, V = score.shape

        # (N, T, V) -> (N * T, V)
        score = score.reshape(-1, V)

        # (N, T) -> (N * T, )
        t_ans = t_ans.flatten()

        # (N * T, V) and (N * T, ) -> float
        loss = self.loss.forward(score, t_ans)

        self.cache = [N, V]

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

        N, V = self.cache

        # float -> (N * T, V)
        dscore = self.loss.backward(dout)

        # (N * T, V) -> (N, T, V)
        dscore = dscore.reshape(N, -1, V)

        # (N, T, V) -> (N, 128, 8)
        dfeature_maps = self.decoder.backward(dscore)

        # (N, 128, 8) -> (N, #, 256, 256)
        dx = self.encoder.backward(dfeature_maps)

        return dx

    def generate(
        self, x: np.ndarray, start_id: int, end_id: int, max_length: int
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Generate captions from image data.

        Parameters
        ----------
        x : np.ndarray
            image data with shape (N, C, H, W).
            Now, the shape (N, C, H, W) must be (N, 3, 256, 256).
        start_id : int
        end_id : int
        max_length : int

        Returns
        -------
        Tuple[List[int], List[List[int]]]
            samples and scores
        """

        # (1, 3, 256, 256) -> (1, 128, 8)
        feature_map = self.encoder.forward(x)

        samples, scores = self.decoder.generage(feature_map, start_id, end_id, max_length)

        return samples, scores

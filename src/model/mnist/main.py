from typing import Tuple

from common.base_model import BaseModel
from common.layers import FeedForward
from common.layers.softmax_with_loss import SoftmaxWithLoss
from common.np import np

from .sub_layer import SubLayer


class Model(BaseModel):
    def __init__(self):

        self.sub_layerA = SubLayer(1, 32, 14, 14, 2, 1)
        self.sub_layerB = SubLayer(32, 64, 7, 7, 2, 1)

        self.feed_forward = FeedForward(64 * 7 * 7, 16 * 7 * 7, 10)
        self.loss = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in [self.sub_layerA, self.sub_layerB, self.feed_forward]:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 1, 28, 28)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 10)
        """

        N = x.shape[0]

        # (N, 1, 28, 28) -> (N, 32, 14, 14)
        x = self.sub_layerA.forward(x)

        # (N, 32, 14, 14) -> (N, 64, 7, 7)
        x = self.sub_layerB.forward(x)

        # (N, 64, 7, 7) -> (N, 64 * 7 * 7)
        x = x.reshape(N, -1)

        # (N, 64 * 7 * 7) -> (N, 16 * 7 * 7) ->  (N, 10)
        x = self.feed_forward.forward(x)

        return x

    def forward(self, x: np.ndarray, t: np.ndarray) -> Tuple[float, int]:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 1, 28, 28)
        t : np.ndarray
            the shape of array is (N, )


        Returns
        -------
        Tuple[float, int]
            loss and correct_num
        """

        # (N, 1, 28, 28) -> (N, 10)
        score = self.predict(x)

        correct_num = np.sum(np.argmax(score, axis=1) == t)

        loss = self.loss.forward(score, t)

        return loss, correct_num

    def backward(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            the shape of array is (N, 1, 28, 28)
        """

        # float -> (N, 10)
        dout = self.loss.backward()

        N = dout.shape[0]

        # (N, 10) -> (N, 16 * 7 * 7) ->  (N, 64 * 7 * 7)
        dout = self.feed_forward.backward(dout)

        # (N, 64 * 7 * 7) -> (N, 64, 7, 7)
        dout = dout.reshape(N, 64, 7, 7)

        # (N, 64, 7, 7) -> (N, 32, 14, 14)
        dout = self.sub_layerB.backward(dout)

        # (N, 32, 14, 14) -> (N, 1, 28, 28)
        dout = self.sub_layerA.backward(dout)

        return dout

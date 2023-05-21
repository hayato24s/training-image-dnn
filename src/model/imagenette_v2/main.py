from typing import Tuple

from common.base_model import BaseModel
from common.layers import FeedForward
from common.layers.softmax_with_loss import SoftmaxWithLoss
from common.np import np

from .sub_layer import SubLayer


class Model(BaseModel):
    def __init__(self):

        self.sub_layerA = SubLayer(3, 8, 128, 128, 2, 1)
        self.sub_layerB = SubLayer(8, 16, 64, 64, 2, 1)

        self.feed_forward = FeedForward(16 * 64 * 64, 32 * 32, 10)
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
            the shape of array is (N, 3, 256, 256)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 10)
        """

        N = x.shape[0]

        # (N, 3, 256, 256) -> (N, 8, 128, 128)
        x = self.sub_layerA.forward(x)

        # (N, 8, 128, 128) -> (N, 16, 64, 64)
        x = self.sub_layerB.forward(x)

        # (N, 16, 64, 64) -> (N, 16 * 64 * 64)
        x = x.reshape(N, -1)

        # (N, 16 * 64 * 64) -> (N, 32 * 32) ->  (N, 10)
        x = self.feed_forward.forward(x)

        return x

    def forward(self, x: np.ndarray, t: np.ndarray) -> Tuple[float, int]:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 3, 256, 256)
        t : np.ndarray
            the shape of array is (N, )


        Returns
        -------
        Tuple[float, int]
            loss and correct_num
        """

        # (N, 3, 256, 256) -> (N, 10)
        score = self.predict(x)

        correct_num = np.sum(np.argmax(score, axis=1) == t)

        loss = self.loss.forward(score, t)

        return loss, correct_num

    def backward(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            the shape of array is (N, 3, 256, 256)
        """

        # float -> (N, 10)
        dout = self.loss.backward()

        N = dout.shape[0]

        # (N, 10) -> (N, 32 * 32) ->  (N, 16 * 64 * 64)
        dout = self.feed_forward.backward(dout)

        # (N, 16 * 64 * 64) -> (N, 16, 64, 64)
        dout = dout.reshape(N, 16, 64, 64)

        # (N, 16, 64, 64) -> (N, 8, 128, 128)
        dout = self.sub_layerB.backward(dout)

        # (N, 8, 128, 128) -> (N, 3, 256, 256)
        dout = self.sub_layerA.backward(dout)

        return dout

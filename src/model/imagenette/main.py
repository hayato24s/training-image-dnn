from typing import Tuple

from common.base_model import BaseModel
from common.layers import Linear, SoftmaxWithLoss
from common.layers.relu import Relu
from common.np import np
from model.v2.base_encoder import BaseEncoder


class Model(BaseModel):
    def __init__(self):

        linear1_W = (np.random.randn(512 * 16 * 16, 16 * 16) / np.sqrt(512 * 16 * 16 / 2)).astype("f")
        linear1_b = np.zeros(16 * 16, dtype="f")

        linear2_W = (np.random.randn(16 * 16, 10) / np.sqrt(16 * 16 / 2)).astype("f")
        linear2_b = np.zeros(10, dtype="f")

        self.base_encoder = BaseEncoder()
        self.linear1 = Linear(linear1_W, linear1_b)
        self.relu = Relu()
        self.linear2 = Linear(linear2_W, linear2_b)
        self.loss = SoftmaxWithLoss()

        self.params = self.base_encoder.params + self.linear1.params + self.linear2.params
        self.grads = self.base_encoder.grads + self.linear1.grads + self.linear2.grads

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

        # (N, 3, 256, 256) -> (N, 512, 16, 16)
        x = self.base_encoder.forward(x)

        # Change shape
        # (N, 512, 16, 16) -> (N, 512 * 16 * 16)
        x = x.reshape(-1, 512 * 16 * 16)

        # Fully Connected 1
        # (N, 512 * 16 * 16) -> (N, 16 * 16)
        x = self.linear1.forward(x)

        # (N, 16 * 16) -> (N, 16 * 16)
        x = self.relu.forward(x)

        # Fully Connected 2
        # (N, 16 * 16) -> (N, 10)
        x = self.linear2.forward(x)

        return x

    def forward(self, x: np.ndarray, t: np.ndarray) -> Tuple[float, int]:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 3, 256, 256)
        t : np.ndarray
            the shape of array is (N, )
            each element of array is int from 0 to 9

        Returns
        -------
        Tuple[float, int]
            first element is loss.
            second element is correct_num
        """

        # (N, 3, 256, 256) -> (N, 10)
        score = self.predict(x)

        correct_num = np.sum(np.argmax(score, axis=1) == t)

        # (N, 10) and (N, ) -> float
        loss = self.loss.forward(score, t)

        return loss, correct_num

    def backward(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            the shape of array is (N, 3, 256, 256)
        """

        # (N, 10)
        dout = self.loss.backward()

        # Fully Connected 2
        # (N, 10) -> (N, 16 * 16)
        dout = self.linear2.backward(dout)

        # (N, 16 * 16) -> (N, 16 * 16)
        dout = self.relu.backward(dout)

        # Fully Connected 1
        # (N, 16 * 16) -> (N, 512 * 16 * 16)
        dout = self.linear1.backward(dout)

        # Change shape
        # (N, 512 * 16 * 16) -> (N, 512, 16, 16)
        dout = dout.reshape(-1, 512, 16, 16)

        # (N, 512, 16, 16) -> (N, 3, 256, 256)
        dout = self.base_encoder.backward(dout)

        return dout

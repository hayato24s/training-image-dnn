from typing import Tuple

from common.base_model import BaseModel
from common.layers import AveragePooling, Convolution, Linear
from common.layers.relu import Relu
from common.layers.softmax_with_loss import SoftmaxWithLoss
from common.np import np


class Lenet(BaseModel):
    """
    Architecture
    ------------
    - Convolution
    - Relu
    - Average Pooling
    - Convolution
    - Relu
    - Average Pooling
    - Convolution
    - Relu
    - Linear
    - Relu
    - Linear

    References
    ----------
    [1] https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
    """

    def __init__(self):
        randn = np.random.randn

        conv1_K = (randn(6, 1, 5, 5) / np.sqrt(1 * 5 * 5 / 2)).astype("f")
        conv1_b = np.zeros(6, dtype="f")

        conv2_K = (randn(16, 6, 5, 5) / np.sqrt(6 * 5 * 5 / 2)).astype("f")
        conv2_b = np.zeros(16, dtype="f")

        conv3_K = (randn(120, 16, 5, 5) / np.sqrt(16 * 5 * 5 / 2)).astype("f")
        conv3_b = np.zeros(120, dtype="f")

        linear1_W = (randn(120, 84) / np.sqrt(120 / 2)).astype("f")
        linear1_b = np.zeros(84, dtype="f")

        linear2_W = (randn(84, 10) / np.sqrt(84 / 2)).astype("f")
        linear2_b = np.zeros(10, dtype="f")

        self.conv1 = Convolution(conv1_K, conv1_b, stride=1, pad=2)
        self.relu1 = Relu()
        self.pool1 = AveragePooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.conv2 = Convolution(conv2_K, conv2_b, stride=1, pad=0)
        self.relu2 = Relu()
        self.pool2 = AveragePooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.conv3 = Convolution(conv3_K, conv3_b, stride=1, pad=0)
        self.relu3 = Relu()
        self.linear1 = Linear(linear1_W, linear1_b)
        self.relu4 = Relu()
        self.linear2 = Linear(linear2_W, linear2_b)
        self.loss = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in [self.conv1, self.conv2, self.conv3, self.linear1, self.linear2]:
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

        # (N, 1, 28, 28) -> (N, 6, 28, 28)
        x = self.conv1.forward(x)

        x = self.relu1.forward(x)

        # (N, 6, 28, 28) -> (N, 6, 14, 14)
        x = self.pool1.forward(x)

        # (N, 6, 14, 14) -> (N, 16, 10, 10)
        x = self.conv2.forward(x)

        x = self.relu2.forward(x)

        # (N, 16, 10, 10) -> (N, 16, 5, 5)
        x = self.pool2.forward(x)

        # (N, 16, 5, 5) -> (N, 120, 1, 1)
        x = self.conv3.forward(x)

        x = self.relu3.forward(x)

        # (N, 120, 1, 1) -> (N, 120)
        x = x.reshape(-1, 120)

        # (N, 120) -> (N, 84)
        x = self.linear1.forward(x)

        x = self.relu4.forward(x)

        # (N, 84) -> (N, 10)
        x = self.linear2.forward(x)

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
            first element is loss
            second element is correct_num
        """

        # (N, 1, 28, 28) -> (N, 10)
        score = self.predict(x)

        correct_num = np.sum(np.argmax(score, axis=1) == t)

        # (N, 10) and (N, ) -> float
        loss = self.loss.forward(score, t)

        return (loss, correct_num)

    def backward(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            the shape of array is (N, 1, 28, 28)
        """

        # (N, 10)
        dout = self.loss.backward()

        # (N, 10) -> (N, 84)
        dout = self.linear2.backward(dout)

        dout = self.relu4.backward(dout)

        # (N, 84) -> (N, 120)
        dout = self.linear1.backward(dout)

        # (N, 120) -> (N, 120, 1, 1)
        dout = dout.reshape(-1, 120, 1, 1)

        dout = self.relu3.backward(dout)

        # (N, 120, 1, 1) -> (N, 16, 5, 5)
        dout = self.conv3.backward(dout)

        # (N, 16, 5, 5) -> (N, 16, 10, 10)
        dout = self.pool2.backward(dout)

        dout = self.relu2.backward(dout)

        # (N, 16, 10, 10) -> (N, 6, 14, 14)
        dout = self.conv2.backward(dout)

        # (N, 6, 14, 14) -> (N, 6, 28, 28)
        dout = self.pool1.backward(dout)

        dout = self.relu1.backward(dout)

        # (N, 6, 28, 28) -> (N, 1, 28, 28)
        dout = self.conv1.backward(dout)

        return dout

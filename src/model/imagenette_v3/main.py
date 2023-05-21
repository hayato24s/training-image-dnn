from typing import Tuple

from common.base_model import BaseModel
from common.layers import AveragePooling, Convolution, FeedForward
from common.layers.softmax_with_loss import SoftmaxWithLoss
from common.np import np
from model.imagenette_v3.instance_norm_and_relu import InstanceNormAndRelu


class Model(BaseModel):
    """
    Architecture
    ------------
    - Convolution
    - Instance Normalization
    - Relu
    - Average Pooling
    - Convolution
    - Instance Normalization
    - Relu
    - Average Pooling
    - Feed Forward
    """

    def __init__(self):
        conv1_K = (np.random.randn(8, 3, 5, 5) / np.sqrt(5 * 5 * 8 / 2)).astype("f")
        conv1_b = np.zeros(8, dtype="f")
        conv1_stride = 1
        conv1_pad = 0

        conv2_K = (np.random.randn(16, 8, 5, 5) / np.sqrt(5 * 5 * 16 / 2)).astype("f")
        conv2_b = np.zeros(16, dtype="f")
        conv2_stride = 1
        conv2_pad = 0

        self.conv1 = Convolution(conv1_K, conv1_b, conv1_stride, conv1_pad)
        self.norm_and_relu1 = InstanceNormAndRelu(252 * 252)

        self.pool1 = AveragePooling(2, 2, 2, 0)

        self.conv2 = Convolution(conv2_K, conv2_b, conv2_stride, conv2_pad)
        self.norm_and_relu2 = InstanceNormAndRelu(122 * 122)

        self.pool2 = AveragePooling(2, 2, 2, 0)

        self.feed_forward = FeedForward(16 * 61 * 61, 5000, 10)

        self.loss = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in [self.conv1, self.norm_and_relu1, self.conv2, self.norm_and_relu2, self.feed_forward]:
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

        # (N, 3, 256, 256) -> (N, 8, 252, 252)
        x = self.conv1.forward(x)

        x = self.norm_and_relu1.forward(x)

        # (N, 8, 252, 252) -> (N, 8, 126, 126)
        x = self.pool1.forward(x)

        # (N, 8, 126, 126) -> (N, 16, 122, 122)
        x = self.conv2.forward(x)

        x = self.norm_and_relu2.forward(x)

        # (N, 16, 122, 122) -> (N, 16, 61, 61)
        x = self.pool2.forward(x)

        # (N, 16, 61, 61) -> (N, 16 * 61 * 61)
        x = x.reshape(-1, 16 * 61 * 61)

        # (N, 16 * 61 * 61) -> (N, 5000) -> (N, 10)
        x = self.feed_forward.forward(x)

        return x

    def forward(self, x: np.ndarray, t: np.ndarray) -> Tuple[float, int]:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 3, 256, 256)
        t : np.ndarray
            the shape of array is (N, 10)

        Returns
        -------
        Tuple[float, int]
            loss and correct_num
        """

        # (N, 3, 256, 256) -> (N, 10)
        score = self.predict(x)

        correct_num = np.sum(np.argmax(score, axis=1) == t)

        loss = self.loss.forward(score, t)

        return (loss, correct_num)

    def backward(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            the shape of array is (N, 3, 256, 256)
        """

        # (N, 10)
        dout = self.loss.backward()

        # (N, 10) -> (N, 16 * 61 * 61)
        dout = self.feed_forward.backward(dout)

        # (N, 16 * 61 * 61) -> (N, 16, 61, 61)
        dout = dout.reshape(-1, 16, 61, 61)

        # (N, 16, 61, 61) -> (N, 16, 122, 122)
        dout = self.pool2.backward(dout)

        dout = self.norm_and_relu2.backward(dout)

        # (N, 16, 122, 122) -> (N, 8, 126, 126)
        dout = self.conv2.backward(dout)

        # (N, 8, 126, 126) -> (N, 8, 252, 252)
        dout = self.pool1.backward(dout)

        dout = self.norm_and_relu1.backward(dout)

        # (N, 8, 252, 252) -> (N, 3, 256, 256)
        dout = self.conv1.backward(dout)

        return dout

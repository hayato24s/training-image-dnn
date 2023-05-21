from common.base_model import BaseModel
from common.np import np
from model.v2.encoder_sub_layer import EncoderSubLayer


class BaseEncoder(BaseModel):
    def __init__(self):

        self.sub_layerA = EncoderSubLayer(3, 64, 128, 128, 2, 2)
        self.sub_layerB = EncoderSubLayer(64, 128, 64, 64, 2, 2)
        self.sub_layerC = EncoderSubLayer(128, 256, 32, 32, 2, 2)
        self.sub_layerD = EncoderSubLayer(256, 512, 16, 16, 2, 2)

        self.params, self.grads = [], []
        for layer in [self.sub_layerA, self.sub_layerB, self.sub_layerC, self.sub_layerD]:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, 3, 256, 256)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 512, 16, 16)
        """

        # (N, 3, 256, 256) -> (N, 64, 128, 128)
        x = self.sub_layerA.forward(x)

        # (N, 64, 128, 128) -> (N, 128, 64, 64)
        x = self.sub_layerB.forward(x)

        # (N, 128, 64, 64) -> (N, 256, 32, 32)
        x = self.sub_layerC.forward(x)

        # (N, 256, 32, 32) -> (N, 512, 16, 16)
        x = self.sub_layerD.forward(x)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, 512, 16, 16)

        Returns
        -------
        np.ndarray
            the shape of array is (N, 3, 256, 256)
        """

        # (N, 512, 16, 16) -> (N, 256, 32, 32)
        dout = self.sub_layerD.backward(dout)

        # (N, 256, 32, 32) -> (N, 128, 64, 64)
        dout = self.sub_layerC.backward(dout)

        # (N, 128, 64, 64) -> (N, 64, 128, 128)
        dout = self.sub_layerB.backward(dout)

        # (N, 64, 128, 128) -> (N, 3, 256, 256)
        dout = self.sub_layerA.backward(dout)

        return dout

import os
import pickle

import numpy

import common.config as config
from common.np import np


def to_cpu(x):
    """
    Convert cupy.ndarray to numpy.ndarray

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/util.py#L170
    """

    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    """
    Convert numpy.ndarray to cupy.ndarray

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/util.py#L177
    """

    import cupy

    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


class BaseModel:
    """
    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/base_model.py#L10
    """

    def __init__(self):
        self.params: np.ndarray = np.array()
        self.grads: np.ndarray = np.array()

        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        params = [p.astype(np.float16) for p in self.params]
        if config.GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        if "/" in file_name:
            file_name = file_name.replace("/", os.sep)

        if not os.path.exists(file_name):
            raise IOError("No file: " + file_name)

        with open(file_name, "rb") as f:
            params = pickle.load(f)

        params = [p.astype("f") for p in params]
        if config.GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]

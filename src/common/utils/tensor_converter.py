from typing import Tuple

from common.np import np


def im2col(img: np.ndarray, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Convert 4 dimensional array into 2 dimensional array.

    Parameters
    ----------
    img : np.ndarray
        the shape of array is (N, C, H, W)
    filter_h : int
    filter_w : int
    stride : int, default 1
    pad : int, default 0

    Returns
    -------
    col : np.ndarray
        the shape of array is (N * OH * OW, C * FH * FW)

    References
    ----------
    https://github.com/oreilly-japan/deep-learning-from-scratch/blob/f549a1886b4c03a252ac7549311e29c1ed78a8a1/common/util.py#L39
    """

    N, C, H, W = img.shape
    FH, FW = filter_h, filter_w
    OH = (H + 2 * pad - FH) // stride + 1
    OW = (W + 2 * pad - FW) // stride + 1

    img = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant", constant_values=0)
    col = np.empty((N, C, FH, FW, OH, OW), dtype="f")

    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, C * FH * FW)

    return col


def col2im(
    col: np.ndarray, input_shape: Tuple[int], filter_h: int, filter_w: int, stride: int = 1, pad: int = 0
) -> np.ndarray:
    """
    Convert 2 dimensional array into 4 dimensional array.

    Parameters
    ----------
    col : np.ndarray
        the shape of array is (N * OH * OW, C * FH * FW)
    input_shape : Tuple[int]
    filter_h : int
    filter_w : int
    stride : int, default 1
    pad : int, default 0

    Returns
    -------
    img : np.ndarray
        the shape of array is (N, C, H, W)

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch/blob/f549a1886b4c03a252ac7549311e29c1ed78a8a1/common/util.py#L71
    """

    N, C, H, W = input_shape
    FH, FW = filter_h, filter_w
    OH = (H + 2 * pad - FH) // stride + 1
    OW = (W + 2 * pad - FW) // stride + 1

    # img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), dtype='f')
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype="f")
    col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    for y in range(FH):
        y_max = y + stride * OH
        for x in range(FW):
            x_max = x + stride * OW
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H + pad, pad : W + pad]

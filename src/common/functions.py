from common.np import np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax function

    Parameters
    ----------
    x : np.ndarray
        the shape of array is (N, D)

    Returns
    -------
    np.ndarray
        the shape of array is (N, D)

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/functions.py#L13
    """

    out = x - np.max(x, axis=1, keepdims=True)
    out = np.exp(out)
    out /= np.sum(out, axis=1, keepdims=True)

    return out


def cross_entropy_error(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate cross entropy error.

    N means batch size.
    D means dimention of vector.

    Parameters
    ----------
    x : np.ndarray
        the shape of array is (N, D)
    t : np.ndarray
        the shape of array is (N, )

    Returns
    -------
    np.ndarray
        the shape of array is (N, )

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/functions.py#L25
    """

    N, _ = x.shape

    return -np.sum(np.log(x[np.arange(N), t] + 1e-7)) / N

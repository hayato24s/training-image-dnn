from typing import List

from common.np import np


def clip_grads(grads: List[np.ndarray], max_norm: float) -> None:
    """
    Parameters
    ----------
    grads : List[np.ndarray]
        gradients
    max_norm : float
        max norm

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/util.py#L184
    """

    total_norm: float = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-7)
    if rate < 1:
        for grad in grads:
            grad *= rate

from typing import List, Tuple

from common.np import np


def remove_duplicate(params: List[np.ndarray], grads: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Summarize duplicated params and grads into one.

    Parameters
    ----------
    params : List[np.ndarray]
    grads : List[np.ndarray]

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/trainer.py#L140
    """

    params, grads = params[:], grads[:]

    i = 0
    while i < len(params) - 1:
        delete_indices: List[int] = list()

        for j in range(i + 1, len(params)):
            # same parameter
            if params[i] is params[j]:
                grads[i] += grads[j]
                delete_indices.append(j)
            # weight tying
            elif (
                (params[i].ndim == 2)
                and (params[j].ndim == 2)
                and (params[i].shape == params[j].T.shape)
                and (np.all(params[i] == params[j].T))
            ):
                grads[i] += grads[j].T
                delete_indices.append(j)

        for idx in reversed(delete_indices):
            params.pop(idx)
            grads.pop(idx)

        i += 1

    return params, grads

from typing import Tuple

from common.np import np


class DateData:
    """
    For data management
    """

    def __init__(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """

        self.x: np.ndarray = x
        self.t: np.ndarray = t
        self.size: int = x.shape[0]
        self.idx: int = 0

    def get_batch(self, batch_size: int, only_once: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch of x and t

        Parameters
        ----------
        batch_size : int
            batch size
        only_once : bool
            If you want to get data at only one epoch, set true.
            by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            first ndarray
            - batch of x
            - shape with (batch size, sequence length)
            - dtype `int32`

            second ndarray
            - batch of t
            - shape with (batch size, sequence length)
            - dtype `int32`
        """

        indices = list()

        while True:
            wanted_size = batch_size - len(indices)

            if self.idx + wanted_size < self.size:
                indices += list(range(self.idx, self.idx + wanted_size))
                self.idx += wanted_size
                break
            else:
                indices += list(range(self.idx, self.size))
                self.idx = 0

                if only_once:
                    break

        batch_x: np.ndarray = self.x[indices]
        batch_t: np.ndarray = self.t[indices]

        return batch_x, batch_t

    def reset_idx(self) -> None:
        self.idx = 0

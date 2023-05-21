from typing import Tuple

from common.np import np


class MnistData:
    """
    For data management
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """
        Parameters
        ----------
        images : np.ndarray
        labels : np.ndarray

        Raises
        ------
        ValueError
            the size of images should be equal the size of labels.
        """

        if images.shape[0] != labels.shape[0]:
            raise ValueError("the size of images should be equal the size of labels.")

        self.images: np.ndarray = images
        self.labels: np.ndarray = labels
        self.size: int = images.shape[0]
        self.idx: int = 0

    def get_batch(self, batch_size: int, only_once: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch of image arrays and labels.

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
            - batch of image array
            - shape with (batch size, 28, 28)
            - dtype `float32`

            second ndarray
            - batch of labels
            - shape with (batch size, )
            - dtype `int32`
        """

        batch_images: np.ndarray = np.empty((batch_size, 1, 28, 28), dtype="f")
        batch_labels: np.ndarray = np.empty((batch_size), dtype="i")

        batch_idx = 0
        while batch_idx < batch_size:
            wanted_size = batch_size - batch_idx

            if self.idx + wanted_size < self.size:
                batch_images[batch_idx:, :, :, :] = self.images[self.idx : self.idx + wanted_size, :, :, :]
                batch_labels[batch_idx:] = self.labels[self.idx : self.idx + wanted_size]
                self.idx += wanted_size
                break
            else:
                batch_images[batch_idx : batch_idx + (self.size - self.idx), :, :, :] = self.images[
                    self.idx :, :, :, :
                ]
                batch_labels[batch_idx : batch_idx + (self.size - self.idx)] = self.labels[self.idx :]
                batch_idx += self.size - self.idx
                self.idx = 0

                if only_once:
                    break

        return batch_images, batch_labels

    def reset_idx(self) -> None:
        self.idx = 0

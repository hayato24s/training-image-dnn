from typing import List, Tuple

from common.np import np
from dataset.imagenette.image_caches import get_image_arrays_by_path_from_file


class ImagenetteData:
    """
    For data management
    """

    def __init__(self, image_paths: List[str], labels: List[int]) -> None:
        """
        Parameters
        ----------
        image_paths : List[str]
        labels : List[int]

        Raises
        ------
        ValueError
            the size of image_paths should be equal the size of labels.
        """

        if len(image_paths) != len(labels):
            raise ValueError("The size of image_paths is not equal to the size of labels.")

        self.image_paths: List[str] = image_paths
        self.labels: List[List[int]] = labels
        self.size: int = len(image_paths)
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
            - shape with (batch size, channel, height, width)
            - dtype `float32`

            second ndarray
            - batch of caption
            - shape with (batch size, sequence length)
            - dtype `int32`
        """

        used_image_paths: List[str] = list()
        used_labels: List[int] = list()

        while len(used_image_paths) < batch_size:
            wanted_size = batch_size - len(used_image_paths)

            if self.idx + wanted_size < self.size:
                used_image_paths += self.image_paths[self.idx : self.idx + wanted_size]
                used_labels += self.labels[self.idx : self.idx + wanted_size]
                self.idx += wanted_size
            else:
                used_image_paths += self.image_paths[self.idx :]
                used_labels += self.labels[self.idx :]
                self.idx = 0

                if only_once:
                    break

        batch_xs: np.ndarray = get_image_arrays_by_path_from_file(used_image_paths)
        batch_ts: np.ndarray = np.array(used_labels, dtype="i")

        return batch_xs, batch_ts

    def reset_idx(self) -> None:
        self.idx = 0

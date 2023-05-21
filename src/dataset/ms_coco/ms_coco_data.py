from typing import List, Tuple

from common.np import np
from dataset.ms_coco.image_cache import (
    get_image_arrays_by_image_ids_from_file,
    get_image_arrays_by_image_ids_from_memory,
    store_image_caches_in_memory,
)


class MSCOCOData:
    """
    For data management
    """

    def __init__(self, image_ids: List[str], captions: List[List[int]], in_memory: bool = True) -> None:
        """
        Parameters
        ----------
        image_ids : List[str]
            image ids
        captions : List[List[int]]
            tokenized captions
        in_memory : bool, optional
            If you want to store data in memory and get data from memory, you should set True.
            If you want to get data from file at every time, you should set False.
            by default True

        Raises
        ------
        ValueError
            the size of image_ids should be equal the size of captions.
        """

        if len(image_ids) != len(captions):
            raise ValueError("The size of image_ids is not equal to the size of captions.")

        self.image_ids: List[str] = image_ids
        self.captions: List[List[int]] = captions
        self.size: int = len(image_ids)
        self.idx: int = 0
        self.get_image_arrays_by_image_ids = (
            get_image_arrays_by_image_ids_from_memory if in_memory else get_image_arrays_by_image_ids_from_file
        )

        if in_memory:
            store_image_caches_in_memory()

    def get_batch(self, batch_size: int, only_once: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch of image arrays and captions.

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

        used_image_ids: List[str] = list()
        used_captions: List[List[int]] = list()

        while True:
            wanted_size = batch_size - len(used_image_ids)

            if self.idx + wanted_size < self.size:
                used_image_ids += self.image_ids[self.idx : self.idx + wanted_size]
                used_captions += self.captions[self.idx : self.idx + wanted_size]
                self.idx += wanted_size
                break
            else:
                used_image_ids += self.image_ids[self.idx :]
                used_captions += self.captions[self.idx :]
                self.idx = 0

                if only_once:
                    break

        batch_xs: np.ndarray = self.get_image_arrays_by_image_ids(used_image_ids)
        batch_ts: np.ndarray = np.array(used_captions, dtype="i")

        return batch_xs, batch_ts

    def reset_idx(self) -> None:
        self.idx = 0

import os
import random

import dataset.ms_coco.const as const
from dataset.ms_coco.caption import create_caption_caches
from dataset.ms_coco.image_cache import create_image_caches


def prepare_dataset() -> None:
    """
    Prepare dataset.
    """

    print("preparing dataset ...")

    if not os.path.exists(const.storage_dir_name):
        os.mkdir(const.storage_dir_name)

    if not os.path.exists(const.image_caches_file_name):
        create_image_caches()

    if not os.path.exists(const.caption_caches_file_name):
        create_caption_caches()

    print("finished preparing dataset.")


def split_data(data, train_size=0, val_size=0, test_size=0, seed=None):
    """
    Divide the data into training, validation and test data.
    """

    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    idx1 = train_size
    idx2 = train_size + val_size
    idx3 = train_size + val_size + test_size

    train_data = data[:idx1]
    val_data = data[idx1:idx2]
    test_data = data[idx2:idx3]

    return train_data, val_data, test_data

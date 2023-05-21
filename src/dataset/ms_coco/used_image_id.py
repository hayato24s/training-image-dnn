import os
import pickle
from typing import List

import dataset.ms_coco.const as const


def save_used_image_ids(used_image_ids: List[str]) -> None:
    with open(const.used_image_ids_file_name, "wb") as f:
        pickle.dump(used_image_ids, f)

    print("saved used image ids.")


def load_used_image_ids() -> List[str]:
    if not os.path.exists(const.used_image_ids_file_name):
        raise FileNotFoundError(const.used_image_ids_file_name)

    with open(const.used_image_ids_file_name, "rb") as f:
        used_image_ids = pickle.load(f)

    return used_image_ids

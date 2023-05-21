import csv
import os
from typing import Any, Dict, List

import numpy
from PIL import Image

import common.config as config
import dataset.imagenette.const as const
from common.base_model import to_gpu
from dataset.imagenette.download import download_imagenette
from dataset.imagenette.image_info import create_image_info, label_dict, label_to_id


def create_image_caches(width: int = 256, height: int = 256) -> None:
    print("creating imagenette image caches ...")

    if not os.path.exists(const.imagenette_storage_dir_name):
        download_imagenette()

    with open(const.imagenette_csv_file_name) as f:
        reader = csv.reader(f)
        data = [row for row in reader][1:]
        all_image_info = [
            {"path": row[0], "label": label_to_id[label_dict[row[1]]], "is_valid": row[6] == "True"} for row in data
        ]

    used_image_info = list()
    image_caches: Dict[str, numpy.ndarray] = dict()

    for image_info in all_image_info:
        image_file_path = os.path.join(const.imagenette_storage_dir_name, image_info["path"])
        image = Image.open(image_file_path)

        # Exclude image whose mode is not RGB.
        if image.mode != "RGB":
            continue

        # Resize image
        image = image.resize((width, height), Image.LANCZOS)
        # Convert image to numpy.ndarray
        image_cache = numpy.array(image, dtype="uint8")
        # (height, width, channel) -> (channel, height, width)
        image_cache = image_cache.transpose(2, 0, 1)

        used_image_info.append(image_info)
        image_caches[image_info["path"]] = image_cache

    create_image_info(used_image_info)
    numpy.savez(const.imagenette_image_caches_without_extension, **image_caches)

    print("finished creating imagenette image caches.")


npz_file: Any or None = None


def get_image_caches_by_path_from_file(paths: List[str]) -> List[numpy.ndarray]:
    if not os.path.exists(const.imagenette_image_caches):
        create_image_caches()

    global npz_file

    if npz_file is None:
        npz_file = numpy.load(const.imagenette_image_caches)

    image_caches = list()
    for path in paths:
        image_caches.append(npz_file[path])

    return image_caches


def get_image_arrays_by_path_from_file(paths: List[str]) -> numpy.ndarray:
    image_caches = get_image_caches_by_path_from_file(paths)
    image_arrays = numpy.array(image_caches, dtype="f") / 255.0

    if config.GPU:
        image_arrays = to_gpu(image_arrays)

    return image_arrays


def prepare_imagenette_dataset():
    if not os.path.exists(const.imagenette_image_caches):
        create_image_caches()

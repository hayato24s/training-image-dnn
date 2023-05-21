import json
import os
from typing import Any, Dict, List

import numpy
from PIL import Image

import common.config as config
import dataset.ms_coco.const as const
from common.base_model import to_gpu
from common.np import np
from dataset.ms_coco.download import download_annotations, download_images
from dataset.ms_coco.used_image_id import save_used_image_ids


def create_compressed_image_caches(width: int = 256, height: int = 256) -> None:
    print("creating compressed image caches ...")

    if not os.path.exists(const.annotations_file_name):
        download_annotations()

    if not os.path.exists(const.images_dir_name):
        download_images()

    with open(const.annotations_file_name) as f:
        annotation_data = json.load(f)

    used_image_ids: List[str] = list()
    image_id_to_image_cache: Dict[str, numpy.ndarray] = dict()

    for image_data in annotation_data["images"]:
        image_id: str = "%012d" % int(image_data["id"])
        image_file_name = os.path.join(const.images_dir_name, image_data["file_name"])
        image = Image.open(image_file_name)

        # Exclude image whose mode is not RGB.
        if image.mode != "RGB":
            continue

        # Resize image
        image = image.resize((width, height), Image.LANCZOS)
        # Convert image to numpy.ndarray
        image_cache = numpy.array(image, dtype="uint8")
        # (height, width, channel) -> (channel, height, width)
        image_cache = image_cache.transpose(2, 0, 1)

        used_image_ids.append(image_id)
        image_id_to_image_cache[image_id] = image_cache

    save_used_image_ids(used_image_ids)
    numpy.savez_compressed(const.compressed_image_caches_file_name_without_extension, **image_id_to_image_cache)

    print("finished creating compressed image caches.")


def load_compressed_image_caches() -> Dict[str, numpy.ndarray]:
    if not os.path.exists(const.compressed_image_caches_file_name):
        create_compressed_image_caches()

    npz_comp = numpy.load(const.compressed_image_caches_file_name)
    image_id_to_image_cache: Dict[str, numpy.ndarray] = dict()

    for image_id in npz_comp.files:
        image_cache: numpy.ndarray = npz_comp[image_id]
        image_id_to_image_cache[image_id] = image_cache

    return image_id_to_image_cache


def create_image_caches() -> None:
    print("creating image caches ...")

    image_id_to_image_cache = load_compressed_image_caches()
    numpy.savez(const.image_caches_file_name_without_extension, **image_id_to_image_cache)

    print("finished creating image caches ...")


def load_image_caches() -> Dict[str, numpy.ndarray]:
    if not os.path.exists(const.image_caches_file_name):
        create_image_caches()
    npz = numpy.load(const.image_caches_file_name)
    image_id_to_image_cache: Dict[str, numpy.ndarray] = dict()

    for image_id in npz.files:
        image_cache: numpy.ndarray = npz[image_id]
        image_id_to_image_cache[image_id] = image_cache

    return image_id_to_image_cache


image_id_to_image_cache: Dict[str, numpy.ndarray] or None = None


def store_image_caches_in_memory() -> None:
    """
    Store image caches in memory.
    """

    print("storing image caches in memory ...")

    global image_id_to_image_cache

    if image_id_to_image_cache is None:
        image_id_to_image_cache = load_image_caches()

    print("finished storing image caches in memory.")


def get_image_caches_by_image_ids_from_memory(image_ids: List[str]) -> List[numpy.ndarray]:
    """
    Get image cache from image ids from memory.

    Parameters
    ----------
    image_ids : List[str]
        list of image id

    Returns
    -------
    List[numpy.ndarray]
        list of image cache.
        the shape of image cache is (channel, height, width).
        the dtype of image cache is `numpy.uint8`.
        the values of elements in the image cache is 0 to 255.
    """

    if image_id_to_image_cache is None:
        store_image_caches_in_memory()

    image_caches = [image_id_to_image_cache[id] for id in image_ids]

    return image_caches


def get_image_arrays_by_image_ids_from_memory(image_ids: List[str]) -> np.ndarray:
    """
    Get image array from image ids from memory.

    Parameters
    ----------
    image_ids : List[str]
        list of image id

    Returns
    -------
    np.ndarray
        the shape of array is (number of image ids, channel, height, width).
        the dtype of this array is `np.float32`.
        the values of elements in this array is 0 to 1.
    """

    image_caches = get_image_caches_by_image_ids_from_memory(image_ids)
    image_arrays = numpy.array(image_caches, dtype="f") / 255.0
    if config.GPU:
        image_arrays = to_gpu(image_arrays)

    return image_arrays


npz_file: Any or None = None


def get_image_caches_by_image_ids_from_file(image_ids: List[str]) -> List[np.ndarray]:
    """
    Get image cache from image ids from memory.

    Parameters
    ----------
    image_ids : List[str]
        list of image id

    Returns
    -------
    List[np.ndarray]
        list of image cache.
        the shape of image cache is (channel, height, width).
        the dtype of image cache is `numpy.uint8`.
        the values of elements in the image cache is 0 to 255.
    """
    global npz_file

    if npz_file is None:
        if not os.path.exists(const.image_caches_file_name):
            create_image_caches()

        npz_file = numpy.load(const.image_caches_file_name)

    image_caches: List[numpy.ndarray] = list()
    for image_id in image_ids:
        image_caches.append(npz_file[image_id])

    return image_caches


def get_image_arrays_by_image_ids_from_file(image_ids: List[str]) -> np.ndarray:
    """
    Get image array from image ids from memory.

    Parameters
    ----------
    image_ids : List[str]
        list of image id

    Returns
    -------
    np.ndarray
        the shape of array is (number of image ids, channel, height, width).
        the dtype of this array is `np.float32`.
        the values of elements in this array is 0 to 1.
    """

    image_caches = get_image_caches_by_image_ids_from_file(image_ids)
    image_arrays = numpy.array(image_caches, dtype="f") / 255.0
    if config.GPU:
        image_arrays = to_gpu(image_arrays)

    return image_arrays

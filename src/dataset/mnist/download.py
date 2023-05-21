import gzip
import os
import urllib.request
from http.client import HTTPResponse
from typing import Dict, Tuple

import numpy

import common.config as config
import dataset.mnist.const as const
from common.base_model import to_gpu
from common.np import np

# https://qiita.com/python_walker/items/e4d2ae5b7196cb07402b

base_url = "http://yann.lecun.com/exdb/mnist/"
file_path = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "text_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(download_url: str, output_file_name: str) -> None:
    response: HTTPResponse = urllib.request.urlopen(download_url)
    data: bytes = response.read()

    with open(output_file_name, "wb") as f:
        f.write(data)


def load_images(file_name: str) -> numpy.ndarray:
    with gzip.open(file_name, "rb") as f:
        array = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
        array = array.reshape(-1, 1, 28, 28)

    return array


def load_labels(file_name: str) -> numpy.ndarray:
    with gzip.open(file_name, "rb") as f:
        array = numpy.frombuffer(f.read(), numpy.uint8, offset=8)

    return array


def donwload_mnist_dataset():
    print("downloading mnist dataset ...")

    dataset: Dict[str, numpy.ndarray] = dict()

    for k, v in file_path.items():
        download_url = base_url + v
        output_file_name = os.path.join(const.mnist_dir_name, v)

        _download(download_url, output_file_name)

        if k[-6:] == "images":
            array = load_images(output_file_name)
        else:
            array = load_labels(output_file_name)

        dataset[k] = array

        os.remove(output_file_name)

    numpy.savez(const.mnist_dataset_without_extension, **dataset)

    print("finished downloading mnist dataset ...")


dataset: Dict[str, np.ndarray] or None = None


def initialize_mnist_dataset():
    if not os.path.exists(const.mnist_dataset_file_name):
        donwload_mnist_dataset()

    global dataset
    dataset = dict()
    npz = numpy.load(const.mnist_dataset_file_name)

    for k in file_path.keys():
        array = npz[k]

        if k[-6:] == "images":
            array = array.astype("f") / 255.0
        else:
            array = array.astype("i")

        if config.GPU:
            array = to_gpu(array)

        dataset[k] = array


def get_mnist_data_from_memory() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    np.ndarray
        (train_images, train_labels, test_images, test_labels)
    """

    if dataset is None:
        initialize_mnist_dataset()

    data = (dataset[k] for k in file_path.keys())

    return data

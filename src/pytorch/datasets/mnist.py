import gzip
import os
import urllib.request
from typing import Dict

import datasets.const as const
import numpy
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from urllib3 import HTTPResponse

# import const


# https://qiita.com/python_walker/items/e4d2ae5b7196cb07402b

base_url = "http://yann.lecun.com/exdb/mnist/"
file_paths = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    os.makedirs(const.mnist_storage_dir_name, exist_ok=True)

    for k, v in file_paths.items():
        download_url = os.path.join(base_url, v)
        output_file_name = os.path.join(const.mnist_storage_dir_name, v)

        response: HTTPResponse = urllib.request.urlopen(download_url)
        data: bytes = response.read()

        with open(output_file_name, "wb") as f:
            f.write(data)


def load_images(file_name: str) -> numpy.ndarray:
    with gzip.open(file_name, "rb") as f:
        array = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=16)
        array = array.reshape(-1, 1, 28, 28)
        array = array.transpose(0, 2, 3, 1)

    return array


def load_labels(file_name: str) -> numpy.ndarray:
    with gzip.open(file_name, "rb") as f:
        array = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=8)

    return array


def get_mnist_dataset() -> Dict[str, numpy.ndarray]:
    if not os.path.exists(const.mnist_storage_dir_name):
        download_mnist()

    dataset: Dict[str, numpy.ndarray] = dict()

    dataset["train_images"] = load_images(os.path.join(const.mnist_storage_dir_name, file_paths["train_images"]))
    dataset["train_labels"] = load_labels(os.path.join(const.mnist_storage_dir_name, file_paths["train_labels"]))
    dataset["test_images"] = load_images(os.path.join(const.mnist_storage_dir_name, file_paths["test_images"]))
    dataset["test_labels"] = load_labels(os.path.join(const.mnist_storage_dir_name, file_paths["test_labels"]))

    return dataset


class MNISTDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
        super().__init__()
        dataset = get_mnist_dataset()

        self.images = (dataset["train_images"] if is_train else dataset["test_images"]).copy()
        self.labels = (dataset["train_labels"] if is_train else dataset["test_labels"]).copy()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])

    mnist_train = MNISTDataset(is_train=True, transform=transform)
    mnist_test = MNISTDataset(is_train=False, transform=transform)

    train_dataloader = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=mnist_test, batch_size=64, shuffle=False)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    img = train_features[0].squeeze()
    label = train_labels[0]

    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

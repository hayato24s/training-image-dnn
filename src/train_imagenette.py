import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy
from PIL import Image

import common.config as config
import dataset.imagenette.const as const
import model.parameters.path as parameters
from common.base_model import to_cpu
from common.functions import cross_entropy_error, softmax
from common.np import np
from common.optimizers import Adam
from dataset.imagenette.image_caches import get_image_arrays_by_path_from_file, prepare_imagenette_dataset
from dataset.imagenette.image_info import get_image_info
from dataset.imagenette.imagenette_data import ImagenetteData
from dataset.imagenette.imagenette_trainer import ImagenetteTrainer, eval_accuracy
from model.imagenette_v2.main import Model

prepare_imagenette_dataset()

train_image_info = get_image_info("train")
val_image_info = get_image_info("val")

print("total train image :", len(train_image_info))
print("total val image :", len(val_image_info))

# Shuffle data
random.shuffle(train_image_info)
random.shuffle(val_image_info)

# Reduce data size
used_train_image_info = train_image_info[:]
used_val_image_info = val_image_info[:1000]

print("used train image :", len(used_train_image_info))
print("used val image :", len(used_val_image_info))

train_paths = [info["path"] for info in used_train_image_info]
val_paths = [info["path"] for info in used_val_image_info]

train_labels = [info["label"] for info in used_train_image_info]
val_labels = [info["label"] for info in used_val_image_info]

# Get Data
train_data = ImagenetteData(train_paths, train_labels)
val_data = ImagenetteData(val_paths, val_labels)

# Create model
model = Model()
parameter = parameters.imagenette_v2
# model.load_params(parameter)


def train():
    # Create trainer
    optimizer = Adam()
    trainer = ImagenetteTrainer(model, optimizer, train_data)

    # Set parameter for trainer
    batch_size = 100
    epochs = 3
    max_grad = 5.0
    eval_interval = 10
    print("batch size :", batch_size)
    print("epochs :", epochs)

    # Train model
    accuracy_list = []
    best_accuracy: float = 0.0
    print("-" * 70)
    print("start training")
    print("-" * 70)
    for _ in range(epochs):
        trainer.fit(batch_size, 1, max_grad, eval_interval)

        accuracy = eval_accuracy(model, val_data, min(batch_size, len(used_val_image_info)))
        accuracy_list.append(accuracy)

        print("-" * 70)
        print("val accuracy : %.2f" % accuracy)
        if accuracy > best_accuracy:
            print("update best val accuracy : %.2f -> %.2f" % (best_accuracy, accuracy))
            best_accuracy = accuracy
            model.save_params(parameter)
        print("-" * 70)

    # Show graph
    x = numpy.arange(len(accuracy_list))
    if config.GPU:
        accuracy_list = [to_cpu(ndarray) for ndarray in accuracy_list]
    plt.plot(x, accuracy_list, marker="o")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("val accuracy")
    plt.show()


def show_image(path: str) -> None:
    file_name = os.path.join(const.imagenette_storage_dir_name, path)

    image = Image.open(file_name)
    image.open()


def check(image_info: Any) -> None:
    path = image_info["path"]
    label = image_info["label"]

    image_arrays = get_image_arrays_by_path_from_file([path])
    score = model.predict(image_arrays)
    prob = softmax(score)
    loss = cross_entropy_error(prob, np.array([label]))
    predicted_label = np.argmax(prob.flatten())

    print("-" * 70)
    print("label: %d, predicred: %d, loss: %f" % (label, predicted_label, loss))
    print(prob)


def check_gradient() -> None:
    batch_size = 100
    train_data.reset_idx()
    batch_x, batch_t = train_data.get_batch(batch_size)
    model.forward(batch_x, batch_t)
    model.backward()

    for grad in model.grads:
        grad = np.abs(grad)
        mean = np.mean(grad)
        std = np.std(grad)
        min = np.min(grad)
        max = np.max(grad)
        median = np.median(grad)

        print("abs\tmean : %f\tstd : %f\tmin : %f\tmedian : %f\tmax : %f" % (mean, std, min, median, max))


if __name__ == "__main__":
    # train()

    check_gradient()

    # for i in range(10):
    #     check(val_image_info[i])

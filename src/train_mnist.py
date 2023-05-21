from typing import List

from matplotlib import pyplot as plt

import common.config as config
import model.parameters.path as parameters
from common.base_model import to_cpu
from common.np import np
from common.optimizers import Adam
from dataset.mnist.download import get_mnist_data_from_memory
from dataset.mnist.mnist_data import MnistData
from dataset.mnist.mnist_trainer import MnistTrainer, eval_accuracy
from model.mnist.main import Model

# donwload_mnist_dataset()
train_images, train_labels, test_images, test_labels = get_mnist_data_from_memory()

print("total train data :", len(train_images))
print("total test data :", len(test_images))

# Reduce data
train_images = train_images[:]
train_labels = train_labels[:]
test_images = test_images[:]
test_labels = test_labels[:]

print("used train data :", len(train_images))
print("used test data :", len(test_images))

# Get data
train_data = MnistData(train_images, train_labels)
test_data = MnistData(test_images, test_labels)

# Create model
model = Model()
parameter = parameters.mnist
# model = LenetWithNorm()
# parameter = parameters.lenet_with_norm
# model.load_params(parameter)


def train():
    # Create trainer
    optimizer = Adam()
    trainer = MnistTrainer(model, optimizer, train_data)

    # Set parameter for trainer
    batch_size = 1000
    epochs = 10
    max_grad = 5.0
    eval_interval = 10
    print("batch size :", batch_size)
    print("epochs :", epochs)

    # Train model
    train_accuracy_list = []
    test_accuracy_list = []
    best_accuracy: float = 0.0
    print("-" * 70)
    print("start training")
    print("-" * 70)

    for _ in range(epochs):
        trainer.fit(batch_size, 1, max_grad, eval_interval)

        train_accuracy = eval_accuracy(model, train_data, min(batch_size, train_data.size))
        test_accuracy = eval_accuracy(model, test_data, min(batch_size, test_data.size))

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

        print("-" * 70)
        print("test accuracy : %.3f" % test_accuracy)
        if test_accuracy > best_accuracy:
            print("update best test accuracy : %.3f -> %.3f" % (best_accuracy, test_accuracy))
            best_accuracy = test_accuracy
            model.save_params(parameter)
        print("-" * 70)

    # Show graph
    if config.GPU:
        train_accuracy_list = [to_cpu(ndarray) for ndarray in train_accuracy_list]
        test_accuracy_list = [to_cpu(ndarray) for ndarray in test_accuracy_list]
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy_list, label="train", color="red")
    plt.plot(test_accuracy_list, label="val", color="blue")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.show()


def check(images: np.ndarray, labels: np.ndarray):
    loss, correct_num = model.forward(images, labels)
    accuracy = correct_num / images.shape[0]
    print("loss : %f, accuracy : %f" % (loss, accuracy))


def calc_loss(prob: float):
    return -np.log(prob + 1e-7)


def check_gradient() -> None:
    batch_size = 1000
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
    train()

    # check_gradient()

    # check(train_images, train_labels)
    # check(test_images, test_labels)

import time

import numpy
from matplotlib import pyplot as plt

from common.np import np
from common.optimizers import BaseOptimizer
from common.utils import clip_grads
from dataset.imagenette.imagenette_data import ImagenetteData
from model.imagenette.main import Model


def eval_accuracy(model: Model, data: ImagenetteData, batch_size: int) -> float:
    if data.size == 0:
        raise ValueError("no data")

    data.reset_idx()
    correct_num = 0

    while True:
        batch_x, batch_t = data.get_batch(batch_size, only_once=True)

        # Calculate accuracy
        score = model.predict(batch_x)
        correct_num += np.sum(np.argmax(score, axis=1) == batch_t)

        if data.idx == 0:
            break

    accuracy = correct_num / data.size

    return accuracy


class ImagenetteTrainer:
    def __init__(self, model: Model, optimizer: BaseOptimizer, data: ImagenetteData) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.accuracy_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def fit(
        self,
        batch_size: int,
        epochs: int,
        max_grad: float = None,
        eval_interval: int = 20,
    ) -> None:
        max_iters = self.data.size // batch_size
        model, optimizer = self.model, self.optimizer
        self.accuracy_list = []
        self.eval_interval = eval_interval

        total_loss = 0
        loss_count = 0
        total_correct_num = 0
        predicted_size = 0

        start_time = time.time()
        for _ in range(epochs):
            for iters in range(max_iters):
                batch_xs, batch_ts = self.data.get_batch(batch_size, only_once=False)

                # Calculate gradients and update parameters.
                loss, correct_num = model.forward(batch_xs, batch_ts)
                model.backward()
                params, grads = model.params, model.grads
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1
                total_correct_num += correct_num
                predicted_size += batch_size

                # Evaluate loss.
                if (eval_interval is not None) and ((iters + 1) % eval_interval) == 0:
                    loss = total_loss / loss_count
                    accuracy = total_correct_num / predicted_size

                    elapsed_time = time.time() - start_time
                    print(
                        "|  epoch %3d  |  iter %4d / %4d  |  time %6d[s] | loss %9.5f  |  accuracy %.3f  |"
                        % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, loss, accuracy)
                    )

                    self.accuracy_list.append(accuracy)

                    total_loss = 0
                    loss_count = 0
                    total_correct_num = 0
                    predicted_size = 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.accuracy_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.accuracy_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("accuracy")
        plt.show()

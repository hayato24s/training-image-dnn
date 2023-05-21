import time

import numpy
from matplotlib import pyplot as plt

from common.np import np
from common.optimizers import BaseOptimizer
from common.utils import clip_grads
from dataset.date.date_data import DateData
from model.date.main import Model


def eval_ppl(model: Model, data: DateData, batch_size: int) -> float:
    if data.size == 0:
        raise ValueError("no data")

    data.reset_idx()
    total_loss = 0
    loss_count = 0

    while True:
        batch_x, batch_t = data.get_batch(batch_size, only_once=True)

        # Calculate loss
        loss = model.forward(batch_x, batch_t)
        total_loss += loss
        loss_count += 1

        if data.idx == 0:
            break

    ppl = np.exp(total_loss / loss_count)

    return ppl


def eval_accuracy(model: Model, data: DateData, batch_size: int) -> float:
    if data.size == 0:
        raise ValueError("no data")

    data.reset_idx()
    correct_num = 0

    while True:
        batch_x, batch_t = data.get_batch(batch_size, only_once=True)

        # (N, T_dec)
        ts_input = batch_t[:, :-1]

        # (N, T_dec)
        ts_ans = batch_t[:, 1:]

        score = model.predict(batch_x, ts_input)
        correct_num += np.sum(np.argmax(score, axis=1) == ts_ans)

        if data.idx == 0:
            break

    # Calculate accuracy
    accuracy = correct_num / data.size

    return accuracy


class DateTrainer:
    def __init__(self, model: Model, optimizer: BaseOptimizer, data: DateData) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.ppl_list = None
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
        self.ppl_list = []
        self.eval_interval = eval_interval

        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for _ in range(epochs):
            for iters in range(max_iters):
                batch_xs, batch_ts = self.data.get_batch(batch_size, only_once=False)

                # Calculate gradients and update parameters.
                loss = model.forward(batch_xs, batch_ts)
                model.backward()
                params, grads = model.params, model.grads
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # Evaluate ppl.
                if (eval_interval is not None) and ((iters + 1) % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)

                    elapsed_time = time.time() - start_time
                    print(
                        "|  epoch %3d  |  iter %4d / %4d  |  time %6d[s] | ppl %9.5f  |"
                        % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl)
                    )

                    self.ppl_list.append(ppl)

                    total_loss = 0
                    loss_count = 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("ppl")
        plt.show()

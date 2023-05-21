import time

import matplotlib.pyplot as plt
import numpy

from common.base_model import BaseModel
from common.np import np
from common.optimizers import BaseOptimizer
from common.utils import Data, clip_grads, remove_duplicate


class Trainer:
    """
    Trainer for model

    References
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/trainer.py#L69
    """

    def __init__(self, model: BaseModel, optimizer: BaseOptimizer, data: Data, remove_duplicated=False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0
        self.remove_duplicated = remove_duplicated

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
                params, grads = (
                    remove_duplicate(model.params, model.grads)
                    if self.remove_duplicated
                    else (model.params, model.grads)
                )
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # Evaluate perplexity.
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "|  epoch %3d  |  iter %4d / %4d  |  time %6d[s]  |  perplexity %8.2f  |"
                        % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl)
                    )
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("perplexity")
        plt.show()

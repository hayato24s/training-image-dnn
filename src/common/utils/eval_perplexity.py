from common.base_model import BaseModel
from common.np import np
from common.utils import Data


def eval_perplexity(model: BaseModel, data: Data, batch_size: int) -> float:
    """
    Evaluate perplexity

    References
    ----------
    [1] https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/484c6e0b45435b1240d3c51f20b6e8357e096fdd/common/util.py#L196
    """

    if data.size == 0:
        raise ValueError("no data")

    data.reset_idx()
    total_loss = 0
    loss_count = 0

    while True:
        batch_xs, batch_ts = data.get_batch(batch_size, only_once=True)

        # Calculate loss
        loss = model.forward(batch_xs, batch_ts)
        total_loss += loss
        loss_count += 1

        if data.idx == 0:
            break

    # Evaluate perplexity.
    ppl = np.exp(total_loss / loss_count)

    return ppl

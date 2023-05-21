import numpy
from matplotlib import pyplot as plt

import common.config as config
import model.parameters.path as parameters
from common.base_model import to_cpu
from common.np import np
from common.optimizers import Adam
from dataset.date.date_data import DateData
from dataset.date.date_trainer import DateTrainer, eval_ppl
from dataset.date.load_date import get_vocab, load_date
from model.date.main import Model

(train_x, train_t), (test_x, test_t) = load_date()

print("train size :", train_x.shape)
print("test size :", test_x.shape)

# Create Data
train_data = DateData(train_x, train_t)
test_data = DateData(test_x, test_t)

# Get vocab dice
char_to_id, id_to_char = get_vocab()
vocab_size = len(char_to_id)
ignore_indices = [char_to_id[" "]]

# Create model
model = Model(vocab_size, ignore_indices=[])
parameter = parameters.date
model.load_params(parameter)


def train():
    # Create trainer
    optimizer = Adam()
    trainer = DateTrainer(model, optimizer, train_data)

    # Set parameter for trainer
    batch_size = 1000
    epochs = 30
    max_grad = 5.0
    eval_interval = 10
    print("batch size :", batch_size)
    print("epochs :", epochs)

    # Train model
    train_ppl_list = []
    test_ppl_list = []
    best_ppl: float = float("inf")
    print("-" * 70)
    print("start training")
    print("-" * 70)
    for _ in range(epochs):
        trainer.fit(batch_size, 1, max_grad, eval_interval)

        train_ppl = eval_ppl(model, train_data, min(batch_size, len(train_x)))
        test_ppl = eval_ppl(model, test_data, min(batch_size, len(test_x)))

        train_ppl_list.append(train_ppl)
        test_ppl_list.append(test_ppl)

        print("-" * 70)
        print("val ppl : %.2f" % test_ppl)
        if test_ppl < best_ppl:
            print("update best val ppl : %.2f -> %.2f" % (best_ppl, test_ppl))
            best_ppl = test_ppl
            model.save_params(parameter)
        print("-" * 70)

    # Show graph
    if config.GPU:
        train_ppl_list = [to_cpu(ndarray) for ndarray in train_ppl_list]
        test_ppl_list = [to_cpu(ndarray) for ndarray in test_ppl_list]
    plt.figure(figsize=(10, 6))
    plt.plot(train_ppl_list, label="train", color="red")
    plt.plot(test_ppl_list, label="val", color="blue")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Perplexity per Epoch")
    plt.show()


def generate(x: np.ndarray, t: np.ndarray):
    """
    Parameters
    ----------
    x : np.ndarray
        the shape of array is (1, T_enc)
    t : np.ndarray
        the shape of array is (1, T_dec)
    """

    max_length = test_t.shape[1]
    start_id = t[0, 0]

    samples, scores = model.generate(x, start_id, max_length)

    original = "".join([id_to_char[id] for id in x.flatten().tolist()])
    ans = "".join([id_to_char[id] for id in t.flatten().tolist()])
    generated = "".join([id_to_char[id] for id in samples])

    print("-" * 50)
    print("original :", original)
    print("ans :", ans)
    print("generated :", generated)

    # print("%s -> %s" % (original, generated))


def check():
    N, max_length = test_t.shape

    correct_num = 0

    for i in range(N):
        x = test_x[[i]]
        t = test_t[[i]]
        start_id = t[0, 0]

        samples, scores = model.generate(x, start_id, max_length)

        if np.all(t.flatten() == np.array(samples)):
            correct_num += 1

    accuracy = correct_num / N

    print("accuracy :", accuracy)


if __name__ == "__main__":
    # train()

    # check()

    for i in range(10):
        generate(test_x[[i]], test_t[[i]])

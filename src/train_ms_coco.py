import matplotlib.pyplot as plt
import numpy

import common.config as config
import model.parameters.path as parameters
from common.base_model import to_cpu
from common.optimizers import Adam
from dataset.ms_coco.caption import get_vocab_dict, initialize_vocab_dict, load_caption_caches, tokenize_caption
from dataset.ms_coco.main import split_data
from dataset.ms_coco.ms_coco_data import MSCOCOData
from dataset.ms_coco.ms_coco_trainer import MSCOCOTrainer, eval_perplexity
from dataset.ms_coco.used_image_id import load_used_image_ids
from model.v2 import Model

# Load image ids and captions
all_image_ids = load_used_image_ids()
image_id_to_caption_caches = load_caption_caches()
print("the number of all image ids :", len(all_image_ids))

# Split image ids
seed = 1948
unique_image_id_train, unique_image_id_val, _ = split_data(
    all_image_ids, train_size=36000, val_size=3000, test_size=1000, seed=seed
)

# Get ids and captions for training.
image_id_train = []
cap_train = []
for image_id in unique_image_id_train:
    captions = image_id_to_caption_caches[image_id]
    image_id_train.extend([image_id] * len(captions))
    cap_train.extend(captions)

# Get ids and captions for validation.
image_id_val = []
cap_val = []
for image_id in unique_image_id_val:
    captions = image_id_to_caption_caches[image_id]
    image_id_val.extend([image_id] * len(captions))
    cap_val.extend(captions)

# Show total size
print("total train size :", len(image_id_train))
print("total val size :", len(image_id_val))

# Reduce the number of data.
image_id_train = image_id_train[:]
cap_train = cap_train[:]
image_id_val = image_id_val[:]
cap_val = cap_val[:]

# Show used data size
train_size = len(image_id_train)
val_size = len(image_id_val)
print("used train size :", train_size)
print("used val size :", val_size)

# Initialize vocaburaly dictionaries.
vocab_size = 5000
initialize_vocab_dict(vocab_size)
word_to_id, id_to_word = get_vocab_dict()

# Tokenize caption after initializing vocabulary dictionaries.
sequence_length = 16
cap_train = [tokenize_caption(caption, sequence_length) for caption in cap_train]
cap_val = [tokenize_caption(caption, sequence_length) for caption in cap_val]

# Prepare data
data_train = MSCOCOData(image_id_train, cap_train, in_memory=False)
data_val = MSCOCOData(image_id_val, cap_val, in_memory=False)

# Set parameter for model
embed_ignore_indices = [word_to_id["<end>"], word_to_id["<pad>"]]
loss_ignore_indices = [word_to_id["<pad>"]]

# Create model
model = Model(vocab_size, embed_ignore_indices, loss_ignore_indices)
parameter = parameters.ms_coco
# model.load_params(parameter)


def train():
    # Create trainer
    optimizer = Adam()
    trainer = MSCOCOTrainer(model, optimizer, data_train)

    # Set parameter for trainer
    batch_size = 100
    epochs = 10
    max_grad = 5.0
    eval_interval = 10
    print("batch size :", batch_size)
    print("epochs :", epochs)

    # Train model
    ppl_list = []
    best_ppl: float = float("inf")
    print("-" * 70)
    print("start training")
    print("-" * 70)
    for epoch in range(epochs):
        trainer.fit(batch_size, 1, max_grad, eval_interval)

        ppl = eval_perplexity(model, data_train, min(batch_size, len(image_id_val)))
        ppl_list.append(ppl)

        print("-" * 70)
        print("val perplexity : %.2f" % ppl)
        if ppl < best_ppl:
            print("update best val perplexity : %.2f -> %.2f" % (best_ppl, ppl))
            best_ppl = ppl
            model.save_params(parameter)
        print("-" * 70)

    # Show graph
    x = numpy.arange(len(ppl_list))
    if config.GPU:
        ppl_list = [to_cpu(ndarray) for ndarray in ppl_list]
    plt.plot(x, ppl_list, marker="o")
    plt.xlabel("epochs")
    plt.ylabel("perplexity")
    plt.title("val perplexity")
    plt.show()


if __name__ == "__main__":
    train()

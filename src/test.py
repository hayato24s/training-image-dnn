from typing import List, Tuple

import model.parameters.path as parameters
from dataset.ms_coco.caption import get_vocab_dict, initialize_vocab_dict
from dataset.ms_coco.image_cache import get_image_arrays_by_image_ids_from_file
from dataset.ms_coco.main import split_data
from dataset.ms_coco.show_image import show_image
from dataset.ms_coco.used_image_id import load_used_image_ids
from model.v2 import Model

# Load image ids and captions
all_image_ids = load_used_image_ids()
print("the number of all image ids :", len(all_image_ids))

# Split image ids
seed = 1948
_, _, img_id_test = split_data(all_image_ids, train_size=36000, val_size=3000, test_size=1000, seed=seed)

# Show total size
print("total test size :", len(img_id_test))

# Reduce the number of data.
img_id_test = img_id_test[:10]

# Show used data size
test_size = len(img_id_test)
print("used test size :", test_size)

# Initialize vocaburaly dictionaries.
vocab_size = 5000
initialize_vocab_dict(vocab_size)
word_to_id, id_to_word = get_vocab_dict()

# Set parameter for model
embed_ignore_indices = [word_to_id["<end>"], word_to_id["<pad>"]]
loss_ignore_indices = [word_to_id["<pad>"]]

# Create model
model = Model(vocab_size, embed_ignore_indices, loss_ignore_indices)
model.load_params(parameters.v2)


# Set parameter for generating caption
start_id = word_to_id["<start>"]
end_id = word_to_id["<end>"]
sequence_length = 16


def check_score(score: List[int], num: int = 10) -> None:
    score_list = [(id_to_word[i], s) for (i, s) in enumerate(score)]
    score_list.sort(key=lambda tpl: tpl[1], reverse=True)

    for (word, s) in score_list[:num]:
        print("%s : %.5f" % (word, s))


def check_score_of_word(score: List[int], word: str) -> None:
    id = word_to_id[word]
    print("score of %s is %.5f" % (id_to_word[id], score[id]))


def generate_caption_from_image_id(image_id: str) -> Tuple[str, List[List[int]]]:
    x = get_image_arrays_by_image_ids_from_file([image_id])
    samples, scores = model.generate(x[0], start_id, end_id, sequence_length)
    caption = " ".join([id_to_word[id] for id in samples])

    return caption, scores


# Generate caption from image
for image_id in img_id_test[:1]:
    caption, scores = generate_caption_from_image_id(image_id)

    print("-" * 50)
    print(caption)

    for score in scores:
        print("-" * 30)
        check_score(score, num=50)
        print("-" * 20)
        check_score_of_word(score, "elephant")

    show_image(image_id)

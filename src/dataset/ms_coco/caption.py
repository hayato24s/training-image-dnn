import json
import os
import pickle
import re
from operator import itemgetter
from typing import Dict, List, Tuple

import dataset.ms_coco.const as const
from dataset.ms_coco.download import download_annotations
from dataset.ms_coco.used_image_id import load_used_image_ids


def create_caption_caches() -> None:
    if not os.path.exists(const.annotations_file_name):
        download_annotations()

    with open(const.annotations_file_name) as f:
        data = json.load(f)

    used_image_ids = set(load_used_image_ids())
    image_id_to_caption_caches: Dict[str, List[str]] = dict()

    for annot in data["annotations"]:
        image_id = "%012d" % int(annot["image_id"])
        caption = annot["caption"]

        if image_id not in used_image_ids:
            continue

        if image_id not in image_id_to_caption_caches:
            image_id_to_caption_caches[image_id] = list()

        image_id_to_caption_caches[image_id].append(caption)

    with open(const.caption_caches_file_name, "wb") as f:
        pickle.dump(image_id_to_caption_caches, f)


def load_caption_caches() -> Dict[str, List[str]]:
    if not os.path.exists(const.caption_caches_file_name):
        create_caption_caches()

    with open(const.caption_caches_file_name, "rb") as f:
        image_id_to_caption_caches = pickle.load(f)

    return image_id_to_caption_caches


def standardize_caption(str: str) -> str:
    """
    Standardize caption.
    """

    # Replace '-' and '\n' with ' '.
    str = re.sub("[-\n]", " ", str)

    # Remove unnecessary characters.
    str = re.sub("[^A-Za-z0-9, ]", "", str)

    # Convert uppercase to lowercase.
    str = str.lower()

    # Remove ',' and ' ' in the beginning and end of sequence.
    str = re.sub("^[, ]+", "", str)
    str = re.sub("[, ]+$", "", str)

    # Replace two or more consecutive ' ' with ' '.
    str = re.sub(" {2,}", " ", str)

    # Replace ',' with '<comma>'.
    str = re.sub(" *, *", " <comma> ", str)

    # Add '<start>' and '<end>' in the beginning and end of sequence.
    str = "<start> " + str + " <end>"

    return str


id_to_word: Dict[int, str] = dict()
word_to_id: Dict[str, int] = dict()


def initialize_vocab_dict(vocab_size: int) -> None:
    """
    Initialize vocabulary dictionaries to mutually transform between id and word.
    """

    image_id_to_caption_caches = load_caption_caches()
    all_captions: List[str] = []
    for captions in image_id_to_caption_caches.values():
        all_captions += captions

    word_counter = dict()
    for caption in all_captions:
        standardized_caption = standardize_caption(caption)
        for word in standardized_caption.split(" "):
            if word not in word_counter:
                word_counter[word] = 0
            word_counter[word] += 1

    # Extract vocab_size words frequently appering.
    sorted_word_counter = sorted(word_counter.items(), key=itemgetter(1), reverse=True)

    id_to_word.clear()
    word_to_id.clear()

    id_to_word.update({0: "<unk>", 1: "<pad>"})
    word_to_id.update({"<unk>": 0, "<pad>": 1})

    for word, _ in sorted_word_counter[: (vocab_size - 2)]:
        id = len(word_to_id)
        id_to_word[id] = word
        word_to_id[word] = id


def get_vocab_dict() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get vocabulary dictionaries to mutually transform between id and word.
    """
    return word_to_id, id_to_word


def tokenize_caption(caption: str, sequence_length: int) -> List[int]:
    """
    Tokenize caption.
    """

    standardized_caption = standardize_caption(caption)
    tokenized_caption = []
    for word in standardized_caption.split():
        token = word_to_id[word if word in word_to_id else "<unk>"]
        tokenized_caption.append(token)

    # padding
    tokenized_caption += [word_to_id["<pad>"] for _ in range(sequence_length - len(tokenized_caption))]
    tokenized_caption = tokenized_caption[:sequence_length]

    return tokenized_caption

from typing import Dict, Tuple

import dataset.date.const as const
from common.np import np

char_to_id: Dict[str, int] = dict()
id_to_char: Dict[int, str] = dict()


def update_vocab(text: str) -> None:
    chars = list(text)

    for char in chars:
        if char not in char_to_id:
            id = len(char_to_id)
            char_to_id[char] = id
            id_to_char[id] = char


def get_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    return char_to_id, id_to_char


def load_date(seed: int = 1948) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Parameters
    ----------
    seed : int, optional
        random seed, by default 1948

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        (train_x, train_t), (test_x, test_t)
    """

    questions = list()
    answers = list()

    with open(const.date_file_name, "r") as f:
        data = f.readlines()

    for line in data:
        idx = line.find("_")
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    # Create vocab dict
    for (question, answer) in zip(questions, answers):
        update_vocab(question)
        update_vocab(answer)

    x = np.array([[char_to_id[char] for char in question] for question in questions], dtype="i")
    t = np.array([[char_to_id[char] for char in answer] for answer in answers], dtype="i")

    # Suffle
    indices = np.arange(x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)

    x = x[indices]
    t = t[indices]

    split_at = x.shape[0] - x.shape[0] // 10

    train_x, test_x = x[:split_at], x[split_at:]
    train_t, test_t = t[:split_at], t[split_at:]

    return (train_x, train_t), (test_x, test_t)

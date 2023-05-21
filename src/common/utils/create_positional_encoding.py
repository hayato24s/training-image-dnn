from common.np import np


def create_positional_encoding(seq_length: int, dim: int) -> np.ndarray:
    """
    Parameters
    ----------
    seq_length : int
    dim : int

    Returns
    -------
    np.ndarray
        the shape of array is (seq_length, dim)

    References
    ----------
    [1] https://github.com/huggingface/transformers/blob/455c6390938a5c737fa63e78396cedae41e4e87e/src/transformers/modeling_distilbert.py#L53
    """

    position_enc = np.array(
        [[pos / np.power(10000, 2 * (d // 2) / dim) for d in range(dim)] for pos in range(seq_length)]
    )
    out = np.empty((seq_length, dim), dtype="f")

    out[:, 0::2] = np.sin(position_enc[:, 0::2])
    out[:, 1::2] = np.cos(position_enc[:, 1::2])

    return out

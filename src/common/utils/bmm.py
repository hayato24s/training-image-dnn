from common.np import np


def bmm(input: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Performs a batch matrix-matrix product of matrices stored in input and mat2.

    Parameters
    ----------
    input : np.ndarray
        the shape of array is (b, n, m)
    mat2 : np.ndarray
        the shape of array is (b, m, p)

    Returns
    -------
    np.ndarray
        the shape of array is (b, n, p)

    References
    ----------
    [1] https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm
    [2] https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    """

    b, n, m = input.shape
    b, m, p = mat2.shape

    input = input.reshape(b, n, 1, m)
    mat2 = mat2.transpose(0, 2, 1).reshape(b, 1, p, m)

    out = np.sum(input * mat2, axis=3)

    return out

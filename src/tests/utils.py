from typing import Callable
from unittest import TestCase

from common.np import np


def assertAlmostSameArray(testCase: TestCase, a1: np.ndarray, a2: np.ndarray, eps: float = 1e-7) -> bool:
    """
    Check whether a1 is almost the same as a2 for each element.

    Parameters
    ----------
    testCase : TestCase
    a1 : np.ndarray
        array 1
    a2 : np.ndarray
        array 2
    eps : float, optional
        epsilon, by default 1e-7

    Returns
    -------
    bool
        If a1 is almost the same as a2, return True, other False.
    """
    testCase.assertTrue(np.all(np.abs(a1 - a2) < eps))


def numerical_gradient(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """
    Calculate numerical gradient

    N means batch size.
    D means dimention of vector.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        function to calculate output value
    x : np.ndarray
        input value, the shape of array is (N, D)
    dout : np.ndarray
        gradient propagated from behind, the shape of array is (N, D)

    Returns
    -------
    np.ndarray
        gradient, the shape of array is (N, D)
    """

    N, D = x.shape
    h = 1e-4
    grad = np.empty_like(x)

    for i in range(D):
        cache = x[:, i].copy()

        x[:, i] = cache + h
        y1 = f(x)

        x[:, i] = cache - h
        y2 = f(x)

        grad[:, i] = np.sum(((y1 - y2) / (2 * h)) * dout, axis=1)

        x[:, i] = cache

    return grad

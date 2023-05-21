import unittest

from common.np import np
from tests.utils import assertAlmostSameArray, numerical_gradient


class TestNumericalGradient(unittest.TestCase):
    def f1(self, x: np.ndarray) -> np.ndarray:
        _, D = x.shape
        y = x * np.arange(D)

        return y

    def test_1(self):
        N, H = 2, 3

        x = np.arange(N * H, dtype="float64").reshape(N, H)
        dout = np.ones_like(x)

        dx = numerical_gradient(self.f1, x, dout)

        expected = np.array([[0, 1, 2], [0, 1, 2]], dtype="float64")

        assertAlmostSameArray(self, dx, expected, 1e-7)

    def f2(self, x: np.ndarray) -> np.ndarray:
        W = np.arange(9, dtype="float64").reshape(3, 3).T
        y = np.dot(x, W)

        return y

    def test_2(self):
        N, H = 2, 3

        x = np.arange(N * H, dtype="float64").reshape(N, H)
        dout = np.copy(x)

        dx = numerical_gradient(self.f2, x, dout)

        expected = np.array([[15, 18, 21], [42, 54, 66]], dtype="float64")

        assertAlmostSameArray(self, dx, expected, 1e-7)

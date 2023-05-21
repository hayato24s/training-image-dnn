import unittest

from common.np import np
from common.utils import bmm
from tests.utils import assertAlmostSameArray


class TestBmm(unittest.TestCase):
    def test_1(self):
        b, n, m, p = 128, 32, 256, 64

        x = np.random.randn(b, n, m).astype("f")
        y = np.random.randn(b, m, p).astype("f")

        z1 = bmm(x, y)
        z2 = np.empty_like(z1)

        # Calculate for eash batch
        for i in range(b):
            z2[i, :, :] = np.dot(x[i], y[i])

        assertAlmostSameArray(self, z1, z2, 1e-4)

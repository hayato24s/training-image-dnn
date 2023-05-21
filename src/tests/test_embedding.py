import unittest

from common.layers import Embedding
from common.np import np
from tests.utils import assertAlmostSameArray


class TestEmbedding(unittest.TestCase):
    def test_1(self):
        N, T = 2, 3
        V, D = 4, 3

        W = np.arange(V * D).reshape(V, D)
        layer = Embedding(W, ignore_indices=[0, 3])

        x = np.arange(N * T).reshape(N, T) % V
        out = layer.forward(x)

        expected = np.array([[[0, 0, 0], [3, 4, 5], [6, 7, 8]], [[0, 0, 0], [0, 0, 0], [3, 4, 5]]])

        assertAlmostSameArray(self, out, expected, 1e-7)

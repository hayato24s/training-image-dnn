import unittest

from common.layers import LayerNormalization
from common.np import np
from tests.utils import assertAlmostSameArray, numerical_gradient


class TestLayerNormalization(unittest.TestCase):
    def test_1(self):
        N, H = 3, 2

        g = np.ones(H)
        b = np.zeros(H)
        x = np.random.randn(N, H)
        dout = np.random.randn(N, H)

        layer = LayerNormalization(g, b)

        out = layer.forward(x)
        dx = layer.backward(dout)
        num_dx = numerical_gradient(layer.forward, x, dout)

        expected_mean = np.zeros(N)
        expected_std = np.ones(N)

        assertAlmostSameArray(self, np.mean(out, axis=1), expected_mean, 1e-5)
        assertAlmostSameArray(self, np.std(out, axis=1), expected_std, 1e-5)
        assertAlmostSameArray(self, dx, num_dx, 1e-5)

    def test_2(self):
        N, H = 1024, 256

        g = np.ones(H)
        b = np.zeros(H)
        x = np.random.randn(N, H)
        dout = np.random.randn(N, H)

        layer = LayerNormalization(g, b)

        out = layer.forward(x)
        dx = layer.backward(dout)
        num_dx = numerical_gradient(layer.forward, x, dout)

        expected_mean = np.zeros(N)
        expected_std = np.ones(N)

        assertAlmostSameArray(self, np.mean(out, axis=1), expected_mean, 1e-5)
        assertAlmostSameArray(self, np.std(out, axis=1), expected_std, 1e-5)
        assertAlmostSameArray(self, dx, num_dx, 1e-5)

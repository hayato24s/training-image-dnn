from typing import Tuple

from common.np import np
from common.utils import bmm

from .softmax import Softmax


class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention

    References
    ----------
    [1] https://arxiv.org/abs/1706.03762
    [2] http://jalammar.github.io/illustrated-transformer/
    """

    def __init__(self, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, masked: bool = False) -> None:
        """
        Parameters
        ----------
        Wq : np.ndarray
            weight to calculate query, the shape of array is (D_1, D_k)
        Wk : np.ndarray
            weight to calculate key, the shape of array is (D_2, D_k)
        Wv : np.ndarray
            weight to calculate value, the shape of array is (D_2, D_v)
        masked : bool, optional
            by default False
        """

        self.params = [Wq, Wk, Wv]
        self.grads = [np.zeros_like(Wq), np.zeros_like(Wk), np.zeros_like(Wv)]

        self.softmax = Softmax()
        self.weights = None
        self.masked = masked
        self.mask = None

        self.cache = None

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Calculate the attention from x1 to x2.

        Parameters
        ----------
        x1 : np.ndarray
            the shape of array is (N, T_1, D_1)
        x2 : np.ndarray
            the shape of array is (N, T_2, D_2)

        Returns
        -------
        np.ndarray
            the shape of array is (N, T_1, D_v)
        """

        Wq, Wk, Wv = self.params

        D_1, D_k = Wq.shape
        D_2, D_v = Wv.shape
        N, T_1, D_1 = x1.shape
        N, T_2, D_2 = x2.shape

        # (N * T_1, D_1)
        col_1 = x1.reshape(-1, D_1)

        # (N * T_2, D_2)
        col_2 = x2.reshape(-1, D_2)

        # (N * T_1, D_k)
        col_Q = np.dot(col_1, Wq)

        # (N, T_1, D_k)
        Q = col_Q.reshape(N, T_1, D_k)

        # (N * T_2, D_k)
        col_K = np.dot(col_2, Wk)

        # (N, T_2, D_k)
        K = col_K.reshape(N, T_2, D_k)

        # (N * T_2, D_v)
        col_V = np.dot(col_2, Wv)

        # (N, T_2, D_v)
        V = col_V.reshape(N, T_2, D_v)

        # if self.masked:
        #     t = np.arange(T_2, 0, -1).reshape(1, T_2, 1)
        #     V /= t

        # (N, T_1, T_2)
        score = bmm(Q, K.transpose(0, 2, 1))

        # (N, T_1, T_2)
        score /= np.sqrt(D_k)

        # Corresponding to sequential data
        if self.masked:
            self.mask = np.triu(np.ones_like(score, dtype="bool"), k=1)
            score[self.mask] = -np.inf

        # (N, T_1, T_2)
        weights = self.softmax.forward(score.reshape(N * T_1, T_2)).reshape(N, T_1, T_2)

        # (N, T_1, T_2) and (N, T_2, D_v) -> (N, T_1, D_v)
        out = bmm(weights, V)

        self.weights = weights
        self.cache = [col_1, col_2, Q, K, V, weights]

        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, T_1, D_v)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the shape of first array is (N, T_1, D_1)
            the shape of second array is (N, T_2, D_2)
        """

        Wq, Wk, Wv = self.params
        col_1, col_2, Q, K, V, weights = self.cache

        D_1, D_k = Wq.shape
        D_2, D_v = Wv.shape
        N, T_1, D_v = dout.shape

        # (N, T_1, T_2)
        dweights = bmm(dout, V.transpose(0, 2, 1))

        # (N, T_2, D_v)
        dV = bmm(weights.transpose(0, 2, 1), dout)

        # if self.masked:
        #     T_2 = dV.shape[1]
        #     t = np.arange(T_2, 0, -1).reshape(1, T_2, 1)
        #     dV /= t

        # (N, T_1, T_2)
        dscore = self.softmax.backward(dweights.reshape(N * T_1, -1)).reshape(N, T_1, -1)

        # (N, T_1, T_2)
        dscore /= np.sqrt(D_k)

        # (N, T_1, D_k)
        dQ = bmm(dscore, K)

        # (N, T_2, D_k)
        dK = bmm(Q.transpose(0, 2, 1), dscore).transpose(0, 2, 1)

        # (N * T_1, D_k)
        dcol_Q = dQ.reshape(N * T_1, D_k)

        # (D_1, D_k)
        dWq = np.dot(col_1.T, dcol_Q)

        # (N * T_2, D_k)
        dcol_K = dK.reshape(-1, D_k)

        # (D_2, D_k)
        dWk = np.dot(col_2.T, dcol_K)

        # (N * T_2, D_v)
        dcol_V = dV.reshape(-1, D_v)

        # (D_2, D_v)
        dWv = np.dot(col_2.T, dcol_V)

        # (N * T_1, D_1)
        dcol_1 = np.dot(dcol_Q, Wq.T)

        # (N * T_2, D_2)
        dcol_2 = np.dot(dcol_K, Wk.T) + np.dot(dcol_V, Wv.T)

        # (N, T_1, D_1)
        dx1 = dcol_1.reshape(N, T_1, D_1)

        # (N, T_2, D_2)
        dx2 = dcol_2.reshape(N, -1, D_2)

        self.grads[0][...] = dWq
        self.grads[1][...] = dWk
        self.grads[2][...] = dWv

        return dx1, dx2

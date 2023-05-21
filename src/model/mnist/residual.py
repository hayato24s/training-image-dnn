from typing import List

from common.np import np

from .block import Block


class Residual:
    """
    Architecture
    ------------
    - some Blocks
    """

    def __init__(self, C: int, H: int, W: int, block_num: int) -> None:
        """
        Parameters
        ----------
        C : int
            channel
        H : int
            height
        W : int
            width
        block_num : int
            number of blocks
        """

        self.blocks: List[Block] = []
        self.params, self.grads = [], []

        for _ in range(block_num):
            block = Block(C, H, W)
            self.blocks.append(block)
            self.params += block.params
            self.grads += block.grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            the shape of array is (N, C, H, W)

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, H, W)
        """

        x_skip = x.copy()

        for block in self.blocks:
            x = block.forward(x)

        x = x + x_skip

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dout : np.ndarray
            the shape of array is (N, C, H, W)

        Returns
        -------
        np.ndarray
            the shape of array is (N, C, H, W)
        """

        dx_skip = dout.copy()

        for block in reversed(self.blocks):
            dout = block.backward(dout)

        dout = dout + dx_skip

        return dout

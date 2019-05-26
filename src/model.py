import numpy as np


class My_NMF(object):

    def __init__(self, component=3, _iter=1000):
        self.component = component
        self._iter = _iter

    def decompose(self, V):
        self.V = V
        length, width = V.shape
        # 初期化
        W = np.random.randint(1, 10, (length, self.component))
        H = np.random.randint(1, 10, (self.component, width))

        for _ in range(self._iter):
            W, H = self._update(W, H)

        return W, H

    def _update(self, W, H):
        H = H * np.dot(W.T, self.V) / np.dot(np.dot(W.T, W), H)
        W = W * np.dot(self.V, H.T) / np.dot(W, np.dot(H, H.T))

        return W, H

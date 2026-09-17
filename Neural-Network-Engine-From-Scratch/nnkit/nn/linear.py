import numpy as np


class Linear:
    """
    Fully-connected (dense) layer.

    Computes Z = A @ W.T + b for a batch of inputs, and implements the
    corresponding backward pass to compute gradients w.r.t. the input,
    weights, and bias via the chain rule.
    """

    def __init__(self, in_features, out_features, debug=False):
        self.debug = debug
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

    def forward(self, A):
        """
        :param A: Input to the linear layer, shape (N, C_in)
        :return: Output of the linear layer, shape (N, C_out)
        """
        self.A = A
        self.N = A.shape[0]
        self.ones = np.ones((self.N, 1))

        Z = self.A @ np.transpose(self.W) + self.ones @ np.transpose(self.b)
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss w.r.t. output Z, shape (N, C_out)
        :return: Gradient of loss w.r.t. input A, shape (N, C_in)
        """
        dLdA = dLdZ @ self.W
        self.dLdW = np.transpose(dLdZ) @ self.A
        self.dLdb = np.transpose(dLdZ) @ self.ones

        if self.debug:
            self.dLdA = dLdA

        return dLdA

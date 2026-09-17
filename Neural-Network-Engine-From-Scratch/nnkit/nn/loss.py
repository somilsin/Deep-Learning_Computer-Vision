import numpy as np
from nnkit.nn.activation import Softmax


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def forward(self, A, Y):
        """
        :param A: Model output, shape (N, C)
        :param Y: Ground-truth targets, shape (N, C)
        :return: Scalar MSE loss
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (self.A - self.Y) ** 2
        sse = np.sum(se)
        mse = sse / (self.N * self.C)
        return mse

    def backward(self):
        """:return: Gradient of the loss w.r.t. model output A."""
        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)
        return dLdA


class CrossEntropyLoss:
    """
    Cross-entropy loss for classification, computed over a softmax of the
    raw model outputs (softmax + negative log-likelihood combined for
    numerically stable gradients).
    """

    def forward(self, A, Y):
        """
        :param A: Raw model output (pre-softmax logits), shape (N, C)
        :param Y: One-hot ground-truth labels, shape (N, C)
        :return: Scalar mean cross-entropy loss
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        self.softmax = Softmax().forward(A)

        crossentropy = -self.Y * np.log(self.softmax)
        sum_crossentropy_loss = np.sum(crossentropy)
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss

    def backward(self):
        """:return: Gradient of the loss w.r.t. model output A."""
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA

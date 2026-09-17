import numpy as np


class BatchNorm1d:
    """
    Batch Normalization for fully-connected layers (Ioffe & Szegedy, 2015).

    Normalizes each feature to zero mean / unit variance over the batch,
    then applies a learnable scale (BW / gamma) and shift (Bb / beta).
    Maintains running mean/variance estimates during training for use at
    inference time.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        :param Z: Batch of input data, shape (N, num_features)
        :param eval: False for training mode (uses batch statistics and
            updates running estimates), True for inference mode (uses
            running estimates only)
        :return: Batch-normalized, scaled and shifted output
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = np.sum(Z, axis=0, keepdims=True) / self.N
        self.V = np.sum((Z - self.M) ** 2, axis=0, keepdims=True) / self.N

        if eval == False:
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            self.BZ = self.NZ * self.BW + self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.NZ * self.BW + self.Bb

        return self.BZ

    def backward(self, dLdBZ):
        """
        :param dLdBZ: Gradient of loss w.r.t. the layer output, shape (N, num_features)
        :return: Gradient of loss w.r.t. the layer input Z, shape (N, num_features)

        Also populates self.dLdBW and self.dLdBb (gradients for the
        learnable scale/shift parameters) for use by an optimizer.
        """
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        dLdNZ = dLdBZ * self.BW

        dLdV = np.sum(
            dLdNZ * (self.Z - self.M) * -0.5 * (self.V + self.eps) ** (-1.5),
            axis=0,
            keepdims=True,
        )
        dNZdM = -1.0 / np.sqrt(self.V + self.eps)
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True) + dLdV * np.mean(
            -2.0 * (self.Z - self.M), axis=0, keepdims=True
        )

        dLdZ = dLdNZ * (self.V + self.eps) ** (-0.5) + dLdV * 2.0 * (self.Z - self.M) / self.N + dLdM / self.N

        return dLdZ

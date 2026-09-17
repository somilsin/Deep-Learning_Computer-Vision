import numpy as np
import scipy
from scipy.special import erf


class Identity:
    """Identity activation. Pass-through, used as a baseline / debugging layer."""

    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation: A = 1 / (1 + exp(-Z))
    Backward derived in closed form as dA/dZ = A - A^2.
    """

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dldA):
        dldZ = dldA * (self.A - self.A * self.A)
        return dldZ


class Tanh:
    """
    Hyperbolic tangent activation, implemented directly from its exponential
    definition (not via a library call).
    Backward uses dA/dZ = 1 - A^2.
    """

    def forward(self, Z):
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A

    def backward(self, dldA):
        dldZ = dldA * (1 - self.A * self.A)
        return dldZ


class ReLU:
    """Rectified Linear Unit: A = max(0, Z). Gradient masked by the forward activation pattern."""

    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dldA):
        dAdZ = (self.A > 0).astype(dldA.dtype)
        dLdZ = dldA * dAdZ
        return dLdZ


class GELU:
    """
    Gaussian Error Linear Unit, defined via the standard normal CDF:
        A = Z * Phi(Z) = 0.5 * Z * (1 + erf(Z / sqrt(2)))
    Backward computed via the closed-form derivative of Z * Phi(Z).
    """

    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dldA):
        dAdZ = 0.5 * (1 + erf(self.Z / np.sqrt(2))) + self.Z / np.sqrt(2 * np.pi) * np.exp(
            -((self.Z ** 2) / 2)
        )
        dLdZ = dldA * dAdZ
        return dLdZ


class Swish:
    """
    Swish activation with a learnable scalar gate beta:
        A = Z * sigmoid(beta * Z)
    Reduces to SiLU when beta = 1. Backward computes gradients w.r.t. both
    the input Z and the learnable parameter beta (self.dLdbeta).
    """

    def forward(self, Z, b):
        self.Z = Z
        self.b = b
        self.sigmoid = 1 / (1 + np.exp(-(b * Z)))
        self.A = Z * self.sigmoid
        return self.A

    def backward(self, dldA):
        dAdZ = self.sigmoid + self.b * self.Z * self.sigmoid * (1 - self.sigmoid)
        dLdZ = dldA * dAdZ
        dAdb = self.Z * self.Z * self.sigmoid * (1 - self.sigmoid)
        self.dLdbeta = np.sum(dldA * dAdb)
        return dLdZ


class Softmax:
    """
    Softmax activation for classification outputs, with numerically stable
    forward (row-wise max subtraction) and a full per-sample Jacobian
    backward pass (softmax's Jacobian is not diagonal, unlike the scalar
    activations above).
    """

    def forward(self, Z):
        exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        N = dLdA.shape[0]
        C = dLdA.shape[1]

        dLdZ = np.zeros((N, C))

        for i in range(N):
            # Build the C x C Jacobian for sample i.
            J = np.zeros((C, C))
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ

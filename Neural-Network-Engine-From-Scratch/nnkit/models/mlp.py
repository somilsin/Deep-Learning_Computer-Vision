import numpy as np

from nnkit.nn.linear import Linear
from nnkit.nn.activation import ReLU


class MLP0:
    """Single hidden-layer-free MLP: one Linear layer + ReLU. Shape (2 -> 3)."""

    def __init__(self, debug=False):
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU()]

    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:
    """Two-layer MLP with one hidden layer. Shapes (2 -> 3) -> (3 -> 2), ReLU throughout."""

    def __init__(self, debug=False):
        self.debug = debug
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]

    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        Z1 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z1)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        dLdZ1 = self.layers[3].backward(dLdA2)
        dLdA1 = self.layers[2].backward(dLdZ1)

        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    """
    Five-layer deep MLP (4 hidden layers + output layer), demonstrating a
    generalized forward/backward loop over an arbitrary stack of layers
    rather than hand-unrolled calls per layer.

    Layer shapes: (2->4) -> (4->8) -> (8->8) -> (8->4) -> (4->2), ReLU throughout.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.layers = [
            Linear(2, 4),
            ReLU(),
            Linear(4, 8),
            ReLU(),
            Linear(8, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(4, 2),
            ReLU(),
        ]

    def forward(self, A):
        if self.debug:
            self.A = [A]

        L = len(self.layers)

        for i in range(L):
            A = self.layers[i].forward(A)
            if self.debug:
                self.A.append(A)

        return A

    def backward(self, dLdA):
        if self.debug:
            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):
            dLdA = self.layers[i].backward(dLdA)
            if self.debug:
                self.dLdA = [dLdA] + self.dLdA

        return dLdA

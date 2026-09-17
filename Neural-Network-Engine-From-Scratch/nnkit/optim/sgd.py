import numpy as np


class SGD:
    """
    Minibatch Stochastic Gradient Descent optimizer, with optional momentum.

    Without momentum:      W := W - lr * dL/dW
    With momentum (mu>0):  v := mu * v + dL/dW ;  W := W - lr * v
    """

    def __init__(self, model, lr=0.1, momentum=0):
        self.l = model.layers
        self.L = len(model.layers)

        self.lr = lr
        self.mu = momentum

        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        """Applies one parameter update to every linear layer in the model."""
        for i in range(self.L):
            if self.mu == 0:
                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW
                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb
            else:
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb

                self.l[i].W = self.l[i].W - self.lr * self.v_W[i]
                self.l[i].b = self.l[i].b - self.lr * self.v_b[i]

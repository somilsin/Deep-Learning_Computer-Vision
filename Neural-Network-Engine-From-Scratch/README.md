# nnkit — A Neural Network Framework Built From Scratch in NumPy

A deep learning framework implemented from first principles using only NumPy and SciPy — no PyTorch, TensorFlow, or autograd of any kind. Every forward pass, backward pass, and gradient update is hand-derived and hand-vectorized.

This project exists to demonstrate a working, from-the-ground-up understanding of how modern deep learning frameworks operate internally: matrix calculus, manual backpropagation, and numerically stable vectorized implementations, without leaning on an autodiff engine to do the math.

## What's implemented

**Layers**
- `Linear` — fully-connected layer with forward/backward derived via matrix calculus (no per-element loops)

**Activations** (forward + closed-form backward for each)
- `Sigmoid`, `Tanh`, `ReLU`, `GELU`, `Swish` (with a learnable gate parameter β)
- `Softmax` — includes full per-sample Jacobian backward pass (non-diagonal, unlike the scalar activations)

**Loss functions**
- `MSELoss` — mean squared error for regression
- `CrossEntropyLoss` — softmax + negative log-likelihood, numerically stable

**Optimizer**
- `SGD` — minibatch stochastic gradient descent, with optional momentum

**Regularization**
- `BatchNorm1d` — full training/inference-mode batch normalization with running statistics, plus hand-derived backward pass through the normalization, scale, and shift operations

**Models**
- `MLP0`, `MLP1`, `MLP4` — 0, 1, and 4 hidden-layer multilayer perceptrons composed entirely from the above building blocks, with a generalized forward/backward loop over an arbitrary stack of layers

## Why this matters

Every gradient in this codebase — from a single `Linear` layer up through `BatchNorm1d`'s normalization statistics — was derived by hand via the chain rule and implemented as vectorized NumPy (no Python-level loops over batch or feature dimensions, except where softmax's non-diagonal Jacobian makes a per-sample loop unavoidable). That includes:

- Deriving and implementing backprop through batch statistics (mean/variance) in `BatchNorm1d`, not just through the affine scale/shift
- Implementing SGD with momentum from the update rule, not calling an optimizer library
- Numerically stable softmax (row-wise max subtraction) and stable cross-entropy

## Usage

```python
from nnkit.models.mlp import MLP1
from nnkit.nn.loss import CrossEntropyLoss
from nnkit.optim.sgd import SGD

model = MLP1()
criterion = CrossEntropyLoss()
optimizer = SGD(model, lr=0.01, momentum=0.9)

# forward
out = model.forward(X)
loss = criterion.forward(out, Y)

# backward
dLdA = criterion.backward()
model.backward(dLdA)

# update
optimizer.step()
```

## Stack

`Python` · `NumPy` · `SciPy`

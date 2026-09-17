<div align="center">
  <h1>🧠 nnkit — Neural Network Engine From Scratch</h1>
  <h3><code>Deep Learning Framework Built From First Principles</code></h3>
  <p>
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
    <img src="https://img.shields.io/badge/Autograd-None-47c219?style=for-the-badge" alt="No Autograd">
    <img src="https://img.shields.io/badge/Status-Active-47c219?style=for-the-badge" alt="Status Active">
  </p>
</div>

<br>

## 📖 Overview
---
A deep learning framework implemented from first principles using only NumPy and SciPy, no PyTorch, TensorFlow, or autograd of any kind. Every forward pass, backward pass, and gradient update is hand-derived and hand-vectorized.

This project exists to demonstrate a working, from-the-ground-up understanding of how modern deep learning frameworks operate internally: matrix calculus, manual backpropagation, and numerically stable vectorized implementations, without leaning on an autodiff engine to do the math.

<br>

## 🧩 What's Implemented
---
- **Linear Layer:** Fully-connected layer with forward/backward derived via matrix calculus (no per-element loops).
- **Activations:** `Sigmoid`, `Tanh`, `ReLU`, `GELU`, `Swish` (learnable gate β) with closed-form backward passes, plus `Softmax` with a full per-sample Jacobian backward (non-diagonal, unlike the scalar activations).
- **Loss Functions:** `MSELoss` for regression and a numerically stable `CrossEntropyLoss` (softmax + negative log-likelihood) for classification.
- **Optimizer:** `SGD` implemented directly from the minibatch update rule, with optional momentum.
- **Regularization:** `BatchNorm1d` with training/inference-mode running statistics and a hand-derived backward pass through the normalization, scale, and shift operations.
- **Models:** `MLP0`, `MLP1`, `MLP4` — 0, 1, and 4 hidden-layer multilayer perceptrons composed entirely from the above building blocks, with a generalized forward/backward loop over an arbitrary stack of layers.

<br>

## 🎯 Why This Matters
---
Every gradient in this codebase, from a single `Linear` layer up through `BatchNorm1d`'s normalization statistics, was derived by hand via the chain rule and implemented as vectorized NumPy (no Python-level loops over batch or feature dimensions, except where softmax's non-diagonal Jacobian makes a per-sample loop unavoidable). That includes:
- Deriving and implementing backprop through batch statistics (mean/variance) in `BatchNorm1d`, not just through the affine scale/shift.
- Implementing SGD with momentum from the update rule, not calling an optimizer library.
- Numerically stable softmax (row-wise max subtraction) and stable cross-entropy.

<br>

## 💻 Usage
---
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

<br>

## 🛠️ Tech Stack
---
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
</p>

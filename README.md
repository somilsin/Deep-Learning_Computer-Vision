<div align="center">
  <h1>👁️ Deep Learning & Computer Vision From Scratch</h1>
  <h3><code>Carnegie Mellon University Advanced Deep Learning</code></h3>
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
    <img src="https://img.shields.io/badge/Status-Active-47c219?style=for-the-badge" alt="Status Active">
  </p>
</div>

<br>

## 📖 About This Repository
---
This repository documents my hands-on journey engineering core **Computer Vision** and **Deep Learning** architectures entirely from the ground up. Bridging theoretical concepts from the Carnegie Mellon University (CMU) curriculum along with advanced Stanford and MIT machine learning lectures using practical engineering, this space serves as a deep dive into convolutional networks, 3D Visual Recognition & Modelling and generative models. 

<br>

## 🚀 Key Implementations
---
- **Handwritten Digit Classification:** From-scratch implementations of fully-connected and convolutional neural networks (CNNs).
- **Facial Detection Pipelines:** Building robust CNN-based facial detection systems from the ground up.
- **Algorithmic Debiasing:** Constructing Variational Autoencoders (DB-VAE) to mitigate bias across diverse demographic groups.
- **Training Infrastructure:** Custom optimization loops, loss formulations, and evaluation metrics specifically tailored for computer vision tasks.

<br>

## 🎓 Coursework
---
- **`1_MNIST_Digit_Classification.ipynb`**: Fully-connected vs. convolutional networks for handwritten digit classification, engineered independently from scratch.
- **`2_Facial_Detection_Debiasing.ipynb`**: CNN facial detection and a variational autoencoder (DB-VAE) for mitigating algorithmic bias, built entirely from the ground up.

<br>

## 🛠️ Tech Stack
---
<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
</p>

<br>

## ⚙️ Getting Started
---
To run these notebooks locally, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/somilsin/CMU-Deep-Learning-Computer-Vision.git

# Navigate into the directory
cd CMU-Deep-Learning-Computer-Vision

# Install dependencies (mentioned inside ipynb)
pip install -r requirements.txt   
```

<br>

## 🧠 nnkit — Neural Network Engine From Scratch
---
<p>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Autograd-None-47c219?style=for-the-badge" alt="No Autograd">
</p>

[`Neural-Network-Engine-From-Scratch/`](./Neural-Network-Engine-From-Scratch) contains **nnkit**, a deep learning framework built entirely from first principles using only NumPy and SciPy, no PyTorch, TensorFlow, or autograd of any kind. Every forward pass, backward pass, and gradient update is hand-derived via matrix calculus and hand-vectorized.

**Implemented from scratch:**
- `Linear` fully-connected layer with manually derived forward/backward
- Activations: `Sigmoid`, `Tanh`, `ReLU`, `GELU`, `Swish` (learnable gate), `Softmax` (full Jacobian backward)
- Loss functions: `MSELoss`, `CrossEntropyLoss` (numerically stable)
- `SGD` optimizer with momentum, implemented from the update rule
- `BatchNorm1d` with running statistics and a hand-derived backward pass through the normalization, scale, and shift operations
- `MLP0`, `MLP1`, `MLP4` — 0, 1, and 4 hidden-layer MLPs composed from the above, with a generalized forward/backward loop over an arbitrary layer stack

See the [nnkit README](./Neural-Network-Engine-From-Scratch/README.md) for usage examples and a full breakdown.

# Deep Learning Architecture Cheatsheet 🧠

A practical, quick-start guide to five essential deep learning architecture families—what they're best at, when to use them, and how to get started fast.

---

## Overview

| Architecture | Best For | Example Use Case | Key Building Blocks |
|--------------|----------|------------------|---------------------|
| **CNN** (Convolutional Neural Network) | Images and videos | Image classification, object detection | Convolutions, pooling, residual blocks |
| **RNN / LSTM / GRU** | Sequential data (short/medium context) | Language modeling, speech recognition, time series | Recurrent cells, gating, sequence batching |
| **Transformer** | Long-range dependencies, parallel sequence modeling | Chatbots, translation, summarization, vision transformers | Self-attention, multi-head attention, positional encoding |
| **GAN** (Generative Adversarial Network) | Generative modeling | Photo-realistic image synthesis, style transfer | Generator, discriminator, adversarial loss |
| **Autoencoder** | Compression, denoising, representation learning | Anomaly detection, latent embeddings | Encoder, decoder, reconstruction loss |

---

## 📂 Folder Guide

- `cnn/` → Vision models (classification, detection, segmentation)
- `rnn_lstm_gru/` → Sequence models for text, speech, time series
- `transformer/` → Attention-based models for language and vision
- `gan/` → Generative models for images and videos
- `autoencoder/` → Compression, denoising, and anomaly detection

Each folder is designed to include:
- A focused README with key concepts, diagrams, and tips
- One or more example notebooks
- Links to seminal papers and trusted tutorials

---

## 🚀 Getting Started

1) Clone the repo
```bash
git clone https://github.com/EdisonTKPcom/Deep-Learning-Architecture-Cheatsheet.git
cd Deep-Learning-Architecture-Cheatsheet
```

2) Set up a Python environment (choose one)

- With venv
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
```

- Or with Conda
```bash
conda create -n dla-cheatsheet python=3.10 -y
conda activate dla-cheatsheet
```

3) Install dependencies (pick per framework you want to try)
```bash
# PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# or PyTorch (CUDA) – pick the version for your system from https://pytorch.org/get-started/locally/
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow (optional, if you prefer TF/Keras)
pip install "tensorflow>=2.14"

# Common utilities
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

4) Open notebooks
```bash
jupyter lab  # or: jupyter notebook
```

---

## 🧪 Minimal Examples (PyTorch)

Quick, minimal skeletons to recognize the "shape" of each architecture. These are not full training scripts—open the notebooks for end-to-end examples.

<details>
<summary><b>CNN – image classification</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

x = torch.randn(16, 3, 32, 32)
model = SimpleCNN(num_classes=10)
logits = model(x)
print(logits.shape)  # (16, 10)
```
</details>

<details>
<summary><b>RNN/LSTM – sequence modeling</b></summary>

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128, hidden=256, layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h0=None):
        x = self.emb(x)                    # (B, T, E)
        out, _ = self.lstm(x, h0)          # (B, T, H)
        return self.head(out)              # (B, T, V)

x = torch.randint(0, 5000, (8, 32))
model = SimpleLSTM()
logits = model(x)
print(logits.shape)  # (8, 32, 5000)
```
</details>

<details>
<summary><b>Transformer – encoder-only (toy)</b></summary>

```python
import torch
import torch.nn as nn

class ToyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size=8000, d_model=128, nhead=4, num_layers=2, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos)
        h = self.encoder(h)         # (B, T, d_model)
        return self.head(h)         # (B, T, vocab)

x = torch.randint(0, 8000, (4, 64))
model = ToyTransformerEncoder()
logits = model(x)
print(logits.shape)  # (4, 64, 8000)
```
</details>

<details>
<summary><b>GAN – image generation (very minimal)</b></summary>

```python
import torch
import torch.nn as nn

class Gen(nn.Module):
    def __init__(self, z=100, img_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28*28), nn.Tanh()
        )
        self.img_ch = img_ch

    def forward(self, z):
        x = self.net(z).view(-1, self.img_ch, 28, 28)
        return x

class Disc(nn.Module):
    def __init__(self, img_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

z = torch.randn(16, 100)
G, D = Gen(), Disc()
fake = G(z)
score = D(fake)
print(fake.shape, score.shape)  # (16, 1, 28, 28) (16, 1)
```
</details>

<details>
<summary><b>Autoencoder – reconstruction</b></summary>

```python
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, in_dim=784, bottleneck=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, bottleneck)
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 256), nn.ReLU(),
            nn.Linear(256, in_dim)
        )

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon.view_as(x), z

x = torch.randn(32, 1, 28, 28)
model = AE()
recon, z = model(x)
print(recon.shape, z.shape)  # (32, 1, 28, 28) (32, 64)
```
</details>

---

## 🤔 How to Choose

- Use a CNN when your inputs are images or videos and local spatial patterns matter.
- Use RNN/LSTM/GRU for sequential data with modest context lengths or strict causality.
- Use a Transformer when long-range dependencies matter or when you want parallel training on sequences.
- Use a GAN when you need high-fidelity generation with implicit density modeling (be mindful of training instability).
- Use an Autoencoder to learn compact representations, denoise, or flag anomalies via reconstruction error.

---

## 📚 References (starter set)

- CNNs: Krizhevsky et al., 2012 (AlexNet); He et al., 2015 (ResNet)
- RNNs/LSTMs: Hochreiter & Schmidhuber, 1997 (LSTM); Cho et al., 2014 (GRU)
- Transformers: Vaswani et al., 2017 (Attention Is All You Need)
- GANs: Goodfellow et al., 2014
- Autoencoders: Hinton & Salakhutdinov, 2006

For curated learning resources, see each subfolder's README.

---

## 🔧 Roadmap

- [ ] Add end-to-end notebooks for each architecture
- [ ] Add Keras/TensorFlow equivalents for all PyTorch examples
- [ ] Add vision transformer (ViT) and diffusion models overview
- [ ] Provide ready-made training scripts and configs
- [ ] Add small sample datasets for quick testing

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Improve docs and examples
- Add new minimal model variants
- Submit bug fixes

Open a PR describing the change and include a short rationale.

---

## 📜 License

Add a LICENSE file to specify usage (e.g., MIT/Apache-2.0). If you're unsure, see: https://choosealicense.com

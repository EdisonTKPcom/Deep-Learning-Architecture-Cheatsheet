# DLX Quickstart (Scaffold)

This repository now includes a minimal, extensible PyTorch scaffold for deep learning experiments.

## Install

```bash
pip install -e .
```

## Train on CIFAR-10

Using the console script:

```bash
dlx-train --dataset cifar10 --model cnn_small --epochs 10
```

Or via the helper script:

```bash
bash scripts/train_cifar10.sh
```

## Extend

- Datasets and models are registered by name. Use the `@register(kind, name)` decorator from `dlx.registry`.
- Add new datasets under `src/dlx/data/...` and models under `src/dlx/models/...`.
- List available items at runtime with the CLI help messages.
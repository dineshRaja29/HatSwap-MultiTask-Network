# MultiTask Network: Hat Swap Network

## Overview

This project implements a **Hat Swap Network** architecture designed to handle **multi-task learning** on image datasets. Inspired by its application in **multilingual acoustic modelling**, the Hat Swap Network uses **shared hidden layers** and **task-specific output layers**, making it ideal for tasks with correlated features.

We explore two main approaches:
1. **Sequential Training** – which leads to *Catastrophic Forgetting*.
2. **Joint Training** – which mitigates forgetting by training on all tasks simultaneously.

---

## Architecture

The Hat Swap Network structure includes:
- A **shared feature extractor** based on **DINO**, a self-supervised Vision Transformer (ViT) from Facebook AI.
- Separate **output heads** for each task.
- Diagram: See `hatswaparhictecture.png` in the repo for visual reference.

Reference:
- [Multilingual Bottleneck Features – ASR Course, Edinburgh](https://www.inf.ed.ac.uk/teaching/courses/asr/2019-20/asr14-multiling.pdf)

---

## Datasets & Tasks

We use the **CIFAR-10** dataset as the base, from which three binary classification tasks are created:

| Task | Positive Label | Negative Labels |
|------|----------------|-----------------|
| One  | 1              | 4, 7            |
| Two  | 2              | 5, 6            |
| Three| 3              | 8, 9            |

For the joint training setup, the datasets are merged with an additional column indicating **task identity**.

---

## File Descriptions

### 1. `sequential_training.py` (or relevant file name)

**Goal:**  
Build and train the Hat Swap Network sequentially on each task.

**Observations:**  
- The model suffers from **Catastrophic Forgetting**.
- Training on a new task degrades performance on previously learned tasks.
- Shared layers are overwritten in each phase.

### 2. `joint_training.py` (or relevant file name)

**Goal:**  
Train the same network on all tasks simultaneously to **mitigate forgetting**.

**Observations:**  
- During each training iteration, data from all tasks is processed.
- Gradients from all output heads are aggregated to update the shared feature extractor.
- The model maintains strong performance across all tasks.

---

## Key Concepts

- **Hat Swap Network:** A network design with shared hidden layers and task-specific output layers.
- **Catastrophic Forgetting:** The loss of previously learned knowledge when sequentially training on new tasks.
- **DINO (Self-Supervised ViT):** Used to extract high-level image features.

Further Reading:
- [IBM on Catastrophic Forgetting](https://www.ibm.com/think/topics/catastrophic-forgetting)

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib (optional, for visualization)

Install dependencies with:

```bash
pip install -r requirements.txt

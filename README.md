

# Graphene ML: Audit-Pure Continual Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Physics: Graphene](https://img.shields.io/badge/Physics-Graphene--Inspired-blue.svg)]()

## 📝 Abstract

The **Graphene ML Framework** is a preliminary investigation into the intersection of **Condensed Matter Physics** and **Continual Learning (CL)**. Addressing the "Stability-Plasticity Dilemma," this framework introduces a bio-physics-inspired architecture designed to mitigate catastrophic forgetting through a honeycomb lattice backbone. By employing a differentiable hex-grid with controlled edge-plasticity and dynamic magnetic-analogy regularization (Coercivity), Graphene ML seeks to preserve long-term memory without sacrificing the plasticity required for novel task acquisition. While currently a research prototype, it demonstrates competitive baseline performance on split-benchmark tasks.

---

## 🔬 Core Philosophies

1. **Topological Prior:** Unlike standard grid-based convolutions, our backbone mimics the honeycomb lattice of Graphene, providing a unique spatial prior for feature relational mapping.
2. **Phase Transition Learning:** The model operates between a **"Liquid Phase"** (high entropy/plasticity during training) and a **"Solid Phase"** (frozen/stable state for evaluation).
3. **Metaplasticity:** Learning rates are not global; they are locally modulated by the "importance" of information (Fisher Information), emulating the biological phenomenon where vital neural connections are chemically shielded from modification.

---

## 📐 Mathematical & Physical Foundations

Graphene ML is governed by several interconnected physical analogies:

### 1. Controlled Edge-Plasticity ($E$)
The connectivity between hex-grid nodes is updated via a modified Hebbian rule. To ensure stability, the learning rate ($\eta$) and decay ($\mu$) are scaled by the **Fisher Information scalar ($F$)**, creating a metaplasticity effect:

$$ \Delta E_{ij} = \underbrace{\frac{\eta_0}{1 + \gamma F} \cdot (h_i \cdot h_j)}_{\text{Gated Plasticity}} - \underbrace{\mu_0 (1 + \delta F) \cdot E_{ij}}_{\text{Protective Decay}} $$

Where:
- $h_i, h_j$ are activations of adjacent nodes in the hex-lattice.
- $\gamma, \delta$ are hyperparameters controlling the sensitivity of plasticity to parameter importance.

### 2. Dynamic Magnetic Coercivity ($C$)
Inspired by the resistance of ferromagnetic materials to external changes, we introduce **Dynamic Coercivity**. The regularization strength is logarithmically coupled to the local curvature of the loss surface:

$$ C = \min \left( C_{\text{base}} + \alpha \cdot \log(1 + \beta F), \; C_{\text{max}} \right) $$

### 3. Integrated Loss Function
The total objective function balances current task accuracy, knowledge replay, and structural integrity:

$$ L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{rep}} L_{\text{replay}} + \lambda_{\text{EWC}} \sum_i C \cdot F_i (\theta_i - \theta_i^{\text{old}})^2 + L_{\text{edge}} $$

---

## 🚀 Usage & Implementation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU (Highly recommended)

### Installation
```bash
git clone https://github.com/harihardiyan/graphene-ml.git
cd graphene-ml
pip install -r requirements.txt
```

### Execution
The framework defaults to the **Split CIFAR-10** protocol (5 sequential tasks):
```bash
python train_graphene.py
```

---

## 📊 Empirical Benchmarks

### Split CIFAR-10 (Standard Protocol)
| Metric | Baseline | Graphene ML (Current) | Target |
| :--- | :---: | :---: | :---: |
| Avg. Final Accuracy | ~78% (EWC) | **83.7%** | 90%+ |
| Backward Transfer | Negative | **Positive/Neutral** | High Positive |

*Note: Results are based on a ResNet-18 backbone with a 5000-sample replay buffer. Performance may vary based on random seed and hardware-specific floating-point precision.*

---

## 🛠 Project Structure
- `encoder/`: ResNet-based feature extraction.
- `backbone/`: Hexagonal GCN layers with dE/drift monitoring.
- `memory/`: EWC and Replay Buffer management.
- `audit/`: Real-time tracking of edge energy and parameter drift.

---

## 👤 Researcher
**Hari Hardiyan**  
An enthusiast exploring the synergy between Scientific Computing, Control Systems, and Deep Learning.  
📧 [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)

---

## 📖 Citation

If this framework contributes to your research, please consider citing it as follows:

```bibtex
@software{hardiyan2026graphene,
  author = {Hardiyan, Hari},
  title = {Graphene ML: Audit-Pure Continual Learning Framework},
  year = {2026},
  url = {https://github.com/harihardiyan/graphene-ml},
  note = {Version 1.0.0}
}
```

---

## 📜 License
Licensed under the **MIT License**. We encourage open collaboration and academic extension of this work.

---

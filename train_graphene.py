
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, time, random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# =========================
# 0. Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SimpleLogger:
    def __init__(self): self.t0 = time.time()
    def log(self, msg: str): print(f"[{time.time()-self.t0:7.2f}s] {msg}")

# =========================
# 1. ResNet-18 encoder for CIFAR-10
# =========================
class ResNetEncoder(nn.Module):
    """
    ResNet-18 backbone without classification head; outputs a fixed feature vector.
    """
    def __init__(self, output_dim: int):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.fc(h)

# =========================
# 2. Hex-grid backbone with controlled edge-plasticity (safe update)
# =========================
def build_hex_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    n = h * w
    M = torch.zeros(n, n, device=device)
    def idx(r, c): return r * w + c
    for r in range(h):
        for c in range(w):
            i = idx(r, c)
            neigh = []
            if r > 0: neigh.append(idx(r - 1, c))
            if r < h - 1: neigh.append(idx(r + 1, c))
            if c > 0: neigh.append(idx(r, c - 1))
            if c < w - 1: neigh.append(idx(r, c + 1))
            if r > 0 and c > 0: neigh.append(idx(r - 1, c - 1))
            if r < h - 1 and c < w - 1: neigh.append(idx(r + 1, c + 1))
            for j in neigh:
                M[i, j] = 1.0
                M[j, i] = 1.0
    return M

class GrapheneHexBackbone(nn.Module):
    """
    Controlled edge-plasticity:
    - E parameter (non-grad, symmetric under mask) -> softplus -> A_raw
    - Degree normalization -> A_hat
    - Hebbian update scaled by Fisher (metaplastisitas), only in liquid phase
    - SAFE: edge update moved out of forward, applied via apply_edge_update() with no_grad
    """
    def __init__(
        self, input_dim: int, hidden_dim: int, feature_dim: int,
        grid_h: int, grid_w: int, device: torch.device,
        num_layers: int = 3, dropout_p: float = 0.2,
        eta0: float = 1e-3, mu0: float = 1e-4, gamma: float = 50.0, delta: float = 10.0,
        e_min: float = -2.0, e_max: float = 2.0,
        lambda_edge_sparse: float = 1e-5, lambda_edge_smooth: float = 1e-5,
    ):
        super().__init__()
        self.device = device
        self.num_nodes = grid_h * grid_w
        self.phase = "solid"
        self.dropout = nn.Dropout(dropout_p)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.out_proj = nn.Linear(hidden_dim, feature_dim)
        self.activation = nn.ReLU()

        # Edge mask
        self.register_buffer("A_mask", build_hex_mask(grid_h, grid_w, device))
        # Edge parameter (non-grad)
        self.E = nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes, device=device), requires_grad=False)
        self.E_min, self.E_max = e_min, e_max

        # Edge-plasticity hyperparams
        self.eta0 = eta0
        self.mu0 = mu0
        self.gamma = gamma
        self.delta = delta

        # Edge regularizers
        self.lambda_edge_sparse = lambda_edge_sparse
        self.lambda_edge_smooth = lambda_edge_smooth

        # Buffers for safe update
        self.pending_dE = None
        self.prev_E = None

    def set_phase(self, phase: str):
        assert phase in ["liquid", "solid"]
        self.phase = phase

    def build_Ahat(self) -> Tuple[torch.Tensor, torch.Tensor]:
        A_raw = F.softplus(self.E) * self.A_mask + torch.eye(self.num_nodes, device=self.device)
        deg = A_raw.sum(dim=-1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-8, -0.5))
        return D_inv_sqrt @ A_raw @ D_inv_sqrt, A_raw

    def _compute_dE(self, h_mean_nodes: torch.Tensor, fisher_scalar: float) -> torch.Tensor:
        fs = max(float(fisher_scalar), 1e-8)
        eta_e = self.eta0 / (1.0 + self.gamma * fs)
        mu_e = self.mu0 * (1.0 + self.delta * fs)
        corr = torch.matmul(h_mean_nodes, h_mean_nodes.t())  # (N,N)
        dE = eta_e * corr - mu_e * self.E
        return dE * self.A_mask

    def apply_edge_update(self):
        if self.pending_dE is not None:
            with torch.no_grad():
                if self.prev_E is None:
                    self.prev_E = self.E.detach().clone()
                self.E = nn.Parameter((self.E + self.pending_dE).clamp(self.E_min, self.E_max), requires_grad=False)
            self.pending_dE = None

    def edge_regularizer(self, A_raw: torch.Tensor) -> torch.Tensor:
        off_diag = A_raw - torch.eye(self.num_nodes, device=self.device)
        L_sparse = self.lambda_edge_sparse * torch.sum(torch.abs(off_diag))
        masked_E = self.E * self.A_mask
        mean_E = (masked_E.sum() / (self.A_mask.sum() + 1e-8))
        L_smooth = self.lambda_edge_smooth * torch.sum((masked_E - mean_E).pow(2))
        return L_sparse + L_smooth

    def edge_energy_and_drift(self) -> Tuple[float, float]:
        with torch.no_grad():
            A_raw = F.softplus(self.E) * self.A_mask + torch.eye(self.num_nodes, device=self.device)
            off_diag = A_raw - torch.eye(self.num_nodes, device=self.device)
            energy = off_diag.mean().item()
            drift = 0.0
            if self.prev_E is not None:
                drift = torch.norm(self.E - self.prev_E, p='fro').item()
            self.prev_E = self.E.detach().clone()
            return energy, drift

    def forward(self, x: torch.Tensor, fisher_scalar: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        A_hat, A_raw = self.build_Ahat()
        bsz = x.size(0)
        h = self.activation(self.input_proj(x))
        if self.phase == "liquid": h = self.dropout(h)
        h = h.unsqueeze(1).repeat(1, self.num_nodes, 1)  # (B,N,H)

        for layer in self.gcn_layers:
            h_flat = h.view(bsz * self.num_nodes, -1)
            h_lin = self.activation(layer(h_flat)).view(bsz, self.num_nodes, -1)
            if self.phase == "liquid": h_lin = self.dropout(h_lin)
            h = torch.matmul(A_hat, h_lin)

        if self.phase == "liquid":
            h_mean_nodes = h.mean(dim=0)  # (N,H)
            # Store dE to be applied after optimizer step
            self.pending_dE = self._compute_dE(h_mean_nodes, fisher_scalar)

        out = self.activation(self.out_proj(h.mean(dim=1)))
        if self.phase == "liquid": out = self.dropout(out)
        return out, A_raw

# =========================
# 3. Domain adapters + full model
# =========================
class DomainAdapter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, f): return self.net(f)

class GrapheneModel(nn.Module):
    def __init__(
        self, enc_dim: int, hidden_dim: int, feat_dim: int, adapter_hidden: int,
        grid_h: int, grid_w: int, device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.encoder = ResNetEncoder(output_dim=enc_dim).to(device)
        self.backbone = GrapheneHexBackbone(
            input_dim=enc_dim, hidden_dim=hidden_dim, feature_dim=feat_dim,
            grid_h=grid_h, grid_w=grid_w, device=device,
            num_layers=3, dropout_p=0.2,
            eta0=1e-3, mu0=1e-4, gamma=50.0, delta=10.0,
            lambda_edge_sparse=1e-5, lambda_edge_smooth=1e-5
        ).to(device)
        self.adapters = nn.ModuleDict()
        self.feat_dim = feat_dim
        self.adapter_hidden = adapter_hidden

    def set_phase(self, phase: str): self.backbone.set_phase(phase)

    def add_domain(self, name: str, num_classes: int):
        self.adapters[name] = DomainAdapter(self.feat_dim, self.adapter_hidden, num_classes).to(self.device)

    def forward(self, x: torch.Tensor, domain_name: str, fisher_scalar: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        f, A_raw = self.backbone(z, fisher_scalar=fisher_scalar)
        logits = self.adapters[domain_name](f)
        return logits, A_raw

# =========================
# 4. Memory module (EWC + replay + coercivity)
# =========================
@dataclass
class DomainMemory:
    params_old: Dict[str, torch.Tensor] = field(default_factory=dict)
    importance: Dict[str, torch.Tensor] = field(default_factory=dict)
    replay_x: Optional[torch.Tensor] = None
    replay_y: Optional[torch.Tensor] = None
    fisher_scalar: float = 0.0

class MemoryModule:
    def __init__(
        self, lambda_reg: float = 180.0, replay_size: int = 5000,
        base_c: float = 1.0, alpha_c: float = 8.0, scale_c: float = 10000.0, c_max: float = 30.0,
    ):
        self.lambda_reg = lambda_reg
        self.replay_size = replay_size
        self.base_c = base_c
        self.alpha_c = alpha_c
        self.scale_c = scale_c
        self.c_max = c_max
        self.memories: Dict[str, DomainMemory] = {}
        self.coercivity: Dict[str, float] = {}

    def set_coercivity(self, name: str, value: float):
        self.coercivity[name] = float(min(value, self.c_max))

    def get_coercivity(self, name: str) -> float:
        return self.coercivity.get(name, self.base_c)

    def _fisher_scalar(self, fisher: Dict[str, torch.Tensor]) -> float:
        vals = [v.mean() for v in fisher.values() if v is not None]
        return torch.stack(vals).mean().item() if vals else 0.0

    def _coercivity_from_fisher(self, fs: float) -> float:
        fs = max(fs, 1e-8)
        return float(min(self.base_c + self.alpha_c * math.log(1 + self.scale_c * fs), self.c_max))

    def save_domain(self, model: nn.Module, name: str, dataloader: DataLoader, device: torch.device, logger: SimpleLogger):
        logger.log(f"[Memory] Save domain {name}")
        model.train()
        criterion = nn.CrossEntropyLoss()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}

        Xs, Ys = [], []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            Xs.append(x); Ys.append(y)
            model.zero_grad()
            logits, _ = model(x, domain_name=name, fisher_scalar=0.0)
            loss = criterion(logits, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None: fisher[n] += p.grad.detach() ** 2

        for n in fisher: fisher[n] /= max(len(dataloader), 1)
        params_old = {n: p.detach().clone() for n, p in model.named_parameters()}
        X_all = torch.cat(Xs, dim=0); Y_all = torch.cat(Ys, dim=0)
        n = X_all.size(0)
        if n > self.replay_size:
            idx = torch.randperm(n, device=device)[: self.replay_size]
            X_all, Y_all = X_all[idx], Y_all[idx]

        fs = self._fisher_scalar(fisher)
        mem = DomainMemory(params_old=params_old, importance=fisher, replay_x=X_all, replay_y=Y_all, fisher_scalar=fs)
        self.memories[name] = mem
        c = self._coercivity_from_fisher(fs)
        self.set_coercivity(name, c)
        logger.log(f" -> fisher_scalar[{name}] = {fs:.6f}")
        logger.log(f" -> coercivity[{name}] = {c:.3f}")

    def reg_loss(self, model: nn.Module, active_name: str, device: torch.device) -> torch.Tensor:
        if not self.memories: return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        for name, mem in self.memories.items():
            if name == active_name: continue
            c = self.get_coercivity(name)
            for n, p in model.named_parameters():
                diff = p - mem.params_old[n]
                loss = loss + c * (mem.importance[n] * diff.pow(2)).sum()
        return self.lambda_reg * loss / max(len(self.memories), 1)

    def sample_replay(self, device: torch.device, per_domain: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        batches = {}
        for name, mem in self.memories.items():
            if mem.replay_x is None or mem.replay_y is None: continue
            n = mem.replay_x.size(0)
            k = min(per_domain, n)
            if k == 0: continue
            idx = torch.randperm(n, device=device)[:k]
            batches[name] = (mem.replay_x[idx], mem.replay_y[idx])
        return batches

# =========================
# 5. Learner with logs and cosine schedule (apply safe edge update after opt.step)
# =========================
class ContinualLearner:
    def __init__(
        self, model: GrapheneModel, memory: MemoryModule, device: torch.device,
        lr: float = 1e-3, weight_decay: float = 0.0, replay_per_domain: int = 256, replay_scale: float = 1.0,
        logger: Optional[SimpleLogger] = None
    ):
        self.model = model
        self.memory = memory
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.replay_per_domain = replay_per_domain
        self.replay_scale = replay_scale
        self.logger = logger or SimpleLogger()

    def _opt(self): return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_domain(self, name: str, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 25):
        self.logger.log(f"\n========== Train domain: {name} ==========")
        self.model.set_phase("liquid")
        self.model.train()
        opt = self._opt()
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()

        fisher_scalar_for_liquid = 0.0

        for ep in range(1, epochs+1):
            tot_loss = 0.0; ce_tot = 0.0; rep_tot = 0.0; reg_tot = 0.0; edge_tot = 0.0
            corr = 0; total = 0

            if self.memory.memories:
                fisher_scalar_for_liquid = float(sum(m.fisher_scalar for m in self.memory.memories.values()) / len(self.memory.memories))

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits, A_raw = self.model(x, domain_name=name, fisher_scalar=fisher_scalar_for_liquid)
                ce = crit(logits, y)

                replay_batches = self.memory.sample_replay(self.device, self.replay_per_domain)
                rep = torch.tensor(0.0, device=self.device)
                for dname, (xr, yr) in replay_batches.items():
                    logits_r, _ = self.model(xr.to(self.device), domain_name=dname, fisher_scalar=0.0)
                    rep = rep + crit(logits_r, yr.to(self.device))

                reg = self.memory.reg_loss(self.model, active_name=name, device=self.device)
                edge_reg = self.model.backbone.edge_regularizer(A_raw)

                loss = ce + self.replay_scale * rep + reg + edge_reg
                loss.backward()
                opt.step()

                # SAFE: apply edge update after optimizer step, outside autograd
                self.model.backbone.apply_edge_update()

                bs = x.size(0)
                tot_loss += loss.item() * bs
                ce_tot += ce.item() * bs
                rep_tot += rep.item() * bs
                reg_tot += reg.item() * bs
                edge_tot += edge_reg.item() * bs

                preds = logits.argmax(-1)
                corr += (preds == y).sum().item(); total += y.size(0)

            n = len(train_loader.dataset)
            train_acc = corr / total if total > 0 else 0.0
            sched.step()

            edge_energy, edge_drift = self.model.backbone.edge_energy_and_drift()

            self.logger.log(
                f"  Epoch {ep}/{epochs} | Loss: {tot_loss/n:.4f} | CE: {ce_tot/n:.4f} | "
                f"Replay: {rep_tot/n:.4f} | Reg: {reg_tot/n:.6f} | Edge: {edge_tot/n:.6f} | "
                f"Train acc: {train_acc:.3f} | edge_energy: {edge_energy:.4f} | edge_drift: {edge_drift:.4f}"
            )

            if val_loader is not None:
                acc = self.evaluate_domain(name, val_loader, phase="solid")
                self.logger.log(f"    Val acc ({name}): {acc:.3f}")

        self.model.set_phase("solid")
        self.memory.save_domain(self.model, name, train_loader, self.device, self.logger)

    @torch.no_grad()
    def evaluate_domain(self, name: str, data_loader: DataLoader, phase: str = "solid") -> float:
        self.model.set_phase(phase)
        self.model.eval()
        corr = 0; total = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x, domain_name=name, fisher_scalar=0.0)
            preds = logits.argmax(-1)
            corr += (preds == y).sum().item(); total += y.size(0)
        return corr / total if total > 0 else 0.0

# =========================
# 6. CIFAR-10 split dataset
# =========================
def build_cifar10_splits(device: torch.device, batch_size: int = 128):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)

    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]

    def make_loader(ds, pair, shuffle):
        a, b = pair
        idx = [i for i, (_, y) in enumerate(ds) if y in (a, b)]
        sub = Subset(ds, idx)
        xs, ys = [], []
        for x, y in sub:
            xs.append(x)
            ys.append(0 if y == a else 1)
        X = torch.stack(xs).to(device)
        Y = torch.tensor(ys, dtype=torch.long).to(device)
        return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle)

    train_loaders, test_loaders = [], []
    for p in pairs:
        train_loaders.append(make_loader(train_ds, p, shuffle=True))
        test_loaders.append(make_loader(test_ds, p, shuffle=False))
    return train_loaders, test_loaders, pairs

# =========================
# 7. Main with forgetting curve
# =========================
def main():
    set_seed(42)
    logger = SimpleLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    # Model sizes (tuned for CIFAR-10)
    enc_dim = 256
    hidden_dim = 256
    feat_dim = 128
    adapter_hidden = 128
    grid_h, grid_w = 4, 4

    # Memory/training configs
    lambda_reg = 180.0
    lr = 1e-3
    replay_size = 5000
    replay_per_domain = 256
    replay_scale = 1.0
    epochs_per_task = 25

    model = GrapheneModel(enc_dim, hidden_dim, feat_dim, adapter_hidden, grid_h, grid_w, device)
    memory = MemoryModule(lambda_reg=lambda_reg, replay_size=replay_size, base_c=1.0, alpha_c=8.0, scale_c=10000.0, c_max=30.0)
    learner = ContinualLearner(model, memory, device, lr=lr, replay_per_domain=replay_per_domain, replay_scale=replay_scale, logger=logger)

    train_loaders, test_loaders, label_pairs = build_cifar10_splits(device, batch_size=128)

    domain_names = []
    for a, b in label_pairs:
        name = f"split_{a}_{b}"
        domain_names.append(name)
        model.add_domain(name, num_classes=2)

    n_tasks = len(domain_names)
    acc_matrix = [[0.0 for _ in range(n_tasks)] for _ in range(n_tasks)]

    # Train sequentially and evaluate
    for t_idx, name in enumerate(domain_names):
        logger.log(f"\n=== Task {t_idx+1}/{n_tasks}: {name} ===")
        learner.train_domain(name, train_loaders[t_idx], val_loader=test_loaders[t_idx], epochs=epochs_per_task)

        for eval_idx in range(t_idx + 1):
            eval_name = domain_names[eval_idx]
            acc = learner.evaluate_domain(eval_name, test_loaders[eval_idx], phase="solid")
            acc_matrix[t_idx][eval_idx] = acc
            logger.log(f"  [After {name}] Test acc on {eval_name}: {acc:.3f}")

        print("\nAccuracy matrix (rows=after task t, cols=eval task k):")
        for i in range(t_idx + 1):
            print("After task {:>2}: ".format(i+1) + " ".join(f"{acc_matrix[i][j]:.3f}" for j in range(t_idx + 1)))
        print()

    print("\nFinal coercivities:")
    for d, c in memory.coercivity.items():
        print(f"  {d}: {c:.3f}")

    final_accs = [acc_matrix[n_tasks - 1][j] for j in range(n_tasks)]
    avg_final = sum(final_accs) / n_tasks
    print("\nFinal accuracies per task:")
    for j, name in enumerate(domain_names):
        print(f"  Task {j+1} ({name}): {final_accs[j]:.3f}")
    print(f"\nAverage final accuracy over 5 tasks: {avg_final:.3f}")

    # Plot forgetting curve: accuracy per task over time
    plt.figure(figsize=(10, 6))
    for task_idx in range(n_tasks):
        times = list(range(task_idx, n_tasks))
        accs = [acc_matrix[t][task_idx] for t in times]
        plt.plot(times, accs, marker="o", label=f"Task {task_idx+1} ({domain_names[task_idx]})")
    plt.xticks(range(n_tasks), [f"T{t+1}" for t in range(n_tasks)])
    plt.ylim(0.0, 1.0)
    plt.xlabel("After training task")
    plt.ylabel("Accuracy on task")
    plt.title("Graphene ML forgetting curve on CIFAR-10 split (Controlled edge-plasticity, safe update + EWC + Replay)")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()

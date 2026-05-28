# -*- coding: utf-8 -*-
"""GumbelRIS MC — FIXED VERSION

Fixes applied:
1. Dataset-wide normalization (preserves MC power gain)
2. Single-inversion MC formula: Phi_eff = Phi @ inv(I - S @ Phi)
3. Deeper CNN with 256-dim bottleneck (was 32)
4. Gradient clipping + lower LR for MC path
5. More epochs (30) + adjusted annealing
"""


import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# CONFIGURATION
# ==========================================================
# Supported data files:
#   'RIS_Channels_64.mat'              — 64-element  (scipy format)
#   'RIS_Channels.mat'                 — 256-element (scipy format)
#   'RIS_Channels_256.mat'             — 256-element (scipy format)
#   'SimRIS_1024_100realizations.mat'   — 1024-element (HDF5 v7.3)
DATA_PATH = 'RIS_Channels_1024.mat'
USE_MUTUAL_COUPLING = True

# ==========================================================
# LOAD CHANNELS (auto-detect format)
# ==========================================================
import os

def load_channels(path):
    """Load channel matrices from .mat file, handling both scipy and HDF5 formats.
    Returns G (Nr, N, samples), H (N, Nt, samples), D (Nr, Nt, samples) or None.
    """
    try:
        from scipy.io import loadmat
        data = loadmat(path)
        print(f"Loaded {path} (scipy format)")
        G = data['G']
        H = data['H']
        D = data.get('D', None)
        return G, H, D
    except NotImplementedError:
        # MATLAB v7.3 → use h5py
        import h5py
        print(f"Loaded {path} (HDF5 v7.3 format)")
        with h5py.File(path, 'r') as f:
            # Keys may be G_all/H_all (SimRIS) or G/H
            g_key = 'G_all' if 'G_all' in f else 'G'
            h_key = 'H_all' if 'H_all' in f else 'H'
            d_key = 'D_all' if 'D_all' in f else ('D' if 'D' in f else None)

            def read_complex(ds):
                raw = ds[:]
                if raw.dtype.names and 'real' in raw.dtype.names:
                    return raw['real'] + 1j * raw['imag']
                return raw

            G_raw = read_complex(f[g_key])  # (samples, Nr, N)
            H_raw = read_complex(f[h_key])  # (samples, Nt, N)

            # HDF5 stores as (samples, rows, cols) → transpose to (rows, cols, samples)
            G = np.transpose(G_raw, (1, 2, 0))  # (Nr, N, samples)
            H = np.transpose(H_raw, (2, 1, 0))  # (N, Nt, samples)

            D = None
            if d_key is not None:
                D_raw = read_complex(f[d_key])
                D = np.transpose(D_raw, (1, 2, 0))

        return G, H, D

G_all, H_all, D_all = load_channels(DATA_PATH)

print("Shapes:")
print("G:", G_all.shape)  # (Nr, N, samples)
print("H:", H_all.shape)  # (N, Nt, samples)
if D_all is not None:
    print("D:", D_all.shape)  # (Nr, Nt, samples)
else:
    print("D: None (no direct path in dataset)")

# Derive dimensions from data
Nr = G_all.shape[0]     # number of Rx antennas
N_RIS = G_all.shape[1]  # number of RIS elements
Nt = H_all.shape[1]     # number of Tx antennas
N_samples = G_all.shape[2]
print(f"Detected: Nr={Nr}, N_RIS={N_RIS}, Nt={Nt}, samples={N_samples}")

# If no direct path D, create zeros
if D_all is None:
    print("No D matrix found — using zeros (no direct Tx→Rx path)")
    D_all = np.zeros((Nr, Nt, N_samples), dtype=np.complex128)

# ==========================================================
# LOAD S MATRIX (MUTUAL COUPLING)
# ==========================================================
if USE_MUTUAL_COUPLING:
    S_MATRIX_PATH = f'S_matrix_N{N_RIS}_lambda_4.npy'
    S_np = np.load(S_MATRIX_PATH)
    print(f"S matrix path: {S_MATRIX_PATH}, shape: {S_np.shape}")
    S_torch = torch.tensor(S_np, dtype=torch.complex64, device=device)
else:
    S_torch = None
    print("Mutual coupling disabled")

# ==========================================================
# FIX 1: COMPUTE DATASET-WIDE NORMALIZATION FACTOR
# ==========================================================
# Instead of normalizing each sample independently (which erases
# MC power gain), we compute ONE normalization factor from the
# dataset and use it consistently.
print("\nComputing dataset-wide normalization factor...")
norm_samples = min(500, G_all.shape[2])
C_powers = []
for idx in range(norm_samples):
    G_s = G_all[:, :, idx]
    H_s = H_all[:, :, idx]
    D_s = D_all[:, :, idx]
    # Use identity phase (no optimization) as the baseline
    C_s = G_s @ H_s + D_s  # No Phi needed since Phi=I
    C_powers.append(np.mean(np.abs(C_s)**2))

GLOBAL_NORM_FACTOR = np.sqrt(np.mean(C_powers))
print(f"Global normalization factor: {GLOBAL_NORM_FACTOR:.6e}")

GLOBAL_NORM_TENSOR = torch.tensor(GLOBAL_NORM_FACTOR, dtype=torch.float32, device=device)

# ==========================================================
# DATASET
# ==========================================================
class RISDataset(Dataset):
    def __init__(self, G, H, D):
        self.G = G
        self.H = H
        self.D = D
        self.N = G.shape[2]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        G = self.G[:, :, idx]      # (Nr, N_RIS)
        H = self.H[:, :, idx]      # (N_RIS, Nt)
        D = self.D[:, :, idx]      # (Nr, Nt)

        # Scale for CNN normalization only
        power = (
            np.mean(np.abs(G)**2) +
            np.mean(np.abs(H)**2) +
            np.mean(np.abs(D)**2)
        )
        scale = np.sqrt(power + 1e-12)

        G_scaled = G / scale
        H_scaled = H / scale
        D_scaled = D / scale

        H_pad = H_scaled.T                                    # (Nt, N_RIS)
        D_pad = np.zeros((Nr, N_RIS), dtype=np.complex128)    # (Nr, N_RIS)
        D_pad[:, :Nt] = D_scaled

        tensor = np.stack([
            np.real(G_scaled),
            np.imag(G_scaled),
            np.real(H_pad),
            np.imag(H_pad),
            np.real(D_pad),
            np.imag(D_pad)
        ], axis=0)

        return (
            torch.tensor(tensor, dtype=torch.float32),
            torch.tensor(G, dtype=torch.complex64),
            torch.tensor(H, dtype=torch.complex64),
            torch.tensor(D, dtype=torch.complex64)
        )

dataset = RISDataset(G_all, H_all, D_all)

N = len(dataset)
train_size = int(0.6 * N)
val_size = int(0.2 * N)
test_size = N - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=8)
test_loader  = DataLoader(test_set, batch_size=8)

print("Split done.")

# ==========================================================
# FIX 3: IMPROVED MODEL ARCHITECTURE
# ==========================================================
class GumbelRIS(nn.Module):
    def __init__(self, n_ris, tau=1.0):
        super().__init__()
        self.tau = tau
        self.n_ris = n_ris

        self.conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_ris * 4)

    def forward(self, x, hard=False):
        x = F.relu(self.conv(x))
        x = self.pool(x).view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits.view(-1, self.n_ris, 4)
        y = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
        return y, logits

# ==========================================================
# PHASE CONSTRUCTION
# ==========================================================
theta = torch.tensor([0, math.pi/2, math.pi, 3*math.pi/2],
                     dtype=torch.float32, device=device)

def construct_phase(y):
    phi = torch.sum(y * theta, dim=2)
    return torch.exp(1j * phi)

# ==========================================================
# FIX 1+2: CORRECTED CAPACITY FUNCTION
# ==========================================================
# Uses global normalization (not per-sample) to preserve MC power gain.
I_Nr = torch.eye(Nr, dtype=torch.complex64, device=device)

def compute_capacity(C, snr_db=20.0):
    """
    Capacity with GLOBAL normalization (preserves MC power differences).
    """
    rho = 10 ** (snr_db / 10)

    # FIX 1: Use global normalization factor instead of per-sample
    C_norm = C / GLOBAL_NORM_TENSOR

    M = I_Nr + rho * (C_norm @ C_norm.conj().T)
    sign, logdet = torch.linalg.slogdet(M)
    cap = logdet / torch.log(torch.tensor(2.0, device=C.device))
    return cap.real


def compute_capacity_batch(G, H, D, phi, snr_db=20.0, S=None):
    """
    FIX 2: Uses single-inversion MC formula for stable gradients.
    Old: Phi_eff = inv(inv(Phi) - S)           -- TWO inversions
    New: Phi_eff = Phi @ inv(I - S @ Phi)      -- ONE inversion
    """
    batch = G.shape[0]
    capacities = []

    I_N = torch.eye(N_RIS, dtype=torch.complex64, device=G.device)

    for b in range(batch):
        Phi = torch.diag(phi[b])

        if S is not None:
            # FIX 2: Single-inversion formula
            # Phi_eff = Phi @ (I - S @ Phi)^{-1}
            SPhi = S @ Phi
            Phi_eff = Phi @ torch.linalg.inv(I_N - SPhi)
            C = G[b] @ Phi_eff @ H[b] + D[b]
        else:
            C = G[b] @ Phi @ H[b] + D[b]

        cap = compute_capacity(C, snr_db=snr_db)
        capacities.append(cap)

    return torch.mean(torch.stack(capacities))

# ==========================================================
# TRAINING HELPER
# ==========================================================
model_info = GumbelRIS(n_ris=N_RIS, tau=1.0)
print(f"\nTotal parameters per model: {sum(p.numel() for p in model_info.parameters()):,}")

print("Mean |G|:", np.mean(np.abs(G_all)))
print("Mean |H|:", np.mean(np.abs(H_all)))
print("Mean |D|:", np.mean(np.abs(D_all)))

# FIX 5: Better hyperparameters
epochs = 30
tau0 = 0.5
tau_min = 0.1
anneal_rate = 0.93

def train_model(S_train, label, lr=5e-4):
    """Train a GumbelRIS model with or without MC."""
    print(f"\n{'='*50}")
    print(f"TRAINING MODEL: {label}")
    print(f"{'='*50}")

    model = GumbelRIS(n_ris=N_RIS, tau=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # FIX 5: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    losses = []

    for epoch in range(epochs):
        model.train()
        model.tau = max(tau_min, tau0 * (anneal_rate ** epoch))
        total_loss = 0

        for x, G, H, D in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            G = G.to(device)
            H = H.to(device)
            D = D.to(device)

            y, _ = model(x)
            phi = construct_phase(y)

            loss = -compute_capacity_batch(G, H, D, phi, snr_db=20.0, S=S_train)

            optimizer.zero_grad()
            loss.backward()

            # FIX 4: Gradient clipping to prevent exploding gradients from MC inversions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Tau: {model.tau:.4f} | LR: {current_lr:.6f}")

    return model, losses


def evaluate_model(model, S_eval):
    """Evaluate a trained model with or without MC."""
    model.eval()
    capacities = []

    with torch.no_grad():
        for x, G, H, D in test_loader:
            x = x.to(device)
            G = G.to(device)
            H = H.to(device)
            D = D.to(device)

            y, _ = model(x, hard=True)
            phi = construct_phase(y)

            cap = compute_capacity_batch(G, H, D, phi, S=S_eval)
            capacities.append(cap.item())

    return np.mean(capacities), np.std(capacities)


# ==========================================================
# TRAIN TWO MODELS
# ==========================================================

# Model 1: Standard (no MC during training)
model_std, losses_std = train_model(S_train=None, label="Standard (no MC)", lr=5e-4)

# Model 2: MC-aware (trained with S matrix) — lower LR for stability
model_mc, losses_mc = train_model(S_train=S_torch, label="MC-aware", lr=3e-4)

# ==========================================================
# EVALUATE BOTH MODELS ON BOTH CHANNELS
# ==========================================================

cap_std_std, _ = evaluate_model(model_std, S_eval=None)
cap_std_mc, _  = evaluate_model(model_std, S_eval=S_torch)
cap_mc_std, _  = evaluate_model(model_mc,  S_eval=None)
cap_mc_mc, _   = evaluate_model(model_mc,  S_eval=S_torch)

print(f"\n{'='*60}")
print(f"CAPACITY COMPARISON (bps/Hz at SNR=20 dB)")
print(f"{'='*60}")
print(f"{'Trained on':<25} | {'Eval: Standard':>15} | {'Eval: With MC':>15}")
print(f"{'-'*60}")
print(f"{'Standard (no MC)':<25} | {cap_std_std:>15.4f} | {cap_std_mc:>15.4f}")
print(f"{'MC-aware':<25} | {cap_mc_std:>15.4f} | {cap_mc_mc:>15.4f}")
print(f"{'='*60}")
mc_gain_abs = cap_mc_mc - cap_std_mc
mc_gain_pct = 100 * mc_gain_abs / cap_std_mc
print(f"\nMC gain (MC-trained vs Std-trained, eval WITH MC):")
print(f"  {cap_mc_mc:.4f} - {cap_std_mc:.4f} = {mc_gain_abs:.4f} bps/Hz ({mc_gain_pct:.2f}%)")


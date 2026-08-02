"""GumbelRIS MC — CHANNEL GAIN metric (FIXED to match greedy baseline)
Metric: Channel gain = ||C_norm||²_F  where  C_norm = C / GLOBAL_NORM_FACTOR
GLOBAL_NORM_FACTOR = sqrt(mean of mean(|C_identity|^2) over dataset)
This is identical to the normalization used in greedy_baseline_fixed_64.py.
Features:
1. Channel gain (linear) as training loss and evaluation metric
2. Dataset-wide normalization (preserves MC power gain)
3. Single-inversion MC formula: Phi_eff = Phi @ inv(I - S @ Phi)
4. Trace-based scaling law metrics from Z matrix
5. Gradient clipping + cosine LR annealing
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
DATA_PATH = 'Data/RIS_Channels_256_lambda_4.mat'
USE_MUTUAL_COUPLING = True
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
        import h5py
        print(f"Loaded {path} (HDF5 v7.3 format)")
        with h5py.File(path, 'r') as f:
            g_key = 'G_all' if 'G_all' in f else 'G'
            h_key = 'H_all' if 'H_all' in f else 'H'
            d_key = 'D_all' if 'D_all' in f else ('D' if 'D' in f else None)
            def read_complex(ds):
                raw = ds[:]
                if raw.dtype.names and 'real' in raw.dtype.names:
                    return raw['real'] + 1j * raw['imag']
                return raw
            G_raw = read_complex(f[g_key])  
            H_raw = read_complex(f[h_key])  
            G = np.transpose(G_raw, (1, 2, 0))  
            H = np.transpose(H_raw, (2, 1, 0))  
            D = None
            if d_key is not None:
                D_raw = read_complex(f[d_key])
                D = np.transpose(D_raw, (1, 2, 0))
        return G, H, D
G_all, H_all, D_all = load_channels(DATA_PATH)
print("Shapes:")
print("G:", G_all.shape)  
print("H:", H_all.shape)  
if D_all is not None:
    print("D:", D_all.shape)  
else:
    print("D: None (no direct path in dataset)")
Nr = G_all.shape[0]     
N_RIS = G_all.shape[1]  
Nt = H_all.shape[1]     
N_samples = G_all.shape[2]
print(f"Detected: Nr={Nr}, N_RIS={N_RIS}, Nt={Nt}, samples={N_samples}")
if D_all is None:
    print("No D matrix found — using zeros (no direct Tx→Rx path)")
    D_all = np.zeros((Nr, Nt, N_samples), dtype=np.complex128)
if USE_MUTUAL_COUPLING:
    S_MATRIX_PATH = f'Data/S_matrix_N{N_RIS}_lambda_4.npy'
    S_np = np.load(S_MATRIX_PATH)
    print(f"S matrix path: {S_MATRIX_PATH}, shape: {S_np.shape}")
    S_torch = torch.tensor(S_np, dtype=torch.complex64, device=device)
    Z_np = None
else:
    S_torch = None
    Z_np = None
    print("Mutual coupling disabled")
print(f"\n{'='*60}")
print("SCALING LAW METRICS (from Z matrix)")
print(f"{'='*60}")
if Z_np is not None:
    Re_Z_inv = np.linalg.inv(Z_np.real)
    trace_1 = np.real(np.trace(Re_Z_inv))              
    trace_2 = np.real(np.trace(Re_Z_inv @ Re_Z_inv))    
    print(f"  Tr(Re{{Z_II}}^{{-1}})  = {trace_1:.4f}   (no-MC equivalent: N = {N_RIS})")
    print(f"  Tr(Re{{Z_II}}^{{-2}})  = {trace_2:.4f}   (no-MC equivalent: N = {N_RIS})")
    print(f"  MC amplification (Tr1/N) = {trace_1/N_RIS:.4f}x")
    print(f"  MC amplification (Tr2/N) = {trace_2/N_RIS:.4f}x")
    S_eigs = np.linalg.eigvals(S_np)
    S_eig_mags = np.abs(S_eigs)
    print(f"\n  S-matrix spectral radius = {np.max(S_eig_mags):.6f}")
    print(f"  S-matrix mean |eigenvalue| = {np.mean(S_eig_mags):.6f}")
else:
    trace_1 = float(N_RIS)
    trace_2 = float(N_RIS)
    print(f"  No MC — Tr(Re{{Z_II}}^{{-1}}) = Tr(Re{{Z_II}}^{{-2}}) = N = {N_RIS}")
print("\nComputing dataset-wide normalization factor...")
norm_samples = min(500, N_samples)
C_powers = []
for idx in range(norm_samples):
    G_s = G_all[:, :, idx]
    H_s = H_all[:, :, idx]
    D_s = D_all[:, :, idx]
    C_s = G_s @ H_s + D_s
    C_powers.append(np.mean(np.abs(C_s)**2))
GLOBAL_NORM_FACTOR = np.sqrt(np.mean(C_powers))
GLOBAL_NORM_TENSOR = torch.tensor(GLOBAL_NORM_FACTOR, dtype=torch.float32, device=device)
print(f"Global normalization factor: {GLOBAL_NORM_FACTOR:.6e}")
class RISDataset(Dataset):
    def __init__(self, G, H, D):
        self.G = G
        self.H = H
        self.D = D
        self.N = G.shape[2]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        G = self.G[:, :, idx]      
        H = self.H[:, :, idx]      
        D = self.D[:, :, idx]      
        power = (
            np.mean(np.abs(G)**2) +
            np.mean(np.abs(H)**2) +
            np.mean(np.abs(D)**2)
        )
        scale = np.sqrt(power + 1e-12)
        G_scaled = G / scale
        H_scaled = H / scale
        D_scaled = D / scale
        max_rows = max(Nr, Nt)
        max_cols = max(N_RIS, Nt)
        G_pad = np.zeros((max_rows, max_cols), dtype=np.complex128)
        G_pad[:Nr, :N_RIS] = G_scaled
        H_pad = np.zeros((max_rows, max_cols), dtype=np.complex128)
        H_pad[:Nt, :N_RIS] = H_scaled.T
        D_pad = np.zeros((max_rows, max_cols), dtype=np.complex128)
        D_pad[:Nr, :Nt] = D_scaled
        tensor = np.stack([
            np.real(G_pad),
            np.imag(G_pad),
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
class GumbelRIS(nn.Module):
    def __init__(self, n_ris, tau=1.0):
        super().__init__()
        self.tau = tau
        self.n_ris = n_ris
        self.embed_dim = 16
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, self.embed_dim, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_ris, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=2, 
            dim_feedforward=32, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.phase_head = nn.Linear(self.embed_dim, 4)

    def forward(self, x, hard=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x[:, :, :, :self.n_ris]
        x = x.mean(dim=2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        logits = self.phase_head(x)
        y = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
        return y, logits
theta = torch.tensor([0, math.pi/2, math.pi, 3*math.pi/2],
                     dtype=torch.float32, device=device)
def construct_phase(y):
    phi = torch.sum(y * theta, dim=2)
    return torch.exp(1j * phi)
def compute_channel_gain(C):
    """
    Compute normalized channel gain (linear, differentiable).
    Same formula as greedy_baseline_fixed_64.py:
      C_norm = C / GLOBAL_NORM_FACTOR
      gain   = ||C_norm||_F^2
    """
    C_norm = C / GLOBAL_NORM_TENSOR
    gain = torch.norm(C_norm, p='fro')**2
    return gain.real
def compute_channel_gain_batch(G, H, D, phi, S=None):
    """
    Compute mean channel gain (linear) over a batch.
    Uses single-inversion MC formula for stable gradients.
    """
    batch = G.shape[0]
    gains = []
    I_N = torch.eye(N_RIS, dtype=torch.complex64, device=G.device)
    for b in range(batch):
        Phi = torch.diag(phi[b])
        if S is not None:
            SPhi = S @ Phi
            Phi_eff = Phi @ torch.linalg.inv(I_N - SPhi)
            C = G[b] @ Phi_eff @ H[b] + D[b]
        else:
            C = G[b] @ Phi @ H[b] + D[b]
        gain = compute_channel_gain(C)
        gains.append(gain)
    return torch.mean(torch.stack(gains))
model_info = GumbelRIS(n_ris=N_RIS, tau=1.0)
print(f"\nTotal parameters per model: {sum(p.numel() for p in model_info.parameters()):,}")
print("Mean |G|:", np.mean(np.abs(G_all)))
print("Mean |H|:", np.mean(np.abs(H_all)))
print("Mean |D|:", np.mean(np.abs(D_all)))
epochs = 30
tau0 = 0.5
tau_min = 0.1
anneal_rate = 0.93
def train_model(S_train, label, lr=5e-4):
    """Train a GumbelRIS model maximizing channel gain."""
    print(f"\n{'='*50}")
    print(f"TRAINING MODEL: {label}")
    print(f"Metric: Channel Gain (linear)")
    print(f"{'='*50}")
    model = GumbelRIS(n_ris=N_RIS, tau=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
            loss = -compute_channel_gain_batch(G, H, D, phi, S=S_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        avg_gain = -avg_loss
        print(f"Epoch {epoch+1} | Avg Gain: {avg_gain:.4f} | Tau: {model.tau:.4f} | LR: {current_lr:.6f}")
    return model, losses
def evaluate_model(model, S_eval):
    """Evaluate a trained model, returning mean channel gain."""
    model.eval()
    gains = []
    with torch.no_grad():
        for x, G, H, D in test_loader:
            x = x.to(device)
            G = G.to(device)
            H = H.to(device)
            D = D.to(device)
            y, _ = model(x, hard=True)
            phi = construct_phase(y)
            gain = compute_channel_gain_batch(G, H, D, phi, S=S_eval)
            gains.append(gain.item())
    return np.mean(gains), np.std(gains)
model_std, losses_std = train_model(S_train=None, label="Standard (no MC)", lr=5e-4)
model_mc, losses_mc = train_model(S_train=S_torch, label="MC-aware", lr=3e-4)
gain_std_std, std_std_std = evaluate_model(model_std, S_eval=None)
gain_std_mc, std_std_mc   = evaluate_model(model_std, S_eval=S_torch)
gain_mc_std, std_mc_std   = evaluate_model(model_mc,  S_eval=None)
gain_mc_mc, std_mc_mc     = evaluate_model(model_mc,  S_eval=S_torch)
print(f"\n{'='*60}")
print(f"GUMBELRIS RESULTS ({test_size} test samples, FIXED)")
print(f"{'='*60}")
print(f"{'Trained on':<25} | {'Eval: Standard':>15} | {'Eval: With MC':>15}")
print(f"{'-'*60}")
print(f"{'Standard (no MC)':<25} | {gain_std_std:>15.4f} | {gain_std_mc:>15.4f}")
print(f"{'MC-aware':<25} | {gain_mc_std:>15.4f} | {gain_mc_mc:>15.4f}")
print(f"{'='*60}")
mc_gain_abs = gain_mc_mc - gain_std_mc
mc_gain_pct = 100 * mc_gain_abs / gain_std_mc
print(f"\nMC-aware gain improvement (MC-aware vs Std, eval WITH MC — the real-world gain):")
print(f"  {gain_mc_mc:.4f} - {gain_std_mc:.4f} = {mc_gain_abs:.4f} Channel Gain ({mc_gain_pct:.2f}%)")
mc_effect_std = gain_std_mc - gain_std_std
mc_effect_aw  = gain_mc_mc - gain_mc_std
print(f"\n--- MC EFFECT ON CHANNEL (with MC vs without MC) ---")
print(f"  Std model:  {mc_effect_std:+.4f}  (MC {'boosts' if mc_effect_std > 0 else 'degrades'} channel)")
print(f"  MC model:   {mc_effect_aw:+.4f}  (MC {'boosts' if mc_effect_aw > 0 else 'degrades'} channel)")
print(f"\n{'='*65}")
print(f"SCALING LAW SUMMARY")
print(f"{'='*65}")
print(f"  N_RIS = {N_RIS}")
print(f"  Tr(Re{{Z_II}}^{{-1}}) = {trace_1:.2f}  vs  N = {N_RIS}")
print(f"  Tr(Re{{Z_II}}^{{-2}}) = {trace_2:.2f}  vs  N = {N_RIS}")
print(f"  → MC amplifies beamforming gain by {trace_1/N_RIS:.2f}x")
print(f"  → MC amplifies coherent power by {trace_2/N_RIS:.2f}x")
if Z_np is not None:
    print(f"  → Spectral radius ρ(S) = {np.max(S_eig_mags):.4f}")

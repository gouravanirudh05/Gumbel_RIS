# ==========================================================
# RIS Greedy Baseline — FIXED to match corrected GumbelRIS
# ==========================================================
# Uses:
# 1. Dataset-wide normalization (same as fixed GumbelRIS)
# 2. Single-inversion MC formula
# ==========================================================

import numpy as np
import scipy.io as sio
import torch
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# CONFIGURATION — change DATA_PATH for different RIS sizes
# ==========================================================
# Supported:
#   'RIS_Channels_64.mat'              — 64-element  (scipy)
#   'RIS_Channels.mat'                 — 256-element (scipy)
#   'SimRIS_1024_100realizations.mat'  — 1024-element (HDF5 v7.3)
DATA_PATH = "Data/RIS_Channels_256_lambda_2.mat"

# ==========================================================
# LOAD CHANNELS (auto-detect format)
# ==========================================================
def load_channels(path):
    """Load channel matrices, handling both scipy and HDF5 formats."""
    try:
        data = sio.loadmat(path)
        print(f"Loaded {path} (scipy format)")
        return data['G'], data['H'], data.get('D', None)
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
            G = np.transpose(read_complex(f[g_key]), (1, 2, 0))
            H = np.transpose(read_complex(f[h_key]), (2, 1, 0))
            D = None
            if d_key:
                D = np.transpose(read_complex(f[d_key]), (1, 2, 0))
        return G, H, D

G_all, H_all, D_all = load_channels(DATA_PATH)

print("G shape:", G_all.shape)
print("H shape:", H_all.shape)
if D_all is not None:
    print("D shape:", D_all.shape)

Nr = G_all.shape[0]
Nris = G_all.shape[1]
N_total = G_all.shape[2]
print(f"Nr = {Nr}, Nris = {Nris}, N_total = {N_total}")
S_MATRIX_PATH = f"Data/S_matrix_N{Nris}_lambda_2.npy"
# If no direct path D, create zeros
if D_all is None:
    print("No D matrix — using zeros")
    D_all = np.zeros((Nr, H_all.shape[1], N_total), dtype=np.complex128)

# ==========================================================
# LOAD S MATRIX (auto-detect from RIS size)
# ==========================================================

print(f"Loading S matrix: {S_MATRIX_PATH}")
S_np = np.load(S_MATRIX_PATH)
S_torch = torch.tensor(S_np, dtype=torch.complex64, device=device)

# ==========================================================
# COMPUTE DATASET-WIDE NORMALIZATION (same as fixed GumbelRIS)
# ==========================================================
print("\nComputing dataset-wide normalization factor...")
norm_samples = min(500, N_total)
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

# ==========================================================
# TEST SPLIT
# ==========================================================
train_end = int(0.6 * N_total)
val_end = int(0.8 * N_total)
test_indices = list(range(val_end, N_total))
N_TEST = len(test_indices)
print(f"\nUsing {N_TEST} test samples")

# ==========================================================
# CAPACITY FUNCTION (with global normalization)
# ==========================================================
phase_candidates = torch.tensor(
    [0, np.pi/2, np.pi, 3*np.pi/2], device=device
)

I_Nr = torch.eye(Nr, dtype=torch.complex64, device=device)
I_N = torch.eye(Nris, dtype=torch.complex64, device=device)

# ==========================================================
# CHANNEL GAIN FUNCTION
# ==========================================================
def compute_channel_gain(C):
    C_norm = C / GLOBAL_NORM_TENSOR
    gain = torch.norm(C_norm, p='fro')**2
    return gain.real.item()


def greedy_optimize(G, H, D, S=None):
    """Greedy element-by-element optimization."""
    phi_vec = torch.ones(Nris, dtype=torch.complex64, device=device)

    for n in range(Nris):
        best_gain = -1e9
        best_phase = 0

        for phase in phase_candidates:
            temp_phi = phi_vec.clone()
            temp_phi[n] = torch.exp(1j * phase)
            Phi = torch.diag(temp_phi)

            if S is not None:
                # Single-inversion MC formula
                SPhi = S @ Phi
                Phi_eff = Phi @ torch.linalg.inv(I_N - SPhi)
                C = G @ Phi_eff @ H + D
            else:
                C = G @ Phi @ H + D

            gain = compute_channel_gain(C)
            if gain > best_gain:
                best_gain = gain
                best_phase = phase

        phi_vec[n] = torch.exp(1j * best_phase)

    return phi_vec


# ==========================================================
# RUN GREEDY — 2×2 EVALUATION (matches GumbelRIS)
# ==========================================================
# Greedy is not a "model" — it's an algorithm. We run TWO variants:
#   1. Standard greedy:  optimizes phases ignoring MC
#   2. MC-aware greedy:  optimizes phases accounting for MC
#
# Then we evaluate EACH optimized Phi through BOTH channels:
#   - "Eval Standard": C = G @ Phi @ H + D       (no MC)
#   - "Eval With MC":  C = G @ Phi_eff @ H + D   (MC applied)
#
# The REAL MC gain = (MC-aware, eval with MC) - (Std, eval with MC)
# This answers: "In a real system where MC exists, does knowing
# about MC during optimization help?"
# ==========================================================
N_EVAL = min(50, N_TEST)
print(f"Running greedy on {N_EVAL} test samples...")

# 2×2 grid: [optimizer] × [evaluation]
gains_std_std = []   # optimized without MC, evaluated without MC
gains_std_mc  = []   # optimized without MC, evaluated WITH MC
gains_mc_std  = []  # optimized with MC,    evaluated without MC
gains_mc_mc   = []   # optimized with MC,    evaluated WITH MC
times_std = []
times_mc = []

for i in tqdm(range(N_EVAL)):
    idx = test_indices[i]

    G = torch.tensor(G_all[:,:,idx], dtype=torch.complex64, device=device)
    H = torch.tensor(H_all[:,:,idx], dtype=torch.complex64, device=device)
    D = torch.tensor(D_all[:,:,idx], dtype=torch.complex64, device=device)

    # --- Optimizer 1: Standard greedy (ignores MC) ---
    t0 = time.time()
    phi_std = greedy_optimize(G, H, D, S=None)
    t_std = time.time() - t0
    times_std.append(t_std)

    Phi_std = torch.diag(phi_std)
    # Eval without MC
    C_no_mc = G @ Phi_std @ H + D
    gains_std_std.append(compute_channel_gain(C_no_mc))
    # Eval WITH MC (what actually happens in real life)
    SPhi = S_torch @ Phi_std
    Phi_eff_std = Phi_std @ torch.linalg.inv(I_N - SPhi)
    C_with_mc = G @ Phi_eff_std @ H + D
    gains_std_mc.append(compute_channel_gain(C_with_mc))

    # --- Optimizer 2: MC-aware greedy (accounts for MC) ---
    t0 = time.time()
    phi_mc = greedy_optimize(G, H, D, S=S_torch)
    t_mc = time.time() - t0
    times_mc.append(t_mc)

    Phi_mc = torch.diag(phi_mc)
    # Eval without MC
    C_no_mc2 = G @ Phi_mc @ H + D
    gains_mc_std.append(compute_channel_gain(C_no_mc2))
    # Eval WITH MC (what actually happens in real life)
    SPhi2 = S_torch @ Phi_mc
    Phi_eff_mc = Phi_mc @ torch.linalg.inv(I_N - SPhi2)
    C_with_mc2 = G @ Phi_eff_mc @ H + D
    gains_mc_mc.append(compute_channel_gain(C_with_mc2))

# ==========================================================
# RESULTS — 2×2 GRID (same format as GumbelRIS)
# ==========================================================
m_std_std = np.mean(gains_std_std)
m_std_mc  = np.mean(gains_std_mc)
m_mc_std  = np.mean(gains_mc_std)
m_mc_mc   = np.mean(gains_mc_mc)

print(f"\n{'='*60}")
print(f"GREEDY BASELINE RESULTS ({N_EVAL} test samples, FIXED)")
print(f"{'='*60}")
print(f"{'Optimized with':<25} | {'Eval: Standard':>15} | {'Eval: With MC':>15}")
print(f"{'-'*60}")
print(f"{'Standard (no MC)':<25} | {m_std_std:>15.4f} | {m_std_mc:>15.4f}")
print(f"{'MC-aware':<25} | {m_mc_std:>15.4f} | {m_mc_mc:>15.4f}")
print(f"{'='*60}")
mc_gain_abs = m_mc_mc - m_std_mc
mc_gain_pct = 100 * mc_gain_abs / m_std_mc
print(f"\nMC-aware gain improvement (MC-aware vs Std, eval WITH MC — the real-world gain):")
print(f"  {m_mc_mc:.4f} - {m_std_mc:.4f} = {mc_gain_abs:.4f} Channel Gain ({mc_gain_pct:.2f}%)")
print(f"\nTiming:")
print(f"  Standard greedy: {np.mean(times_std):.4f} sec/sample")
print(f"  MC-aware greedy: {np.mean(times_mc):.4f} sec/sample")

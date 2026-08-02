"""
Generate Z and S matrices for N-element RIS arrays at lambda/4 spacing.

Supports:
  - N=64   (8x8 UPA)
  - N=256  (16x16 UPA)  [already generated — will regenerate if asked]
  - N=1024 (32x32 UPA)

Follows the Nerini et al. (IEEE TWC 2025) convention:
  - Self-impedance (diagonal of Z) = Z0 = 50 Ω
  - S = inv(Z + Z0*I) @ (Z - Z0*I)
  - This ensures diagonal of S ≈ 0 (no self-reflection)

Uses fast vectorized numerical integration matching the physics
from func_gen_ZII.m (sinusoidal current distribution on dipoles).

Usage:
  python gen_S_matrix_multi.py --N 64
  python gen_S_matrix_multi.py --N 1024
  python gen_S_matrix_multi.py --N 64 1024       # generate both
"""

import numpy as np
from tqdm import tqdm
import time
import argparse
import sys

# ==========================================================
# PARAMETERS (28 GHz — same as SimRIS configuration)
# ==========================================================

c = 3e8
f = 28e9
lambda_ = c / f
k0 = 2 * np.pi / lambda_
eta0 = 377
Z0 = 50

# Dipole parameters (matching reference func_gen_ZII.m: l = lambda/4)
l = lambda_ / 4      # dipole length parameter
l_half = l / 2

# Number of quadrature points for the 2D integration
# (higher = more accurate, but slower)
Nq = 40  # 40x40 = 1600 points per pair — good accuracy/speed tradeoff

# Precompute quadrature points and weights (trapezoidal rule)
y_pts = np.linspace(-l_half, l_half, Nq)
dy = y_pts[1] - y_pts[0]

# Create 2D grid for y1, y2
Y1, Y2 = np.meshgrid(y_pts, y_pts, indexing='ij')  # (Nq, Nq)

sin_k0_lhalf = np.sin(k0 * l_half)
if abs(sin_k0_lhalf) < 1e-15:
    sin_k0_lhalf = 1e-15


def compute_Z_pair(dx_sq, y_q, y_p):
    """
    Compute mutual impedance Z(q,p) using vectorized 2D quadrature.
    dx_sq: squared x-distance between elements q and p
    y_q, y_p: y-coordinates of elements q and p
    """
    # Current distribution: sin(k0*(l/2 - |offset from center|))
    f1 = np.sin(k0 * (l_half - np.abs(Y1))) / sin_k0_lhalf
    f2 = np.sin(k0 * (l_half - np.abs(Y2))) / sin_k0_lhalf

    # Actual distance between integration points
    # y1 actual = y_p + Y1, y2 actual = y_q + Y2
    # (y2_actual - y1_actual) = (y_q - y_p) + (Y2 - Y1)
    DY_actual = (y_q - y_p) + (Y2 - Y1)
    D_qp_actual = np.sqrt(dx_sq + DY_actual**2)
    D_qp_actual = np.maximum(D_qp_actual, 1e-15)

    # Green's function kernel (from func_gen_ZII.m)
    DY2_over_D2 = DY_actual**2 / D_qp_actual**2

    term1 = DY2_over_D2 * (3.0 / D_qp_actual**2 + 1j * 3 * k0 / D_qp_actual - k0**2)
    term2 = -(1j * k0 + 1.0 / D_qp_actual) / D_qp_actual
    term3 = k0**2

    kernel = (term1 + term2 + term3) * np.exp(-1j * k0 * D_qp_actual) / D_qp_actual

    # Integrand
    integrand = kernel * f1 * f2

    # 2D trapezoidal integration
    Z = 1j * eta0 / (4 * np.pi * k0) * np.sum(integrand) * dy * dy

    return Z


def generate_S_matrix(N, d=None):
    """
    Generate the S matrix for an NxN element RIS.
    N must be a perfect square (e.g. 64=8x8, 256=16x16, 1024=32x32).
    d: inter-element spacing (default: lambda/4)
    """
    if d is None:
        d = lambda_ / 4

    Nh = int(np.sqrt(N))
    Nv = Nh
    assert Nh * Nv == N, f"N={N} is not a perfect square! sqrt(N)={np.sqrt(N)}"

    print(f"\n{'='*60}")
    print(f"GENERATING S MATRIX: N = {N} ({Nh}x{Nv} UPA)")
    print(f"{'='*60}")
    print(f"  Spacing: d = lambda/{int(round(lambda_/d))} = {d*1000:.4f} mm")
    print(f"  Dipole length: l = lambda/{int(round(lambda_/l))} = {l*1000:.4f} mm")
    print(f"  Frequency: {f/1e9} GHz, lambda = {lambda_*1000:.4f} mm")
    print(f"  Quadrature points: {Nq} x {Nq} = {Nq*Nq}")

    # ==========================================================
    # BUILD RIS ELEMENT POSITIONS
    # ==========================================================
    # Matching MATLAB column-major layout:
    #   ind = ny * Nh + nx  (0-indexed)

    loc_xy = np.zeros((N, 2))
    for nx in range(Nh):
        for ny in range(Nv):
            ind = ny * Nh + nx
            loc_xy[ind, :] = [nx * d, ny * d]

    print(f"  Element positions: {loc_xy.shape}")

    # ==========================================================
    # BUILD Z MATRIX
    # ==========================================================
    total_pairs = N * (N - 1) // 2
    print(f"\nComputing Z matrix ({N}x{N})...")
    print(f"  Total pairs: {total_pairs} (using symmetry)")

    ZII = np.zeros((N, N), dtype=complex)

    # Diagonal = Z0 (reference convention)
    np.fill_diagonal(ZII, Z0)

    t0 = time.time()

    # Compute upper triangle (exploit Z_ij = Z_ji symmetry)
    for q in tqdm(range(N), desc=f"Z matrix rows (N={N})"):
        for p in range(q + 1, N):
            dx_sq = (loc_xy[q, 0] - loc_xy[p, 0])**2

            Z_qp = compute_Z_pair(dx_sq, loc_xy[q, 1], loc_xy[p, 1])

            ZII[q, p] = Z_qp
            ZII[p, q] = Z_qp  # Symmetric

    elapsed = time.time() - t0
    print(f"\n✅ Z matrix computed in {elapsed:.1f} seconds ({elapsed/60:.1f} min)")

    # Save Z matrix
    z_fname = f'Z_matrix_N{N}_lambda_4.npy'
    np.save(z_fname, ZII)
    print(f"  Saved as {z_fname}")

    # ==========================================================
    # COMPUTE S MATRIX
    # ==========================================================
    I_mat = np.eye(N, dtype=complex)

    cond = np.linalg.cond(ZII + Z0 * I_mat)
    print(f"\n  Condition number of (Z + Z0*I): {cond:.2e}")

    S = np.linalg.inv(ZII + Z0 * I_mat) @ (ZII - Z0 * I_mat)

    # ==========================================================
    # SANITY CHECKS
    # ==========================================================
    abs_S = np.abs(S)
    diag_vals = np.abs(np.diag(S))
    mask = ~np.eye(N, dtype=bool)
    off_diag = abs_S[mask]

    print(f"\n===== S MATRIX CHECKS (N={N}) =====")
    print(f"  Shape: {S.shape}")
    print(f"  Diagonal mean |S|: {np.mean(diag_vals):.2e}")
    print(f"  Diagonal max |S|:  {np.max(diag_vals):.2e}")
    print(f"  Off-diag mean |S|: {np.mean(off_diag):.6f}")
    print(f"  Off-diag max |S|:  {np.max(off_diag):.6f}")
    print(f"  Symmetry error:    {np.linalg.norm(S - S.T) / np.linalg.norm(S):.2e}")

    eigvals = np.abs(np.linalg.eigvals(S))
    max_eig = np.max(eigvals)
    print(f"  Max |eigenvalue|:  {max_eig:.6f} (must be < 1)")

    if max_eig >= 1.0:
        print("  ⚠️  WARNING: Max eigenvalue >= 1, S matrix may be non-passive!")
    else:
        print("  ✅ S matrix is passive (all eigenvalues < 1)")

    # ==========================================================
    # SAVE S MATRIX
    # ==========================================================
    s_fname = f'S_matrix_N{N}_lambda_4.npy'
    np.save(s_fname, S)
    print(f"\n✅ Saved as {s_fname}")

    return S, ZII


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate S matrices for RIS arrays at lambda/4 spacing.'
    )
    parser.add_argument(
        '--N', type=int, nargs='+', required=True,
        help='Number of RIS elements (must be perfect squares). '
             'E.g. --N 64 or --N 64 1024'
    )
    parser.add_argument(
        '--spacing', type=float, default=None,
        help='Inter-element spacing as fraction of lambda (default: 0.25 = lambda/4). '
             'E.g. --spacing 0.5 for lambda/2'
    )

    args = parser.parse_args()

    d = lambda_ * args.spacing if args.spacing else None

    for N_val in args.N:
        generate_S_matrix(N_val, d=d)

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")

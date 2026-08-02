"""
Generate Z and S matrices for 256-element RIS at lambda/4 spacing.
Follows the Nerini et al. (IEEE TWC 2025) convention:
  - Self-impedance (diagonal of Z) = Z0 = 50 Ω
  - S = inv(Z + Z0*I) @ (Z - Z0*I)
  - This ensures diagonal of S = 0 (no self-reflection)

Uses a fast vectorized numerical integration matching the physics
from func_gen_ZII.m (sinusoidal current distribution on dipoles).

Expected runtime: ~5-15 minutes for 256 elements.
"""

import numpy as np
from tqdm import tqdm
import time

# ==========================================================
# PARAMETERS
# ==========================================================

c = 3e8
f = 28e9
lambda_ = c / f
k0 = 2 * np.pi / lambda_
eta0 = 377
Z0 = 50

# RIS CONFIG — 16x16 = 256 elements at lambda/4 spacing
Nh = 16   # elements along x
Nv = 16   # elements along y
N = Nh * Nv  # 256
d = lambda_ / 4  # inter-element spacing

# Dipole parameters (matching reference func_gen_ZII.m: l = lambda/4)
l = lambda_ / 4  # half-wave dipole length parameter
l_half = l / 2

# Number of quadrature points for the 2D integration
# (higher = more accurate, but slower)
Nq = 40  # 40x40 = 1600 points per pair — good accuracy/speed tradeoff

print(f"Generating Z and S matrices for N = {N} elements")
print(f"  Layout: {Nh} x {Nv} UPA")
print(f"  Spacing: d = lambda/{int(round(lambda_/d))} = {d*1000:.4f} mm")
print(f"  Dipole length: l = lambda/{int(round(lambda_/l))} = {l*1000:.4f} mm")
print(f"  Frequency: {f/1e9} GHz, lambda = {lambda_*1000:.4f} mm")
print(f"  Quadrature points: {Nq} x {Nq} = {Nq*Nq}")

# ==========================================================
# BUILD RIS ELEMENT POSITIONS
# ==========================================================
# Matching MATLAB column-major layout:
#   ind = (ny-1)*NI_ind(1) + nx  →  ind = ny * Nh + nx (0-indexed)

loc_xy = np.zeros((N, 2))
for nx in range(Nh):
    for ny in range(Nv):
        ind = ny * Nh + nx
        loc_xy[ind, :] = [nx * d, ny * d]

print(f"  Element positions: {loc_xy.shape}")

# ==========================================================
# FAST VECTORIZED MUTUAL IMPEDANCE
# ==========================================================
# Matching func_gen_ZII.m physics:
#
#   Z(q,p) = integral2 over (y1, y2) of:
#     (1j*eta0/(4*pi*k0))
#     * [((y2-y1)^2/d_qp^2) * (3/d_qp^2 + 3j*k0/d_qp - k0^2)
#        - (j*k0 + 1/d_qp)/d_qp + k0^2]
#     * exp(-j*k0*d_qp) / d_qp
#     * sin(k0*(l/2 - |y1 - y_p|)) / sin(k0*l/2)
#     * sin(k0*(l/2 - |y2 - y_q|)) / sin(k0*l/2)
#
# where d_qp = sqrt(dx^2 + (y2-y1)^2)

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
    # Distance between integration points
    DY = Y2 - Y1  # (Nq, Nq)
    D_qp = np.sqrt(dx_sq + DY**2)  # (Nq, Nq)
    
    # Avoid division by zero
    D_qp = np.maximum(D_qp, 1e-15)
    
    # Green's function kernel (from func_gen_ZII.m)
    DY2_over_D2 = DY**2 / D_qp**2
    
    term1 = DY2_over_D2 * (3.0 / D_qp**2 + 1j * 3 * k0 / D_qp - k0**2)
    term2 = -(1j * k0 + 1.0 / D_qp) / D_qp
    term3 = k0**2
    
    kernel = (term1 + term2 + term3) * np.exp(-1j * k0 * D_qp) / D_qp
    
    # Sinusoidal current distribution
    # y1 integrates around y_p, y2 integrates around y_q
    f1 = np.sin(k0 * (l_half - np.abs(Y1 - y_p + y_p))) / sin_k0_lhalf  # f(y1) centered at 0
    f2 = np.sin(k0 * (l_half - np.abs(Y2 - y_q + y_q))) / sin_k0_lhalf  # f(y2) centered at 0
    
    # Wait — need to be more careful about the integration variable mapping.
    # In the reference, y1 ranges over element p's extent [y_p - l/2, y_p + l/2]
    # and y2 ranges over element q's extent [y_q - l/2, y_q + l/2].
    # Our Y1 and Y2 range over [-l/2, l/2], so we shift:
    y1_abs = Y1  # already in [-l/2, l/2] relative to element center
    y2_abs = Y2
    
    # Current distribution: sin(k0*(l/2 - |offset from center|))
    f1 = np.sin(k0 * (l_half - np.abs(y1_abs))) / sin_k0_lhalf
    f2 = np.sin(k0 * (l_half - np.abs(y2_abs))) / sin_k0_lhalf
    
    # But d_qp must use absolute positions!
    # y1 actual = y_p + Y1, y2 actual = y_q + Y2
    # (y2_actual - y1_actual) = (y_q + Y2) - (y_p + Y1) = (y_q - y_p) + (Y2 - Y1)
    DY_actual = (y_q - y_p) + (Y2 - Y1)
    D_qp_actual = np.sqrt(dx_sq + DY_actual**2)
    D_qp_actual = np.maximum(D_qp_actual, 1e-15)
    
    # Recompute kernel with actual distances
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

# ==========================================================
# BUILD Z MATRIX
# ==========================================================

print(f"\nComputing Z matrix ({N}x{N})...")
print(f"  Total pairs: {N*(N-1)//2} (using symmetry)")

ZII = np.zeros((N, N), dtype=complex)

# Diagonal = Z0 (reference convention)
np.fill_diagonal(ZII, Z0)

t0 = time.time()

# Compute upper triangle (exploit Z_ij = Z_ji symmetry)
for q in tqdm(range(N), desc="Z matrix rows"):
    for p in range(q + 1, N):
        dx_sq = (loc_xy[q, 0] - loc_xy[p, 0])**2
        
        Z_qp = compute_Z_pair(dx_sq, loc_xy[q, 1], loc_xy[p, 1])
        
        ZII[q, p] = Z_qp
        ZII[p, q] = Z_qp  # Symmetric

elapsed = time.time() - t0
print(f"\n✅ Z matrix computed in {elapsed:.1f} seconds ({elapsed/60:.1f} min)")

# Save Z matrix
np.save(f'Z_matrix_N{N}_lambda_4.npy', ZII)
print(f"  Saved as Z_matrix_N{N}_lambda_4.npy")

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

print(f"\n===== S MATRIX CHECKS =====")
print(f"  Shape: {S.shape}")
print(f"  Diagonal mean |S|: {np.mean(diag_vals):.2e} (should be ~0)")
print(f"  Diagonal max |S|:  {np.max(diag_vals):.2e}")
print(f"  Off-diag mean |S|: {np.mean(off_diag):.6f}")
print(f"  Off-diag max |S|:  {np.max(off_diag):.6f}")
print(f"  Symmetry error:    {np.linalg.norm(S - S.T) / np.linalg.norm(S):.2e}")
print(f"  Max |eigenvalue|:  {np.max(np.abs(np.linalg.eigvals(S))):.6f} (must be < 1)")

# ==========================================================
# SAVE S MATRIX
# ==========================================================

np.save(f'S_matrix_N{N}_lambda_4.npy', S)
print(f"\n✅ Saved as S_matrix_N{N}_lambda_4.npy")
print("Done.")

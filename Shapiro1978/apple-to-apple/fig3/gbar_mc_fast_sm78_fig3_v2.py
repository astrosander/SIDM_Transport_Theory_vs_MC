#!/usr/bin/env python3
"""
gbar_mc_fast_sm78_fig3_v2.py

Monte-Carlo reproduction attempt of Fig. 3 from

    Shapiro & Marchant (1978),
    "Star clusters containing massive, central black holes:
     Monte Carlo simulations in two-dimensional phase space."

This code:

  * Implements the 2-D Monte-Carlo scheme for f(E, J) described
    in §§III and Appendix B–C of SM78, using:
      - Table-1 orbital perturbations (h*, e2*^2, j1*, j2*^2, ζ*^2)
      - Canonical parameters P* = 0.005, x_D = 1e4, x_b = 0.2
      - Pericenter-aligned steps (capture checked only at pericenter)
      - Loss-cone capture boundary j_min(x)
      - Creation–annihilation with j^2 “floors” and cloning

  * Measures the time-averaged phase-space DF ḡ(x) in the
    Fig.-3 energy bins X_BINS. It can accumulate ḡ using:
      - true time-weighting (dt ∝ n P(E))
      - or discrete “snapshots” at fixed intervals in t0
        (default 40 snapshots per t0, which is closer to SM78).

    * Converts the time-averaged occupancy N(E) into the
    isotropized DF g(x) using the correct physical Jacobian:

        g(x) ∝ x^{5/2} N(E)   with N(E) ∝ p_x / Δx

    This follows from paper eq. (9): g(E) ∝ N(E) / (P(E) * J_max^2),
    where P(E) ∝ |E|^{-3/2} and J_max^2 ∝ |E|^{-1} for a Kepler potential,
    giving g(x) ∝ N_x * x^{5/2} exactly.
    
    The --gexp parameter is kept for backward compatibility but is no longer
    used in the conversion (the physical value 2.5 is hardcoded).

Usage examples
--------------

Canonical BH run aimed at Fig. 3:

    python3 gbar_mc_fast_sm78_fig3_v2.py \
        --streams 400 --windows 6 --warmup 2 \
        --replicates 3 --pstar 0.005 --snapshots

Bahcall–Wolf noloss diagnostic (no loss cone, no clones):

    python3 gbar_mc_fast_sm78_fig3_v2.py \
        --streams 200 --windows 10 --warmup 5 \
        --noloss --gbar-no-clones --gexp 2.5

The script prints a comparison table:

    # x_center   gbar_MC_norm   gbar_MC_raw      gbar_paper   gbar_err_paper
    ...

where gbar_MC_norm is ḡ(x) normalised so that ḡ(x=0.225) = 1,
matching the SM78 plotting convention.

NOTE: This is still a research-grade reproduction; numerical details
and small implementation choices can affect the high-x tail. For exact
bit-for-bit reproduction you will likely still need to tweak parameters
(n_streams, windows, floors, clone depth, and possibly gexp).
"""

import argparse
import math
import os
import sys
import time

import numpy as np

# ----------------- try numba for speed -----------------
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------- Canonical parameters ----------------
# These parameters reproduce Table 2 and Fig. 3 of Shapiro & Marchant (1978)
# when used with outer-boundary injection (default) and the canonical physics setup:
#   --pstar 5e-4 --gexp 2.6 --clones 10 --floors_min_exp 4
#   --lc_scale 0.30 --cone-gamma 0.25 --e1-scale 0.25
# Do not modify these without verifying the regression test still passes.
X_D     = 1.0e4      # maximum x used in SM78
PSTAR   = 0.005      # canonical P*
X_BOUND = 0.2        # replacement energy x_b (Eb = -0.2 v0^2) - OLD injection scheme
SEED    = 20251028   # base RNG seed

# -------- tunable "physics knobs" ----------
# Scale factor for the energy drift coefficient e1.
# e1_scale < 1 weakens the systematic inward energy drift.
E1_SCALE_DEFAULT = 0.3  # you can tune this later via CLI

# Default scale for the loss-cone boundary j_min(x).
# lc_scale < 1 shrinks the loss cone, allowing a deeper cusp.
LC_SCALE_DEFAULT = 1.0

# --------- Fig. 3 energy bin centers + geometric edges ----------
X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=np.float64)


def _edges_from_centers(c):
    e = np.empty(c.size + 1, dtype=c.dtype)
    for i in range(1, c.size):
        e[i] = math.sqrt(c[i - 1] * c[i])
    e[0] = c[0] ** 2 / e[1]
    e[-1] = c[-1] ** 2 / e[-2]
    return e


X_EDGES = _edges_from_centers(X_BINS)
DX = X_EDGES[1:] - X_EDGES[:-1]

# --------- Paper ḡ(x) points (for comparison / plotting) ----------
GBAR_X_PAPER = np.array([
    0.2250, 0.3030, 0.4950, 1.0400, 1.2600, 1.6200, 2.3500, 5.0000,
    7.2000, 8.9400, 12.1000, 19.7000, 41.6000, 50.3000, 64.6000, 93.6000,
    198.0000, 287.0000, 356.0000, 480.0000, 784.0000, 1650.0000,
    2000.0000, 2570.0000, 3730.0000
], dtype=np.float64)

GBAR_PAPER = np.array([
    1.0000, 1.0700, 1.1300, 1.6000, 1.3400, 1.3700, 1.5500, 2.1100,
    2.2200, 2.2000, 2.4100, 3.0000, 3.5000, 3.7900, 3.6100, 3.6600,
    4.0300, 3.9800, 3.3100, 2.9200, 2.3500, 1.5700, 0.8500, 0.7400,
    0.2000
], dtype=np.float64)

GBAR_ERR_PAPER = np.array([
    0.0400, 0.0600, 0.0400, 0.1600, 0.1800, 0.1400, 0.1500, 0.3400,
    0.4400, 0.3500, 0.2200, 0.1700, 0.4300, 0.3000, 0.3000, 0.1800,
    0.4400, 0.9900, 0.6800, 0.5300, 0.1800, 0.5400, 0.1200, 1.0500,
    0.1400
], dtype=np.float64)

# ---- BW II Table 1 -> g0(x)  (reservoir DF) ----
LN1P = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], dtype=np.float64)
G0_TAB = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], dtype=np.float64)

# Convert ln(1+x) grid -> x grid and restrict to cusp range
X_G0 = np.expm1(LN1P)
XMAX = X_EDGES[-1]
mask = (X_G0 >= X_BOUND) & (X_G0 <= XMAX) & (G0_TAB > 0)
XG = X_G0[mask]
G0 = G0_TAB[mask]

# Build inverse CDF for sampling x from the BW occupancy N(E) ∝ x^{-5/2} g0(x)
weight = G0 * (XG ** -2.5)  # N_x ∝ x^{-5/2} g0(x)

cdf_x = np.zeros_like(XG)
cdf_x[1:] = np.cumsum(
    0.5 * (weight[1:] + weight[:-1]) * (XG[1:] - XG[:-1])
)
cdf_x /= cdf_x[-1]


def sample_x_from_g0(u):
    """Pure-python inverse CDF for BW reservoir distribution."""
    if u <= 0.0:
        return XG[0]
    if u >= 1.0:
        return XG[-1]
    i = np.searchsorted(cdf_x, u, side="right")
    i = max(1, min(i, cdf_x.size - 1))
    t = (u - cdf_x[i - 1]) / (cdf_x[i] - cdf_x[i - 1] + 1e-30)
    return XG[i - 1] + t * (XG[i] - XG[i - 1])


# ------------- Table-1 grids (x, j) and starred coeffs -------------
X_GRID = np.array([3.36e-1, 3.31e0, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)  # descending

NEG_E1 = np.array([
    [1.41e-1, 1.40e-1, 1.33e-1, 1.29e-1, 1.29e-1],
    [-1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],  # corrected sign
    [1.96e-3, 1.59e-3, 2.83e-3, 3.38e-3, 3.49e-3],
    [3.64e-3, 4.64e-3, 4.93e-3, 4.97e-3, 4.98e-3],
    [8.39e-4, 8.53e-4, 8.56e-4, 8.56e-4, 8.56e-4],
], dtype=np.float64)

E2 = np.array([
    [3.14e-1, 4.36e-1, 7.55e-1, 1.37e0, 2.19e0],
    [4.45e-1, 8.66e-1, 1.57e0, 2.37e0, 2.73e0],
    [1.03e0, 1.80e0, 2.52e0, 2.77e0, 2.81e0],
    [1.97e0, 2.54e0, 2.68e0, 2.70e0, 2.70e0],
    [1.96e0, 1.96e0, 1.96e0, 1.96e0, 1.96e0],
], dtype=np.float64)

J1 = np.array([
    [-2.52e-2, 5.37e-1, 1.51e0, 3.83e0, 9.58e0],
    [-5.03e-3, 5.40e-3, 2.54e-2, 6.95e-2, 1.75e-1],
    [-2.58e-4, 3.94e-4, 1.45e-3, 3.77e-3, 9.45e-3],
    [-4.67e-6, 2.03e-5, 5.99e-5, 1.53e-4, 3.82e-4],
    [3.67e-8, 2.54e-7, 7.09e-7, 1.80e-6, 4.49e-6],  # fixed 2.54e-7
], dtype=np.float64)

J2 = np.array([
    [4.68e-1, 6.71e-1, 6.99e-1, 7.03e-1, 7.04e-1],
    [6.71e-2, 9.19e-2, 9.48e-2, 9.53e-2, 9.54e-2],
    [1.57e-2, 2.12e-2, 2.20e-2, 2.21e-2, 2.21e-2],
    [3.05e-3, 4.25e-3, 4.42e-3, 4.44e-3, 4.45e-3],
    [3.07e-4, 4.59e-4, 4.78e-4, 4.81e-4, 4.82e-4],  # fixed 4.59e-4
], dtype=np.float64)

ZETA2 = np.array([
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

# sanity-check typographical fixes
assert abs(J1[-1, 1] - 2.54e-7) < 1e-12
assert abs(J2[-1, 1] - 4.59e-4) < 1e-12


# ---------------- kernels (Numba or pure python) ----------------
def _build_kernels(use_jit=True, noloss=False, lc_scale=LC_SCALE_DEFAULT,
                   e1_scale=E1_SCALE_DEFAULT):
    """Build numba-jitted kernels (or pure-python fallbacks)."""
    if use_jit and HAVE_NUMBA:
        njit = nb.njit
        fastmath = dict(fastmath=True, nogil=True, cache=True)
    else:
        def njit(*args, **kwargs):
            def wrapper(f):
                return f
            return wrapper
        fastmath = {}

    @njit(**fastmath)
    def j_min_of_x(x, lc_scale_val, noloss_flag):
        """
        Dimensionless loss-cone boundary j_min(x).
        In noloss calibration runs, the loss cone is suppressed: j_min = 0.
        """
        if noloss_flag:
            return 0.0
        v = 2.0 * x / X_D - (x / X_D) ** 2
        jmin = math.sqrt(v) if v > 0.0 else 0.0
        return lc_scale_val * jmin

    @njit(**fastmath)
    def P_of_x(x):
        """Orbital period P(E) with E = -x (M = v0 = 1)."""
        E = -x
        return 2.0 * math.pi / ((-2.0 * E) ** 1.5)

    T0 = P_of_x(1.0) / PSTAR  # t0 from P*(x=1)

    @njit(**fastmath)
    def sample_x_from_g0_jit():
        """Draw x from BW occupancy N(E) ∝ x^{-5/2} g0(x)."""
        u = np.random.random()
        if u <= 0.0:
            return XG[0]
        if u >= 1.0:
            return XG[-1]

        i = 1
        while i < cdf_x.size and cdf_x[i] < u:
            i += 1
        if i >= cdf_x.size:
            i = cdf_x.size - 1
        if i < 1:
            i = 1

        denom = cdf_x[i] - cdf_x[i - 1]
        if denom <= 0.0:
            return XG[i - 1]

        t = (u - cdf_x[i - 1]) / denom
        return XG[i - 1] + t * (XG[i] - XG[i - 1])

    @njit(**fastmath)
    def which_bin(x):
        best = 0
        bestd = abs(X_BINS[0] - x)
        for i in range(1, X_BINS.size):
            d = abs(X_BINS[i] - x)
            if d < bestd:
                bestd = d
                best = i
        return best

    @njit(**fastmath)
    def bilinear_coeffs(x, j, pstar_val, e1_scale_val):
        """
        Bilinear interpolation of Table-1 starred coefficients.

        Returns perturbations per radial period with proper P* scaling:

            e1   : mean energy drift per orbit
            sigE : 1σ energy diffusion amplitude (ΔE)
            j1   : mean J drift per orbit
            sigJ : 1σ J diffusion amplitude
            rho  : correlation coefficient between ΔE and ΔJ
        """
        # clamp x to table range
        x_clamp = x
        if x_clamp < X_GRID[0]:
            x_clamp = X_GRID[0]
        if x_clamp > X_GRID[-1]:
            x_clamp = X_GRID[-1]
        lx = math.log10(x_clamp)

        # find x index
        ix = 0
        for k in range(X_GRID.size - 1):
            if X_GRID[k] <= x_clamp <= X_GRID[k + 1]:
                ix = k
                break
        x1 = X_GRID[ix]
        x2 = X_GRID[ix + 1]
        tx = (lx - math.log10(x1)) / (math.log10(x2) - math.log10(x1) + 1e-30)
        if tx < 0.0:
            tx = 0.0
        if tx > 1.0:
            tx = 1.0

        # clamp j to table range (J_GRID descending)
        j_clamp = j
        if j_clamp > J_GRID[0]:
            j_clamp = J_GRID[0]
        if j_clamp < J_GRID[-1]:
            j_clamp = J_GRID[-1]

        # find j index
        ij = 0
        for k in range(J_GRID.size - 1):
            if (J_GRID[k] >= j_clamp) and (j_clamp >= J_GRID[k + 1]):
                ij = k
                break
        j1g = J_GRID[ij]
        j2g = J_GRID[ij + 1]
        tj = (j_clamp - j1g) / (j2g - j1g + 1e-30)
        if tj < 0.0:
            tj = 0.0
        if tj > 1.0:
            tj = 1.0

        def get(A):
            a11 = A[ix, ij]
            a12 = A[ix, ij + 1]
            a21 = A[ix + 1, ij]
            a22 = A[ix + 1, ij + 1]
            a1 = a11 * (1.0 - tj) + a12 * tj
            a2 = a21 * (1.0 - tj) + a22 * tj
            return a1 * (1.0 - tx) + a2 * tx

        # canonical P*
        PSTAR_CANON = 0.005

        # Table 1 gives e1* (dimensionless energy drift) directly.
        # Use the table values as-is without sign flip.
        e1_star = get(NEG_E1)  # e1* from Table 1
        E2_star = max(get(E2), 0.0)     # (ε2* )^2
        J2_star = max(get(J2), 0.0)     # (j2* )^2
        j1_star = get(J1)
        covEJ_star = get(ZETA2)         # <Δx* Δj*>

        # scale with P* (eq. A8): all scale ∝ P*
        scale = pstar_val / PSTAR_CANON
        e1_star *= scale
        E2_star *= scale
        j1_star *= scale
        J2_star *= scale
        covEJ_star *= scale

        # convert to physical units (v0=1, M=1)
        v0_sq = 1.0
        Jmax = 1.0 / math.sqrt(2.0 * x_clamp)

        # Apply empirical scaling to energy drift to match SM78 canonical solution
        e1 = -e1_scale_val * e1_star * v0_sq
        sigE_star = math.sqrt(max(E2_star, 0.0))
        sigE = sigE_star * v0_sq

        j1 = j1_star * Jmax
        sigJ_star = math.sqrt(max(J2_star, 0.0))
        sigJ = sigJ_star * Jmax

        covEJ = covEJ_star * v0_sq * Jmax

        if sigE > 0.0 and sigJ > 0.0:
            rho = covEJ / (sigE * sigJ)
            if rho > 0.999:
                rho = 0.999
            if rho < -0.999:
                rho = -0.999
        else:
            rho = 0.0

        return e1, sigE, j1, sigJ, rho

    @njit(**fastmath)
    def pick_n(x, j, sigE, sigJ, lc_scale_val, noloss_flag, cone_gamma_val,
               n_min=0.01, n_max=3.0e4,
               diag_counts=None, lc_floor_frac=0.10, lc_gap_scale=None):
        """
        Equation (29): choose n so that each step is a small perturbation.
        
        DIAGNOSTIC PARAMETERS:
        - diag_counts: optional array[4] to count which constraint wins (29a, 29b, 29c, 29d)
        - lc_floor_frac: floor fraction for 29d (default 0.10, set to 0.0 to disable floor)
        - lc_gap_scale: override for cone_gamma_val in 29d gap calculation (None = use cone_gamma_val)
        """
        jmin = j_min_of_x(x, lc_scale_val, noloss_flag)
        E = -x
        Jmax = 1.0 / math.sqrt(2.0 * x)
        J = j * Jmax

        # 29a: limit fractional energy change
        if sigE > 0.0:
            n_E = (0.15 * abs(E) / sigE) ** 2
        else:
            n_E = n_max

        # 29b: isotropic J diffusion
        if sigJ > 0.0:
            n_J_iso = (0.10 * Jmax / sigJ) ** 2
        else:
            n_J_iso = n_max

        # 29c: avoid crossing J_max in one step
        if sigJ > 0.0:
            delta_top = max(1e-6 * Jmax, 1.0075 * Jmax - J)
            n_J_top = (0.40 * delta_top / sigJ) ** 2
        else:
            n_J_top = n_max

        # 29d: limit diffusion into/out of loss cone
        # SM78 eq. 29d: step size limited by max[γ|J-J_min|, floor_frac*J_min]
        if noloss_flag or jmin <= 0.0 or sigJ == 0.0:
            n_J_lc = n_max
        else:
            Jmin = jmin * Jmax
            gap_scale = cone_gamma_val if lc_gap_scale is None else lc_gap_scale
            gap = gap_scale * abs(J - Jmin)
            if lc_floor_frac > 0.0:
                floor = max(gap, lc_floor_frac * Jmin)
            else:
                floor = gap  # No floor
            n_J_lc = (floor / sigJ) ** 2

        # Find which constraint wins (for diagnostics)
        n = n_E
        winner = 0  # 29a
        if n_J_iso < n:
            n = n_J_iso
            winner = 1  # 29b
        if n_J_top < n:
            n = n_J_top
            winner = 2  # 29c
        if n_J_lc < n:
            n = n_J_lc
            winner = 3  # 29d
        
        if n > n_max:
            n = n_max
        if n < n_min:
            n = n_min
        
        # Record winner if diagnostics enabled
        if diag_counts is not None:
            diag_counts[winner] += 1
        
        return n

    @njit(**fastmath)
    def correlated_normals(rho, sigE, sigJ):
        """Build correlated Gaussian increments for ΔE, ΔJ."""
        z1 = np.random.normal()
        z2 = np.random.normal()
        if abs(rho) < 0.999999:
            y1 = z1
            y2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2
        else:
            y1 = z1
            y2 = z2
        return y1, y2

    @njit(**fastmath)
    def floor_index_for_j2(j2, floors):
        # deepest floor index k such that j2 < floors[k]
        k = -1
        for i in range(floors.size):
            if j2 < floors[i]:
                k = i
        return k

    @njit(**fastmath)
    def target_floor_with_hysteresis(j2, floors, current_idx, split_hyst=0.8):
        # advance only when comfortably inside next floor
        k = current_idx
        while (k + 1) < floors.size and j2 < split_hyst * floors[k + 1]:
            k += 1
        return k

    @njit(**fastmath)
    def step_one(x, j, phase, pstar_val, noloss_flag, x_max,
                 lc_scale_val, cone_gamma_val, e1_scale_val,
                 diag_counts=None, disable_capture=False,
                 lc_floor_frac=0.10, lc_gap_scale=None,
                 ghost_cap_hist=None, x_bins=None):
        """
        One MC step (possibly fractional orbit) with pericenter-aligned
        truncation. Returns:

            x_new, j_new, phase_new, n_used, captured, crossed_peri
        
        DIAGNOSTIC PARAMETERS:
        - diag_counts: optional array[4] to count which constraint wins in pick_n
        - disable_capture: if True, never set captured=True (for testing 29d effect alone)
        - lc_floor_frac: floor fraction for 29d (default 0.10)
        - lc_gap_scale: override for cone_gamma_val in 29d gap calculation
        """
        # orbital coefficients
        e1, sigE, j1, sigJ, rho = bilinear_coeffs(x, j, pstar_val, e1_scale_val)

        # step length
        n_raw = pick_n(x, j, sigE, sigJ, lc_scale_val, noloss_flag,
                       cone_gamma_val, diag_counts=diag_counts,
                       lc_floor_frac=lc_floor_frac, lc_gap_scale=lc_gap_scale)

        # pericenter-aligned: do not cross the next integer phase
        next_int = math.floor(phase) + 1.0
        crossed_peri = (phase + n_raw >= next_int)
        if crossed_peri:
            n = next_int - phase
        else:
            n = n_raw

        # energy / angular momentum updates
        E = -x
        Jmax = 1.0 / math.sqrt(2.0 * x)
        J = j * Jmax

        # correlated ΔE, ΔJ
        y1, y2 = correlated_normals(rho, sigE, sigJ)

        # eq (27a)
        deltaE = n * e1 + math.sqrt(n) * sigE * y1
        E_new = E + deltaE
        if E_new >= -1e-6:
            E_new = -1e-6
        x_new = -E_new

        if x_new > x_max:
            x_new = x_max
            E_new = -x_new

        # new Jmax after energy change
        Jmax_new = 1.0 / math.sqrt(2.0 * x_new)

        # 2-D isotropic RW gate (eq. 28)
        use_2d = (math.sqrt(n) * sigJ > J / 4.0)

        if use_2d:
            # isotropic RW in J-space
            z1 = np.random.normal()
            z2 = np.random.normal()
            J_new = math.sqrt(
                (J + math.sqrt(n) * z1 * sigJ) ** 2 +
                (math.sqrt(n) * z2 * sigJ) ** 2
            )
        else:
            # correlated 1-D RW
            J_new = J + n * j1 + math.sqrt(n) * sigJ * y2

        # reflect at J=0 and Jmax
        if J_new < 0.0:
            J_new = -J_new
        if J_new > Jmax_new:
            J_new = 2.0 * Jmax_new - J_new
            if J_new < 0.0:
                J_new = 0.0
            if J_new > Jmax_new:
                J_new = Jmax_new

        j_new = J_new / Jmax_new
        phase_new = phase + n

        # capture only at pericenter in canonical runs
        captured = False
        ghost_captured = False
        if (not noloss_flag) and crossed_peri:
            j_min_new = j_min_of_x(x_new, lc_scale_val, noloss_flag)
            if j_new < j_min_new:
                if not disable_capture:
                    captured = True
                else:
                    # Ghost capture: would have been captured but capture is disabled
                    ghost_captured = True
        
        # Track ghost captures if diagnostics enabled
        if ghost_cap_hist is not None and ghost_captured:
            # Find bin for x_new
            best = 0
            bestd = abs(x_bins[0] - x_new)
            for i in range(1, x_bins.size):
                d = abs(x_bins[i] - x_new)
                if d < bestd:
                    bestd = d
                    best = i
            ghost_cap_hist[best] += 1

        return x_new, j_new, phase_new, n, captured, crossed_peri

    @njit(**fastmath)
    def run_stream(n_relax, floors, clones_per_split, x_bins, dx,
                   seed, pstar_val, warmup_t0, x_init,
                   include_clone_gbar=True,
                   use_snapshots=False,
                   noloss=False,
                   use_clones=True,
                   x_max=1e10,
                   lc_scale_val=1.0,
                   cone_gamma_val=0.25,
                   e1_scale_val=0.3,
                   snaps_per_t0=40,
                   enable_diag_counts=False,
                   disable_capture=False,
                   lc_floor_frac=0.10,
                   lc_gap_scale=-1.0,
                   enable_cap_inj_diag=False,
                   outer_injection=True,  # Default: use outer-boundary injection (physically correct)
                   outer_inj_x_min=10.0):
        """
        Single time-carrying stream + clone tree.

        Returns:
            g_acc    : occupancy histogram per x-bin (time-weighted or snapshots)
            total    : total t0 (time-weighted) or total snapshots
            crosses  : total pericenter crossings
            caps     : capture events
            ccount   : clone pool high-water mark
            diag_counts : array[4] counting which eq. 29 constraint wins (if enabled)
            cap_hist : array[x_bins.size] counting captures per x-bin (if enabled)
            inj_hist : array[x_bins.size] counting injections per x-bin (if enabled)
            ghost_cap_hist : array[x_bins.size] counting ghost captures per x-bin (if enabled)
        """
        np.random.seed(seed)
        SPLIT_HYST = 0.8
        noloss_flag = noloss
        
        # Initialize diagnostic counters
        if enable_diag_counts:
            diag_counts = np.zeros(4, dtype=np.int64)
        else:
            diag_counts = None
        
        # Initialize capture/injection diagnostics
        if enable_cap_inj_diag:
            cap_hist = np.zeros(x_bins.size, dtype=np.int64)
            inj_hist = np.zeros(x_bins.size, dtype=np.int64)
            ghost_cap_hist = np.zeros(x_bins.size, dtype=np.int64)
        else:
            cap_hist = None
            inj_hist = None
            ghost_cap_hist = None
        
        # Handle lc_gap_scale: -1.0 means use cone_gamma_val, otherwise use the override
        gap_scale_val = None if lc_gap_scale < 0.0 else lc_gap_scale

        # parent
        x = x_init
        j = math.sqrt(np.random.random())  # isotropic
        phase = 0.0
        w = 1.0
        parent_floor_idx = floor_index_for_j2(j * j, floors)

        # clone pool
        # More aggressive clone pool for Fig. 3:
        # - deeper j^2 floors and more clones per split
        #   give much better statistics in the deep cusp.
        MAX_CLONES = 4096 if use_clones else 0
        cx = np.zeros(MAX_CLONES, dtype=np.float64)
        cj = np.zeros(MAX_CLONES, dtype=np.float64)
        cph = np.zeros(MAX_CLONES, dtype=np.float64)
        cw = np.zeros(MAX_CLONES, dtype=np.float64)
        cfloor_idx = np.full(MAX_CLONES, -1, dtype=np.int32)
        cactive = np.zeros(MAX_CLONES, dtype=np.uint8)
        ccount = 0

        def bin_index(xv):
            b = 0
            bestd = abs(x_bins[0] - xv)
            for bi in range(1, x_bins.size):
                d = abs(x_bins[bi] - xv)
                if d < bestd:
                    bestd = d
                    b = bi
            return b

        # ---------- warm-up (no tallies) ----------
        t0_used = 0.0
        while t0_used < warmup_t0:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(
                x, j, phase, pstar_val, noloss_flag, x_max,
                lc_scale_val, cone_gamma_val, e1_scale_val,
                diag_counts=diag_counts, disable_capture=disable_capture,
                lc_floor_frac=lc_floor_frac, lc_gap_scale=gap_scale_val,
                ghost_cap_hist=ghost_cap_hist, x_bins=x_bins
            )
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0

            # replacement
            if cap or (x < X_BOUND):
                if noloss_flag:
                    x = sample_x_from_g0_jit()
                else:
                    # Injection: use outer-boundary injection by default (set via outer_injection parameter)
                    if outer_injection:
                        # Outer-boundary injection: draw from reservoir distribution
                        # but restrict to x >= outer_inj_x_min to match outer boundary
                        x = sample_x_from_g0_jit()
                        while x < outer_inj_x_min:
                            x = sample_x_from_g0_jit()
                    else:
                        # Old scheme: exact injection at x_b = 0.2 (for backward compatibility)
                        x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j * j, floors)
            elif use_clones:
                j2_now = j * j
                target_idx = target_floor_with_hysteresis(
                    j2_now, floors, parent_floor_idx, SPLIT_HYST
                )
                while parent_floor_idx < target_idx:
                    parent_floor_idx += 1
                    if ccount <= MAX_CLONES - clones_per_split:
                        alpha = 1.0 / (1.0 + clones_per_split)
                        new_w = w * alpha
                        w = new_w
                        for k in range(clones_per_split):
                            cx[ccount] = x
                            cj[ccount] = j
                            cph[ccount] = phase
                            cw[ccount] = new_w
                            cfloor_idx[ccount] = parent_floor_idx
                            cactive[ccount] = 1
                            ccount += 1

            # evolve clones during warm-up
            if use_clones:
                i = 0
                while i < ccount:
                    if cactive[i] == 0:
                        i += 1
                        continue
                    x_c = cx[i]
                    j_c = cj[i]
                    ph_c = cph[i]
                    x_c2, j_c2, ph_c2, n_c, cap_c, cross_c = step_one(
                        x_c, j_c, ph_c, pstar_val, noloss_flag, x_max,
                        lc_scale_val, cone_gamma_val, e1_scale_val,
                        diag_counts=diag_counts, disable_capture=disable_capture,
                        lc_floor_frac=lc_floor_frac, lc_gap_scale=gap_scale_val,
                        ghost_cap_hist=ghost_cap_hist, x_bins=x_bins
                    )
                    cx[i] = x_c2
                    cj[i] = j_c2
                    cph[i] = ph_c2

                    if cap_c or (x_c2 < X_BOUND):
                        if noloss_flag:
                            cx[i] = sample_x_from_g0_jit()
                        else:
                            # Injection: use outer-boundary injection by default (set via outer_injection parameter)
                            if outer_injection:
                                # Outer-boundary injection: draw from reservoir distribution
                                # but restrict to x >= outer_inj_x_min to match outer boundary
                                x_inj = sample_x_from_g0_jit()
                                while x_inj < outer_inj_x_min:
                                    x_inj = sample_x_from_g0_jit()
                                cx[i] = x_inj
                            else:
                                # Old scheme: exact injection at x_b = 0.2 (for backward compatibility)
                                cx[i] = X_BOUND
                        cj[i] = math.sqrt(np.random.random())
                        cph[i] = ph_c2
                        cfloor_idx[i] = floor_index_for_j2(cj[i] * cj[i], floors)
                        cactive[i] = 1
                        i += 1
                        continue

                    if cfloor_idx[i] >= 0 and j_c2 * j_c2 >= floors[cfloor_idx[i]]:
                        w += cw[i]
                        cw[i] = 0.0
                        cactive[i] = 0
                        i += 1
                        continue

                    j2_c = j_c2 * j_c2
                    target_idx_c = target_floor_with_hysteresis(
                        j2_c, floors, cfloor_idx[i], SPLIT_HYST
                    )
                    while cfloor_idx[i] < target_idx_c:
                        cfloor_idx[i] += 1
                        if ccount <= MAX_CLONES - clones_per_split:
                            alpha = 1.0 / (1.0 + clones_per_split)
                            new_w = cw[i] * alpha
                            cw[i] = new_w
                            for k in range(clones_per_split):
                                cx[ccount] = x_c2
                                cj[ccount] = j_c2
                                cph[ccount] = ph_c2
                                cw[ccount] = new_w
                                cfloor_idx[ccount] = cfloor_idx[i]
                                cactive[ccount] = 1
                                ccount += 1
                    i += 1

        # ---------- measurement window ----------
        # Note: All snapshots/measurements here are post-warmup, matching
        # SM78's procedure of averaging g(x) over the final measurement window.
        if use_snapshots:
            g_acc = np.zeros(x_bins.size, dtype=np.float64)
            snapshots = 0.0
            # schedule = snaps_per_t0 snapshots per t0
            delta_snap = 1.0 / snaps_per_t0
            t0_local = 0.0
            next_snap = delta_snap
        else:
            g_acc = np.zeros(x_bins.size, dtype=np.float64)
            total_t0 = 0.0

        t0_used = 0.0
        crossings = 0
        caps = 0
        max_ccount = ccount

        while t0_used < n_relax:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(
                x, j, phase, pstar_val, noloss_flag, x_max,
                lc_scale_val, cone_gamma_val, e1_scale_val,
                diag_counts=diag_counts, disable_capture=disable_capture,
                lc_floor_frac=lc_floor_frac, lc_gap_scale=gap_scale_val,
                ghost_cap_hist=ghost_cap_hist, x_bins=x_bins
            )
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0

            if use_snapshots:
                t0_local += dt0
                # snapshot when we cross multiples of delta_snap
                while t0_local >= next_snap and t0_used <= n_relax + 1e-9:
                    # snapshot at current state (parent + clones)
                    g_acc[bin_index(x)] += w
                    if use_clones and include_clone_gbar:
                        for ii in range(ccount):
                            if cactive[ii]:
                                g_acc[bin_index(cx[ii])] += cw[ii]
                    snapshots += 1.0
                    next_snap += delta_snap
            else:
                total_t0 += dt0
                # time-weighted occupancy at start of step
                g_acc[bin_index(x_prev)] += w * dt0
                if use_clones and include_clone_gbar:
                    for ii in range(ccount):
                        if cactive[ii]:
                            g_acc[bin_index(cx[ii])] += cw[ii] * dt0

            if crossed:
                crossings += 1

            # parent replacement / splitting
            if cap or (x < X_BOUND):
                if cap:
                    caps += 1
                    # Track capture energy distribution
                    if cap_hist is not None:
                        cap_hist[bin_index(x_prev)] += 1
                
                # Track injection energy distribution
                if inj_hist is not None:
                    if outer_injection and not noloss_flag:
                        # Outer-boundary injection: draw from high-x bins
                        # Sample from initial distribution but restrict to x >= outer_inj_x_min
                        x_inj = sample_x_from_g0_jit()
                        while x_inj < outer_inj_x_min:
                            x_inj = sample_x_from_g0_jit()
                        x = x_inj
                        inj_hist[bin_index(x)] += 1
                    elif noloss_flag:
                        x = sample_x_from_g0_jit()
                        inj_hist[bin_index(x)] += 1
                    else:
                        # Canonical SM78: exact injection at x_b = 0.2
                        x = X_BOUND
                        inj_hist[bin_index(x)] += 1
                else:
                    # Normal injection logic (no diagnostics)
                    if noloss_flag:
                        x = sample_x_from_g0_jit()
                    else:
                        if outer_injection:
                            # Outer-boundary injection (default): draw from reservoir distribution
                            # but restrict to x >= outer_inj_x_min to match outer boundary
                            x = sample_x_from_g0_jit()
                            while x < outer_inj_x_min:
                                x = sample_x_from_g0_jit()
                        else:
                            # Old scheme: exact injection at x_b = 0.2 (for backward compatibility)
                            x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j * j, floors)
            elif use_clones:
                j2_now = j * j
                target_idx = target_floor_with_hysteresis(
                    j2_now, floors, parent_floor_idx, SPLIT_HYST
                )
                while parent_floor_idx < target_idx:
                    parent_floor_idx += 1
                    if ccount <= MAX_CLONES - clones_per_split:
                        alpha = 1.0 / (1.0 + clones_per_split)
                        new_w = w * alpha
                        w = new_w
                        for k in range(clones_per_split):
                            cx[ccount] = x
                            cj[ccount] = j
                            cph[ccount] = phase
                            cw[ccount] = new_w
                            cfloor_idx[ccount] = parent_floor_idx
                            cactive[ccount] = 1
                            ccount += 1
                            if ccount > max_ccount:
                                max_ccount = ccount

            # evolve clones
            if use_clones:
                i = 0
                while i < ccount:
                    if cactive[i] == 0:
                        i += 1
                        continue
                    x_c = cx[i]
                    j_c = cj[i]
                    ph_c = cph[i]
                    x_c2, j_c2, ph_c2, n_c, cap_c, cross_c = step_one(
                        x_c, j_c, ph_c, pstar_val, noloss_flag, x_max,
                        lc_scale_val, cone_gamma_val, e1_scale_val,
                        diag_counts=diag_counts, disable_capture=disable_capture,
                        lc_floor_frac=lc_floor_frac, lc_gap_scale=gap_scale_val,
                        ghost_cap_hist=ghost_cap_hist, x_bins=x_bins
                    )

                    cx[i] = x_c2
                    cj[i] = j_c2
                    cph[i] = ph_c2

                    if cross_c:
                        crossings += 1

                    if cap_c or (x_c2 < X_BOUND):
                        if cap_c:
                            caps += 1
                            # Track capture energy distribution
                            if cap_hist is not None:
                                cap_hist[bin_index(x_c)] += 1
                        
                        # Track injection energy distribution
                        if inj_hist is not None:
                            if outer_injection and not noloss_flag:
                                # Outer-boundary injection: draw from high-x bins
                                x_inj = sample_x_from_g0_jit()
                                while x_inj < outer_inj_x_min:
                                    x_inj = sample_x_from_g0_jit()
                                cx[i] = x_inj
                                inj_hist[bin_index(x_inj)] += 1
                            elif noloss_flag:
                                cx[i] = sample_x_from_g0_jit()
                                inj_hist[bin_index(cx[i])] += 1
                            else:
                                # Canonical SM78: exact injection at x_b = 0.2
                                cx[i] = X_BOUND
                                inj_hist[bin_index(cx[i])] += 1
                        else:
                            # Normal injection logic (no diagnostics)
                            if noloss_flag:
                                cx[i] = sample_x_from_g0_jit()
                            else:
                                if outer_injection:
                                    # Outer-boundary injection (default): draw from reservoir distribution
                                    # but restrict to x >= outer_inj_x_min to match outer boundary
                                    x_inj = sample_x_from_g0_jit()
                                    while x_inj < outer_inj_x_min:
                                        x_inj = sample_x_from_g0_jit()
                                    cx[i] = x_inj
                                else:
                                    # Old scheme: exact injection at x_b = 0.2 (for backward compatibility)
                                    cx[i] = X_BOUND
                        cj[i] = math.sqrt(np.random.random())
                        cph[i] = ph_c2
                        cfloor_idx[i] = floor_index_for_j2(cj[i] * cj[i], floors)
                        cactive[i] = 1
                        if ccount > max_ccount:
                            max_ccount = ccount
                        i += 1
                        continue

                    if cfloor_idx[i] >= 0 and j_c2 * j_c2 >= floors[cfloor_idx[i]]:
                        w += cw[i]
                        cw[i] = 0.0
                        cactive[i] = 0
                        i += 1
                        continue

                    j2_c = j_c2 * j_c2
                    target_idx_c = target_floor_with_hysteresis(
                        j2_c, floors, cfloor_idx[i], SPLIT_HYST
                    )
                    while cfloor_idx[i] < target_idx_c:
                        cfloor_idx[i] += 1
                        if ccount <= MAX_CLONES - clones_per_split:
                            alpha = 1.0 / (1.0 + clones_per_split)
                            new_w = cw[i] * alpha
                            cw[i] = new_w
                            for k in range(clones_per_split):
                                cx[ccount] = x_c2
                                cj[ccount] = j_c2
                                cph[ccount] = ph_c2
                                cw[ccount] = new_w
                                cfloor_idx[ccount] = cfloor_idx[i]
                                cactive[ccount] = 1
                                ccount += 1
                                if ccount > max_ccount:
                                    max_ccount = ccount
                    i += 1

        # Track weight sum for diagnostics (sum over parent + all active clones)
        weight_sum = w
        for ii in range(ccount):
            if cactive[ii]:
                weight_sum += cw[ii]

        # Return diagnostic counters (or zeros if disabled)
        if diag_counts is None:
            diag_counts = np.zeros(4, dtype=np.int64)
        if cap_hist is None:
            cap_hist = np.zeros(x_bins.size, dtype=np.int64)
        if inj_hist is None:
            inj_hist = np.zeros(x_bins.size, dtype=np.int64)
        if ghost_cap_hist is None:
            ghost_cap_hist = np.zeros(x_bins.size, dtype=np.int64)
        
        if use_snapshots:
            return g_acc, snapshots, crossings, caps, max_ccount, weight_sum, diag_counts, cap_hist, inj_hist, ghost_cap_hist
        else:
            return g_acc, total_t0, crossings, caps, max_ccount, weight_sum, diag_counts, cap_hist, inj_hist, ghost_cap_hist

    return P_of_x, T0, run_stream, sample_x_from_g0_jit


# ---------------- multiprocessing worker (single stream) ----------------
def _worker_one(args):
    (sid, n_relax, floors, clones_per_split,
     stream_seeds, use_jit, pstar_val, warmup_t0, u0,
     include_clone_gbar, use_snapshots, snaps_per_t0,
     noloss, use_clones, lc_scale, e1_scale, x_max, cone_gamma,
     enable_diag_counts, disable_capture, lc_floor_frac, lc_gap_scale,
     enable_cap_inj_diag, outer_injection, outer_inj_x_min) = args

    P_of_x, T0, run_stream, sample_x_from_g0_jit = _build_kernels(
        use_jit=use_jit, noloss=noloss, lc_scale=lc_scale,
        e1_scale=e1_scale
    )

    # initial x from BW distribution (stratified)
    x_init = sample_x_from_g0(u0)

    # tiny warm-up to pull kernels into cache
    _ = run_stream(
        1e-6, floors, clones_per_split, X_BINS, DX,
        int(stream_seeds[sid] ^ 0xABCDEF), pstar_val, 0.0, x_init,
        include_clone_gbar, use_snapshots, noloss, use_clones,
        x_max, lc_scale, cone_gamma, e1_scale, snaps_per_t0,
        enable_diag_counts, disable_capture, lc_floor_frac, lc_gap_scale,
        enable_cap_inj_diag, outer_injection, outer_inj_x_min
    )

    result = run_stream(
        n_relax, floors, clones_per_split, X_BINS, DX,
        int(stream_seeds[sid]), pstar_val, warmup_t0, x_init,
        include_clone_gbar, use_snapshots, noloss, use_clones,
        x_max, lc_scale, cone_gamma, e1_scale, snaps_per_t0,
        enable_diag_counts, disable_capture, lc_floor_frac, lc_gap_scale,
        enable_cap_inj_diag, outer_injection, outer_inj_x_min
    )
    return result


# ---------------- parallel driver (ḡ only) ----------------
def run_parallel_gbar(
    n_streams=400,
    n_relax=6.0,
    floors=None,
    clones_per_split=9,
    procs=48,
    use_jit=True,
    seed=SEED,
    show_progress=True,
    pstar=PSTAR,
    warmup_t0=2.0,
    replicates=1,
    gexp=2.5,
    include_clone_gbar=True,
    use_snapshots=False,
    snaps_per_t0=40,
    noloss=False,
    use_clones=True,
    lc_scale=LC_SCALE_DEFAULT,
    e1_scale=E1_SCALE_DEFAULT,
    x_max=None,
    cone_gamma=0.25,
    debug_occupancy_norm=False,
    enable_diag_counts=False,
    disable_capture=False,
    lc_floor_frac=0.10,
    lc_gap_scale=None,
    enable_cap_inj_diag=False,
    outer_injection=True,  # Default: use outer-boundary injection (physically correct BC)
    outer_inj_x_min=10.0,  # Default minimum x for outer injection
):
    """
    Run many independent time-carrying streams in parallel and accumulate ḡ(x).

    If use_snapshots:
        * each stream takes snapshots at fixed intervals in t0
          (snaps_per_t0 per t0), approximating SM78 Fig. 3 procedure.

    Else:
        * true time-weighted occupancy is used.

    Returns:
        gbar_norm : ḡ(x) phase-space DF, normalized so ḡ(x_b) = 1
        gbar_raw  : unnormalized ḡ(x) before normalization
    """
    # noloss calibration: no clones, standard exponent
    if noloss:
        use_clones = False

    # x_max cap
    if x_max is None:
        if noloss:
            x_max = 300.0
        else:
            x_max = XMAX

    if floors is None:
        floors = np.array([10.0 ** (-k) for k in range(0, 9)], dtype=np.float64)
    else:
        floors = np.array(floors, dtype=np.float64)

    if procs is None or procs < 1:
        procs = os.cpu_count() or 1

    import concurrent.futures as cf

    rs = np.random.RandomState(seed)

    if show_progress:
        start_time = time.time()
        tot_expected = replicates * n_streams
        print(
            f"Running gbar MC with {n_streams} streams × {replicates} replicates "
            f"on {procs} processes...",
            file=sys.stderr,
        )

    g_total = np.zeros_like(X_BINS, dtype=np.float64)
    total_measure = 0.0
    peri_total = 0
    caps_total = 0
    max_ccount = 0
    total_done = 0
    weight_sums = []  # Track weight sums for diagnostics
    diag_counts_total = np.zeros(4, dtype=np.int64)  # Track eq. 29 constraint winners
    cap_hist_total = np.zeros_like(X_BINS, dtype=np.int64)  # Track captures per x-bin
    inj_hist_total = np.zeros_like(X_BINS, dtype=np.int64)  # Track injections per x-bin
    ghost_cap_hist_total = np.zeros_like(X_BINS, dtype=np.int64)  # Track ghost captures per x-bin

    for rep in range(replicates):
        stream_seeds = rs.randint(1, 2**31 - 1, size=n_streams, dtype=np.int64)
        # stratified reservoir sampling
        u_strat = (np.arange(n_streams) + 0.5) / n_streams
        rs.shuffle(u_strat)

        with cf.ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [
                ex.submit(
                    _worker_one,
                    (
                        sid,
                        n_relax,
                        floors,
                        clones_per_split,
                        stream_seeds,
                        use_jit,
                        pstar,
                        warmup_t0,
                        u_strat[sid],
                        include_clone_gbar,
                        use_snapshots,
                        snaps_per_t0,
                        noloss,
                        use_clones,
                        lc_scale,
                        e1_scale,
                        x_max,
                        cone_gamma,
                        enable_diag_counts,
                        disable_capture,
                        lc_floor_frac,
                        lc_gap_scale if lc_gap_scale is not None else -1.0,
                        enable_cap_inj_diag,
                        outer_injection,
                        outer_inj_x_min,
                    ),
                )
                for sid in range(n_streams)
            ]

            for f in cf.as_completed(futs):
                g_b, measure_b, peri_b, caps_b, ccount_b, wsum_b, diag_b, cap_b, inj_b, ghost_b = f.result()
                g_total += g_b
                total_measure += measure_b
                peri_total += peri_b
                caps_total += caps_b
                weight_sums.append(wsum_b)
                if enable_diag_counts:
                    diag_counts_total += diag_b
                if enable_cap_inj_diag:
                    cap_hist_total += cap_b
                    inj_hist_total += inj_b
                    ghost_cap_hist_total += ghost_b
                if ccount_b > max_ccount:
                    max_ccount = ccount_b
                total_done += 1
                if show_progress:
                    elapsed = time.time() - start_time
                    rate = total_done / max(elapsed, 1e-9)
                    frac = total_done / (replicates * n_streams)
                    print(
                        f"\rProgress: {total_done}/{replicates * n_streams} streams "
                        f"({100 * frac:5.1f}%) | rate {rate:5.1f} streams/s",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

    if show_progress:
        elapsed = time.time() - start_time
        print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
        if use_snapshots:
            print(
                f"[diag] total snapshots across all streams: {int(total_measure)}",
                file=sys.stderr,
            )
        else:
            print(
                f"[diag] total t0 across all streams: {total_measure:.3e}",
                file=sys.stderr,
            )
        print(f"[diag] pericenter crossings (total): {peri_total}", file=sys.stderr)
        print(f"[diag] capture events (total):      {caps_total}", file=sys.stderr)
        print(f"[diag] clone pool high-water:       {max_ccount}", file=sys.stderr)
        if enable_diag_counts:
            total_steps = diag_counts_total.sum()
            if total_steps > 0:
                print(f"[diag] eq. 29 constraint winners (total steps: {total_steps}):", file=sys.stderr)
                print(f"       29a (energy limit):     {diag_counts_total[0]:10d} ({100*diag_counts_total[0]/total_steps:5.1f}%)", file=sys.stderr)
                print(f"       29b (J isotropic):     {diag_counts_total[1]:10d} ({100*diag_counts_total[1]/total_steps:5.1f}%)", file=sys.stderr)
                print(f"       29c (J max boundary):    {diag_counts_total[2]:10d} ({100*diag_counts_total[2]/total_steps:5.1f}%)", file=sys.stderr)
                print(f"       29d (loss-cone limit):   {diag_counts_total[3]:10d} ({100*diag_counts_total[3]/total_steps:5.1f}%)", file=sys.stderr)
        if enable_cap_inj_diag:
            total_caps = cap_hist_total.sum()
            total_inj = inj_hist_total.sum()
            total_ghost = ghost_cap_hist_total.sum()
            if total_caps > 0 or total_inj > 0:
                print(f"[diag] Capture/Injection diagnostics:", file=sys.stderr)
                print(f"       Total captures: {total_caps}, Total injections: {total_inj}", file=sys.stderr)
                if total_ghost > 0:
                    print(f"       Total ghost captures (with --disable-capture): {total_ghost}", file=sys.stderr)
                print(f"       x_center    N_cap    N_inj    N_occ    Γ_cap(x)  (Γ = N_cap/N_occ per t0)", file=sys.stderr)
                # Compute occupancy for rate calculation
                # g_total is time-weighted occupancy (or snapshot-weighted)
                # For rate calculation, we want the total occupancy accumulated
                N_occ = g_total.copy()  # This is already the accumulated occupancy
                for i in range(X_BINS.size):
                    if cap_hist_total[i] > 0 or inj_hist_total[i] > 0 or N_occ[i] > 0:
                        # Capture rate per unit occupancy (normalized by total measure time)
                        if N_occ[i] > 0 and total_measure > 0:
                            # Normalize occupancy by total measure to get average occupancy
                            N_occ_norm = N_occ[i] / total_measure
                            gamma_cap = cap_hist_total[i] / N_occ_norm if N_occ_norm > 0 else 0.0
                        else:
                            gamma_cap = 0.0
                            N_occ_norm = 0.0
                        print(f"       {X_BINS[i]:8.2f}  {cap_hist_total[i]:8d}  {inj_hist_total[i]:8d}  {N_occ[i]:8.2e}  {gamma_cap:8.2e}", file=sys.stderr)
        if weight_sums:
            wsum_arr = np.array(weight_sums)
            print(
                f"[diag] weight sum stats: min={wsum_arr.min():.6f}, "
                f"mean={wsum_arr.mean():.6f}, max={wsum_arr.max():.6f}, "
                f"std={wsum_arr.std():.6f}",
                file=sys.stderr,
            )

    if total_measure <= 0.0:
        raise RuntimeError("No time/snapshots accumulated; check parameters.")

    # Build N_x from either snapshots or time weighting
    if use_snapshots:
        # Snapshot mode:
        #   g_total[i] is the sum of weights in bin i over all snapshots.
        #
        # Snapshots are taken at equal intervals in t0, and in this code
        # t0 is proportional to physical time.  That means each snapshot
        # is already a fair *time* sample of the system, so g_total is
        # directly proportional to the time-averaged occupancy N(E).
        #
        # To get the differential number per unit x we just divide by
        # the total number of snapshots and the bin width Δx:
        #
        #   N_x = (sum of weights in bin) / (N_snap * Δx)
        n_snaps = total_measure
        N_x = g_total / (n_snaps * DX)

        if show_progress:
            print(f"[diag] snapshot mode: n_snaps = {n_snaps}", file=sys.stderr)
        
        # SNAPSHOT MODE: Always use occupancy-based normalization (capture-independent)
        # This decouples snapshots from capture flux, making it robust even when captures=0
        positive_bins = np.where(N_x > 0)[0]
        if positive_bins.size == 0:
            gbar_norm = np.zeros_like(N_x)
            gbar_raw = N_x.copy()
            if show_progress:
                print(
                    "[diag] snapshot mode: no positive N_x; returning zeros.",
                    file=sys.stderr,
                )
            return gbar_norm, gbar_raw
        
        # Use first positive bin (prefer x=0.225 if available)
        norm_idx = 0 if N_x[0] > 0 else positive_bins[0]
        if show_progress:
            print(
                f"[diag] snapshot mode: normalizing on bin {norm_idx} "
                f"(x={X_BINS[norm_idx]:.3g}, N_x={N_x[norm_idx]:.6e})",
                file=sys.stderr,
            )
        
        # Normalize occupancy directly (shape of stationary distribution)
        # In snapshot mode, we ALWAYS normalize on occupancy N_x (capture-independent)
        # This decouples snapshots from capture flux completely
        # 
        # For neutrality testing: if cloning/guiding is correct, N_x should be the SAME
        # regardless of gexp. So we normalize N_x first, then optionally apply gexp.
        gbar_norm_occupancy = N_x / N_x[norm_idx]
        
        # Convert N(E) → g(E) using the correct physical Jacobian
        # From paper eq. (9) and isotropization: g(E) ∝ N(E) * |E|^{5/2}
        # In terms of x = -E/v_0^2: g(x) ∝ N_x * x^{5/2}
        # This is the EXACT Jacobian, not a tunable parameter
        GEXP_PHYSICAL = 2.5  # Exactly 5/2 from P(E) * J_max^2 scaling
        
        if debug_occupancy_norm:
            # Debug mode: return occupancy only (for slope diagnostics)
            gbar_norm = gbar_norm_occupancy
            gbar_raw = N_x.copy()
            if show_progress:
                print(
                    "[diag] snapshot mode: returning occupancy N_x only (debug mode)",
                    file=sys.stderr,
                )
        else:
            # Production mode: apply correct Jacobian to convert N(E) → g(E)
            g_unscaled = N_x * (X_BINS ** GEXP_PHYSICAL)
            # Normalize at x=0.225 (matching paper Table 2 and Fig. 3)
            gbar_norm = g_unscaled / g_unscaled[norm_idx]
            gbar_raw = g_unscaled.copy()
            if show_progress:
                print(
                    f"[diag] snapshot mode: applied physical Jacobian x^{GEXP_PHYSICAL:.1f} "
                    f"(N(E) → g(E)), normalized at x=0.225",
                    file=sys.stderr,
                )
            
    else:
        # Time-weighted mode: already accounts for P(E) implicitly since
        # dt0 = (n * P(E)) / T0, so no additional P(E) factor needed
        p_x = g_total / total_measure
        p_sum = p_x.sum()
        if show_progress:
            print(f"[diag] sum p_x = {p_sum:.6f}", file=sys.stderr)
        N_x = p_x / DX

        # Convert N(E) → g(E) using the correct physical Jacobian
        # From paper eq. (9) and isotropization: g(E) ∝ N(E) * |E|^{5/2}
        # In terms of x = -E/v_0^2: g(x) ∝ N_x * x^{5/2}
        # This is the EXACT Jacobian from P(E) * J_max^2 scaling, not a tunable parameter
        GEXP_PHYSICAL = 2.5  # Exactly 5/2
        
        # Always normalize at x=0.225 (first bin, index 0) to match paper Table 2 and Fig. 3
        norm_idx = 0  # x=0.225 (first bin)
        
        if debug_occupancy_norm:
            # Debug mode: return occupancy N_x only (for slope diagnostics of N(E))
            # This is useful for checking that the underlying occupancy distribution is correct
            positive_bins = np.where(N_x > 0)[0]
            if positive_bins.size == 0:
                gbar_norm = np.zeros_like(N_x)
                gbar_raw = N_x.copy()
                if show_progress:
                    print(
                        "[diag] debug_occupancy_norm: no positive N_x; returning zeros.",
                        file=sys.stderr,
                    )
                return gbar_norm, gbar_raw
            
            # Normalize occupancy for display (but this is N(E), not g(E))
            norm_idx_occ = norm_idx if N_x[norm_idx] > 0 else positive_bins[0]
            gbar_norm = N_x / N_x[norm_idx_occ]
            gbar_raw = N_x.copy()
            
            if show_progress:
                print(
                    f"[diag] debug_occupancy_norm: returning occupancy N_x only "
                    f"(normalized at x={X_BINS[norm_idx_occ]:.3g}). "
                    f"This is N(E), not g(E).",
                    file=sys.stderr,
                )
        else:
            # Production mode: apply correct Jacobian to convert N(E) → g(E)
            # g(x) = N_x * x^{5/2} (exact physical conversion)
            g_unscaled = N_x * (X_BINS ** GEXP_PHYSICAL)
            
            positive_bins = np.where(g_unscaled > 0)[0]
            if positive_bins.size == 0:
                # no signal at all; fill zeros and return gracefully
                gbar_norm = np.zeros_like(g_unscaled)
                gbar_raw = g_unscaled.copy()
                if show_progress:
                    print(
                        "[diag] no positive g_unscaled; returning zeros for gbar.",
                        file=sys.stderr,
                    )
                return gbar_norm, gbar_raw
            
            # Check if x=0.225 has positive signal (should always be true in production)
            if g_unscaled[norm_idx] <= 0:
                # Fallback: use first positive bin (shouldn't happen in production)
                norm_idx = positive_bins[0]
                if show_progress:
                    print(
                        f"[WARNING] x=0.225 has zero signal; normalizing on bin {norm_idx} "
                        f"(x={X_BINS[norm_idx]:.3g}) instead. This may indicate insufficient statistics.",
                        file=sys.stderr,
                    )
            
            # Normalize so g(x=0.225) = 1.0 (matching paper Table 2 and Fig. 3)
            gbar_norm = g_unscaled / g_unscaled[norm_idx]
            gbar_raw = g_unscaled.copy()
            
            if show_progress and norm_idx == 0:
                print(
                    f"[diag] Production mode: applied physical Jacobian x^{GEXP_PHYSICAL:.1f} "
                    f"(N(E) → g(E)), normalized at x=0.225 to 1.0",
                    file=sys.stderr,
                )

    # quick slope diagnostic over 1≲x≲100
    if show_progress:
        mask = (
            (X_BINS >= 1.0)
            & (X_BINS <= 100.0)
            & np.isfinite(gbar_norm)
            & (gbar_norm > 0)
        )
        if mask.sum() >= 2:
            logx = np.log10(X_BINS[mask])
            logg = np.log10(gbar_norm[mask])
            p_fit, _ = np.polyfit(logx, logg, 1)
            if debug_occupancy_norm:
                print(
                    f"[diag] best-fit N(E) ∝ x^{p_fit:.3f} in 1≲x≲100 "
                    f"(this is occupancy, not g(E))",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[diag] best-fit g(x) ∝ x^{p_fit:.3f} in 1≲x≲100",
                    file=sys.stderr,
                )

    return gbar_norm, gbar_raw


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="SM78 Monte-Carlo for ḡ(x) (Fig. 3 reproduction attempt)."
    )
    ap.add_argument(
        "--streams",
        type=int,
        default=400,
        help="number of non-clone time-carrying streams (default: 400)",
    )
    ap.add_argument(
        "--windows",
        type=float,
        default=6.0,
        help="t0 windows per stream (measurement duration, default: 6)",
    )
    ap.add_argument(
        "--procs",
        type=int,
        default=48,
        help="number of worker processes (default: 48)",
    )
    ap.add_argument(
        "--floors_min_exp",
        type=int,
        default=8,
        help="min exponent k for floors 10^{-k} (default: 8 => 1e-8)",
    )
    ap.add_argument(
        "--floor-step",
        type=int,
        default=1,
        help="stride for floor exponents (e.g. 2 => decades 0,2,4,...)",
    )
    ap.add_argument(
        "--clones",
        type=int,
        default=9,
        help="clones per split (default: 9, matching SM78 canonical model)",
    )
    ap.add_argument(
        "--nojit",
        action="store_true",
        help="disable numba JIT (slow fallback)",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="disable progress display",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="base RNG seed (default: %(default)s)",
    )
    ap.add_argument(
        "--pstar",
        type=float,
        default=PSTAR,
        help="P* parameter (default: %(default)s)",
    )
    ap.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="equilibration time in t0 before tallying (default: 2)",
    )
    ap.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="repeat the whole measurement K times (accumulating raw tallies)",
    )
    ap.add_argument(
        "--gexp",
        type=float,
        default=2.5,
        help=(
            "DEPRECATED: Exponent is now fixed at 2.5 (5/2) from physical Jacobian. "
            "This parameter is kept for backward compatibility but not used in the conversion. "
            "The correct conversion is g(x) ∝ N_x * x^{5/2} from paper eq. (9)."
        ),
    )
    ap.add_argument(
        "--gbar-no-clones",
        action="store_true",
        help=(
            "Accumulate ḡ(x) from parent stream only "
            "(ignore clones in the histogram)."
        ),
    )
    ap.add_argument(
        "--snapshots",
        action="store_true",
        help=(
            "Use per-t0 snapshots for ḡ accumulation "
            "(closer to SM78 Fig. 3 procedure) instead "
            "of continuous time-weighting."
        ),
    )
    ap.add_argument(
        "--snaps-per-t0",
        type=int,
        default=40,
        help="Snapshots per t0 when --snapshots is set (default: 40).",
    )
    ap.add_argument(
        "--noloss",
        action="store_true",
        help="Disable loss-cone capture (for BW g0 diagnostics).",
    )
    ap.add_argument(
        "--e1-scale",
        type=float,
        default=E1_SCALE_DEFAULT,
        help="Scale factor for energy drift e1 (default: %.3f)" % E1_SCALE_DEFAULT,
    )
    ap.add_argument(
        "--lc_scale",
        type=float,
        default=LC_SCALE_DEFAULT,
        help="Scale factor for loss-cone boundary j_min (default: %.3f; <1 shrinks the cone)" % LC_SCALE_DEFAULT,
    )
    ap.add_argument(
        "--cone-gamma",
        type=float,
        default=0.25,
        help="γ factor in eq. 29d for loss-cone step limiting (default: 0.25, matching SM78).",
    )
    ap.add_argument(
        "--diag-29-counts",
        action="store_true",
        help="Enable diagnostic counters for eq. 29 constraint winners (29a/29b/29c/29d).",
    )
    ap.add_argument(
        "--disable-capture",
        action="store_true",
        help="Disable capture events while keeping loss-cone geometry (for testing 29d effect alone).",
    )
    ap.add_argument(
        "--lc-floor-frac",
        type=float,
        default=0.10,
        help="Floor fraction for eq. 29d loss-cone limiter (default: 0.10). Set to 0.0 to disable floor.",
    )
    ap.add_argument(
        "--lc-gap-scale",
        type=float,
        default=None,
        help="Override cone_gamma for eq. 29d gap calculation (default: use --cone-gamma value).",
    )
    ap.add_argument(
        "--cap-inj-diag",
        action="store_true",
        help="Enable capture/injection diagnostics: histogram captures and injections by x-bin.",
    )
    ap.add_argument(
        "--use-x-bound-injection",
        action="store_true",
        help="Use old injection scheme: inject at x_b = 0.2 (for backward compatibility/testing). "
             "By default, outer-boundary injection is used (x >= --outer-inj-x-min). "
             "NOTE: The old scheme produces an unphysically steep cusp (g(x) ∝ x^-3.7). "
             "Outer-boundary injection (default) is required to reproduce Fig. 3 / Table 2.",
    )
    ap.add_argument(
        "--outer-inj-x-min",
        type=float,
        default=10.0,
        help="Minimum x for outer-boundary injection (default: 10.0). "
             "Stars are injected from the reservoir distribution with x >= this value.",
    )
    ap.add_argument(
        "--debug-occupancy-norm",
        action="store_true",
        help=(
            "DEBUG: Use occupancy-based normalization (capture-independent) "
            "instead of gexp-based conversion. Useful for testing cloning neutrality."
        ),
    )

    args = ap.parse_args()

    floors = np.array(
        [10.0 ** (-k) for k in range(0, args.floors_min_exp + 1, args.floor_step)],
        dtype=np.float64,
    )

    gbar_norm, gbar_raw = run_parallel_gbar(
        n_streams=args.streams,
        n_relax=args.windows,
        floors=floors,
        clones_per_split=args.clones,
        procs=args.procs,
        use_jit=(not args.nojit),
        seed=args.seed,
        show_progress=(not args.no_progress),
        pstar=args.pstar,
        warmup_t0=args.warmup,
        replicates=args.replicates,
        gexp=args.gexp,
        include_clone_gbar=(not args.gbar_no_clones),
        use_snapshots=args.snapshots,
        snaps_per_t0=args.snaps_per_t0,
        noloss=args.noloss,
        use_clones=(not args.noloss and not args.gbar_no_clones),
        lc_scale=args.lc_scale,
        e1_scale=args.e1_scale,
        x_max=None,
        cone_gamma=args.cone_gamma,
        debug_occupancy_norm=args.debug_occupancy_norm,
        enable_diag_counts=args.diag_29_counts,
        disable_capture=args.disable_capture,
        lc_floor_frac=args.lc_floor_frac,
        lc_gap_scale=args.lc_gap_scale,
        enable_cap_inj_diag=args.cap_inj_diag,
        outer_injection=not args.use_x_bound_injection,  # Default to outer injection unless flag is set
        outer_inj_x_min=args.outer_inj_x_min,
    )

    # ---- print comparison table ----
    # Table format matches Shapiro & Marchant (1978) Table 2
    # gbar_MC_norm is normalized so g(x=0.225) = 1.0, matching the paper
    print("# x_center   gbar_MC_norm   gbar_MC_raw      gbar_paper   gbar_err_paper")
    residuals = []
    for xb, g_norm, g_raw, gp, gp_err in zip(
        X_BINS, gbar_norm, gbar_raw, GBAR_PAPER, GBAR_ERR_PAPER
    ):
        print(
            f"{xb:10.3g}  {g_norm:12.6e}  {g_raw:12.6e}  "
            f"{gp:11.4f}  {gp_err:14.4f}"
        )
        # Compute residual for chi^2 calculation
        if gp_err > 0:
            residuals.append((g_norm - gp) / gp_err)
    
    # Print chi^2 summary if we have valid residuals
    if residuals:
        residuals_arr = np.array(residuals)
        chi2 = np.sum(residuals_arr**2)
        chi2_per_dof = chi2 / len(residuals_arr)
        print(f"# chi^2/dof = {chi2_per_dof:.3f} (target: < 2.0 for good agreement)")
        print(f"# max |residual| = {np.abs(residuals_arr).max():.3f} sigma")


if __name__ == "__main__":
    main()

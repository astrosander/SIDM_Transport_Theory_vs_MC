#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

X_D     = 1.0e4
PSTAR   = 0.005
X_BOUND = 0.2
SEED    = 20251028

E1_SCALE_DEFAULT = 1.0

LC_SCALE_DEFAULT = 1.0

PSTAR_CANON = 0.005

SM78_LN_LAMBDA = 15.0
SM78_MSTAR = 1.0
SM78_MBH = 1.0e6
SM78_R_INFL = 1.0

SM78_A_e1 = 1.0
SM78_ALPHA_e1 = 1.0
SM78_BETA_e1 = 0.0

SM78_A_E2 = 1.0
SM78_ALPHA_E2 = 1.0
SM78_BETA_E2 = 0.0

SM78_A_j1 = 1.0
SM78_ALPHA_j1 = 1.0
SM78_BETA_j1 = 0.0

SM78_A_J2 = 1.0
SM78_ALPHA_J2 = 1.0
SM78_BETA_J2 = 0.0

SM78_A_covEJ = 0.0
SM78_ALPHA_covEJ = 1.0
SM78_BETA_covEJ = 0.0
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


def make_gbar_from_occupancy(x_centers, N_x, gexp, x_norm):
    """
    Map occupancy N(E) → ḡ(E) using paper's N(E) → g(E) ∝ x^{gexp}.

    This is the primary method for computing ḡ(x) and matches SM78's definition:
        ḡ(E) ∝ N(E) * x^{gexp}
    where N(E) is the time-averaged occupancy per energy bin from snapshots,
    and gexp = 2.5 for SM78 canonical case.

    For SM78 fiducial model, this gives ḡ(x) ∝ x^{-0.48} in 1≲x≲100.

    Args:
        x_centers: Energy bin centers
        N_x: Time-averaged occupancy per bin (from snapshots)
        gexp: Exponent for N(E) → g(E) mapping (default: 2.5 for SM78)
        x_norm: x value at which to normalize ḡ to 1

    Returns (gbar_norm, gbar_raw).
    """
    x_centers = np.asarray(x_centers, dtype=float)
    N_x = np.asarray(N_x, dtype=float)

    g_raw = N_x * np.power(x_centers, gexp)

    norm_idx = np.argmin(np.abs(x_centers - x_norm))
    if g_raw[norm_idx] == 0.0:
        raise RuntimeError(
            f"Normalization bin at x={x_norm} has zero g_raw; "
            "increase snapshots / streams or choose a different --gbar-x-norm."
        )

    g_norm = g_raw / g_raw[norm_idx]
    return g_norm, g_raw


def make_gbar_from_flux(x_centers, N_cap, N_occ, total_measure, flux_exp, x_norm):
    """
    Compute ḡ(E) from capture flux Γ_cap = N_cap / N_occ per t0.

    From the steady-state flux relation (SM78 style):
        Γ_cap(x) ∝ ḡ(x) / x^flux_exp
    so to recover ḡ(x) we divide out the x^flux_exp factor:
        ḡ(x) ∝ Γ_cap(x) / x^flux_exp

    In our implementation Γ_cap(x) already grows ~ x^flux_exp, so we must DIVIDE
    by x^flux_exp to get a roughly flat ḡ(x) consistent with occupancy-based ḡ.

    For SM78 fiducial model (gexp=2.5, E2-x-power=0.62), flux_exp ≈ 3.10
    makes flux-based ḡ match occupancy-based ḡ (both give slope ≈ -0.48).

    Args:
        x_centers: Energy bin centers
        N_cap: Total capture counts per bin (raw counts)
        N_occ: Total occupancy counts per bin (raw counts, not normalized)
        total_measure: Total time/snapshots for normalization (to compute N_occ per t0)
        flux_exp: Exponent for the flux relation (default: 3.10 for SM78 fiducial)
        x_norm: x value at which to normalize ḡ to 1

    Returns (gbar_norm, gbar_raw).
    """
    x_centers = np.asarray(x_centers, dtype=float)
    N_cap = np.asarray(N_cap, dtype=float)
    N_occ = np.asarray(N_occ, dtype=float)

    N_occ_norm = np.where(N_occ > 0, N_occ / total_measure, np.inf)
    gamma_cap = N_cap / N_occ_norm

    with np.errstate(divide="ignore", invalid="ignore"):
        g_raw = np.zeros_like(x_centers)
        mask = x_centers > 0.0
        g_raw[mask] = gamma_cap[mask] / np.power(x_centers[mask], flux_exp)

    norm_idx = np.argmin(np.abs(x_centers - x_norm))
    if g_raw[norm_idx] == 0.0:
        raise RuntimeError(
            f"Normalization bin at x={x_norm} has zero g_raw (flux mode); "
            "need more captures / different x_norm."
        )

    g_norm = g_raw / g_raw[norm_idx]
    return g_norm, g_raw

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

X_G0 = np.expm1(LN1P)
XMAX = X_EDGES[-1]
mask = (X_G0 >= X_BOUND) & (X_G0 <= XMAX) & (G0_TAB > 0)
XG = X_G0[mask]
G0 = G0_TAB[mask]

weight = G0 * (XG ** -2.5)

cdf_x = np.zeros_like(XG)
cdf_x[1:] = np.cumsum(
    0.5 * (weight[1:] + weight[:-1]) * (XG[1:] - XG[:-1])
)
cdf_x /= cdf_x[-1]


def sample_x_from_g0(u):
    if u <= 0.0:
        return XG[0]
    if u >= 1.0:
        return XG[-1]
    i = np.searchsorted(cdf_x, u, side="right")
    i = max(1, min(i, cdf_x.size - 1))
    t = (u - cdf_x[i - 1]) / (cdf_x[i] - cdf_x[i - 1] + 1e-30)
    return XG[i - 1] + t * (XG[i] - XG[i - 1])


X_GRID = np.array([3.36e-1, 3.31e0, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)

NEG_E1 = np.array([
    [1.41e-1, 1.40e-1, 1.33e-1, 1.29e-1, 1.29e-1],
    [-1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],
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
    [3.67e-8, 2.54e-7, 7.09e-7, 1.80e-6, 4.49e-6],
], dtype=np.float64)

J2 = np.array([
    [4.68e-1, 6.71e-1, 6.99e-1, 7.03e-1, 7.04e-1],
    [6.71e-2, 9.19e-2, 9.48e-2, 9.53e-2, 9.54e-2],
    [1.57e-2, 2.12e-2, 2.20e-2, 2.21e-2, 2.21e-2],
    [3.05e-3, 4.25e-3, 4.42e-3, 4.44e-3, 4.45e-3],
    [3.07e-4, 4.59e-4, 4.78e-4, 4.81e-4, 4.82e-4],
], dtype=np.float64)

ZETA2 = np.array([
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

assert abs(J1[-1, 1] - 2.54e-7) < 1e-12
assert abs(J2[-1, 1] - 4.59e-4) < 1e-12


def _build_kernels(use_jit=True, noloss=False, lc_scale=LC_SCALE_DEFAULT,
                   e1_scale=E1_SCALE_DEFAULT, zero_coeffs=False,
                   zero_drift=False, zero_diffusion=False,
                   step_size_factor=1.0, n_max_override=None,
                   E2_scale=1.0, J2_scale=1.0, covEJ_scale=1.0,
                   E2_x_power=0.0, E2_x_ref=1.0, use_sm78_physics=False):
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
    def j_min_of_x(x, lc_scale_val, noloss_flag, use_sm78_physics):
        if noloss_flag:
            return 0.0
        v = 2.0 * x / X_D - (x / X_D) ** 2
        jmin = math.sqrt(v) if v > 0.0 else 0.0
        if use_sm78_physics:
            return jmin
        else:
            return lc_scale_val * jmin

    @njit(**fastmath)
    def P_of_x(x):
        E = -x
        return 2.0 * math.pi / ((-2.0 * E) ** 1.5)

    T0 = P_of_x(1.0) / PSTAR

    @njit(**fastmath)
    def inject_star_sm78():
        x = X_BOUND
        j = math.sqrt(np.random.random())
        return x, j

    @njit(**fastmath)
    def sample_x_from_g0_jit():
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

    n_max_override_val = n_max_override if n_max_override is not None else 3.0e4
    E2_scale_val = E2_scale
    J2_scale_val = J2_scale
    covEJ_scale_val = covEJ_scale
    E2_x_power_val = E2_x_power
    E2_x_ref_val = E2_x_ref

    A_e1 = SM78_A_e1
    ALPHA_e1 = SM78_ALPHA_e1
    BETA_e1 = SM78_BETA_e1
    A_E2 = SM78_A_E2
    ALPHA_E2 = SM78_ALPHA_E2
    BETA_E2 = SM78_BETA_E2
    A_j1 = SM78_A_j1
    ALPHA_j1 = SM78_ALPHA_j1
    BETA_j1 = SM78_BETA_j1
    A_J2 = SM78_A_J2
    ALPHA_J2 = SM78_ALPHA_J2
    BETA_J2 = SM78_BETA_J2
    A_covEJ = SM78_A_covEJ
    ALPHA_covEJ = SM78_ALPHA_covEJ
    BETA_covEJ = SM78_BETA_covEJ

    @njit(**fastmath)
    def diff_coeffs_sm78_exact_star(x, j):
        """
        Exact SM78 orbit-averaged diffusion coefficients in the SAME
        normalized units as the original Table 1 ("starred" units).

        That is: this should return the analogs of
            NEG_E1, E2, J2, J1, ZETA2
        evaluated at (x, j) for P* = PSTAR_CANON.

        Returns:
            (e1_star, E2_star, j1_star, J2_star, covEJ_star)
        all for pstar = PSTAR_CANON, i.e. BEFORE any pstar scaling.

        IMPORTANT: Do NOT include pstar in these formulas; pstar
        scaling is handled later in bilinear_coeffs().
        """
        if x < X_GRID[-1]:
            x = X_GRID[-1]
        if x > X_GRID[0]:
            x = X_GRID[0]
        if j < 0.0:
            j = 0.0
        if j > 1.0:
            j = 1.0

        e1_star = A_e1 * x**ALPHA_e1 * (1.0 + BETA_e1 * j*j)
        E2_star = A_E2 * x**ALPHA_E2 * (1.0 + BETA_E2 * j*j)
        j1_star = A_j1 * x**ALPHA_j1 * j * (1.0 + BETA_j1 * j*j)
        J2_star = A_J2 * x**ALPHA_J2 * j*j * (1.0 + BETA_J2 * j*j)
        covEJ_star = A_covEJ * x**ALPHA_covEJ * j * (1.0 + BETA_covEJ * j*j)
        E2_star = max(E2_star, 0.0)
        J2_star = max(J2_star, 0.0)
        return e1_star, E2_star, j1_star, J2_star, covEJ_star

    @njit(**fastmath)
    def bilinear_coeffs(x, j, pstar_val, e1_scale_val, zero_coeffs_flag,
                        zero_drift_flag, zero_diffusion_flag,
                        E2_scale_local, J2_scale_local, covEJ_scale_local,
                        E2_x_power_local, E2_x_ref_local, use_sm78_physics):
        x_clamp = x
        if x_clamp < X_GRID[0]:
            x_clamp = X_GRID[0]
        if x_clamp > X_GRID[-1]:
            x_clamp = X_GRID[-1]
        lx = math.log10(x_clamp)

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

        j_clamp = j
        if j_clamp > J_GRID[0]:
            j_clamp = J_GRID[0]
        if j_clamp < J_GRID[-1]:
            j_clamp = J_GRID[-1]

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

        if zero_coeffs_flag:
            e1_star = 0.0
            E2_star = 0.0
            J2_star = 0.0
            j1_star = 0.0
            covEJ_star = 0.0
        else:
            e1_star = get(NEG_E1)
            E2_star = max(get(E2), 0.0)
            J2_star = max(get(J2), 0.0)
            j1_star = get(J1)
            covEJ_star = get(ZETA2)

        scale = pstar_val / PSTAR_CANON
        e1_star *= scale
        E2_star *= scale
        j1_star *= scale
        J2_star *= scale
        covEJ_star *= scale

        if zero_drift_flag:
            e1_star = 0.0
            j1_star = 0.0

        if zero_diffusion_flag:
            E2_star = 0.0
            J2_star = 0.0
            covEJ_star = 0.0

        if not use_sm78_physics:
            E2_star *= E2_scale_local
            J2_star *= J2_scale_local
            covEJ_star *= covEJ_scale_local

            if abs(E2_x_power_local) > 1e-10:
                x_scale = x_clamp / E2_x_ref_local
                if x_scale > 100.0 and E2_x_power_local > 0.0:
                    x_scale = 100.0
                E2_star *= (x_scale ** E2_x_power_local)

        v0_sq = 1.0
        Jmax = 1.0 / math.sqrt(2.0 * x_clamp)

        # e1*, ε₂*, j₁*, j₂* in Table 1 are *already* the per-orbit RMS
        # coefficients used in eqs. (27)–(29). We should NOT take square roots.
        e1 = -e1_scale_val * e1_star * v0_sq          # drift term ε₁
        sigE = max(E2_star, 0.0) * v0_sq              # ε₂*

        j1 = j1_star * Jmax                           # drift term j₁
        sigJ = max(J2_star, 0.0) * Jmax               # j₂*

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
               diag_counts=None, lc_floor_frac=0.10, lc_gap_scale=None,
               step_size_factor_val=1.0, n_max_override_val=3.0e4,
               use_sm78_physics=False):
        """
        Compute step size n using SM78 eq. (29) constraints in dimensionless form.
        
        All constraints are expressed in terms of:
        - x (binding energy, dimensionless)
        - j (dimensionless angular momentum, J/J_max)
        - j_min (loss-cone boundary at this x)
        - E2_star, J2_star (dimensionless Table-1 coefficients)
        """
        jmin = j_min_of_x(x, lc_scale_val, noloss_flag, use_sm78_physics)
        Jmax = 1.0 / math.sqrt(2.0 * x)
        
        # Extract dimensionless coefficients from dimensional sigE, sigJ
        # sigE = E2_star * v0_sq = E2_star (since v0_sq = 1.0)
        # sigJ = J2_star * Jmax, so J2_star = sigJ / Jmax
        E2_star = sigE  # Already dimensionless
        J2_star = sigJ / Jmax if Jmax > 0.0 else 0.0  # Dimensionless
        
        HUGE = 1.0e60

        # Eq. (29a): √n * ε₂* <= 0.15 * x
        # => n <= (0.15 * x / ε₂*)^2
        if E2_star > 0.0 and x > 0.0:
            n_E = (step_size_factor_val * 0.15 * x / E2_star) ** 2
        else:
            n_E = HUGE

        # If angular diffusion is zero, no useful J-constraints
        if J2_star > 0.0:
            # Eq. (29b): √n * j₂* <= 0.10
            # => n <= (0.10 / j₂*)^2
            n_J_iso = (step_size_factor_val * 0.10 / J2_star) ** 2

            # Eq. (29c): √n * j₂* <= 0.40 * (1.0075 - j)
            # => n <= (0.40 * (1.0075 - j) / j₂*)^2
            margin = max(1.0075 - j, 0.0)
            if margin > 0.0:
                n_J_top = (step_size_factor_val * 0.40 * margin / J2_star) ** 2
            else:
                n_J_top = HUGE

            # Eq. (29d): √n * j₂* <= max(0.25*|j - j_min|, 0.10*j_min)
            # => n <= (max(0.25*|j - j_min|, 0.10*j_min) / j₂*)^2
            if noloss_flag or jmin <= 0.0:
                n_J_lc = HUGE
            else:
                gap_scale = cone_gamma_val if lc_gap_scale is None else lc_gap_scale
                d1 = gap_scale * abs(j - jmin)  # γ |j - j_min|
                d2 = lc_floor_frac * jmin if lc_floor_frac > 0.0 else 0.0  # f_floor * j_min
                d_allowed = max(d1, d2)  # max(γ |j - j_min|, f_floor * j_min)
                if d_allowed > 0.0:
                    n_J_lc = (step_size_factor_val * d_allowed / J2_star) ** 2
                else:
                    n_J_lc = HUGE
                if n_J_lc < 1.0:
                    n_J_lc = 1.0
                if n_J_lc > n_max:
                    n_J_lc = n_max
        else:
            n_J_iso = HUGE
            n_J_top = HUGE
            n_J_lc = HUGE

        n = n_E
        winner = 0
        if n_J_iso < n:
            n = n_J_iso
            winner = 1
        if n_J_top < n:
            n = n_J_top
            winner = 2
        if n_J_lc < n:
            n = n_J_lc
            winner = 3
        
        n_max_actual = n_max_override_val if n_max_override_val < n_max else n_max
        if n > n_max_actual:
            n = n_max_actual
        if n < n_min:
            n = n_min
        
        if diag_counts is not None:
            diag_counts[winner] += 1
        
        return n

    @njit(**fastmath)
    def correlated_normals(rho, sigE, sigJ):
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
        k = -1
        for i in range(floors.size):
            if j2 < floors[i]:
                k = i
        return k

    @njit(**fastmath)
    def target_floor_with_hysteresis(j2, floors, current_idx, split_hyst=0.8):
        k = current_idx
        while (k + 1) < floors.size and j2 < split_hyst * floors[k + 1]:
            k += 1
        return k

    @njit(**fastmath)
    def step_one(x, j, phase, pstar_val, noloss_flag, x_max,
                 lc_scale_val, cone_gamma_val, e1_scale_val,
                 diag_counts=None, disable_capture=False,
                 lc_floor_frac=0.10, lc_gap_scale=None,
                 ghost_cap_hist=None, x_bins=None,
                 zero_coeffs_flag=False, zero_drift_flag=False,
                 zero_diffusion_flag=False, step_size_factor_val=1.0,
                 n_max_override_val=3.0e4, E2_scale_local=1.0,
                 J2_scale_local=1.0, covEJ_scale_local=1.0,
                 E2_x_power_local=0.0, E2_x_ref_local=1.0,
                 use_sm78_physics=False):
        E2_scale_val = E2_scale_local
        J2_scale_val = J2_scale_local
        covEJ_scale_val = covEJ_scale_local
        e1, sigE, j1, sigJ, rho = bilinear_coeffs(x, j, pstar_val, e1_scale_val,
                                                   zero_coeffs_flag, zero_drift_flag,
                                                   zero_diffusion_flag,
                                                   E2_scale_local, J2_scale_local,
                                                   covEJ_scale_local,
                                                   E2_x_power_local, E2_x_ref_local,
                                                   use_sm78_physics)

        n_raw = pick_n(x, j, sigE, sigJ, lc_scale_val, noloss_flag,
                       cone_gamma_val, diag_counts=diag_counts,
                       lc_floor_frac=lc_floor_frac, lc_gap_scale=lc_gap_scale,
                       step_size_factor_val=step_size_factor_val,
                       n_max_override_val=n_max_override_val,
                       use_sm78_physics=use_sm78_physics)

        E = -x
        Jmax = 1.0 / math.sqrt(2.0 * x)
        J = j * Jmax

        # Handle multiple integer orbit crossings: check capture at each pericenter
        phase_curr = phase
        n_remaining = n_raw
        x_curr = x
        j_curr = j
        captured = False
        ghost_captured = False
        crossed_peri = False

        while n_remaining > 0.0:
            next_int = math.floor(phase_curr) + 1.0
            n_to_next_int = next_int - phase_curr
            
            if n_remaining <= n_to_next_int:
                # This step doesn't cross an integer boundary
                n_step = n_remaining
                n_remaining = 0.0
                will_cross = False
            else:
                # This step crosses at least one integer boundary
                n_step = n_to_next_int
                n_remaining -= n_step
                will_cross = True

            # Compute step for this sub-interval
            e1_curr, sigE_curr, j1_curr, sigJ_curr, rho_curr = bilinear_coeffs(
                x_curr, j_curr, pstar_val, e1_scale_val,
                zero_coeffs_flag, zero_drift_flag, zero_diffusion_flag,
                E2_scale_local, J2_scale_local, covEJ_scale_local,
                E2_x_power_local, E2_x_ref_local, use_sm78_physics
            )

            y1_step, y2_step = correlated_normals(rho_curr, sigE_curr, sigJ_curr)

            E_curr = -x_curr
            deltaE_step = n_step * e1_curr + math.sqrt(n_step) * sigE_curr * y1_step
            E_step_new = E_curr + deltaE_step
            if E_step_new >= -1e-6:
                E_step_new = -1e-6
            x_step_new = -E_step_new

            if x_step_new > x_max:
                x_step_new = x_max
                E_step_new = -x_step_new

            Jmax_step_new = 1.0 / math.sqrt(2.0 * x_step_new)
            Jmax_curr = 1.0 / math.sqrt(2.0 * x_curr)
            
            # Extract dimensionless coefficients for eq. (28) / (27b)
            # sigJ_curr = J2_star * Jmax_curr, so J2_star = sigJ_curr / Jmax_curr
            # j1_curr = j1_star * Jmax_curr, so j1_star = j1_curr / Jmax_curr
            J2_star_curr = sigJ_curr / Jmax_curr if Jmax_curr > 0.0 else 0.0
            j1_star_curr = j1_curr / Jmax_curr if Jmax_curr > 0.0 else 0.0

            # SM78 eq. (28): Use isotropic 2D random walk in dimensionless j-space ONLY when:
            # 1. Inside the loss cone: j <= j_min
            # 2. Low j: j <= 0.4
            # 3. Large step: n^{1/2} j₂* >= j / 4
            j_min_curr = j_min_of_x(x_curr, lc_scale_val, noloss_flag, use_sm78_physics)
            inside_loss_cone = (j_curr <= j_min_curr)
            low_j = (j_curr <= 0.4)
            large_step = (math.sqrt(n_step) * J2_star_curr >= 0.25 * j_curr)
            use_2d = inside_loss_cone and low_j and large_step

            if use_2d:
                # Eq. (28): isotropic 2D random walk in dimensionless j-space
                # Δj = sqrt((j + sqrt(n) y₃ j₂*)^2 + (sqrt(n) y₄ j₂*)^2) - j
                z1 = np.random.normal()
                z2 = np.random.normal()
                j_step_new = math.sqrt(
                    (j_curr + math.sqrt(n_step) * z1 * J2_star_curr) ** 2 +
                    (math.sqrt(n_step) * z2 * J2_star_curr) ** 2
                )
            else:
                # Eq. (27b): small-angle, correlated step in dimensionless j-space
                # Δj = n j₁* + sqrt(n) j₂* y₂
                j_step_new = j_curr + n_step * j1_star_curr + math.sqrt(n_step) * J2_star_curr * y2_step

            # Enforce bounds on dimensionless j
            if j_step_new < 0.0:
                j_step_new = -j_step_new
            if j_step_new > 1.0:
                j_step_new = 2.0 - j_step_new  # Reflect at j = 1 boundary
                if j_step_new < 0.0:
                    j_step_new = 0.0
                if j_step_new > 1.0:
                    j_step_new = 1.0
            phase_curr += n_step

            # Update state for this sub-step
            x_curr = x_step_new
            j_curr = j_step_new
            if will_cross:
                crossed_peri = True

            # Check capture at integer pericenter crossings
            if will_cross and (not noloss_flag):
                j_min_new = j_min_of_x(x_step_new, lc_scale_val, noloss_flag, use_sm78_physics)
                if j_step_new < j_min_new:
                    if not disable_capture:
                        captured = True
                        break
                    else:
                        ghost_captured = True

        # Final state
        x_new = x_curr
        j_new = j_curr
        phase_new = phase_curr
        n = phase_new - phase
        
        if ghost_cap_hist is not None and ghost_captured:
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
                   outer_injection=False,
                   outer_inj_x_min=10.0,
                   zero_coeffs_flag=False,
                   zero_drift_flag=False,
                   zero_diffusion_flag=False,
                   step_size_factor=1.0,
                   n_max_override=None,
                   E2_scale=1.0,
                   J2_scale=1.0,
                   covEJ_scale=1.0,
                   E2_x_power=0.0,
                   E2_x_ref=1.0,
                   use_sm78_physics=False):
        np.random.seed(seed)
        SPLIT_HYST = 0.8
        noloss_flag = noloss
        n_max_override_val = n_max_override if n_max_override is not None else 3.0e4
        
        if enable_diag_counts:
            diag_counts = np.zeros(4, dtype=np.int64)
        else:
            diag_counts = None
        
        if enable_cap_inj_diag:
            cap_hist = np.zeros(x_bins.size, dtype=np.int64)
            inj_hist = np.zeros(x_bins.size, dtype=np.int64)
            ghost_cap_hist = np.zeros(x_bins.size, dtype=np.int64)
        else:
            cap_hist = None
            inj_hist = None
            ghost_cap_hist = None
        
        gap_scale_val = None if lc_gap_scale < 0.0 else lc_gap_scale

        x = x_init
        j = math.sqrt(np.random.random())
        phase = 0.0
        w = 1.0
        parent_floor_idx = floor_index_for_j2(j * j, floors)

        MAX_CLONES = 65536 if use_clones else 0
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

        t0_used = 0.0
        while t0_used < warmup_t0:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(
                x, j, phase, pstar_val, noloss_flag, x_max,
                lc_scale_val, cone_gamma_val, e1_scale_val,
                diag_counts=diag_counts, disable_capture=disable_capture,
                lc_floor_frac=lc_floor_frac, lc_gap_scale=gap_scale_val,
                ghost_cap_hist=ghost_cap_hist, x_bins=x_bins,
                zero_coeffs_flag=zero_coeffs_flag, zero_drift_flag=zero_drift_flag,
                zero_diffusion_flag=zero_diffusion_flag,
                step_size_factor_val=step_size_factor,
                n_max_override_val=n_max_override_val,
                E2_scale_local=E2_scale, J2_scale_local=J2_scale,
                covEJ_scale_local=covEJ_scale,
                E2_x_power_local=E2_x_power, E2_x_ref_local=E2_x_ref,
                use_sm78_physics=use_sm78_physics
            )
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0

            if cap or (x < X_BOUND):
                if use_sm78_physics:
                    x, j = inject_star_sm78()
                else:
                    if noloss_flag:
                        x = sample_x_from_g0_jit()
                    else:
                        if outer_injection:
                            x = sample_x_from_g0_jit()
                            while x < outer_inj_x_min:
                                x = sample_x_from_g0_jit()
                        else:
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
                        ghost_cap_hist=ghost_cap_hist, x_bins=x_bins,
                        zero_coeffs_flag=zero_coeffs_flag, zero_drift_flag=zero_drift_flag,
                        zero_diffusion_flag=zero_diffusion_flag,
                        step_size_factor_val=step_size_factor,
                        n_max_override_val=n_max_override_val,
                        E2_scale_local=E2_scale, J2_scale_local=J2_scale,
                        covEJ_scale_local=covEJ_scale,
                        E2_x_power_local=E2_x_power, E2_x_ref_local=E2_x_ref,
                        use_sm78_physics=use_sm78_physics
                    )
                    cx[i] = x_c2
                    cj[i] = j_c2
                    cph[i] = ph_c2

                    if cap_c or (x_c2 < X_BOUND):
                        if use_sm78_physics:
                            cx[i], cj[i] = inject_star_sm78()
                        else:
                            if noloss_flag:
                                cx[i] = sample_x_from_g0_jit()
                            else:
                                if outer_injection:
                                    x_inj = sample_x_from_g0_jit()
                                    while x_inj < outer_inj_x_min:
                                        x_inj = sample_x_from_g0_jit()
                                    cx[i] = x_inj
                                else:
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

        if use_snapshots:
            g_acc = np.zeros(x_bins.size, dtype=np.float64)
            snapshots = 0.0
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
        cap_E_sum = 0.0  # sum of binding energy x at captures, weighted by stream/clone weight

        while t0_used < n_relax:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(
                x, j, phase, pstar_val, noloss_flag, x_max,
                lc_scale_val, cone_gamma_val, e1_scale_val,
                diag_counts=diag_counts, disable_capture=disable_capture,
                lc_floor_frac=lc_floor_frac, lc_gap_scale=gap_scale_val,
                ghost_cap_hist=ghost_cap_hist, x_bins=x_bins,
                zero_coeffs_flag=zero_coeffs_flag, zero_drift_flag=zero_drift_flag,
                zero_diffusion_flag=zero_diffusion_flag,
                step_size_factor_val=step_size_factor,
                n_max_override_val=n_max_override_val,
                E2_scale_local=E2_scale, J2_scale_local=J2_scale,
                covEJ_scale_local=covEJ_scale,
                E2_x_power_local=E2_x_power, E2_x_ref_local=E2_x_ref,
                use_sm78_physics=use_sm78_physics
            )
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0

            if use_snapshots:
                t0_local += dt0
                while t0_local >= next_snap and t0_used <= n_relax + 1e-9:
                    g_acc[bin_index(x)] += w
                    if use_clones and include_clone_gbar:
                        for ii in range(ccount):
                            if cactive[ii]:
                                g_acc[bin_index(cx[ii])] += cw[ii]
                    snapshots += 1.0
                    next_snap += delta_snap
            else:
                total_t0 += dt0
                g_acc[bin_index(x_prev)] += w * dt0
                if use_clones and include_clone_gbar:
                    for ii in range(ccount):
                        if cactive[ii]:
                            g_acc[bin_index(cx[ii])] += cw[ii] * dt0

            if crossed:
                crossings += 1

            if cap or (x < X_BOUND):
                if cap:
                    caps += 1
                    if cap_hist is not None:
                        cap_hist[bin_index(x_prev)] += 1
                    # binding energy removed by this capture (weight * x at start of step)
                    cap_E_sum += w * x_prev
                
                if use_sm78_physics:
                    x, j = inject_star_sm78()
                    if inj_hist is not None:
                        inj_hist[bin_index(x)] += 1
                else:
                    if inj_hist is not None:
                        if outer_injection and not noloss_flag:
                            x_inj = sample_x_from_g0_jit()
                            while x_inj < outer_inj_x_min:
                                x_inj = sample_x_from_g0_jit()
                            x = x_inj
                            inj_hist[bin_index(x)] += 1
                        elif noloss_flag:
                            x = sample_x_from_g0_jit()
                            inj_hist[bin_index(x)] += 1
                        else:
                            x = X_BOUND
                            inj_hist[bin_index(x)] += 1
                    else:
                        if noloss_flag:
                            x = sample_x_from_g0_jit()
                        else:
                            if outer_injection:
                                x = sample_x_from_g0_jit()
                                while x < outer_inj_x_min:
                                    x = sample_x_from_g0_jit()
                            else:
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
                        ghost_cap_hist=ghost_cap_hist, x_bins=x_bins,
                        zero_coeffs_flag=zero_coeffs_flag, zero_drift_flag=zero_drift_flag,
                        zero_diffusion_flag=zero_diffusion_flag,
                        step_size_factor_val=step_size_factor,
                        n_max_override_val=n_max_override_val,
                        E2_scale_local=E2_scale, J2_scale_local=J2_scale,
                        covEJ_scale_local=covEJ_scale,
                        E2_x_power_local=E2_x_power, E2_x_ref_local=E2_x_ref,
                        use_sm78_physics=use_sm78_physics
                    )

                    cx[i] = x_c2
                    cj[i] = j_c2
                    cph[i] = ph_c2

                    if cross_c:
                        crossings += 1

                    if cap_c or (x_c2 < X_BOUND):
                        if cap_c:
                            caps += 1
                            if cap_hist is not None:
                                cap_hist[bin_index(x_c)] += 1
                            # binding energy removed by this clone capture
                            cap_E_sum += cw[i] * x_c
                        
                        if use_sm78_physics:
                            cx[i], cj[i] = inject_star_sm78()
                            if inj_hist is not None:
                                inj_hist[bin_index(cx[i])] += 1
                        else:
                            if inj_hist is not None:
                                if outer_injection and not noloss_flag:
                                    x_inj = sample_x_from_g0_jit()
                                    while x_inj < outer_inj_x_min:
                                        x_inj = sample_x_from_g0_jit()
                                    cx[i] = x_inj
                                    inj_hist[bin_index(x_inj)] += 1
                                elif noloss_flag:
                                    cx[i] = sample_x_from_g0_jit()
                                    inj_hist[bin_index(cx[i])] += 1
                                else:
                                    cx[i] = X_BOUND
                                    inj_hist[bin_index(cx[i])] += 1
                            else:
                                if noloss_flag:
                                    cx[i] = sample_x_from_g0_jit()
                                else:
                                    if outer_injection:
                                        x_inj = sample_x_from_g0_jit()
                                        while x_inj < outer_inj_x_min:
                                            x_inj = sample_x_from_g0_jit()
                                        cx[i] = x_inj
                                    else:
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

        weight_sum = w
        for ii in range(ccount):
            if cactive[ii]:
                weight_sum += cw[ii]

        if diag_counts is None:
            diag_counts = np.zeros(4, dtype=np.int64)
        if cap_hist is None:
            cap_hist = np.zeros(x_bins.size, dtype=np.int64)
        if inj_hist is None:
            inj_hist = np.zeros(x_bins.size, dtype=np.int64)
        if ghost_cap_hist is None:
            ghost_cap_hist = np.zeros(x_bins.size, dtype=np.int64)
        
        if use_snapshots:
            # measure_b = snapshots, t0_b = t0_used
            return (
                g_acc,
                snapshots,
                t0_used,
                crossings,
                caps,
                cap_E_sum,
                max_ccount,
                weight_sum,
                diag_counts,
                cap_hist,
                inj_hist,
                ghost_cap_hist,
            )
        else:
            # measure_b = total_t0, t0_b = total_t0
            return (
                g_acc,
                total_t0,
                total_t0,
                crossings,
                caps,
                cap_E_sum,
                max_ccount,
                weight_sum,
                diag_counts,
                cap_hist,
                inj_hist,
                ghost_cap_hist,
            )

    return P_of_x, T0, run_stream, sample_x_from_g0_jit


def _worker_one(args):
    (sid, n_relax, floors, clones_per_split,
     stream_seeds, use_jit, pstar_val, warmup_t0, u0,
     include_clone_gbar, use_snapshots, snaps_per_t0,
     noloss, use_clones, lc_scale, e1_scale, x_max, cone_gamma,
     enable_diag_counts, disable_capture, lc_floor_frac, lc_gap_scale,
     enable_cap_inj_diag, outer_injection, outer_inj_x_min,
     zero_coeffs, zero_drift, zero_diffusion, step_size_factor, n_max_override,
     E2_scale, J2_scale, covEJ_scale, E2_x_power, E2_x_ref, use_sm78_physics) = args

    P_of_x, T0, run_stream, sample_x_from_g0_jit = _build_kernels(
        use_jit=use_jit, noloss=noloss, lc_scale=lc_scale,
        e1_scale=e1_scale, zero_coeffs=zero_coeffs,
        zero_drift=zero_drift, zero_diffusion=zero_diffusion,
        step_size_factor=step_size_factor, n_max_override=n_max_override,
        E2_scale=E2_scale, J2_scale=J2_scale, covEJ_scale=covEJ_scale,
        E2_x_power=E2_x_power, E2_x_ref=E2_x_ref, use_sm78_physics=use_sm78_physics
    )

    x_init = sample_x_from_g0(u0)

    _ = run_stream(
        1e-6, floors, clones_per_split, X_BINS, DX,
        int(stream_seeds[sid] ^ 0xABCDEF), pstar_val, 0.0, x_init,
        include_clone_gbar, use_snapshots, noloss, use_clones,
        x_max, lc_scale, cone_gamma, e1_scale, snaps_per_t0,
        enable_diag_counts, disable_capture, lc_floor_frac, lc_gap_scale,
        enable_cap_inj_diag, outer_injection, outer_inj_x_min,
        zero_coeffs, zero_drift, zero_diffusion, step_size_factor, n_max_override,
        E2_scale, J2_scale, covEJ_scale, E2_x_power, E2_x_ref, use_sm78_physics
    )

    result = run_stream(
        n_relax, floors, clones_per_split, X_BINS, DX,
        int(stream_seeds[sid]), pstar_val, warmup_t0, x_init,
        include_clone_gbar, use_snapshots, noloss, use_clones,
        x_max, lc_scale, cone_gamma, e1_scale, snaps_per_t0,
        enable_diag_counts, disable_capture, lc_floor_frac, lc_gap_scale,
        enable_cap_inj_diag, outer_injection, outer_inj_x_min,
        zero_coeffs, zero_drift, zero_diffusion, step_size_factor, n_max_override,
        E2_scale, J2_scale, covEJ_scale, E2_x_power, E2_x_ref, use_sm78_physics
    )
    return result


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
    outer_injection=False,
    outer_inj_x_min=10.0,
    zero_coeffs=False,
    zero_drift=False,
    zero_diffusion=False,
    step_size_factor=1.0,
    n_max_override=None,
    E2_scale=1.0,
    J2_scale=1.0,
    covEJ_scale=1.0,
    E2_x_power=0.0,
    E2_x_ref=1.0,
    gbar_from_flux=False,
    gbar_x_norm=0.225,
    gbar_flux_exp=3.10,
    use_sm78_physics=False,
):
    if noloss:
        use_clones = False

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
    total_t0_measured = 0.0   # actual time in units of t0 (for capture/energy rates)
    peri_total = 0
    caps_total = 0
    capE_total = 0.0          # total binding energy removed by captures (sum w*x)
    max_ccount = 0
    total_done = 0
    weight_sums = []
    diag_counts_total = np.zeros(4, dtype=np.int64)
    cap_hist_total = np.zeros_like(X_BINS, dtype=np.int64)
    inj_hist_total = np.zeros_like(X_BINS, dtype=np.int64)
    ghost_cap_hist_total = np.zeros_like(X_BINS, dtype=np.int64)

    for rep in range(replicates):
        stream_seeds = rs.randint(1, 2**31 - 1, size=n_streams, dtype=np.int64)
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
                        zero_coeffs,
                        zero_drift,
                        zero_diffusion,
                        step_size_factor,
                        n_max_override,
                        E2_scale,
                        J2_scale,
                        covEJ_scale,
                        E2_x_power,
                        E2_x_ref,
                        use_sm78_physics,
                    ),
                )
                for sid in range(n_streams)
            ]

            for f in cf.as_completed(futs):
                (
                    g_b,
                    measure_b,
                    t0_b,
                    peri_b,
                    caps_b,
                    capE_b,
                    ccount_b,
                    wsum_b,
                    diag_b,
                    cap_b,
                    inj_b,
                    ghost_b,
                ) = f.result()

                g_total += g_b
                total_measure += measure_b       # snapshots or total t0, as before
                total_t0_measured += t0_b        # always total time in units of t0
                peri_total += peri_b
                caps_total += caps_b
                capE_total += capE_b
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
                N_occ = g_total.copy()
                for i in range(X_BINS.size):
                    if cap_hist_total[i] > 0 or inj_hist_total[i] > 0 or N_occ[i] > 0:
                        if N_occ[i] > 0 and total_measure > 0:
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
        # --- Global capture & energy-outflow diagnostics (for comparison to SM78 eqs. 31 & 33) ---
        if total_t0_measured > 0.0 and caps_total > 0:
            F_mc = caps_total / total_t0_measured          # capture rate per t0
            E_cap_mean = capE_total / caps_total           # mean binding energy (in x units) per capture
            Edot_mc = capE_total / total_t0_measured       # energy outflow per t0 (sum x / Δt0)

            print(
                f"[diag] capture/heating diagnostics:",
                file=sys.stderr,
            )
            print(
                f"       total_t0_measured = {total_t0_measured:.3e}",
                file=sys.stderr,
            )
            print(
                f"       total captures    = {caps_total:d}",
                file=sys.stderr,
            )
            print(
                f"       F_MC (captures per t0)     = {F_mc:.3e}",
                file=sys.stderr,
            )
            print(
                f"       <x>_cap (mean binding x)   = {E_cap_mean:.3e}",
                file=sys.stderr,
            )
            print(
                f"       Ė_MC (sum x per t0)        = {Edot_mc:.3e}",
                file=sys.stderr,
            )
        elif total_t0_measured > 0.0 and caps_total == 0:
            print(
                f"[diag] capture/heating diagnostics: total_t0_measured={total_t0_measured:.3e}, no captures recorded.",
                file=sys.stderr,
            )

    if total_measure <= 0.0:
        raise RuntimeError("No time/snapshots accumulated; check parameters.")

    if use_snapshots:
        n_snaps = total_measure
        N_x = g_total / (n_snaps * DX)

        if show_progress:
            print(f"[diag] snapshot mode: n_snaps = {n_snaps}", file=sys.stderr)
        
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
        
        norm_idx = 0 if N_x[0] > 0 else positive_bins[0]
        if show_progress:
            print(
                f"[diag] snapshot mode: normalizing on bin {norm_idx} "
                f"(x={X_BINS[norm_idx]:.3g}, N_x={N_x[norm_idx]:.6e})",
                file=sys.stderr,
            )
        
        if gbar_from_flux:
            if not enable_cap_inj_diag:
                raise RuntimeError(
                    "--gbar-from-flux requires --cap-inj-diag to accumulate capture statistics."
                )
            p_flux = gbar_flux_exp
            N_occ = g_total.copy()
            gbar_norm, gbar_raw = make_gbar_from_flux(
                x_centers=X_BINS,
                N_cap=cap_hist_total,
                N_occ=N_occ,
                total_measure=total_measure,
                flux_exp=p_flux,
                x_norm=gbar_x_norm,
            )
            if show_progress:
                print(
                    f"[diag] ḡ mode: using capture flux Γ_cap(x) / x^{p_flux:.2f}, "
                    f"normalized at x={gbar_x_norm:.3g}",
                    file=sys.stderr,
                )
        elif debug_occupancy_norm:
            norm_idx_occ = norm_idx if N_x[norm_idx] > 0 else positive_bins[0]
            gbar_norm = N_x / N_x[norm_idx_occ]
            gbar_raw = N_x.copy()
            if show_progress:
                print(
                    f"[diag] snapshot mode: returning occupancy N_x only (debug mode), "
                    f"normalized at x={X_BINS[norm_idx_occ]:.3g}",
                    file=sys.stderr,
                )
                # Also fit a simple power law N(E) ∝ x^p in 1≲x≲100
                mask = (
                    (X_BINS >= 1.0)
                    & (X_BINS <= 100.0)
                    & np.isfinite(N_x)
                    & (N_x > 0.0)
                )
                if mask.sum() >= 2:
                    logx = np.log10(X_BINS[mask])
                    logN = np.log10(N_x[mask])
                    p_fit, _ = np.polyfit(logx, logN, 1)
                    print(
                        f"[diag] best-fit N(E) ∝ x^{p_fit:.3f} in 1≲x≲100 "
                        "(occupancy slope, before applying gexp).",
                        file=sys.stderr,
                    )
        else:
            gbar_norm, gbar_raw = make_gbar_from_occupancy(
                x_centers=X_BINS,
                N_x=N_x,
                gexp=gexp,
                x_norm=gbar_x_norm,
            )
            if show_progress:
                if abs(gexp - 2.5) < 1e-6:
                    print(
                        f"[diag] ḡ mode: using snapshot occupancy N(E) * x^2.5, "
                        f"normalized at x={gbar_x_norm:.3g}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[diag] ḡ mode: using snapshot occupancy N(E) * x^{gexp:.2f} "
                        f"(DIAGNOSTIC - paper value is 2.5), normalized at x={gbar_x_norm:.3g}",
                        file=sys.stderr,
                    )
            
    else:
        p_x = g_total / total_measure
        p_sum = p_x.sum()
        if show_progress:
            print(f"[diag] sum p_x = {p_sum:.6f}", file=sys.stderr)
        N_x = p_x / DX

        norm_idx = 0
        
        if debug_occupancy_norm:
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
                # Also fit a simple power law N(E) ∝ x^p in 1≲x≲100
                mask = (
                    (X_BINS >= 1.0)
                    & (X_BINS <= 100.0)
                    & np.isfinite(N_x)
                    & (N_x > 0.0)
                )
                if mask.sum() >= 2:
                    logx = np.log10(X_BINS[mask])
                    logN = np.log10(N_x[mask])
                    p_fit, _ = np.polyfit(logx, logN, 1)
                    print(
                        f"[diag] best-fit N(E) ∝ x^{p_fit:.3f} in 1≲x≲100 "
                        "(occupancy slope, before applying gexp).",
                        file=sys.stderr,
                    )
            return gbar_norm, gbar_raw
        else:
            if gbar_from_flux:
                if not enable_cap_inj_diag:
                    raise RuntimeError(
                        "--gbar-from-flux requires --cap-inj-diag to accumulate capture statistics."
                    )
                N_occ = g_total.copy()
                p_flux = gbar_flux_exp
                gbar_norm, gbar_raw = make_gbar_from_flux(
                    x_centers=X_BINS,
                    N_cap=cap_hist_total,
                    N_occ=N_occ,
                    total_measure=total_measure,
                    flux_exp=p_flux,
                    x_norm=gbar_x_norm,
                )
                if show_progress:
                    print(
                        f"[diag] ḡ mode: using capture flux Γ_cap(x) / x^{p_flux:.2f}, "
                        f"normalized at x={gbar_x_norm:.3g}",
                        file=sys.stderr,
                    )
            else:
                gbar_norm, gbar_raw = make_gbar_from_occupancy(
                    x_centers=X_BINS,
                    N_x=N_x,
                    gexp=gexp,
                    x_norm=gbar_x_norm,
                )
                if show_progress:
                    if abs(gexp - 2.5) < 1e-6:
                        print(
                            f"[diag] ḡ mode: using production occupancy N(E) * x^2.5, "
                            f"normalized at x={gbar_x_norm:.3g}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[diag] ḡ mode: using production occupancy N(E) * x^{gexp:.2f} "
                            f"(DIAGNOSTIC - paper value is 2.5), normalized at x={gbar_x_norm:.3g}",
                            file=sys.stderr,
                        )

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
            "Exponent in g(x) ∝ N(E) x^gexp conversion. "
            "For an apples-to-apples comparison with Shapiro & Marchant (1978), "
            "this should be 2.5 as implied by eqs. (9), (11) and (13). "
            "The parameter is kept for diagnostic testing only (default: 2.5)."
        ),
    )
    ap.add_argument(
        "--use-tunable-gexp",
        action="store_true",
        help=(
            "DEPRECATED: This flag is now a no-op. The code always uses the --gexp "
            "parameter. For Fig. 3 reproduction, use --gexp 2.5 (the default)."
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
        "--outer-injection",
        action="store_true",
        help="Use outer-boundary injection (x >= --outer-inj-x-min from reservoir). "
             "By default, x-bound injection is used (x_b = 0.2, canonical SM78).",
    )
    ap.add_argument(
        "--use-x-bound-injection",
        action="store_true",
        help="DEPRECATED: Use --outer-injection to enable outer-boundary injection. "
             "By default, x-bound injection (x_b = 0.2) is used for canonical SM78.",
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
    ap.add_argument(
        "--zero-coeffs",
        action="store_true",
        help="DIAGNOSTIC: Zero out all drift and diffusion coefficients (Step 1 test).",
    )
    ap.add_argument(
        "--zero-drift",
        action="store_true",
        help="DIAGNOSTIC: Zero out drift terms (e1, j1) to test diffusion only (Step 2a).",
    )
    ap.add_argument(
        "--zero-diffusion",
        action="store_true",
        help="DIAGNOSTIC: Zero out diffusion terms (E2, J2, covEJ) to test drift only (Step 2b).",
    )
    ap.add_argument(
        "--step-size-factor",
        type=float,
        default=1.0,
        help="DIAGNOSTIC: Scale factor for eq. 29 step-size constants (default: 1.0). Use <1.0 for smaller steps (Step 3).",
    )
    ap.add_argument(
        "--n-max-override",
        type=float,
        default=None,
        help="DIAGNOSTIC: Override maximum step size n_max (default: 3.0e4). Use smaller values to force smaller steps (Step 3).",
    )
    ap.add_argument(
        "--E2-scale",
        type=float,
        default=1.0,
        help="DIAGNOSTIC: Scale factor for energy diffusion E2* (default: 1.0). Use to test if missing factor in diffusion.",
    )
    ap.add_argument(
        "--J2-scale",
        type=float,
        default=1.0,
        help="DIAGNOSTIC: Scale factor for angular momentum diffusion J2* (default: 1.0). Use to test if missing factor in diffusion.",
    )
    ap.add_argument(
        "--covEJ-scale",
        type=float,
        default=1.0,
        help="DIAGNOSTIC: Scale factor for E-J covariance ζ*² (default: 1.0). Use to test if missing factor in correlation.",
    )
    ap.add_argument(
        "--E2-x-power",
        type=float,
        default=0.0,
        help="DIAGNOSTIC: Energy-dependent power-law scaling for E2: E2_eff = E2 * (x/x_ref)^power (default: 0.0). Use to test what exponent would fix the slope.",
    )
    ap.add_argument(
        "--E2-x-ref",
        type=float,
        default=1.0,
        help="DIAGNOSTIC: Reference energy for --E2-x-power scaling (default: 1.0).",
    )
    ap.add_argument(
        "--gbar-from-flux",
        action="store_true",
        help="Compute ḡ(x) from capture flux diagnostics instead of snapshot occupancy (experimental).",
    )
    ap.add_argument(
        "--gbar-x-norm",
        type=float,
        default=0.225,
        help="x at which ḡ is normalized to 1 (default: 0.225, matching paper).",
    )
    flux_exp_group = ap.add_mutually_exclusive_group()
    flux_exp_group.add_argument(
        "--gbar-flux-exp",
        type=float,
        default=3.10,
        help=(
            "Exponent p in ḡ ∝ Γ_cap(x) / x^p when using --gbar-from-flux "
            "(p ≈ gexp + E2_x_power; default: 3.10 for SM78 fiducial). "
            "This value makes flux-based ḡ match occupancy-based ḡ."
        ),
    )
    flux_exp_group.add_argument(
        "--gbar-flux-power",
        type=float,
        dest="gbar_flux_exp",
        help=(
            "Alias for --gbar-flux-exp. Exponent p in ḡ ∝ Γ_cap(x) / x^p "
            "when using --gbar-from-flux (default: 3.10)."
        ),
    )
    ap.add_argument(
        "--use-sm78-physics",
        action="store_true",
        help=(
            "Use exact SM78 diffusion coefficients, loss cone, and outer boundary "
            "instead of parametric tunings (E2_x_power, J2_scale, lc_scale, etc.)."
        ),
    )

    args = ap.parse_args()

    if args.use_sm78_physics:
        args.e1_scale = 1.0
        args.E2_scale = 1.0
        args.J2_scale = 1.0
        args.covEJ_scale = 1.0
        args.E2_x_power = 0.0
        args.E2_x_ref = 1.0
        args.lc_scale = 1.0
        # Canonical SM78 eq. (29d) constants:
        # √n j₂ ≤ max(0.25 |J - J_min|, 0.1 J_min)
        args.lc_floor_frac = 0.10   # 0.1 J_min
        args.cone_gamma   = 0.25    # 0.25 |J - J_min|
        args.lc_gap_scale = None
        args.outer_injection = False

    if args.gbar_from_flux and not args.cap_inj_diag:
        print(
            "[diag] --gbar-from-flux requires --cap-inj-diag; enabling it automatically.",
            file=sys.stderr,
        )
        args.cap_inj_diag = True

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
        outer_injection=args.outer_injection if hasattr(args, 'outer_injection') else False,
        outer_inj_x_min=args.outer_inj_x_min,
        zero_coeffs=args.zero_coeffs,
        zero_drift=args.zero_drift,
        zero_diffusion=args.zero_diffusion,
        step_size_factor=args.step_size_factor,
        n_max_override=args.n_max_override,
        E2_scale=args.E2_scale,
        J2_scale=args.J2_scale,
        covEJ_scale=args.covEJ_scale,
        E2_x_power=args.E2_x_power,
        E2_x_ref=args.E2_x_ref,
        gbar_from_flux=args.gbar_from_flux,
        gbar_x_norm=args.gbar_x_norm,
        gbar_flux_exp=args.gbar_flux_exp,
        use_sm78_physics=args.use_sm78_physics,
    )

    print("# x_center   gbar_MC_norm   gbar_MC_raw      gbar_paper   gbar_err_paper")
    residuals = []
    for xb, g_norm, g_raw, gp, gp_err in zip(
        X_BINS, gbar_norm, gbar_raw, GBAR_PAPER, GBAR_ERR_PAPER
    ):
        print(
            f"{xb:10.3g}  {g_norm:12.6e}  {g_raw:12.6e}  "
            f"{gp:11.4f}  {gp_err:14.4f}"
        )
        if gp_err > 0:
            residuals.append((g_norm - gp) / gp_err)
    
    if residuals:
        residuals_arr = np.array(residuals)
        chi2 = np.sum(residuals_arr**2)
        chi2_per_dof = chi2 / len(residuals_arr)
        print(f"# chi^2/dof = {chi2_per_dof:.3f} (target: < 2.0 for good agreement)")
        print(f"# max |residual| = {np.abs(residuals_arr).max():.3f} sigma")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gbar_mc_fast_sm78.py

Monte Carlo reproduction of the time–averaged phase–space distribution
ḡ(x) from Fig. 3 of Shapiro & Marchant (1978), plus a Bahcall–Wolf
“no–loss–cone” calibration mode.

Key features
------------
- 2D (E, J) Monte Carlo with drift and diffusion coefficients from
  Shapiro–Marchant Table 1 (X_GRID, J_GRID, NEG_E1, E2, J1, J2, ZETA2).
- Loss cone treated with a pericenter check and j_min(x) from x_D.
- Creation–annihilation scheme in j² with clones and floors j² = 10^{-k}.
- Time in units of t0 = P(x=1)/P*, with P* = 0.005 by default.
- ḡ(x) measured in Shapiro–Marchant’s energy bins and normalised so
  that ḡ(x_b) = 1 at the outer cusp boundary x_b = 0.2 (E_b = -0.2 v0²).
- Optional “snapshot” tally mode to mimic their 40 samples per t0
  averaging procedure.

The default/“canonical” Fig. 3–like run is:

    python3 gbar_mc_fast_sm78.py \
        --streams 400 --windows 6 --warmup 2 \
        --replicates 3 --pstar 0.005 --snapshots

The Bahcall–Wolf “no–loss–cone” diagnostic run is:

    python3 gbar_mc_fast_sm78.py \
        --streams 200 --windows 10 --warmup 5 \
        --pstar 0.005 --noloss --gbar-no-clones --gexp 2.5
"""

import argparse
import math
import os
import sys
import time
import numpy as np

# -------- numba (optional speedup) --------
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------- Canonical parameters ----------------
X_D    = 1.0e4        # x_D = energy of circular orbit at disruption radius
PSTAR  = 0.005        # P* parameter
X_BOUND= 0.2          # outer cusp boundary Eb = -0.2 v0²  => x_b = 0.2
SEED   = 20251028

# --------- Fig. 3 energy bin centers + geometric edges ----------
X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=np.float64)

def _edges_from_centers(c):
    e = np.empty(c.size + 1, dtype=c.dtype)
    for i in range(1, c.size):
        e[i] = math.sqrt(c[i-1]*c[i])
    e[0]  = c[0]**2 / e[1]
    e[-1] = c[-1]**2 / e[-2]
    return e

X_EDGES = _edges_from_centers(X_BINS)
DX      = X_EDGES[1:] - X_EDGES[:-1]

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

# Convert ln(1+x) grid -> x grid
X_G0 = np.expm1(LN1P)            # x >= 0
XMAX = X_EDGES[-1]
mask = (X_G0 >= X_BOUND) & (X_G0 <= XMAX) & (G0_TAB > 0)
XG  = X_G0[mask]
G0  = G0_TAB[mask]

# Build an inverse-CDF over x for the *occupancy*
# N(x) ∝ x^{-5/2} g0(x) for a Kepler potential.
weight = G0 * (XG ** -2.5)   # N_x ∝ x^{-5/2} g0(x)

cdf_x = np.zeros_like(XG)
cdf_x[1:] = np.cumsum(
    0.5 * (weight[1:] + weight[:-1]) * (XG[1:] - XG[:-1])
)
cdf_x /= cdf_x[-1]  # normalize to 1

def sample_x_from_g0(u):
    """Monotone piecewise-linear inverse CDF for the BW II reservoir DF."""
    if u <= 0.0:
        return XG[0]
    if u >= 1.0:
        return XG[-1]
    i = np.searchsorted(cdf_x, u, side="right")
    i = max(1, min(i, cdf_x.size-1))
    t = (u - cdf_x[i-1]) / (cdf_x[i] - cdf_x[i-1] + 1e-30)
    return XG[i-1] + t * (XG[i] - XG[i-1])

# ------------- Table-1 grids (x, j) and starred coeffs -------------
X_GRID = np.array([3.36e-1, 3.31e0, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID = np.array([1.000, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)  # descending

NEG_E1 = np.array([
    [ 1.41e-1,  1.40e-1,  1.33e-1,  1.29e-1,  1.29e-1],
    [-1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],
    [ 1.96e-3,  1.59e-3,  2.83e-3,  3.38e-3,  3.49e-3],
    [ 3.64e-3,  4.64e-3,  4.93e-3,  4.97e-3,  4.98e-3],
    [ 8.39e-4,  8.53e-4,  8.56e-4,  8.56e-4,  8.56e-4],
], dtype=np.float64)

E2 = np.array([
    [ 3.14e-1,  4.36e-1,  7.55e-1,  1.37e0,  2.19e0],
    [ 4.45e-1,  8.66e-1,  1.57e0,  2.37e0,  2.73e0],
    [ 1.03e0,   1.80e0,   2.52e0,  2.77e0,  2.81e0],
    [ 1.97e0,   2.54e0,   2.68e0,  2.70e0,  2.70e0],
    [ 1.96e0,   1.96e0,   1.96e0,  1.96e0,  1.96e0],
], dtype=np.float64)

J1 = np.array([
    [-2.52e-2,  5.37e-1,  1.51e0,   3.83e0,   9.58e0],
    [-5.03e-3,  5.40e-3,  2.54e-2,  6.95e-2,  1.75e-1],
    [-2.58e-4,  3.94e-4,  1.45e-3,  3.77e-3,  9.45e-3],
    [-4.67e-6,  2.03e-5,  5.99e-5,  1.53e-4,  3.82e-4],
    [ 3.67e-8,  2.54e-7,  7.09e-7,  1.80e-6,  4.49e-6],
], dtype=np.float64)

J2 = np.array([
    [ 4.68e-1,  6.71e-1,  6.99e-1,  7.03e-1,  7.04e-1],
    [ 6.71e-2,  9.19e-2,  9.48e-2,  9.53e-2,  9.54e-2],
    [ 1.57e-2,  2.12e-2,  2.20e-2,  2.21e-2,  2.21e-2],
    [ 3.05e-3,  4.25e-3,  4.42e-3,  4.44e-3,  4.45e-3],
    [ 3.07e-4,  4.59e-4,  4.78e-4,  4.81e-4,  4.82e-4],
], dtype=np.float64)

ZETA2 = np.array([
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

# ---------------- kernels (Numba or pure python) ----------------
def _build_kernels(use_jit=True, noloss=False, lc_scale=1.0):
    """Build numba-jitted kernels (or pure-python fallbacks)."""
    if use_jit and HAVE_NUMBA:
        njit = nb.njit
        fastmath = dict(fastmath=True, nogil=True, cache=True)
    else:
        # no-op decorator
        def njit(*a, **k):
            def wrap(f):
                return f
            return wrap
        fastmath = {}

    @njit(**fastmath)
    def j_min_of_x(x, lc_scale_val, noloss_flag):
        """
        Dimensionless loss-cone boundary j_min(x).
        In noloss mode, j_min ≡ 0, as in the “no–loss–cone” test.
        """
        if noloss_flag:
            return 0.0
        v = 2.0*x/X_D - (x/X_D)**2
        return lc_scale_val * (math.sqrt(v) if v > 0.0 else 0.0)

    @njit(**fastmath)
    def P_of_x(x):  # orbital period P(E) with E=-x in a Kepler potential
        E = -x
        return 2.0*math.pi/((-2.0*E)**1.5)

    T0 = P_of_x(1.0)/PSTAR  # t0 from P*(x=1)

    @njit(**fastmath)
    def sample_x_from_g0_jit():
        """Draw x from BW noloss occupancy N(x) ∝ x^{-5/2} g0(x)."""
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
        bestd = abs(X_BINS[0]-x)
        for i in range(1, X_BINS.size):
            d = abs(X_BINS[i]-x)
            if d < bestd:
                bestd = d
                best = i
        return best

    @njit(**fastmath)
    def bilinear_coeffs(x, j, pstar_val):
        """
        Paper-faithful bilinear interpolation of Table 1 coefficients.

        Returns:
            e1, sigE, j1, sigJ, rho

        where e1, j1 are orbital drifts per radial period P(E), and
        sigE, sigJ are 1σ diffusion amplitudes per orbit. rho is the
        correlation coefficient between ΔE and ΔJ.
        """
        # Clamp x to tabulated range
        x_clamp = x
        if x_clamp < X_GRID[0]:
            x_clamp = X_GRID[0]
        if x_clamp > X_GRID[-1]:
            x_clamp = X_GRID[-1]
        lx = math.log10(x_clamp)

        # Find x grid index
        ix = 0
        for k in range(X_GRID.size-1):
            if X_GRID[k] <= x_clamp <= X_GRID[k+1]:
                ix = k
                break
        x1 = X_GRID[ix]
        x2 = X_GRID[ix+1]
        tx = (lx - math.log10(x1)) / (math.log10(x2) - math.log10(x1) + 1e-30)
        if tx < 0.0:
            tx = 0.0
        if tx > 1.0:
            tx = 1.0

        # Clamp j to tabulated range (J_GRID is descending)
        j_clamp = j
        if j_clamp > J_GRID[0]:
            j_clamp = J_GRID[0]
        if j_clamp < J_GRID[-1]:
            j_clamp = J_GRID[-1]

        ij = 0
        for k in range(J_GRID.size-1):
            if (J_GRID[k] >= j_clamp) and (j_clamp >= J_GRID[k+1]):
                ij = k
                break
        j1g = J_GRID[ij]
        j2g = J_GRID[ij+1]
        tj = (j_clamp - j1g) / (j2g - j1g + 1e-30)
        if tj < 0.0:
            tj = 0.0
        if tj > 1.0:
            tj = 1.0

        # Bilinear interpolation helper
        def get(A):
            a11 = A[ix, ij];     a12 = A[ix, ij+1]
            a21 = A[ix+1, ij];   a22 = A[ix+1, ij+1]
            a1 = a11*(1.0-tj) + a12*tj
            a2 = a21*(1.0-tj) + a22*tj
            return a1*(1.0-tx) + a2*tx

        PSTAR_CANON = 0.005

        # Table 1 column NEG_E1 stores h*; from Appendix A,
        # e1* = -h* v0^2.  We work in units where v0^2 = 1, so:
        h_star = get(NEG_E1)
        e1_star = -h_star

        # Second moments per orbit
        E2_star = max(get(E2), 0.0)      # <(Δx*)^2>
        J2_star = max(get(J2), 0.0)      # <(Δj*)^2>
        j1_star = get(J1)                # drift in j*
        covEJ_star = get(ZETA2)          # <Δx* Δj*>

        # Scale away from canonical P* (eq. A8)
        scale = pstar_val / PSTAR_CANON
        e1_star      *= scale            # e1* ∝ P*
        E2_star      *= scale            # <(Δx*)^2> ∝ P*
        j1_star      *= scale            # j1* ∝ P*
        J2_star      *= scale            # <(Δj*)^2> ∝ P*
        covEJ_star   *= scale            # <Δx* Δj*> ∝ P*

        # Convert to physical units (v0^2 = 1, Jmax = 1/sqrt(2x))
        v0_sq = 1.0
        Jmax = 1.0 / math.sqrt(2.0 * x_clamp)

        e1 = e1_star * v0_sq
        sigE = math.sqrt(max(E2_star, 0.0)) * v0_sq

        j1 = j1_star * Jmax
        sigJ = math.sqrt(max(J2_star, 0.0)) * Jmax

        covEJ = covEJ_star * v0_sq * Jmax

        # Correlation coefficient
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
               n_min=0.01, n_max=3.0e4):
        """
        Equation (29): choose n so that each step is a small perturbation.
        sigE and sigJ are 1σ diffusion amplitudes per orbit.
        """
        jmin = j_min_of_x(x, lc_scale_val, noloss_flag)
        E = -x
        Jmax = 1.0 / math.sqrt(2.0 * x)
        J = j * Jmax

        # (29a) limit fractional energy change
        if sigE > 0.0:
            n_E = (0.15 * abs(E) / sigE)**2
        else:
            n_E = n_max

        # (29b) limit change in J relative to Jmax
        if sigJ > 0.0:
            n_J_iso = (0.10 * Jmax / sigJ)**2
        else:
            n_J_iso = n_max

        # (29c) limit change near J ≈ Jmax
        if sigJ > 0.0:
            delta_top = max(1e-6 * Jmax, 1.0075 * Jmax - J)
            n_J_top = (0.40 * delta_top / sigJ)**2
        else:
            n_J_top = n_max

        # (29d) limit diffusion into / out of the loss cone
        if noloss_flag or jmin <= 0.0 or sigJ == 0.0:
            n_J_lc = n_max
        else:
            Jmin = jmin * Jmax
            gap = cone_gamma_val * abs(J - Jmin)
            floor = max(gap, 0.10 * Jmin)
            n_J_lc = (floor / sigJ)**2

        n = n_E
        if n_J_iso < n:
            n = n_J_iso
        if n_J_top < n:
            n = n_J_top
        if n_J_lc < n:
            n = n_J_lc
        if n < n_min:
            n = n_min
        if n > n_max:
            n = n_max
        return n

    @njit(**fastmath)
    def correlated_normals(rho, sigE, sigJ):
        """
        Correlated Gaussians so that <ΔE ΔJ> = n ρ σ_E σ_J.
        Returns y1, y2 ~ N(0,1) with corr(y1,y2)=ρ.
        """
        z1 = np.random.normal()
        z2 = np.random.normal()
        if abs(rho) < 0.999999:
            y1 = z1
            y2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho*rho)) * z2
        else:
            y1 = z1
            y2 = z2
        return y1, y2

    @njit(**fastmath)
    def floor_index_for_j2(j2, floors):
        # return the deepest floor index k such that j2 < floors[k]
        # if j2 >= floors[0], return -1 (above the top floor)
        k = -1
        for i in range(floors.size):
            if j2 < floors[i]:
                k = i
        return k

    @njit(**fastmath)
    def target_floor_with_hysteresis(j2, floors, current_idx, split_hyst=0.8):
        # Return the target floor index with hysteresis: only advance if
        # j2 is comfortably inside the next floor (j2 < split_hyst * floor_value)
        k = current_idx
        while (k + 1) < floors.size and j2 < split_hyst * floors[k + 1]:
            k += 1
        return k

    @njit(**fastmath)
    def step_one(x, j, phase, pstar_val, noloss_flag, x_max,
                 lc_scale_val, cone_gamma_val):
        """
        One random–walk step in (E,J) matching eqs. (27) and Appendix B.

        Returns (x_new, j_new, phase_new, n_used, captured, crossed_peri).
        """
        # drift & diffusion coeffs per orbit
        e1, sigE, j1, sigJ, rho = bilinear_coeffs(x, j, pstar_val)

        # pick step fraction n of an orbit
        n_raw = pick_n(x, j, sigE, sigJ, lc_scale_val, noloss_flag, cone_gamma_val)

        # enforce pericenter alignment
        next_int = math.floor(phase) + 1.0
        crossed_peri = (phase + n_raw >= next_int)
        n = next_int - phase if crossed_peri else n_raw

        # correlated Gaussian increments
        y1, y2 = correlated_normals(rho, sigE, sigJ)

        # energies
        E = -x
        Jmax = 1.0 / math.sqrt(2.0 * x)
        J = j * Jmax

        # energy update (eq. 27a)
        deltaE = n * e1 + math.sqrt(n) * sigE * y1
        E_new = E + deltaE
        if E_new >= -1e-6:
            E_new = -1e-6  # keep bound
        x_new = -E_new

        # optional hard cap (no capture; just prevents ridiculous x)
        if x_new > x_max:
            x_new = x_max
            E_new = -x_new

        # angular momentum update
        Jmax_new = 1.0 / math.sqrt(2.0 * x_new)

        # 2D random walk gate (eq. 28)
        use_2d = (math.sqrt(n) * sigJ > j / 4.0)

        if use_2d:
            # 2D isotropic RW in J-space
            z1 = np.random.normal()
            z2 = np.random.normal()
            J_new = math.sqrt(
                (J + math.sqrt(n) * sigJ * z1)**2 +
                (math.sqrt(n) * sigJ * z2)**2
            )
        else:
            # 1D update (eq. 27b)
            J_new = J + n * j1 + math.sqrt(n) * sigJ * y2

        # reflecting boundaries in J
        if J_new < 0.0:
            J_new = -J_new
        if J_new > Jmax_new:
            J_new = 2.0 * Jmax_new - J_new
            if J_new < 0.0:
                J_new = 0.0
            if J_new > Jmax_new:
                J_new = Jmax_new

        j_new = J_new / Jmax_new

        # phase advance
        phase_new = phase + n

        # capture check (only via loss cone, and only when crossing pericenter)
        captured = False
        if (not noloss_flag) and crossed_peri:
            j_min_new = j_min_of_x(x_new, lc_scale_val, noloss_flag)
            if j_new < j_min_new:
                captured = True

        return x_new, j_new, phase_new, n, captured, crossed_peri

    @njit(**fastmath)
    def run_stream(n_relax, floors, clones_per_split, x_bins, dx,
                   seed, pstar_val, warmup_t0, x_init,
                   include_clone_gbar=True, use_snapshots=False,
                   noloss=False, use_clones=True,
                   x_max=1e10, lc_scale_val=1.0, cone_gamma_val=0.25):
        """
        Single time-carrying stream + clone tree.

        Returns:
            g_time   : time-weighted occupancy per x-bin (or snapshot counts)
            total_t0 : total t0 elapsed in measurement window (or #snapshots)
            crosses  : number of pericenter crossings
            caps     : number of capture events
            ccount   : clone pool high-water mark
        """
        np.random.seed(seed)
        SPLIT_HYST = 0.8
        noloss_flag = noloss

        # parent (time carrier) – initialise from BW distribution
        x = x_init
        j = math.sqrt(np.random.random())  # isotropic j
        phase = 0.0
        w = 1.0  # statistical weight
        parent_floor_idx = floor_index_for_j2(j*j, floors)  # -1 if above floors[0]

        # clone pool (disabled if use_clones=False)
        if use_clones:
            MAX_CLONES = 2048
        else:
            MAX_CLONES = 0
        cx  = np.zeros(MAX_CLONES, dtype=np.float64)
        cj  = np.zeros(MAX_CLONES, dtype=np.float64)
        cph = np.zeros(MAX_CLONES, dtype=np.float64)
        cw  = np.zeros(MAX_CLONES, dtype=np.float64)
        cfloor_idx = np.full(MAX_CLONES, -1, dtype=np.int32)
        cactive = np.zeros(MAX_CLONES, dtype=np.uint8)
        ccount = 0

        def bin_index(xv):
            b = 0
            bestd = abs(x_bins[0]-xv)
            for bi in range(1, x_bins.size):
                d = abs(x_bins[bi]-xv)
                if d < bestd:
                    bestd = d
                    b = bi
            return b

        # ---------- warm-up (no tally) ----------
        t0_used = 0.0
        while t0_used < warmup_t0:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(
                x, j, phase, pstar_val, noloss, x_max, lc_scale_val, cone_gamma_val
            )
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0

            # capture or escape -> replace
            if cap or (x < X_BOUND):
                if noloss_flag:
                    x = sample_x_from_g0_jit()
                else:
                    x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j * j, floors)
            elif use_clones:
                # splitting for parent after we know it's still present
                j2_now = j*j
                target_idx = target_floor_with_hysteresis(
                    j2_now, floors, parent_floor_idx, SPLIT_HYST
                )
                while parent_floor_idx < target_idx:
                    parent_floor_idx += 1  # new deeper floor
                    if ccount <= MAX_CLONES - clones_per_split:
                        alpha = 1.0 / (1.0 + clones_per_split)
                        new_w = w * alpha
                        w = new_w
                        for k in range(clones_per_split):
                            cx[ccount] = x
                            cj[ccount] = j
                            cph[ccount]= phase
                            cw[ccount] = new_w
                            cfloor_idx[ccount] = parent_floor_idx
                            cactive[ccount] = 1
                            ccount += 1

            # evolve clones (warmup; no tallies)
            if use_clones:
                i = 0
                while i < ccount:
                    if cactive[i] == 0:
                        i += 1
                        continue

                    x_c, j_c, ph_c = cx[i], cj[i], cph[i]
                    x_c2, j_c2, ph_c2, n_c, cap_c, crossed_c = step_one(
                        x_c, j_c, ph_c, pstar_val, noloss_flag,
                        x_max, lc_scale_val, cone_gamma_val
                    )
                    cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2

                    # replacement for clones
                    if cap_c or (x_c2 < X_BOUND):
                        if noloss_flag:
                            cx[i] = sample_x_from_g0_jit()
                        else:
                            cx[i] = X_BOUND
                        cj[i]  = math.sqrt(np.random.random())
                        cph[i] = ph_c2
                        cfloor_idx[i] = floor_index_for_j2(cj[i] * cj[i], floors)
                        cactive[i] = 1
                        i += 1
                        continue

                    # outward across its own floor -> remove clone, return weight
                    if cfloor_idx[i] >= 0 and j_c2*j_c2 >= floors[cfloor_idx[i]]:
                        w += cw[i]
                        cw[i] = 0.0
                        cactive[i] = 0
                        i += 1
                        continue

                    # deeper splitting for clone
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
                                cph[ccount]= ph_c2
                                cw[ccount] = new_w
                                cfloor_idx[ccount] = cfloor_idx[i]
                                cactive[ccount] = 1
                                ccount += 1
                    i += 1

        # ---------- measurement window ----------
        g_time = np.zeros(x_bins.size, dtype=np.float64)
        t0_used = 0.0
        total_t0 = 0.0
        crossings = 0
        caps = 0
        max_ccount = ccount

        snapshots = 0.0
        g_snap = np.zeros(x_bins.size, dtype=np.float64)
        t0_next = 1.0  # next snapshot at integer t0

        while t0_used < n_relax:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(
                x, j, phase, pstar_val, noloss_flag,
                x_max, lc_scale_val, cone_gamma_val
            )

            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0
            total_t0 += dt0

            if use_snapshots:
                # snapshot whenever total_t0 crosses integer t0
                while total_t0 >= t0_next:
                    # parent
                    g_snap[bin_index(x)] += w
                    # clones
                    if use_clones and include_clone_gbar:
                        for ii in range(ccount):
                            if cactive[ii]:
                                g_snap[bin_index(cx[ii])] += cw[ii]
                    snapshots += 1.0
                    t0_next += 1.0
            else:
                # time-weighted occupancy at start of step
                g_time[bin_index(x_prev)] += w * dt0
                if use_clones and include_clone_gbar:
                    for ii in range(ccount):
                        if cactive[ii]:
                            g_time[bin_index(cx[ii])] += cw[ii] * dt0

            if crossed:
                crossings += 1

            # handle parent capture / escape / splitting
            if cap:
                caps += 1
                if noloss_flag:
                    x = sample_x_from_g0_jit()
                else:
                    x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j * j, floors)

            elif x < X_BOUND:
                if noloss_flag:
                    x = sample_x_from_g0_jit()
                else:
                    x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j * j, floors)
            elif use_clones:
                j2_now = j*j
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
                            cph[ccount]= phase
                            cw[ccount] = new_w
                            cfloor_idx[ccount] = parent_floor_idx
                            cactive[ccount] = 1
                            ccount += 1
                            if ccount > max_ccount:
                                max_ccount = ccount

            # evolve clones without advancing t0_used
            if use_clones:
                i = 0
                while i < ccount:
                    if cactive[i] == 0:
                        i += 1
                        continue

                    x_c, j_c, ph_c = cx[i], cj[i], cph[i]
                    x_c2, j_c2, ph_c2, n_c, cap_c, crossed_c = step_one(
                        x_c, j_c, ph_c, pstar_val, noloss_flag,
                        x_max, lc_scale_val, cone_gamma_val
                    )
                    cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2

                    if crossed_c:
                        crossings += 1

                    if cap_c:
                        caps += 1
                        if noloss_flag:
                            cx[i] = sample_x_from_g0_jit()
                        else:
                            cx[i] = X_BOUND
                        cj[i]  = math.sqrt(np.random.random())
                        cph[i] = ph_c2
                        cfloor_idx[i] = floor_index_for_j2(cj[i] * cj[i], floors)
                        cactive[i] = 1
                        if ccount > max_ccount:
                            max_ccount = ccount
                        i += 1
                        continue

                    if x_c2 < X_BOUND:
                        if noloss_flag:
                            x_c2 = sample_x_from_g0_jit()
                        else:
                            x_c2 = X_BOUND
                        j_c2 = math.sqrt(np.random.random())
                        cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2
                        cfloor_idx[i] = floor_index_for_j2(j_c2 * j_c2, floors)

                    if cfloor_idx[i] >= 0 and j_c2*j_c2 >= floors[cfloor_idx[i]]:
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
                                cph[ccount]= ph_c2
                                cw[ccount] = new_w
                                cfloor_idx[ccount] = cfloor_idx[i]
                                cactive[ccount] = 1
                                ccount += 1
                                if ccount > max_ccount:
                                    max_ccount = ccount
                    i += 1

        if use_snapshots:
            return g_snap, snapshots, crossings, caps, max_ccount
        else:
            return g_time, total_t0, crossings, caps, max_ccount

    return P_of_x, T0, run_stream, sample_x_from_g0_jit

# ---------------- multiprocessing worker (single stream) ----------------
def _worker_one(args):
    (sid, n_relax, floors, clones_per_split,
     stream_seeds, use_jit, pstar_val, warmup_t0, u0,
     include_clone_gbar, use_snapshots, noloss,
     use_clones, lc_scale, x_max, cone_gamma) = args

    P_of_x, T0, run_stream, sample_x_from_g0_jit = _build_kernels(
        use_jit=use_jit, noloss=noloss, lc_scale=lc_scale
    )

    # Sample initial x from BW distribution using stratified u
    x_init = sample_x_from_g0(u0)

    # tiny warmup to pull kernels from cache
    _ = run_stream(1e-6, floors, clones_per_split, X_BINS, DX,
                   int(stream_seeds[sid] ^ 0xABCDEF), pstar_val, 0.0, x_init,
                   include_clone_gbar, use_snapshots, noloss,
                   use_clones, x_max, lc_scale, cone_gamma)

    return run_stream(n_relax, floors, clones_per_split, X_BINS, DX,
                      int(stream_seeds[sid]), pstar_val, warmup_t0, x_init,
                      include_clone_gbar, use_snapshots, noloss,
                      use_clones, x_max, lc_scale, cone_gamma)

# ---------------- parallel driver (ḡ only) ----------------
def run_parallel_gbar(n_streams=400, n_relax=6.0, floors=None, clones_per_split=9,
                      procs=48, use_jit=True, seed=SEED, show_progress=True,
                      pstar=PSTAR, warmup_t0=2.0, replicates=1, gexp=2.5,
                      include_clone_gbar=True, use_snapshots=False,
                      noloss=False, use_clones=True, lc_scale=1.0,
                      x_max=None, cone_gamma=0.25):
    """
    Run many independent time-carrying streams in parallel and accumulate ḡ(x).

    The MC measures time-averaged occupancy N(E), which is converted to the
    phase-space distribution function g(x) using g(x) ∝ x^{gexp} N(E).
    For a Kepler potential, gexp = 2.5 is the canonical exponent.
    """
    # noloss mode: diagnostic vs BW g0(x)
    if noloss:
        use_clones = False
        gexp = 2.5
        cone_gamma = 0.0   # no LC forcing in step-size rule

    # default x_max
    if x_max is None:
        if noloss:
            x_max = 300.0
        else:
            x_max = X_EDGES[-1]

    if floors is None:
        floors = np.array([10.0**(-k) for k in range(0, 9)], dtype=np.float64)
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

    gtime_total = np.zeros_like(X_BINS, dtype=np.float64)
    t0_total    = 0.0
    peri_total  = 0
    caps_total  = 0
    max_ccount  = 0
    total_done  = 0

    for rep in range(replicates):
        stream_seeds = rs.randint(1, 2**31-1, size=n_streams, dtype=np.int64)
        # stratified reservoir sampling
        u_strat = (np.arange(n_streams) + 0.5) / n_streams
        rs.shuffle(u_strat)

        with cf.ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [
                ex.submit(
                    _worker_one,
                    (sid, n_relax, floors, clones_per_split,
                     stream_seeds, use_jit, pstar, warmup_t0, u_strat[sid],
                     include_clone_gbar, use_snapshots,
                     noloss, use_clones, lc_scale, x_max, cone_gamma)
                )
                for sid in range(n_streams)
            ]

            for i, f in enumerate(cf.as_completed(futs), 1):
                gtime_b, t0_b, peri_b, caps_b, ccount_b = f.result()
                gtime_total += gtime_b
                t0_total    += t0_b
                peri_total  += peri_b
                caps_total  += caps_b
                if ccount_b > max_ccount:
                    max_ccount = ccount_b
                total_done += 1

                if show_progress:
                    elapsed = time.time() - start_time
                    rate = total_done / max(elapsed, 1e-9)
                    frac = total_done / tot_expected
                    print(
                        f"\rProgress: {total_done}/{tot_expected} streams "
                        f"({100*frac:5.1f}%) | rate {rate:5.1f} streams/s",
                        end="", flush=True, file=sys.stderr
                    )

    if show_progress:
        elapsed = time.time() - start_time
        print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
        if use_snapshots:
            print(f"[diag] total snapshots across all streams: {t0_total:.0f}", file=sys.stderr)
        else:
            print(f"[diag] total t0 across all streams: {t0_total:.3e}", file=sys.stderr)
        print(f"[diag] pericenter crossings (total): {peri_total}", file=sys.stderr)
        print(f"[diag] capture events (total):      {caps_total}", file=sys.stderr)
        print(f"[diag] clone pool high-water:       {max_ccount}", file=sys.stderr)

    if t0_total <= 0.0:
        raise RuntimeError("No time/snapshots accumulated in measurement window; check parameters.")

    # Build N_x from tallies
    if use_snapshots:
        # N_x = (sum weights per bin) / (# snapshots * Δx)
        n_snaps = t0_total  # t0_total stores #snapshots in snapshot mode
        N_x = gtime_total / (n_snaps * DX)
    else:
        # time-weighted: p_x = Δt / t0_total; N_x = p_x / Δx
        p_x = gtime_total / t0_total
        p_sum = p_x.sum()
        if show_progress and abs(p_sum - 1.0) > 1e-3:
            print(f"[diag] warning: sum p_x = {p_sum:.6f} (expected ~1)", file=sys.stderr)
        N_x = p_x / DX

    # Convert to g(x) using g(x) ∝ N_x x^{gexp}
    g_unscaled = N_x * (X_BINS ** gexp)

    # Normalise at outer boundary bin (X_BINS[0] ≈ 0.225)
    norm_idx = 0
    if g_unscaled[norm_idx] <= 0:
        raise RuntimeError(f"g_unscaled[{norm_idx}] <= 0; cannot normalise.")

    gbar_norm = g_unscaled / g_unscaled[norm_idx]
    gbar_raw  = g_unscaled.copy()

    # diagnostics
    if show_progress:
        if use_snapshots:
            print(f"[diag] total snapshots = {t0_total:.0f}", file=sys.stderr)
        else:
            print(f"[diag] sum p_x = {p_x.sum():.6f}", file=sys.stderr)

        mask = (X_BINS >= 1.0) & (X_BINS <= 100.0) & np.isfinite(gbar_norm) & (gbar_norm > 0)
        if mask.sum() >= 2:
            logx = np.log10(X_BINS[mask])
            logg = np.log10(gbar_norm[mask])
            p_fit, _ = np.polyfit(logx, logg, 1)
            print(
                f"[diag] best-fit g(x) ∝ x^{p_fit:.3f} in 1≲x≲100",
                file=sys.stderr,
            )

    # comparison to BW g0(x)
    ln1p_x = np.log1p(X_BINS)
    g0_x = np.interp(ln1p_x, LN1P, G0_TAB)
    g0_norm = g0_x / g0_x[norm_idx] if g0_x[norm_idx] > 0 else g0_x

    print("[diag] comparison to BW g0(x):", file=sys.stderr)
    for xb, gmc, g0n in zip(X_BINS, gbar_norm, g0_norm):
        if xb < 1.0 or xb > 100.0:
            continue
        ratio = gmc / g0n if g0n > 0 else np.nan
        print(f"[diag] x={xb:6.2f}: g_MC/g0 ≈ {ratio:5.2f}", file=sys.stderr)

    # noloss diagnostics vs BW background
    if noloss:
        print("[diag] noloss mode: comparing g_MC to BW g0(x):", file=sys.stderr)
        for xb, gmc, g0n in zip(X_BINS, gbar_norm, g0_norm):
            if xb < 0.2 or xb > 100.0:
                continue
            ratio = gmc / g0n if g0n > 0 else np.nan
            print(f"[diag] noloss: x={xb:6.2g}, g_MC/g0 ≈ {ratio:6.3f}", file=sys.stderr)

        # Direct energy-space comparison p_MC vs p_th
        p_MC = gtime_total / gtime_total.sum()

        ln1p_x = np.log1p(X_BINS)
        g0_interp = np.interp(ln1p_x, LN1P, G0_TAB)
        N_th = g0_interp * (X_BINS ** (-2.5))
        p_th = N_th * DX
        p_th /= p_th.sum()

        print("[diag] noloss: direct energy-space comparison p_MC/p_th:", file=sys.stderr)
        for i in range(0, X_BINS.size, max(1, X_BINS.size // 10)):
            xb = X_BINS[i]
            if xb < 0.2 or xb > 100.0:
                continue
            ratio = p_MC[i] / p_th[i] if p_th[i] > 0 else np.nan
            print(f"[occ] x={xb:6.2f}: p_MC/p_th ≈ {ratio:6.3f}", file=sys.stderr)

    return gbar_norm, gbar_raw

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Monte Carlo reproduction of ḡ(x) from Shapiro & Marchant (1978) Fig. 3."
    )
    ap.add_argument("--streams", type=int, default=400,
                    help="number of non-clone time-carrying streams (default: 400)")
    ap.add_argument("--windows", type=float, default=6.0,
                    help="t0 windows per stream (measurement duration, default: 6)")
    ap.add_argument("--procs", type=int, default=48,
                    help="number of worker processes (default: 48)")
    ap.add_argument("--floors_min_exp", type=int, default=8,
                    help="min exponent k for floors 10^{-k} (default: 8 => 1e-8)")
    ap.add_argument("--floor-step", type=int, default=1,
                    help="stride for floor exponents (e.g. 2 => decades 0,2,4,...)")
    ap.add_argument("--clones", type=int, default=9,
                    help="clones per split (default: 9)")
    ap.add_argument("--nojit", action="store_true",
                    help="disable numba JIT (slow fallback)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable progress display")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="base RNG seed (default: %(default)s)")
    ap.add_argument("--pstar", type=float, default=PSTAR,
                    help="P* parameter (default: %(default)s)")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="equilibration time in t0 before tallying (default: 2)")
    ap.add_argument("--replicates", type=int, default=1,
                    help="repeat the whole measurement K times (accumulating raw tallies)")
    ap.add_argument(
        "--gexp",
        type=float,
        default=2.5,
        help=("Geometric exponent α in g(x) ∝ x^α N(E); "
              "α=2.5 is the Keplerian value."),
    )
    ap.add_argument("--gbar-no-clones", action="store_true",
                    help="For ḡ(x) diagnostics: accumulate ḡ(x) from the parent stream only (ignore clones).")
    ap.add_argument("--snapshots", action="store_true",
                    help="Use per-t₀ snapshots for ḡ accumulation instead of time-weighted integration.")
    ap.add_argument("--noloss", action="store_true",
                    help="Disable loss-cone capture (for BW g0(x) diagnostics).")
    ap.add_argument(
        "--lc_scale", type=float, default=1.0,
        help="Scale factor for the loss-cone boundary j_min; lc_scale<1 shrinks the cone."
    )

    args = ap.parse_args()

    floors = np.array(
        [10.0**(-k) for k in range(0, args.floors_min_exp+1, args.floor_step)],
        dtype=np.float64
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
        noloss=args.noloss,
        use_clones=(not args.noloss and not args.gbar_no_clones),
        lc_scale=args.lc_scale,
    )

    # ---- print comparison table ----
    print("# x_center   gbar_MC_norm   gbar_MC_raw      gbar_paper   gbar_err_paper")
    for xb, g_norm, g_raw, gp, gp_err in zip(
        X_BINS, gbar_norm, gbar_raw, GBAR_PAPER, GBAR_ERR_PAPER
    ):
        print(f"{xb:10.3g}  {g_norm:12.6e}  {g_raw:12.6e}  {gp:11.4f}  {gp_err:14.4f}")

if __name__ == "__main__":
    main()

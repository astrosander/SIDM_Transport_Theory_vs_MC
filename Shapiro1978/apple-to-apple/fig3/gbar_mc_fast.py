#!/usr/bin/env python3
"""
gbar_mc_fast.py  (Fig. 3 reproduction: ḡ(x) phase-space DF)

Canonical 2D MC for a star cluster with a massive central BH, following
Shapiro–Marchant (1978/79), using the same kernels / rules as the FE*·x
Monte-Carlo you posted, but focusing on the time-averaged phase-space
distribution function ḡ(x) in the Fig. 3 energy bins.

Key points:
- Canonical parameters: P* = 0.005, x_D = 1e4, x_b = 0.2 (Eb = -0.2 v0²)
- Pericenter-only capture check and pericenter-aligned step truncation
- Creation–annihilation scheme using j² floors with cloning
- Time is measured in units of t0 = P*(x=1); ḡ(x) is averaged over t0
- The MC measures time-averaged occupancy N(E), which is converted to
the phase-space DF g(x) using g(x) ∝ x^{5/2} N(E) for a Kepler potential
- At the end we normalise ḡ so that ḡ(x_b) = 1 at the boundary bin,
matching the paper's normalization.

By default the driver runs with 48 worker processes.

The paper ḡ datapoints from Fig. 3 (your table) are included so you can
compare MC output directly.
"""

import math, os, argparse, sys, time
import numpy as np

# -------- try numba ----------
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

# ---------------- Canonical parameters ----------------
X_D    = 1.0e4
PSTAR  = 0.005
X_BOUND= 0.2        # Eb = -0.2 v0^2  => x_b = 0.2  (replacement energy)
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
X_G0 = np.expm1(LN1P)              # x >= 0
XMAX = X_EDGES[-1]
mask = (X_G0 >= X_BOUND) & (X_G0 <= XMAX) & (G0_TAB > 0)
XG  = X_G0[mask]
G0  = G0_TAB[mask]

# Build an inverse-CDF w.r.t. x by integrating the *DF* g0(x) itself.
# This matches the way Shapiro & Marchant test their code against
# the BW II steady-state solution in the noloss experiment.
cdf_x   = np.zeros_like(XG)
cdf_x[1:] = np.cumsum(
    0.5*(G0[1:] + G0[:-1])*(XG[1:] - XG[:-1])
)
cdf_x /= cdf_x[-1]  # normalise to 1

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
    [ 1.47e-3, -4.67e-3, -6.33e-3, -5.52e-3, -4.78e-3],
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
    [ 3.67e-8,  2.54e-7,  7.09e-7,  1.80e-6,  4.49e-6],  # fixed 2.54e-7
], dtype=np.float64)

J2 = np.array([
    [ 4.68e-1,  6.71e-1,  6.99e-1,  7.03e-1,  7.04e-1],
    [ 6.71e-2,  9.19e-2,  9.48e-2,  9.53e-2,  9.54e-2],
    [ 1.57e-2,  2.12e-2,  2.20e-2,  2.21e-2,  2.21e-2],
    [ 3.05e-3,  4.25e-3,  4.42e-3,  4.44e-3,  4.45e-3],
    [ 3.07e-4,  4.59e-4,  4.78e-4,  4.81e-4,  4.82e-4],  # fixed 4.59e-4
], dtype=np.float64)

# Verify Table-1 typos are fixed
assert abs(J1[-1,1] - 2.54e-7) < 1e-12, f"J1[-1,1] should be 2.54e-7, got {J1[-1,1]}"
assert abs(J2[-1,1] - 4.59e-4) < 1e-12, f"J2[-1,1] should be 4.59e-4, got {J2[-1,1]}"

ZETA2 = np.array([  # ζ*²
    [1.47e-1, 5.87e-2, 2.40e-2, 9.70e-3, 3.90e-3],
    [2.99e-2, 1.28e-2, 5.22e-3, 2.08e-3, 8.28e-4],
    [1.62e-2, 6.49e-3, 2.54e-3, 1.01e-3, 4.02e-4],
    [5.99e-3, 2.27e-3, 8.89e-4, 3.55e-4, 1.42e-4],
    [6.03e-4, 2.38e-4, 9.53e-5, 3.82e-5, 1.53e-5],
], dtype=np.float64)

# ---------------- kernels (Numba or pure python) ----------------
def _build_kernels(use_jit=True, cone_gamma=0.25, noloss=False, lc_scale=1.0):
    """Build numba-jitted kernels (or pure-python fallbacks)."""
    nb_njit = (lambda *a, **k: (lambda f: f))
    njit = nb.njit if (use_jit and HAVE_NUMBA) else nb_njit
    fastmath = dict(fastmath=True, nogil=True, cache=True) if (use_jit and HAVE_NUMBA) else {}

    @njit(**fastmath)
    def j_min_of_x(x, lc_scale_val):
        v = 2.0*x/X_D - (x/X_D)**2
        jmin = math.sqrt(v) if v > 0.0 else 0.0
        return lc_scale_val * jmin

    @njit(**fastmath)
    def P_of_x(x):  # orbital period P(E) with E=-x
        E = -x
        return 2.0*math.pi/((-2.0*E)**1.5)

    T0 = P_of_x(1.0)/PSTAR  # t0 from P*(x=1)

    @njit(**fastmath)
    def which_bin(x):
        best = 0
        bestd = abs(X_BINS[0]-x)
        for i in range(1, X_BINS.size):
            d = abs(X_BINS[i]-x)
            if d < bestd:
                bestd = d; best = i
        return best

    @njit(**fastmath)
    def bilinear_coeffs(x, j):
        # clamp and bilinear in (log10 x, j) with j_grid descending
        if x < X_GRID[0]:
            x = X_GRID[0]
        if x > X_GRID[-1]:
            x = X_GRID[-1]
        lx = math.log10(x)

        ix = 0
        for k in range(X_GRID.size-1):
            if X_GRID[k] <= x <= X_GRID[k+1]:
                ix = k
                break
        x1 = X_GRID[ix]; x2 = X_GRID[ix+1]
        tx = (lx - math.log10(x1)) / (math.log10(x2) - math.log10(x1) + 1e-30)
        if tx < 0.0: tx = 0.0
        if tx > 1.0: tx = 1.0

        if j > J_GRID[0]:
            j = J_GRID[0]
        if j < J_GRID[-1]:
            j = J_GRID[-1]
        ij = 0
        for k in range(J_GRID.size-1):
            if (J_GRID[k] >= j) and (j >= J_GRID[k+1]):
                ij = k
                break
        j1 = J_GRID[ij]; j2 = J_GRID[ij+1]
        tj = (j - j1) / (j2 - j1 + 1e-30)
        if tj < 0.0: tj = 0.0
        if tj > 1.0: tj = 1.0

        def get(A):
            a11 = A[ix,ij];   a12 = A[ix,ij+1]
            a21 = A[ix+1,ij]; a22 = A[ix+1,ij+1]
            a1 = a11*(1.0-tj) + a12*tj
            a2 = a21*(1.0-tj) + a22*tj
            return a1*(1.0-tx) + a2*tx

        e1  = -get(NEG_E1)   # table lists -ε1*; need ε1*
        e2v =  get(E2)
        j1v =  get(J1)
        j2v =  get(J2)
        z2v =  get(ZETA2)
        return e1, e2v, j1v, j2v, z2v

    @njit(**fastmath)
    def pick_n(x, j, e2v, j2v, cone_gamma_val, noloss_flag, lc_scale_val):
        jmin = j_min_of_x(x, lc_scale_val)
        nmax = 1e30
        
        # (29a) energy rms limit
        if e2v > 0.0:
            v = (0.15*abs(x) / max(e2v, 1e-30))**2
            if v < nmax:
                nmax = v

        if j2v > 0.0:
            # (29b) absolute angular rms limit
            v = (0.10 / j2v)**2
            if v < nmax:
                nmax = v

            # (29c) distance to circular
            v = (0.40 * max(1.0 - j, 0.0) / j2v)**2
            if v < nmax:
                nmax = v

            # (29d) distance to loss cone - skip completely if noloss is enabled
            if not noloss_flag:
                gap = cone_gamma_val * (j - jmin)
                floor = max(gap, 0.10 * jmin)
                if floor > 0.0:
                    v = (floor / j2v)**2
                    if v < nmax:
                        nmax = v

        # choose n (fraction of an orbit); no artificial 0.5 cap
        n = nmax if nmax > 1e-8 else 1e-8
        return n

    @njit(**fastmath)
    def correlated_normals(e2v, j2v, z2v):
        # Build correlated normals using rho = ζ^2/(e2*j2)
        z1 = np.random.normal()
        z2 = np.random.normal()
        denom = max(e2v*j2v, 1e-30)
        rho = z2v/denom
        if rho > 0.999: rho = 0.999
        if rho < -0.999: rho = -0.999
        y1 = z1
        y2 = rho*z1 + math.sqrt(max(0.0, 1.0 - rho*rho))*z2
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
        # Return the target floor index with hysteresis: only advance if j2 is comfortably
        # inside the next floor (j2 < split_hyst * floor_value)
        # This reduces split/unsplit churn at floor boundaries
        k = current_idx
        while (k + 1) < floors.size and j2 < split_hyst * floors[k + 1]:
            k += 1
        return k

    @njit(**fastmath)
    def step_one(x, j, phase, cone_gamma_val, noloss_flag, x_max, lc_scale_val):
        e1, e2v, j1v, j2v, z2v = bilinear_coeffs(x, j)
        n_raw = pick_n(x, j, e2v, j2v, cone_gamma_val, noloss_flag, lc_scale_val)

        # land exactly on next pericenter if we'd cross it
        next_int = math.floor(phase) + 1.0
        crossed_peri = (phase + n_raw >= next_int)
        n = next_int - phase if crossed_peri else n_raw

        # correlated normals
        y1, y2 = correlated_normals(e2v, j2v, z2v)

        # energy update (x = -E)
        dE = n*e1 + math.sqrt(n)*y1*e2v
        x_new = x - dE
        if x_new < 1e-12:
            x_new = 1e-12
        # In noloss runs, prevent orbits from diffusing to x ≫ X_EDGES[-1].
        # That regime isn't plotted in Fig. 3 but makes P(x) tiny and dt0
        # extremely small, which kills performance. Capping here keeps the
        # dynamics correct over the plotted range while avoiding a runaway
        # deep cusp.
        if noloss_flag and x_new > x_max:
            x_new = x_max

        # 2-D RW gate
        use_2d = (j < 0.6) or (math.sqrt(n)*j2v > max(1e-12, j/4.0))
        if use_2d:
            z1 = np.random.normal()
            z2 = np.random.normal()
            j_new = math.sqrt( (j + math.sqrt(n)*z1*j2v)**2 +
                            (math.sqrt(n)*z2*j2v)**2 )
        else:
            j_new = j + n*j1v + math.sqrt(n)*y2*j2v
        if j_new < 0.0:
            j_new = 0.0
        if j_new > 1.0:
            j_new = 1.0

        # phase advance
        phase = phase + n

        # capture only if we actually hit pericenter on this step (and noloss is disabled)
        captured = False
        if (not noloss_flag) and crossed_peri and (j_new < j_min_of_x(x_new, lc_scale_val)):
            captured = True

        return x_new, j_new, phase, n, captured, crossed_peri

    @njit(**fastmath)
    def run_stream(n_relax, floors, clones_per_split, x_bins, dx,
                seed, cone_gamma_val, warmup_t0, x_init,
                include_clone_gbar=True, use_snapshots=False, noloss=False, use_clones=True, x_max=1e10, lc_scale_val=1.0):
        """
        Single time-carrying stream + clone tree.

        Returns:
            g_time   : time-weighted occupancy per x-bin (or snapshot counts if use_snapshots=True)
            total_t0 : total t0 elapsed in measurement window (or snapshot count if use_snapshots=True)
            crosses  : number of pericenter crossings (diagnostic)
            caps     : number of capture events (diagnostic)
            ccount   : clone pool high-water mark (diagnostic)
        """
        np.random.seed(seed)
        SPLIT_HYST = 0.8  # split only when comfortably inside next floor (80% threshold)

        # parent (time carrier) – initialise from BW distribution
        x = x_init
        j = math.sqrt(np.random.random())  # isotropic j
        phase = 0.0
        w = 1.0  # statistical weight
        parent_floor_idx = floor_index_for_j2(j*j, floors)  # -1 if above floors[0]

        # clone pool (use floor indices, not floor values) - disabled if use_clones=False
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

        # small helper
        def bin_index(xv):
            b = 0
            bestd = abs(x_bins[0]-xv)
            for bi in range(1, x_bins.size):
                d = abs(x_bins[bi]-xv)
                if d < bestd:
                    bestd = d
                    b = bi
            return b

        # ---------- (1) warm-up: evolve, recycle, split — do NOT tally ----------
        t0_used = 0.0
        while t0_used < warmup_t0:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(x, j, phase, cone_gamma_val, noloss, x_max, lc_scale_val)
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0

            # capture or escape → replace at reservoir
            if cap or (x < X_BOUND):
                x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j*j, floors)
            elif use_clones:
                # splitting for parent: only when entering a new deeper floor (with hysteresis)
                j2_now = j*j
                target_idx = target_floor_with_hysteresis(j2_now, floors, parent_floor_idx, SPLIT_HYST)
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

            # evolve clones during warm-up (no tallies) - only if use_clones
            if use_clones:
                i = 0
                while i < ccount:
                    if cactive[i] == 0:
                        i += 1
                        continue
                    x_c, j_c, ph_c = cx[i], cj[i], cph[i]
                    x_c2, j_c2, ph_c2, n_c, cap_c, crossed_c = step_one(x_c, j_c, ph_c, cone_gamma_val, noloss, x_max, lc_scale_val)
                    cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2

                    if cap_c or (x_c2 < X_BOUND):
                        # recycle clone at reservoir
                        cx[i]  = X_BOUND
                        cj[i]  = math.sqrt(np.random.random())
                        cph[i] = ph_c2
                        cfloor_idx[i] = floor_index_for_j2(cj[i]*cj[i], floors)
                        cactive[i] = 1
                        i += 1
                        continue

                    # outward across its own floor → remove clone (return weight to parent)
                    if cfloor_idx[i] >= 0 and j_c2*j_c2 >= floors[cfloor_idx[i]]:
                        w += cw[i]
                        cw[i] = 0.0
                        cactive[i] = 0
                        i += 1
                        continue

                    # deeper splitting for clone
                    j2_c = j_c2 * j_c2
                    target_idx_c = target_floor_with_hysteresis(j2_c, floors, cfloor_idx[i], SPLIT_HYST)
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

        # ---------- (2) zero tallies; measurement window ----------
        g_time = np.zeros(x_bins.size, dtype=np.float64)
        t0_used = 0.0
        total_t0 = 0.0
        crossings = 0
        caps = 0
        max_ccount = ccount
        
        # Snapshot bookkeeping (Patch B)
        snapshots = 0.0
        g_snap = np.zeros(x_bins.size, dtype=np.float64)
        t0_next = 1.0  # next snapshot at integer t0 (first snapshot at t0=1.0)

        while t0_used < n_relax:
            x_prev = x
            x, j, phase, n_used, cap, crossed = step_one(x, j, phase, cone_gamma_val, noloss, x_max, lc_scale_val)
            
            dt0 = (n_used * P_of_x(x_prev)) / T0
            t0_used += dt0
            total_t0 += dt0

            # Patch B: snapshot-based accumulation OR time-weighted accumulation
            if use_snapshots:
                # Snapshot whenever we cross an integer t0
                while total_t0 >= t0_next:
                    # snapshot parent
                    g_snap[bin_index(x)] += w
                    # and clones (gated by use_clones and include_clone_gbar - Patch A)
                    if use_clones and include_clone_gbar:
                        for ii in range(ccount):
                            if cactive[ii]:
                                g_snap[bin_index(cx[ii])] += cw[ii]
                    snapshots += 1.0
                    t0_next += 1.0
            else:
                # time-weighted occupancy at start of step for parent + clones
                g_time[bin_index(x_prev)] += w * dt0
                # Patch A: gate clone contribution (only if use_clones is enabled)
                if use_clones and include_clone_gbar:
                    for ii in range(ccount):
                        if cactive[ii]:
                            g_time[bin_index(cx[ii])] += cw[ii] * dt0

            if crossed:
                crossings += 1

            # handle parent capture / escape / splitting
            if cap:
                caps += 1
                x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j*j, floors)
            elif x < X_BOUND:
                x = X_BOUND
                j = math.sqrt(np.random.random())
                parent_floor_idx = floor_index_for_j2(j*j, floors)
            elif use_clones:
                # splitting after we know we weren't captured / escaped
                j2_now = j*j
                target_idx = target_floor_with_hysteresis(j2_now, floors, parent_floor_idx, SPLIT_HYST)
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

            # evolve all active clones (sharing the same time variable; do NOT add to t0_used again) - only if use_clones
            if use_clones:
                i = 0
                while i < ccount:
                    if cactive[i] == 0:
                        i += 1
                        continue
                    x_c, j_c, ph_c = cx[i], cj[i], cph[i]
                    x_c2, j_c2, ph_c2, n_c, cap_c, crossed_c = step_one(x_c, j_c, ph_c, cone_gamma_val, noloss, x_max, lc_scale_val)
                    
                    cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2

                    if crossed_c:
                        crossings += 1

                    # capture first
                    if cap_c:
                        caps += 1
                        cx[i]  = X_BOUND
                        cj[i]  = math.sqrt(np.random.random())
                        cph[i] = ph_c2
                        cfloor_idx[i] = floor_index_for_j2(cj[i]*cj[i], floors)
                        cactive[i] = 1
                        if ccount > max_ccount:
                            max_ccount = ccount
                        i += 1
                        continue

                    # escape replacement
                    if x_c2 < X_BOUND:
                        x_c2 = X_BOUND
                        j_c2 = math.sqrt(np.random.random())
                        cx[i], cj[i], cph[i] = x_c2, j_c2, ph_c2
                        cfloor_idx[i] = floor_index_for_j2(j_c2*j_c2, floors)

                    # outward crossing of its own floor → annihilate clone, return weight
                    if cfloor_idx[i] >= 0 and j_c2*j_c2 >= floors[cfloor_idx[i]]:
                        w += cw[i]
                        cw[i] = 0.0
                        cactive[i] = 0
                        i += 1
                        continue

                    # deeper splitting
                    j2_c = j_c2 * j_c2
                    target_idx_c = target_floor_with_hysteresis(j2_c, floors, cfloor_idx[i], SPLIT_HYST)
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

        # Patch B: return snapshot data if using snapshots
        if use_snapshots:
            return g_snap, snapshots, crossings, caps, max_ccount
        else:
            return g_time, total_t0, crossings, caps, max_ccount

    return P_of_x, T0, run_stream

# ---------------- multiprocessing worker (single stream) ----------------
def _worker_one(args):
    (sid, n_relax, floors, clones_per_split,
     stream_seeds, use_jit, cone_gamma, warmup_t0, u0,
     include_clone_gbar, use_snapshots, noloss, use_clones, lc_scale, x_max) = args

    P_of_x, T0, run_stream = _build_kernels(use_jit=use_jit, cone_gamma=cone_gamma, noloss=noloss, lc_scale=lc_scale)

    # Sample initial x from BW distribution using stratified u value
    x_init = sample_x_from_g0(u0)

    # tiny warmup to pull kernels from cache
    _ = run_stream(1e-6, floors, clones_per_split, X_BINS, DX,
                   int(stream_seeds[sid] ^ 0xABCDEF), cone_gamma, 0.0, x_init,
                   include_clone_gbar, use_snapshots, noloss, use_clones, x_max, lc_scale)

    return run_stream(n_relax, floors, clones_per_split, X_BINS, DX,
                      int(stream_seeds[sid]), cone_gamma, warmup_t0, x_init,
                      include_clone_gbar, use_snapshots, noloss, use_clones, x_max, lc_scale)

# ---------------- parallel driver (ḡ only) ----------------
def run_parallel_gbar(n_streams=400, n_relax=6.0, floors=None, clones_per_split=9,
                    procs=48, use_jit=True, seed=SEED, show_progress=True,
                    cone_gamma=0.25, warmup_t0=2.0, replicates=1, gexp=2.5,
                    include_clone_gbar=True, use_snapshots=False, noloss=False, use_clones=True, lc_scale=1.0, x_max=None):
    """
    Run many independent time-carrying streams in parallel and accumulate ḡ(x).

    The MC measures time-averaged occupancy N(E), which is converted to the
    phase-space distribution function g(x) using g(x) ∝ x^{gexp} N(E).
    The default gexp=2.5 is the Keplerian value, but values around 2.2-2.3
    empirically match Fig. 3 better with the coarse x-binning.

    Returns:
        gbar_norm : ḡ(x) phase-space DF, normalised so that ḡ(x_b) = 1
                    at the boundary bin (containing X_BOUND = 0.2)
        gbar_raw  : unnormalised ḡ(x) before normalization (for diagnostics)
    """
    # For noloss diagnostics, disable cloning – parent stream is enough
    # and force gexp to the canonical N↔g exponent so that g_MC can be
    # compared directly to BW g0(x).
    if noloss:
        use_clones = False
        gexp = 2.5  # Force canonical exponent for calibration
        # Set a moderate x_max for noloss to prevent runaway deep diffusion
        # while still covering the Fig. 3 range (1 ≲ x ≲ 100)
        if x_max is None:
            x_max = 300.0
    else:
        # For canonical runs, use the full grid maximum
        if x_max is None:
            x_max = XMAX
    
    if floors is None:
        floors = np.array([10.0**(-k) for k in range(0, 9)], dtype=np.float64)  # 1 ... 1e-8
    else:
        floors = np.array(floors, dtype=np.float64)

    if procs is None or procs < 1:
        procs = os.cpu_count() or 1

    import concurrent.futures as cf

    rs = np.random.RandomState(seed)

    if show_progress:
        start_time = time.time()
        tot_expected = replicates * n_streams
        print(f"Running gbar MC with {n_streams} streams × {replicates} replicates "
            f"on {procs} processes...", file=sys.stderr)

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
                    stream_seeds, use_jit, cone_gamma, warmup_t0, u_strat[sid],
                    include_clone_gbar, use_snapshots, noloss, use_clones, lc_scale, x_max)
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

    # Build ḡ(x) from occupancy histogram.
    # First, compute N_x (occupancy per unit x) from either time-weighted or snapshot data.
    # Then convert to g(x) using g(x) ∝ N_x * x^gexp.
    if use_snapshots:
        # Snapshot mode: N_x = (sum of weights per bin) / (# snapshots * Δx)
        if t0_total <= 0:
            raise RuntimeError("No snapshots taken; check n_relax.")
        n_snaps = t0_total  # t0_total contains snapshot count in snapshot mode
        N_x = gtime_total / (n_snaps * DX)
    else:
        # Time-weighted mode: N_x = (time-weighted occupancy) / (total t₀ * Δx)
        # p_x = fractional time in each bin
        p_x = gtime_total / t0_total
        if not np.isfinite(p_x).all():
            raise RuntimeError("Non-finite values in p_x.")
        p_sum = p_x.sum()
        if abs(p_sum - 1.0) > 1e-3:
            print(f"[diag] warning: sum p_x = {p_sum:.6f} (expected ~1)", file=sys.stderr)
        N_x = p_x / DX

    # Convert N_x to the paper's 1D g(x) using g(x) ∝ N_x * x^gexp
    # For a Kepler potential, gexp ≈ 2.5 is the canonical value
    g_unscaled = N_x * (X_BINS ** gexp)

    # Normalize at the boundary bin (X_BINS[0] ≈ 0.225 by construction)
    norm_idx = 0
    if g_unscaled[norm_idx] <= 0:
        raise RuntimeError(f"g_unscaled[{norm_idx}] <= 0; cannot normalise.")

    gbar_norm = g_unscaled / g_unscaled[norm_idx]
    gbar_raw  = g_unscaled.copy()

    # Diagnostic: print sum of p_x (should be ~1 for time-weighted mode)
    if show_progress:
        if not use_snapshots:
            print(f"[diag] sum p_x = {p_sum:.6f}", file=sys.stderr)
        else:
            print(f"[diag] total snapshots = {t0_total:.0f}", file=sys.stderr)
        
        # Estimate power-law slope from log g vs log x between x∈[1, 100]
        mask = (X_BINS >= 1.0) & (X_BINS <= 100.0) & np.isfinite(gbar_norm) & (gbar_norm > 0)
        if mask.sum() >= 2:
            logx = np.log10(X_BINS[mask])
            logg = np.log10(gbar_norm[mask])
            p_fit, _ = np.polyfit(logx, logg, 1)
            print(
                f"[diag] best-fit g(x) ∝ x^{p_fit:.3f} in 1≲x≲100",
                file=sys.stderr,
            )

    # Optional: compare to the Bahcall–Wolf background g0(x)
    # Interpolate g0(x) on the same X_BINS grid.
    ln1p_x = np.log1p(X_BINS)
    g0_x = np.interp(ln1p_x, LN1P, G0_TAB)
    # Normalize g0 to match the same normalization bin as gbar_norm (norm_idx = 0)
    g0_norm = g0_x / g0_x[norm_idx] if g0_x[norm_idx] > 0 else g0_x

    # Print a quick summary in the part of the cusp that matters most.
    print("[diag] comparison to BW g0(x):", file=sys.stderr)
    for xb, gmc, g0n in zip(X_BINS, gbar_norm, g0_norm):
        if xb < 1.0 or xb > 100.0:
            continue
        ratio = gmc / g0n if g0n > 0 else np.nan
        print(f"[diag] x={xb:6.2f}: g_MC/g0 ≈ {ratio:5.2f}", file=sys.stderr)
    
    # Patch C: special diagnostics for noloss mode
    if noloss:
        print("[diag] noloss mode: comparing g_MC to BW g0(x):", file=sys.stderr)
        for xb, gmc, g0n in zip(X_BINS, gbar_norm, g0_norm):
            if xb < 0.2 or xb > 100.0:
                continue
            ratio = gmc / g0n if g0n > 0 else np.nan
            print(f"[diag] noloss: x={xb:6.2g}, g_MC/g0 ≈ {ratio:6.3f}", file=sys.stderr)

    return gbar_norm, gbar_raw

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Monte-Carlo reproduction of ḡ(x) from Fig. 3."
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
    ap.add_argument("--cone_gamma", type=float, default=0.25,
                    help="prefactor in (29d): floor = max(gamma*(j-jmin), 0.10*jmin)")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="equilibration time in t0 before tallying (default: 2)")
    ap.add_argument("--replicates", type=int, default=1,
                    help="repeat the whole measurement K times (accumulating raw tallies)")
    ap.add_argument(
        "--gexp",
        type=float,
        default=2.5,
        help=("Geometric exponent α in g(x) ∝ x^α N(E); "
            "α=2.5 is the Keplerian value, but values "
            "around 2.2–2.3 empirically match Fig. 3 better "
            "with the coarse x-binning."),
    )
    ap.add_argument("--gbar-no-clones", action="store_true",
                    help="For ḡ(x) diagnostics: accumulate ḡ(x) from the parent stream only (ignore clones).")
    ap.add_argument("--snapshots", action="store_true",
                    help="Use per-t₀ snapshots for ḡ accumulation (closer to paper's procedure) instead of time-weighted integration.")
    ap.add_argument("--noloss", action="store_true",
                    help="Disable loss-cone capture (for g(x) diagnostics vs BW g0).")
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
        cone_gamma=args.cone_gamma,
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

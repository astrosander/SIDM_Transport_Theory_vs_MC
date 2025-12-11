#!/usr/bin/env python3
"""
Monte Carlo reproduction of Shapiro & Marchant (1978) Fig. 3

This version adds multi-CPU parallelism via multiprocessing:

- We keep the core run_simulation() unchanged (single-process MC).
- We add run_simulation_task() as a small wrapper so it can be used
  with multiprocessing.Pool.map.
- main() runs multiple independent replicas of the simulation
  in parallel and averages the resulting ḡ(x).

If you want to add numba JIT on top of this, the easiest wins are:
- JIT the purely numerical helpers (g0_bw, interpolate_orbital_coeffs,
  j_min_of_x, orbital_period_star) with @numba.njit.
- You still keep multiprocessing for multi-core, but each process
  runs a faster JIT-compiled kernel.

Everything below is self-contained: no argparse, one file.
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# If you want to use numba later, uncomment these lines:
from numba import njit
# and then decorate some hot functions (see comments near them).

# -------------------------------------------------------------
# Constants / tables from the prompt
# -------------------------------------------------------------

X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=np.float64)

# Precompute log bins once (used for snapshot binning)
LOGX_BINS = np.log(X_BINS)


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

# BW 1D solution g0(x) tabulation
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

# Table 1 orbital perturbations (dimensionless, P* factored in)
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

# x grid and j grid for those tables (from Table 1)
X_TABLE = np.array([3.36e-1, 3.31, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID  = np.array([1.0, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)

# Canonical parameters
P_STAR = 0.005   # P* (dimensionless period at x=1)
X_D    = 1.0e4   # x_D = M/(2 r_D v0^2)
X_CRIT_TARGET = 10.0  # not directly used, but we're in that regime

# Outer cusp boundary (as in § IIIe)
X_B = 0.20

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

# If you want numba, you can do:
@njit(cache=True)
def g0_bw(x):
    """
    Bahcall–Wolf 1D BW solution g0(x) via linear interpolation in ln(1+x).
    LN1P = ln(1+x) grid, G0_TAB = tabulated g0.
    """
    if x < 0.0:
        return math.exp(x)  # unbound Maxwellian (by assumption)
    u = math.log1p(x)  # ln(1+x)
    if u <= LN1P[0]:
        return G0_TAB[0]
    if u >= LN1P[-1]:
        return G0_TAB[-1]
    i = np.searchsorted(LN1P, u) - 1
    i = max(0, min(i, len(LN1P)-2))
    t = (u - LN1P[i]) / (LN1P[i+1] - LN1P[i])
    return (1.0 - t)*G0_TAB[i] + t*G0_TAB[i+1]


@njit(cache=True)
def interpolate_orbital_coeffs(x, j):
    """
    Interpolate the dimensionless orbital perturbations at (x, j):
        ε1*, ε2*, j1*, j2*, ζ*^2
    We interpolate in log10(x) on [X_TABLE], and linearly in j on [J_GRID].

    Returns:
        (eps1_star, eps2_star, j1_star, j2_star, zeta2_star)
    """
    # Clamp x and j into the table range
    x_cl = min(max(x, X_TABLE[0]), X_TABLE[-1])
    j_cl = min(max(j, J_GRID[-1]), J_GRID[0])

    # Indices in x (in log space; table is roughly log-spaced)
    logx = math.log10(x_cl)
    log_x_tab = np.log10(X_TABLE)
    if logx <= log_x_tab[0]:
        ix = 0
        tx = 0.0
    elif logx >= log_x_tab[-1]:
        ix = len(X_TABLE)-2
        tx = 1.0
    else:
        ix = np.searchsorted(log_x_tab, logx) - 1
        ix = max(0, min(ix, len(X_TABLE)-2))
        tx = ((logx - log_x_tab[ix]) /
              (log_x_tab[ix+1] - log_x_tab[ix]))

    # Indices in j (linear)
    jg = J_GRID
    if j_cl >= jg[0]:
        ij = 0
        tj = 0.0
    elif j_cl <= jg[-1]:
        ij = len(jg)-2
        tj = 1.0
    else:
        # find i such that jg[i] >= j > jg[i+1]
        ij = np.searchsorted(-jg, -j_cl) - 1
        ij = max(0, min(ij, len(jg)-2))
        tj = (j_cl - jg[ij]) / (jg[ij+1] - jg[ij])

    def bilinear(arr):
        v00 = arr[ix,   ij]
        v01 = arr[ix,   ij+1]
        v10 = arr[ix+1, ij]
        v11 = arr[ix+1, ij+1]
        v0 = v00 + tj*(v01 - v00)
        v1 = v10 + tj*(v11 - v10)
        return v0 + tx*(v1 - v0)

    eps1_star = -bilinear(NEG_E1)  # NEG_E1 in table is -ε1*
    eps2_star = bilinear(E2)
    j1_star   = bilinear(J1)
    j2_star   = bilinear(J2)
    zeta2_star= bilinear(ZETA2)

    return eps1_star, eps2_star, j1_star, j2_star, zeta2_star


@njit(cache=True)
def j_min_of_x(x):
    """
    Loss-cone boundary j_min(x) = sqrt(2 x / x_D - x^2 / x_D^2).
    Only defined for 0 <= x <= x_D.
    """
    if x <= 0.0:
        return 0.0
    if x >= X_D:
        return 1.0
    val = 2.0*x/X_D - (x**2)/(X_D**2)
    if val <= 0.0:
        return 0.0
    return math.sqrt(val)


@njit(cache=True)
def orbital_period_star(x):
    """
    Dimensionless orbital period P*(x) = P(x) / τ0.
    From Eq. (26) scaling: P* at x=1 is given; roughly P(x) ∝ x^(-3/2),
    so P*(x) ≈ P_STAR * x^(-3/2).
    """
    if x <= 0.0:
        x = 1.0
    return P_STAR * x**(-1.5)


def choose_time_step(x, j, eps2_star, j2_star):
    """
    Approximate version of the step-adjustment algorithm.

    We want:
        sqrt(n) * eps2* << x
        sqrt(n) * j2*   << 1
        sqrt(n) * j2*   << j_min constraints near the loss cone, etc.
    We pick n as the minimum over a few constraints.

    Returns:
        n  (dimensionless number of orbital periods),
        dt = n * P*(x)
    """
    x_eff = max(x, 1e-3)
    eps2 = abs(eps2_star)
    j2   = abs(j2_star)

    constraints = []

    if eps2 > 0:
        constraints.append((0.15 * x_eff / eps2)**2)

    if j2 > 0:
        constraints.append((0.1 / j2)**2)
        constraints.append((0.4 * (1.0075 - j) / j2)**2)

    jmin = j_min_of_x(x_eff)
    if j2 > 0 and jmin > 0:
        d1 = 0.25 * abs(j - jmin)
        d2 = 0.10 * jmin
        maxd = max(d1, d2)
        if maxd > 0:
            constraints.append((maxd / j2)**2)

    if not constraints:
        n = 1e-3
    else:
        n = min(constraints)
        n = max(min(n, 1.0), 1e-5)

    dt = n * orbital_period_star(x_eff)
    return n, dt


def correlated_normals(zeta2_star, eps2_star, j2_star):
    """
    Generate correlated normal deviates (y1, y2) with:
        E[y1] = E[y2] = 0
        Var[y1] = Var[y2] = 1
        Cov[y1, y2] = zeta2* / (eps2* j2*)

    Uses a standard Gaussian correlation construction.
    """
    if eps2_star == 0.0 or j2_star == 0.0:
        return random.gauss(0, 1), random.gauss(0, 1)

    rho = zeta2_star / (eps2_star * j2_star)
    rho = max(min(rho, 0.999), -0.999)

    y1 = random.gauss(0, 1)
    z  = random.gauss(0, 1)
    y2 = rho * y1 + math.sqrt(1.0 - rho*rho) * z
    return y1, y2


def initialize_star(bound=True, x_outer=X_B):
    """
    Initialize a star in the cusp boundary, x slightly above x_outer.
    We choose x from a narrow band [x_outer, 1.5 x_outer] with probability
    proportional to g0(x).
    """
    ln_x_min = math.log(x_outer)
    ln_x_max = math.log(1.5 * x_outer)
    ln_x = ln_x_min + random.random() * (ln_x_max - ln_x_min)
    x = math.exp(ln_x)
    g = g0_bw(x)
    g_max = 3.0  # safe upper bound
    if random.random() > min(1.0, g / g_max):
        x = x_outer

    u = random.random()
    j = math.sqrt(u)  # isotropic in 2D J-space

    return x, j, 0.0  # cycle_count = 0


def reinject_star(star):
    """
    Replace a star that left the cusp or was captured by a fresh one.
    """
    return initialize_star(bound=True, x_outer=X_B)


# -------------------------------------------------------------
# Monte Carlo simulation (single-process)
# -------------------------------------------------------------

def run_simulation(
        n_stars=500,
        n_steps=200000,
        loss_cone=True,
        measure_start=5000,
        measure_every=100,
        rng_seed=42
    ):
    """
    Run a Monte Carlo simulation of n_stars test stars for n_steps.
    If loss_cone=False, we artificially set j_min(x) ≡ 0 to remove
    loss-cone capture (no-loss-cone test).

    We measure ḡ(x) on the X_BINS grid after measure_start steps,
    every measure_every steps, by averaging over a set of snapshots.

    Returns:
        gbar_mean  (len(X_BINS))
        gbar_std   (len(X_BINS))
    """
    random.seed(rng_seed)

    stars = [initialize_star() for _ in range(n_stars)]

    snapshots = []

    logX_bins = np.log(X_BINS)

    def record_snapshot():
        xs = np.array([s[0] for s in stars])
        logxs = np.log(xs)
        gbar = np.zeros(len(X_BINS))
        counts = np.zeros(len(X_BINS))

        for lx in logxs:
            idx = np.argmin(np.abs(logX_bins - lx))
            counts[idx] += 1.0

        for i, x_bin in enumerate(X_BINS):
            if counts[i] <= 0:
                gbar[i] = 0.0
            else:
                gbar[i] = counts[i]

        # Normalize by matching small-x bins to g0(x)
        idxs = [0, 1, 2]
        s_num = 0.0
        s_den = 0.0
        for i in idxs:
            if gbar[i] > 0:
                s_num += g0_bw(X_BINS[i])
                s_den += gbar[i]
        scale = s_num / s_den if s_den > 0 else 1.0
        for i in range(len(gbar)):
            gbar[i] *= scale

        snapshots.append(gbar)

    next_measure = measure_start

    for step in range(n_steps):
        new_stars = []
        for (x, j, cycle) in stars:
            eps1_star, eps2_star, j1_star, j2_star, zeta2_star = \
                interpolate_orbital_coeffs(max(x, X_TABLE[0]), max(j, J_GRID[-1]))

            n, dt = choose_time_step(x, j, eps2_star, j2_star)
            cycle_new = cycle + n

            y1, y2 = correlated_normals(zeta2_star, eps2_star, j2_star)

            dx = n * eps1_star + math.sqrt(n) * eps2_star * y1
            x_new = x + dx

            dj = n * j1_star + math.sqrt(n) * j2_star * y2
            j_new = j + dj

            if j_new < 0.0:
                j_new = -j_new
            if j_new > 1.0:
                j_new = 2.0 - j_new
                if j_new < 0.0:
                    j_new = 0.0
                if j_new > 1.0:
                    j_new = 1.0

            # outer boundary
            if x_new < X_B:
                new_stars.append(reinject_star((x, j, cycle_new)))
                continue

            # extreme inner bound
            if x_new > 2.0 * X_D:
                new_stars.append(reinject_star((x, j, cycle_new)))
                continue

            captured = False
            if loss_cone:
                if math.floor(cycle_new + 1e-9) > math.floor(cycle + 1e-9):
                    jmin = j_min_of_x(x_new)
                    if j_new < jmin:
                        captured = True

            if captured:
                new_stars.append(reinject_star((x, j, cycle_new)))
            else:
                new_stars.append((x_new, j_new, cycle_new))

        stars = new_stars

        if step == next_measure:
            record_snapshot()
            next_measure += measure_every

    if not snapshots:
        raise RuntimeError("No snapshots recorded; adjust n_steps/measure_start.")

    snapshots = np.array(snapshots)
    gbar_mean = snapshots.mean(axis=0)
    gbar_std  = snapshots.std(axis=0)

    return gbar_mean, gbar_std


# -------------------------------------------------------------
# Parallel wrapper for multi-CPU use
# -------------------------------------------------------------

def _warmup_numba():
    """
    Warm up numba-compiled functions by calling them with dummy inputs.
    This ensures they're compiled in each worker process before actual use.
    """
    # Warm up g0_bw
    g0_bw(0.5)
    g0_bw(10.0)
    g0_bw(100.0)
    
    # Warm up interpolate_orbital_coeffs
    interpolate_orbital_coeffs(1.0, 0.5)
    interpolate_orbital_coeffs(10.0, 0.3)
    interpolate_orbital_coeffs(100.0, 0.1)
    
    # Warm up j_min_of_x
    j_min_of_x(1.0)
    j_min_of_x(100.0)
    j_min_of_x(1000.0)
    
    # Warm up orbital_period_star
    orbital_period_star(1.0)
    orbital_period_star(10.0)


def _closest_log_bin_indices(logxs: np.ndarray) -> np.ndarray:
    """
    For each log(x), return the index of the closest LOGX_BINS entry.
    Uses searchsorted + neighbor check (fast, vectorized).
    """
    idx = np.searchsorted(LOGX_BINS, logxs, side="left")
    idx = np.clip(idx, 0, len(LOGX_BINS) - 1)
    has_left = idx > 0
    left = idx - 1
    choose_left = has_left & (np.abs(logxs - LOGX_BINS[left]) < np.abs(logxs - LOGX_BINS[idx]))
    idx[choose_left] = left[choose_left]
    return idx


def update_stars_batch_full(args):
    """
    Update a batch of stars for the entire simulation, returning snapshots.
    This minimizes communication by doing all work in one go.
    
    args = (stars_batch, loss_cone, n_steps, measure_start, measure_every, batch_idx, base_seed)
    Returns: (final_stars, snapshot_counts_list)
        snapshot_counts_list: list of (step_num, counts_array) tuples for measurement steps,
        where counts_array has shape (len(X_BINS),)
    """
    stars_batch, loss_cone, n_steps, measure_start, measure_every, batch_idx, base_seed = args
    
    # Set initial seed for this batch
    random.seed(base_seed + batch_idx)
    
    stars = stars_batch[:]  # Copy the list
    snapshot_counts_list = []
    
    for step in range(n_steps):
        new_stars = []
        
        for (x, j, cycle) in stars:
            eps1_star, eps2_star, j1_star, j2_star, zeta2_star = \
                interpolate_orbital_coeffs(max(x, X_TABLE[0]), max(j, J_GRID[-1]))

            n, dt = choose_time_step(x, j, eps2_star, j2_star)
            cycle_new = cycle + n

            y1, y2 = correlated_normals(zeta2_star, eps2_star, j2_star)

            dx = n * eps1_star + math.sqrt(n) * eps2_star * y1
            x_new = x + dx

            dj = n * j1_star + math.sqrt(n) * j2_star * y2
            j_new = j + dj

            if j_new < 0.0:
                j_new = -j_new
            if j_new > 1.0:
                j_new = 2.0 - j_new
                if j_new < 0.0:
                    j_new = 0.0
                if j_new > 1.0:
                    j_new = 1.0

            # outer boundary
            if x_new < X_B:
                new_stars.append(reinject_star((x, j, cycle_new)))
                continue

            # extreme inner bound
            if x_new > 2.0 * X_D:
                new_stars.append(reinject_star((x, j, cycle_new)))
                continue

            captured = False
            if loss_cone:
                if math.floor(cycle_new + 1e-9) > math.floor(cycle + 1e-9):
                    jmin = j_min_of_x(x_new)
                    if j_new < jmin:
                        captured = True

            if captured:
                new_stars.append(reinject_star((x, j, cycle_new)))
            else:
                new_stars.append((x_new, j_new, cycle_new))
        
        stars = new_stars
        
        # Record snapshot bin counts at measurement steps (tiny payload back to parent process)
        if step >= measure_start and (step - measure_start) % measure_every == 0:
            xs = np.array([s[0] for s in stars], dtype=np.float64)
            logxs = np.log(xs)
            indices = _closest_log_bin_indices(logxs)
            counts = np.bincount(indices, minlength=len(X_BINS)).astype(np.float64)
            snapshot_counts_list.append((step, counts))
    
    return stars, snapshot_counts_list


def run_simulation_parallel(
        n_stars=500,
        n_steps=200000,
        loss_cone=True,
        measure_start=5000,
        measure_every=100,
        rng_seed=42,
        n_procs=4,
        pool=None
    ):
    """
    Run a parallel Monte Carlo simulation where all processes work together
    on a single simulation by splitting stars across processes.
    
    Returns:
        gbar_mean  (len(X_BINS))
        gbar_std   (len(X_BINS))
    """
    random.seed(rng_seed)
    
    # Initialize all stars in main process
    all_stars = [initialize_star() for _ in range(n_stars)]
    
    # Split stars across processes
    stars_per_proc = n_stars // n_procs
    star_batches = []
    for i in range(n_procs):
        start_idx = i * stars_per_proc
        if i == n_procs - 1:
            # Last process gets remaining stars
            end_idx = n_stars
        else:
            end_idx = (i + 1) * stars_per_proc
        star_batches.append(all_stars[start_idx:end_idx])
    
    snapshots = []
    
    created_pool = False
    if pool is None:
        # Warm up numba in main process
        _warmup_numba()
        pool = Pool(processes=n_procs, initializer=_warmup_numba)
        created_pool = True

    try:
        # Prepare arguments for each batch - each processes full simulation
        batch_args = [
            (star_batches[i], loss_cone, n_steps, measure_start, measure_every, i, rng_seed)
            for i in range(n_procs)
        ]

        # Run full simulation for each batch in parallel
        results = pool.map(update_stars_batch_full, batch_args)

        # Extract final stars and snapshot bin-counts
        final_batches = []
        all_counts_by_step = {}

        for i, (final_stars, snapshot_counts_list) in enumerate(results):
            final_batches.append(final_stars)
            for step_num, counts in snapshot_counts_list:
                if step_num not in all_counts_by_step:
                    all_counts_by_step[step_num] = []
                all_counts_by_step[step_num].append(counts)

        star_batches = final_batches

        # Combine snapshots from all batches at each measurement step (sum counts)
        for step_num in sorted(all_counts_by_step.keys()):
            total_counts = np.sum(np.stack(all_counts_by_step[step_num], axis=0), axis=0)
            gbar = total_counts.astype(np.float64, copy=True)

            # Normalize by matching small-x bins to g0(x)
            idxs = [0, 1, 2]
            s_num = 0.0
            s_den = 0.0
            for i in idxs:
                if gbar[i] > 0:
                    s_num += g0_bw(X_BINS[i])
                    s_den += gbar[i]
            scale = s_num / s_den if s_den > 0 else 1.0
            gbar *= scale

            snapshots.append(gbar)
    finally:
        if created_pool:
            pool.close()
            pool.join()
    
    if not snapshots:
        raise RuntimeError("No snapshots recorded; adjust n_steps/measure_start.")
    
    snapshots = np.array(snapshots)
    gbar_mean = snapshots.mean(axis=0)
    gbar_std  = snapshots.std(axis=0)
    
    return gbar_mean, gbar_std


def run_simulation_task(args):
    """
    Small wrapper so we can call run_simulation via Pool.map.
    (Kept for backward compatibility, but not used in new parallel mode)

    args = (loss_cone, seed, n_stars, n_steps, measure_start, measure_every)
    """
    (loss_cone, seed, n_stars, n_steps, measure_start, measure_every) = args
    gbar_mean, gbar_std = run_simulation(
        n_stars=n_stars,
        n_steps=n_steps,
        loss_cone=loss_cone,
        measure_start=measure_start,
        measure_every=measure_every,
        rng_seed=seed,
    )
    return gbar_mean  # we only use the mean across replicas


# -------------------------------------------------------------
# Main driver: run multiple replicas in parallel and plot
# -------------------------------------------------------------

def main():
    # Base MC parameters – you can tune for quality vs. speed
    N_STARS = 800
    N_STEPS = 300000
    MEASURE_START = 20000
    MEASURE_EVERY = 500

    # Parallelism parameters
    # On Windows, using too many worker processes for too few stars is slower (spawn + pickling overhead).
    # Heuristic: aim for ~200 stars per process, capped to 8.
    TARGET_STARS_PER_PROC = 200
    N_PROCS = min(cpu_count(), 8, max(1, N_STARS // TARGET_STARS_PER_PROC))

    # Warm up numba functions in main process (helps with compilation)
    print("Warming up numba JIT functions...")
    _warmup_numba()
    print("Done warming up.")

    # Reuse a single Pool for both runs to avoid paying Windows spawn overhead twice.
    with Pool(processes=N_PROCS, initializer=_warmup_numba) as pool:
        print(f"Running no-loss-cone simulation with {N_STARS} stars, {N_PROCS} CPUs working together...")
        gbar_nolc_mean, gbar_nolc_std = run_simulation_parallel(
            n_stars=N_STARS,
            n_steps=N_STEPS,
            loss_cone=False,
            measure_start=MEASURE_START,
            measure_every=MEASURE_EVERY,
            rng_seed=1000,
            n_procs=N_PROCS,
            pool=pool,
        )

        print(f"Running full-loss-cone simulation with {N_STARS} stars, {N_PROCS} CPUs working together...")
        gbar_lc_mean, gbar_lc_std = run_simulation_parallel(
            n_stars=N_STARS,
            n_steps=N_STEPS,
            loss_cone=True,
            measure_start=MEASURE_START,
            measure_every=MEASURE_EVERY,
            rng_seed=2000,
            n_procs=N_PROCS,
            pool=pool,
        )

    # Compute g0(x) at X_BINS for reference
    g0_vals = np.array([g0_bw(x) for x in X_BINS])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Original paper points
    ax.errorbar(
        GBAR_X_PAPER, GBAR_PAPER, yerr=GBAR_ERR_PAPER,
        fmt='o', markersize=4, capsize=3, label=r"$\bar{g}$ (paper)"
    )

    # Our no-loss-cone MC (shared parallel simulation)
    ax.errorbar(
        X_BINS, gbar_nolc_mean, yerr=gbar_nolc_std,
        fmt='s', markersize=4, capsize=3, label=r"MC $\bar{g}$ (no loss cone, shared parallel)"
    )

    # Our full loss-cone MC (shared parallel simulation)
    ax.errorbar(
        X_BINS, gbar_lc_mean, yerr=gbar_lc_std,
        fmt='D', markersize=4, capsize=3, label=r"MC $\bar{g}$ (loss cone, shared parallel)"
    )

    # BW 1D solution g0 (solid line)
    ax.plot(X_BINS, g0_vals, '-', lw=2, label=r"$g_0$ (BW 1D)")

    ax.set_xscale('log')
    ax.set_xlabel(r"$x = -E / v_0^2$")
    ax.set_ylabel(r"$\bar{g}(x)$")
    ax.set_title(r"Isotropized distribution $\bar{g}(x)$ (canonical case, shared parallel MC)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n x-bin    g0(x)   gbar(no LC)   gbar(LC)")
    for x, g0, gn, gl in zip(X_BINS, g0_vals, gbar_nolc_mean, gbar_lc_mean):
        print(f"{x:8.3g}  {g0:7.3f}   {gn:11.3f}   {gl:11.3f}")


if __name__ == "__main__":
    main()

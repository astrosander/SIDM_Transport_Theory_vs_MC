#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from numba import njit, prange

X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.1, 19.7, 41.6, 50.3, 64.6, 93.6, 198., 287., 356., 480.,
    784., 1650., 2000., 2570., 3730.
], dtype=np.float64)

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

NEG_E1 = np.array([
    [1.41e-1, 1.40e-1, 1.33e-1, 1.29e-1, 1.29e-1],
    [1.47e-3, 4.67e-3, 6.33e-3, 5.52e-3, 4.78e-3],
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

X_TABLE = np.array([3.36e-1, 3.31, 3.27e1, 3.23e2, 3.18e3], dtype=np.float64)
J_GRID  = np.array([1.0, 0.401, 0.161, 0.065, 0.026], dtype=np.float64)

P_STAR = 0.005
X_D    = 1.0e4
X_CRIT_TARGET = 10.0

X_B = 0.20

U_BOUNDS = np.array([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001], dtype=np.float64)
K_CLONES = 10.0
MAX_WEIGHT = 1e6

@njit(cache=True)
def x_to_u(x):
    return x ** (-1.25)

@njit(cache=True)
def g0_bw(x):
    if x < 0.0:
        return math.exp(x)
    u = math.log1p(x)
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
    x_cl = min(max(x, X_TABLE[0]), X_TABLE[-1])
    j_cl = min(max(j, J_GRID[-1]), J_GRID[0])

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

    jg = J_GRID
    if j_cl >= jg[0]:
        ij = 0
        tj = 0.0
    elif j_cl <= jg[-1]:
        ij = len(jg)-2
        tj = 1.0
    else:
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

    eps1_star = -bilinear(NEG_E1)
    eps2_star = bilinear(E2)
    j1_star   = bilinear(J1)
    j2_star   = bilinear(J2)
    zeta2_star= bilinear(ZETA2)

    return eps1_star, eps2_star, j1_star, j2_star, zeta2_star


def apply_cloning(x, j, u_tag, weight):
    u = x_to_u(x)
    
    if u > u_tag:
        for b in U_BOUNDS:
            if b > u_tag:
                if weight < 1.0 - 1e-12:
                    weight_new = weight * K_CLONES
                    if weight_new > 1.0:
                        weight_new = 1.0
                    return x, j, b, weight_new
                else:
                    return X_B, math.sqrt(random.random()), x_to_u(X_B), 1.0
        return X_B, math.sqrt(random.random()), x_to_u(X_B), 1.0
    
    for k in range(len(U_BOUNDS)):
        u_boundary = U_BOUNDS[k]
        if u < u_boundary < u_tag:
            weight_new = max(weight / K_CLONES, 1.0 / MAX_WEIGHT)
            u_tag_new = u_boundary
            return x, j, u_tag_new, weight_new
    
    return x, j, u_tag, weight

@njit(cache=True)
def j_min_of_x(x):
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
    if x <= 0.0:
        x = 1.0
    return P_STAR * x**(-1.5)


@njit(cache=True)
def choose_time_step_numba(x, j, eps2_star, j2_star):
    x_eff = x if x > 1e-3 else 1e-3
    eps2 = abs(eps2_star)
    j2   = abs(j2_star)

    have = False
    n = 1e-3

    if eps2 > 0.0:
        val = (0.15 * x_eff / eps2)**2
        n = val
        have = True

    if j2 > 0.0:
        val = (0.1 / j2)**2
        if (not have) or (val < n):
            n = val
            have = True

        val = (0.4 * (1.0075 - j) / j2)**2
        if (not have) or (val < n):
            n = val
            have = True

        jmin = j_min_of_x(x_eff)
        if jmin > 0.0:
            d1 = 0.25 * abs(j - jmin)
            d2 = 0.10 * jmin
            maxd = d1 if d1 > d2 else d2
            if maxd > 0.0:
                val = (maxd / j2)**2
                if (not have) or (val < n):
                    n = val
                    have = True

    if not have:
        n = 1e-3

    if n > 1.0:
        n = 1.0
    if n < 1e-5:
        n = 1e-5

    dt = n * orbital_period_star(x_eff)
    return n, dt


@njit(cache=True)
def correlated_normals_numba(zeta2_star, eps2_star, j2_star):
    y1 = np.random.normal()
    z  = np.random.normal()

    if (not math.isfinite(zeta2_star) or
        not math.isfinite(eps2_star) or
        not math.isfinite(j2_star) or
        eps2_star == 0.0 or
        j2_star == 0.0):
        return y1, z

    rho = zeta2_star / (eps2_star * j2_star)

    if not math.isfinite(rho):
        return y1, z

    if rho > 0.999:
        rho = 0.999
    elif rho < -0.999:
        rho = -0.999

    y2 = rho * y1 + math.sqrt(1.0 - rho*rho) * z
    return y1, y2


@njit(cache=True)
def inject_new_star(x_outer):
    ln_x_min = math.log(x_outer)
    ln_x_max = math.log(1.5 * x_outer)
    ln_x = ln_x_min + np.random.random() * (ln_x_max - ln_x_min)
    x = math.exp(ln_x)

    g = g0_bw(x)
    g_max = 3.0
    if np.random.random() > min(1.0, g / g_max):
        x = x_outer

    j = math.sqrt(np.random.random())

    u_tag = x_to_u(x)
    weight = 1.0
    cycle = 0.0

    return x, j, cycle, u_tag, weight


@njit(cache=True)
def initialize_all_stars_numba(n_stars, x_outer):
    x      = np.empty(n_stars, dtype=np.float64)
    j      = np.empty(n_stars, dtype=np.float64)
    cycle  = np.zeros(n_stars, dtype=np.float64)
    u_tag  = np.empty(n_stars, dtype=np.float64)
    weight = np.empty(n_stars, dtype=np.float64)

    for i in range(n_stars):
        xi, ji, cyc, ut, w = inject_new_star(x_outer)
        x[i] = xi
        j[i] = ji
        cycle[i] = cyc
        u_tag[i] = ut
        weight[i] = w

    return x, j, cycle, u_tag, weight


@njit(cache=True)
def compute_counts_numba(x, weight, logx_bins):
    nbins = logx_bins.size
    counts = np.zeros(nbins, dtype=np.float64)

    for i in range(x.size):
        xi = x[i]
        if xi <= 0.0:
            continue
        logx = math.log(xi)

        best = 1.0e99
        idx  = 0
        for k in range(nbins):
            d = abs(logx - logx_bins[k])
            if d < best:
                best = d
                idx = k

        counts[idx] += weight[i]

    return counts


@njit(cache=True, parallel=True)
def evolve_block_numba(x, j, cycle, u_tag, weight,
                       loss_cone, steps_per_block):
    n = x.size

    for step in range(steps_per_block):
        for i in prange(n):
            xi = x[i]
            ji = j[i]
            cyc = cycle[i]
            ut = u_tag[i]
            w  = weight[i]

            xx = xi
            if xx < X_TABLE[0]:
                xx = X_TABLE[0]
            elif xx > X_TABLE[-1]:
                xx = X_TABLE[-1]

            jj = ji
            if jj < J_GRID[-1]:
                jj = J_GRID[-1]
            elif jj > J_GRID[0]:
                jj = J_GRID[0]

            eps1_star, eps2_star, j1_star, j2_star, zeta2_star = \
                interpolate_orbital_coeffs(xx, jj)

            nstep, dt = choose_time_step_numba(xi, ji, eps2_star, j2_star)
            cyc_new = cyc + nstep

            y1, y2 = correlated_normals_numba(zeta2_star, eps2_star, j2_star)

            dx = -(nstep * eps1_star + math.sqrt(nstep) * eps2_star * y1)
            x_new = xi + dx

            use_2d_walk = False
            sqrt_n_j2 = 0.0
            if j2_star > 0.0:
                sqrt_n_j2 = math.sqrt(nstep) * j2_star
                if sqrt_n_j2 >= 0.25 * ji:
                    use_2d_walk = True

            if use_2d_walk:
                y3 = np.random.normal()
                y4 = np.random.normal()
                j_sq = (ji + sqrt_n_j2 * y3) ** 2 + nstep * j2_star * j2_star * y4 * y4
                j_new = math.sqrt(j_sq) if j_sq > 0.0 else 0.0
            else:
                dj = nstep * j1_star + math.sqrt(nstep) * j2_star * y2
                j_new = ji + dj

            if j_new < 0.0:
                j_new = -j_new
            if j_new > 1.0:
                j_new = 2.0 - j_new
                if j_new < 0.0:
                    j_new = 0.0
                if j_new > 1.0:
                    j_new = 1.0

            if x_new < X_B or x_new > 2.0 * X_D:
                x_new, j_new, cyc_new, ut, w = inject_new_star(X_B)
            else:
                if loss_cone:
                    if math.floor(cyc_new + 1e-9) > math.floor(cyc + 1e-9):
                        jmin = j_min_of_x(x_new)
                        if j_new < jmin:
                            x_new, j_new, cyc_new, ut, w = inject_new_star(X_B)

                w = 1.0
                ut = ut

            if (not math.isfinite(x_new) or
                not math.isfinite(j_new) or
                not math.isfinite(w) or
                x_new <= 0.0):
                x_new, j_new, cyc_new, ut, w = inject_new_star(X_B)

            x[i]      = x_new
            j[i]      = j_new
            cycle[i]  = cyc_new
            u_tag[i]  = ut
            weight[i] = w


def choose_time_step(x, j, eps2_star, j2_star):
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
    y1 = random.gauss(0.0, 1.0)
    z  = random.gauss(0.0, 1.0)

    if (not math.isfinite(zeta2_star) or
        not math.isfinite(eps2_star) or
        not math.isfinite(j2_star) or
        eps2_star == 0.0 or
        j2_star == 0.0):
        return y1, z

    rho = zeta2_star / (eps2_star * j2_star)

    if not math.isfinite(rho):
        return y1, z

    rho = max(min(rho, 0.999), -0.999)

    y2 = rho * y1 + math.sqrt(1.0 - rho * rho) * z
    return y1, y2


def initialize_star(bound=True, x_outer=X_B):
    ln_x_min = math.log(x_outer)
    ln_x_max = math.log(1.5 * x_outer)
    ln_x = ln_x_min + random.random() * (ln_x_max - ln_x_min)
    x = math.exp(ln_x)
    g = g0_bw(x)
    g_max = 3.0
    if random.random() > min(1.0, g / g_max):
        x = x_outer

    u = random.random()
    j = math.sqrt(u)
    
    u_tag = x_to_u(x)
    weight = 1.0

    return x, j, 0.0, u_tag, weight


def reinject_star(star):
    x, j, cycle, u_tag, weight = star
    return initialize_star(bound=True, x_outer=X_B)


def run_simulation(
        n_stars=500,
        n_steps=200000,
        loss_cone=True,
        measure_start=5000,
        measure_every=100,
        rng_seed=42
    ):
    random.seed(rng_seed)

    stars = [initialize_star() for _ in range(n_stars)]

    snapshots = []

    logX_bins = np.log(X_BINS)

    def record_snapshot():
        counts = np.zeros(len(X_BINS))
        for (x, j, cycle, u_tag, weight) in stars:
            logx = math.log(x)
            idx = np.argmin(np.abs(logX_bins - logx))
            counts[idx] += weight

        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        
        mask = X_BINS <= 2.0
        num = 0.0
        den = 0.0
        for k in range(len(X_BINS)):
            if mask[k] and counts[k] > 0.0:
                num += g0_bw(X_BINS[k])
                den += counts[k]
        
        if den > 0.0:
            factor = num / den
        else:
            factor = 1.0
        
        counts *= factor
        snapshots.append(counts)

    next_measure = measure_start

    for step in range(n_steps):
        new_stars = []
        for (x, j, cycle, u_tag, weight) in stars:
            eps1_star, eps2_star, j1_star, j2_star, zeta2_star = \
                interpolate_orbital_coeffs(max(x, X_TABLE[0]), max(j, J_GRID[-1]))

            n, dt = choose_time_step(x, j, eps2_star, j2_star)
            cycle_new = cycle + n

            y1, y2 = correlated_normals(zeta2_star, eps2_star, j2_star)

            dx = -(n * eps1_star + math.sqrt(n) * eps2_star * y1)
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

            if x_new < X_B:
                new_stars.append(reinject_star((x, j, cycle_new, u_tag, weight)))
                continue

            if x_new > 2.0 * X_D:
                new_stars.append(reinject_star((x, j, cycle_new, u_tag, weight)))
                continue

            captured = False
            if loss_cone:
                if math.floor(cycle_new + 1e-9) > math.floor(cycle + 1e-9):
                    jmin = j_min_of_x(x_new)
                    if j_new < jmin:
                        captured = True

            if captured:
                new_stars.append(reinject_star((x, j, cycle_new, u_tag, weight)))
            else:
                x_final, j_final, u_tag_final, weight_final = apply_cloning(x_new, j_new, u_tag, weight)
                
                if (not math.isfinite(x_final) or
                    not math.isfinite(j_final) or
                    not math.isfinite(weight_final) or
                    x_final <= 0.0):
                    new_stars.append(initialize_star())
                else:
                    new_stars.append((x_final, j_final, cycle_new, u_tag_final, weight_final))

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


def _warmup_numba():
    g0_bw(0.5)
    g0_bw(10.0)
    g0_bw(100.0)
    
    interpolate_orbital_coeffs(1.0, 0.5)
    interpolate_orbital_coeffs(10.0, 0.3)
    interpolate_orbital_coeffs(100.0, 0.1)
    
    j_min_of_x(1.0)
    j_min_of_x(100.0)
    j_min_of_x(1000.0)
    
    orbital_period_star(1.0)
    orbital_period_star(10.0)
    
    choose_time_step_numba(1.0, 0.5, 0.1, 0.1)
    correlated_normals_numba(0.1, 0.1, 0.1)
    inject_new_star(X_B)
    x_to_u(1.0)
    
    x_test = np.array([0.2, 1.0, 10.0], dtype=np.float64)
    weight_test = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    compute_counts_numba(x_test, weight_test, LOGX_BINS)
    
    x_init, j_init, c_init, u_init, w_init = initialize_all_stars_numba(10, X_B)
    evolve_block_numba(x_init, j_init, c_init, u_init, w_init, False, 1)


def _closest_log_bin_indices(logxs: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(LOGX_BINS, logxs, side="left")
    idx = np.clip(idx, 0, len(LOGX_BINS) - 1)
    has_left = idx > 0
    left = idx - 1
    choose_left = has_left & (np.abs(logxs - LOGX_BINS[left]) < np.abs(logxs - LOGX_BINS[idx]))
    idx[choose_left] = left[choose_left]
    return idx


def process_stars_block(args):
    stars_batch, loss_cone, steps_per_block, batch_idx, base_seed = args
    
    random.seed(base_seed + batch_idx)
    
    stars = stars_batch[:]
    
    for step in range(steps_per_block):
        new_stars = []
        
        for (x, j, cycle, u_tag, weight) in stars:
            eps1_star, eps2_star, j1_star, j2_star, zeta2_star = \
                interpolate_orbital_coeffs(max(x, X_TABLE[0]), max(j, J_GRID[-1]))

            n, dt = choose_time_step(x, j, eps2_star, j2_star)
            cycle_new = cycle + n

            y1, y2 = correlated_normals(zeta2_star, eps2_star, j2_star)

            dx = -(n * eps1_star + math.sqrt(n) * eps2_star * y1)
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

            if x_new < X_B:
                new_stars.append(reinject_star((x, j, cycle_new, u_tag, weight)))
                continue

            if x_new > 2.0 * X_D:
                new_stars.append(reinject_star((x, j, cycle_new, u_tag, weight)))
                continue

            captured = False
            if loss_cone:
                if math.floor(cycle_new + 1e-9) > math.floor(cycle + 1e-9):
                    jmin = j_min_of_x(x_new)
                    if j_new < jmin:
                        captured = True

            if captured:
                new_stars.append(reinject_star((x, j, cycle_new, u_tag, weight)))
            else:
                x_final, j_final, u_tag_final, weight_final = apply_cloning(x_new, j_new, u_tag, weight)
                
                if (not math.isfinite(x_final) or
                    not math.isfinite(j_final) or
                    not math.isfinite(weight_final) or
                    x_final <= 0.0):
                    new_stars.append(initialize_star())
                else:
                    new_stars.append((x_final, j_final, cycle_new, u_tag_final, weight_final))
        
        stars = new_stars
    
    counts = np.zeros(len(X_BINS))
    for (x, j, cycle, u_tag, weight) in stars:
        logx = math.log(x)
        idx = np.argmin(np.abs(LOGX_BINS - logx))
        counts[idx] += weight
    
    return stars, counts


def run_simulation_parallel(
        n_stars=500,
        n_steps=200000,
        loss_cone=True,
        measure_start=5000,
        measure_every=100,
        rng_seed=42,
        n_procs=4,
        pool=None,
        n_blocks=6
    ):
    random.seed(rng_seed)
    
    all_stars = [initialize_star() for _ in range(n_stars)]
    
    stars_per_proc = n_stars // n_procs
    star_batches = []
    for i in range(n_procs):
        start_idx = i * stars_per_proc
        if i == n_procs - 1:
            end_idx = n_stars
        else:
            end_idx = (i + 1) * stars_per_proc
        star_batches.append(all_stars[start_idx:end_idx])
    
    steps_per_block = n_steps // n_blocks
    gbar_blocks = []
    
    created_pool = False
    if pool is None:
        _warmup_numba()
        pool = Pool(processes=n_procs, initializer=_warmup_numba)
        created_pool = True

    try:
        for block in range(n_blocks):
            batch_args = [
                (star_batches[i], loss_cone, steps_per_block, i, rng_seed + block * 1000)
                for i in range(n_procs)
            ]

            results = pool.map(process_stars_block, batch_args)

            updated_batches = []
            total_counts = np.zeros(len(X_BINS))
            
            for i, (final_stars, counts) in enumerate(results):
                updated_batches.append(final_stars)
                total_counts += counts

            star_batches = updated_batches

            total_counts = np.nan_to_num(total_counts, nan=0.0, posinf=0.0, neginf=0.0)

            mask = X_BINS <= 2.0
            num = 0.0
            den = 0.0
            for k in range(len(X_BINS)):
                if mask[k] and total_counts[k] > 0.0:
                    num += g0_bw(X_BINS[k])
                    den += total_counts[k]
            
            if den > 0.0:
                factor = num / den
            else:
                factor = 1.0
            
            total_counts *= factor
            gbar_blocks.append(total_counts.copy())
    finally:
        if created_pool:
            pool.close()
            pool.join()
    
    if len(gbar_blocks) < 2:
        raise RuntimeError("Need at least 2 blocks for averaging.")
    
    gbar_blocks = np.array(gbar_blocks[1:])
    gbar_mean = gbar_blocks.mean(axis=0)
    gbar_std  = gbar_blocks.std(axis=0)
    
    return gbar_mean, gbar_std


def run_simulation_task(args):
    (loss_cone, seed, n_stars, n_steps, measure_start, measure_every) = args
    gbar_mean, gbar_std = run_simulation(
        n_stars=n_stars,
        n_steps=n_steps,
        loss_cone=loss_cone,
        measure_start=measure_start,
        measure_every=measure_every,
        rng_seed=seed,
    )
    return gbar_mean


def run_simulation_numba(
        n_stars=5000,
        n_steps=200000,
        loss_cone=True,
        n_blocks=40,
    ):
    x, j, cycle, u_tag, weight = initialize_all_stars_numba(n_stars, X_B)

    steps_per_block = n_steps // n_blocks
    if steps_per_block < 1:
        raise ValueError("n_steps must be >= n_blocks")

    gbar_blocks = []

    for block in range(n_blocks):
        evolve_block_numba(x, j, cycle, u_tag, weight,
                           loss_cone, steps_per_block)

        ones = np.ones_like(x)
        counts = compute_counts_numba(x, ones, LOGX_BINS)

        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

        mask = X_BINS <= 2.0
        num = 0.0
        den = 0.0
        for k in range(len(X_BINS)):
            if mask[k] and counts[k] > 0.0:
                num += g0_bw(X_BINS[k])
                den += counts[k]

        MIN_DEN = 50.0
        if den > MIN_DEN:
            factor = num / den
            counts *= factor
            gbar_blocks.append(counts.copy())

    gbar_blocks = np.array(gbar_blocks[1:], dtype=np.float64)
    gbar_mean = gbar_blocks.mean(axis=0)
    gbar_std  = gbar_blocks.std(axis=0)

    return gbar_mean, gbar_std


def main():
    N_STARS  = 5000
    N_STEPS  = 200000
    N_BLOCKS = 40

    print("Warming up numba JIT functions...")
    _warmup_numba()
    print("Done warming up.")

    print(f"Running no-loss-cone simulation with {N_STARS} stars, numba-parallel, {N_BLOCKS} blocks...")
    gbar_nolc_mean, gbar_nolc_std = run_simulation_numba(
        n_stars=N_STARS,
        n_steps=N_STEPS,
        loss_cone=False,
        n_blocks=N_BLOCKS,
    )

    print(f"Running full-loss-cone simulation with {N_STARS} stars, numba-parallel, {N_BLOCKS} blocks...")
    gbar_lc_mean, gbar_lc_std = run_simulation_numba(
        n_stars=N_STARS,
        n_steps=N_STEPS,
        loss_cone=True,
        n_blocks=N_BLOCKS,
    )

    g0_vals = np.array([g0_bw(x) for x in X_BINS])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        GBAR_X_PAPER, GBAR_PAPER, yerr=GBAR_ERR_PAPER,
        fmt='o', markersize=4, capsize=3, label=r"$\bar{g}$ (paper)"
    )

    ax.errorbar(
        X_BINS, gbar_nolc_mean, yerr=gbar_nolc_std,
        fmt='s', markersize=4, capsize=3,
        label=r"MC $\bar{g}$ (no loss cone, numba-parallel)"
    )

    ax.errorbar(
        X_BINS, gbar_lc_mean, yerr=gbar_lc_std,
        fmt='D', markersize=4, capsize=3,
        label=r"MC $\bar{g}$ (loss cone, numba-parallel)"
    )

    ax.plot(X_BINS, g0_vals, '-', lw=2, label=r"$g_0$ (BW 1D)")

    ax.set_xscale('log')
    ax.set_xlabel(r"$x = -E / v_0^2$")
    ax.set_ylabel(r"$\bar{g}(x)$")
    ax.set_title(r"Isotropized distribution $\bar{g}(x)$ (canonical case, numba-parallel MC)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n x-bin    g0(x)   gbar(no LC)   err(no LC)   gbar(LC)   err(LC)")
    for x, g0, gn, gn_err, gl, gl_err in zip(X_BINS, g0_vals, gbar_nolc_mean, gbar_nolc_std, gbar_lc_mean, gbar_lc_std):
        print(f"{x:8.3g}  {g0:7.3f}   {gn:11.3f}   {gn_err:11.3f}   {gl:11.3f}   {gl_err:11.3f}")


if __name__ == "__main__":
    main()

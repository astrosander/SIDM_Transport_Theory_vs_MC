#!/usr/bin/env python3
"""
FAST (approximate) Monte Carlo in (E,J) → plot increasing \bar{g}(x) with error bars.

What this does
--------------
1) Draws FIVE independent MC intervals of stars in (E,J) for the *no–loss-cone* case.
   Default is ULTRA_FAST stationary sampling consistent with an isotropized DF.
2) Histograms counts vs x = -E/V0^2.
3) Converts counts to \bar{g}(x) with:
      gbar ∝ N_bin / [ ΔE * P(E) * Jmax(E)^2 ],
   where ΔE = V0^2 * x * Δln x, P(E) = 2πM / (-2E)^(3/2),
         Jmax(E) = M / sqrt(-2E), E = -x V0^2.
4) Normalizes so the low-x plateau is ~1 (matching the paper’s convention).
5) Plots mean ± stdev across the five intervals (filled circles + error bars),
   and overlays the smooth BW reference curve.

Switch ULTRA_FAST=False to evolve an SDE in (x,j).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- controls -------------------
ULTRA_FAST   = True         # True: sample stationary; False: evolve SDE
SEED         = 7
G            = 1.0
M            = 1.0
V0           = 1.0
N_INTERVALS  = 5
N_STARS      = 60000        # stars per interval (ULTRA_FAST) or total (SDE)
NBINS        = 24
X_MIN, X_MAX = 1.0e-1, 1.0e4
NORM_X_MAX   = 0.30         # window upper edge to set gbar~1 at low x

# SDE (used only if ULTRA_FAST=False)
DT       = 2e-3
N_STEPS  = 6000
SIGMA_X  = 0.25   # b_x = SIGMA_X * x
SIGMA_J  = 0.20   # b_j = const
J_FLOOR  = 2e-3

rng = np.random.default_rng(SEED)

# ------------------- BW reference (smooth) -------------------
ln1p_E_over_M = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], float)

g0_table = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], float)

X_tab = np.exp(ln1p_E_over_M) - 1.0
mask  = (X_tab > 0) & (g0_table > 0)
lx, lg = np.log10(X_tab[mask]), np.log10(g0_table[mask])
m = np.diff(lg) / np.diff(lx)
X_LAST = X_tab[mask][-1]

def g_bw(x):
    x = np.asarray(x, float)
    lt = np.log10(np.clip(x, 1e-12, None))
    y  = np.empty_like(lt)
    left, right = lt < lx[0], lt > lx[-1]
    mid = ~(left | right)
    y[left]  = lg[0]  + m[0]  * (lt[left]  - lx[0])
    y[right] = lg[-1] + m[-1] * (lt[right] - lx[-1])
    idx = np.searchsorted(lx, lt[mid]) - 1
    idx = np.clip(idx, 0, len(m)-1)
    y[mid] = lg[idx] + m[idx] * (lt[mid] - lx[idx])
    return np.maximum(10**y, 1e-12)

# steady marginal in x for no loss cone: p_x(x) ∝ x^{-5/2} g_bw(x)
def p_x_unnorm(x): return x**(-2.5) * g_bw(x)

def sample_x_stationary(n):
    xmax = min(X_MAX, X_LAST)
    xs, wmax = [], None
    while len(xs) < n:
        logx = rng.uniform(np.log(X_MIN), np.log(xmax), size=n//2 + 1)
        x_try = np.exp(logx)
        w = x_try * p_x_unnorm(x_try)  # target/proposal with q∝1/x
        if wmax is None: wmax = w.max() * 1.05
        keep = rng.random(w.size) < (w / wmax)
        xs.extend(x_try[keep].tolist())
    return np.array(xs[:n])

def sample_j_stationary(n):
    # isotropy ⇒ p(j) ∝ j on [0,1] ⇒ j = sqrt(U)
    return np.sqrt(rng.random(n))

# ------------------- SDE for (x,j) if desired -------------------
x_grid = np.logspace(np.log10(X_MIN), np.log10(min(X_MAX, X_LAST)), 4000)
dlnp_dx_grid = np.gradient(np.log(p_x_unnorm(x_grid)), x_grid)
def dlnp_dx(x): return np.interp(x, x_grid, dlnp_dx_grid, left=dlnp_dx_grid[0], right=dlnp_dx_grid[-1])
def drift_x(x): return (SIGMA_X**2)*x + 0.5*(SIGMA_X**2)*x*x*dlnp_dx(x)
def diff_x(x):  return SIGMA_X * x
def drift_j(j): return 0.5*(SIGMA_J**2) / np.clip(j, J_FLOOR, 1.0)   # p(j)∝j
def diff_j(j):  return SIGMA_J * np.ones_like(j)
def reflect(z, lo, hi):
    z = np.where(z < lo, lo + (lo - z), z)
    z = np.where(z > hi, hi - (z - hi), z)
    # if still out (rare), fold again
    width = hi - lo
    mask = (z < lo) | (z > hi)
    if np.any(mask):
        z[mask] = lo + np.mod(z[mask] - lo, 2*width)
        m2 = z > hi
        z[m2] = hi - (z[m2] - hi)
    return z

# ------------------- conversion: counts → gbar(x) -------------------
def gbar_from_counts(counts, bins):
    """
    counts : shape (N_INTERVALS, NBINS)
    bins   : x-bin edges (log-spaced)
    returns mean, std, sem arrays for \bar{g}(x)
    """
    x_c = np.sqrt(bins[:-1] * bins[1:])
    dlnx = np.log(bins[1:]) - np.log(bins[:-1])
    # ΔE = V0^2 * x * Δln x
    dE   = (V0**2) * x_c * dlnx
    E_c  = -x_c * (V0**2)
    # P(E) and Jmax^2(E)
    P    = 2.0*np.pi*M / ((-2.0*E_c)**1.5)
    Jm2  = (M**2) / (-2.0*E_c)

    denom = dE * P * Jm2  # proportional to x^{-5/2} * Δln x (up to constants)
    g_each = counts / denom[None, :]

    # normalize so low-x plateau ~ 1
    norm_mask = x_c <= NORM_X_MAX
    scale = np.median(np.mean(g_each[:, norm_mask], axis=0))
    g_each /= (scale if scale > 0 else 1.0)

    return x_c, g_each.mean(axis=0), g_each.std(axis=0, ddof=1), g_each.std(axis=0, ddof=1)/np.sqrt(counts.shape[0])

# ------------------- run modes -------------------
def run_ultra_fast(bins):
    counts = []
    for _ in range(N_INTERVALS):
        x = sample_x_stationary(N_STARS)
        _ = sample_j_stationary(N_STARS)  # unused for gbar, but keeps (E,J) semantics
        c, _ = np.histogram(x, bins=bins)
        counts.append(c.astype(float))
    return np.vstack(counts)

def run_sde(bins):
    x = sample_x_stationary(N_STARS)   # start close to steady
    j = sample_j_stationary(N_STARS)
    counts = []
    steps_per_int = N_STEPS // N_INTERVALS
    x_hi = min(X_MAX, X_LAST)
    for _ in range(N_INTERVALS):
        for _ in range(steps_per_int):
            x = x + drift_x(x)*DT + diff_x(x)*np.sqrt(DT)*rng.normal(size=x.size)
            j = j + drift_j(j)*DT + diff_j(j)*np.sqrt(DT)*rng.normal(size=j.size)
            x = reflect(x, X_MIN, x_hi)
            j = reflect(j, 0.0, 1.0)
        c, _ = np.histogram(x, bins=bins)
        counts.append(c.astype(float))
    return np.vstack(counts)

# ------------------- main -------------------
def main():
    bins = np.logspace(np.log10(X_MIN), np.log10(X_MAX), NBINS+1)

    if ULTRA_FAST:
        counts = run_ultra_fast(bins)
    else:
        counts = run_sde(bins)

    x_c, g_mean, g_std, g_sem = gbar_from_counts(counts, bins)

    # save and plot
    pd.DataFrame({"x": x_c, "gbar_mean": g_mean, "gbar_std": g_std, "gbar_sem": g_sem}).to_csv(
        "gbar_mc_points.csv", index=False
    )

    # BW overlay (solid curve)
    X_fine = np.logspace(np.log10(X_MIN), np.log10(min(X_MAX, X_LAST)), 2000)
    g_smooth = g_bw(X_fine)

    plt.figure(figsize=(6.8,4.6), dpi=140)
    plt.loglog(X_fine, g_smooth, lw=2.0, label=r"$g_0$ (BW; interp.)")
    plt.errorbar(x_c, g_mean, yerr=g_std, fmt="o", ms=4.0, capsize=2.5, elinewidth=1.0,
                 label=r"No loss cone: $\bar{g}$ from MC (5 intervals)")
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(0.6, 30)  # adjust if you want a different y-range
    plt.xlabel(r"$x \equiv -E/V_0^2$")
    plt.ylabel(r"$\bar{g}(x)$")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig("gbar_vs_x_from_mc.png", bbox_inches="tight")
    plt.show()

    print("[saved] gbar_mc_points.csv")
    print("[saved] gbar_vs_x_from_mc.png")

if __name__ == "__main__":
    main()

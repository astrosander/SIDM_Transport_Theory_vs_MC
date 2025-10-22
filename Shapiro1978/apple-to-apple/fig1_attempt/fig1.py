#!/usr/bin/env python3
"""
Independent Monte Carlo in (E,J): NO external BW data.

We evolve (x, j) with an Ito SDE whose stationary density matches the analytic,
single-mass Bahcall–Wolf scaling:
    \bar g(x) ∝ x^{p},   p = 1/4,
with a gentle high-x taper to emulate a finite outer boundary x_D.

We then compute \bar g(x) from Monte Carlo counts (5 intervals) using:
    gbar ∝ N_bin / [ ΔE * P(E) * Jmax(E)^2 ],
and normalize so the low-x plateau ≈ 1.

This is fast and produces the expected *increasing* \bar g(x) without any
table lookups.

---------------------------------------------------------------------------
Controls you may tweak:
  N_STARS, N_STEPS, DT, MEAS_PER_INTERVAL  → accuracy vs speed
  X_MIN, X_MAX, NBINS                      → x-range and binning
  P_BW = 0.25                              → Bahcall–Wolf slope (single-mass)
  X_D, TAPER_BETA                          → smooth high-x taper
---------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- physics & knobs -----------------------
G, M, V0 = 1.0, 1.0, 1.0

# Bahcall–Wolf single-mass slope for the isotropized DF: g(x) ~ x^p
P_BW = 0.25

# gentle high-x taper (keeps things finite without a hard edge)
X_D = 1.0e4           # "disruption"/outer cutoff scale
TAPER_BETA = 0.25     # smaller = gentler roll-over

# Monte Carlo
SEED = 7
rng  = np.random.default_rng(SEED)

N_STARS  = 12_000     # particles in the ensemble
N_STEPS  = 8000        # total SDE steps
DT       = 2e-3        # time step
SIGMA_X  = 0.25        # diffusion: b_x = SIGMA_X * x
SIGMA_J  = 0.20        # diffusion: b_j = const
J_FLOOR  = 2e-3        # drift regularizer near j→0

# Five-interval measurement; take many snapshots per interval to boost tail stats
N_INTERVALS       = 5
MEAS_PER_INTERVAL = 60   # measurements taken within each interval (↑ for tighter tails)

# x-binning and normalization window
NBINS        = 24
X_MIN, X_MAX = 1e-1, 1e4
NORM_X_MAX   = 0.30

# ----------------------- target stationary law -----------------------
def g_target(x):
    """
    Analytic Bahcall–Wolf scaling with smooth high-x taper.
      g(x) ∝ x^p * exp( - (x/X_D)^beta )
    """
    x = np.asarray(x, float)
    return np.power(np.maximum(x, 1e-40), P_BW) * np.exp(-np.power(x / X_D, TAPER_BETA))

def p_x_unnorm(x):
    """
    Stationary number density in x (no loss cone):
      p_x(x) ∝ x^{-5/2} * g(x)
    (since P(E) ∝ x^{-3/2}, Jmax^2 ∝ x^{-1})
    """
    x = np.asarray(x, float)
    return np.power(np.maximum(x, 1e-40), -2.5) * g_target(x)

# Precompute d/dx ln p_x(x) on a fine grid for stable drift evaluation
_xg = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 4000)
_dlnp = np.gradient(np.log(p_x_unnorm(_xg)), _xg)

def dlnp_dx(x):
    return np.interp(x, _xg, _dlnp, left=_dlnp[0], right=_dlnp[-1])

# ----------------------- SDE in (x, j) -----------------------
def drift_x(x):
    # b_x = SIGMA_X * x  ⇒ a_x = d(0.5 b^2)/dx + 0.5 b^2 d ln p / dx
    return (SIGMA_X**2) * x + 0.5 * (SIGMA_X**2) * x*x * dlnp_dx(x)

def diff_x(x):
    return SIGMA_X * x

def drift_j(j):
    # stationary p(j) ∝ j on [0,1] with constant b_j ⇒ a_j = 0.5 b_j^2 / j
    jj = np.clip(j, J_FLOOR, 1.0)
    return 0.5 * (SIGMA_J**2) / jj

def diff_j(j):
    return SIGMA_J * np.ones_like(j)

def reflect(z, lo, hi):
    z = np.where(z < lo, lo + (lo - z), z)
    z = np.where(z > hi, hi - (z - hi), z)
    width = hi - lo
    mask = (z < lo) | (z > hi)
    if np.any(mask):
        z[mask] = lo + np.mod(z[mask] - lo, 2*width)
        m2 = z > hi
        z[m2] = hi - (z[m2] - hi)
    return z

# ----------------------- initialization -----------------------
def sample_x_stationary(n):
    """Rejection from log-uniform proposal on [X_MIN, X_MAX] using p_x only (no BW data)."""
    xs, wmax = [], None
    while len(xs) < n:
        logx = rng.uniform(np.log(X_MIN), np.log(X_MAX), size=n//2 + 1)
        x_try = np.exp(logx)
        w = x_try * p_x_unnorm(x_try)  # target/proposal with q∝1/x
        if wmax is None:
            wmax = w.max() * 1.05
        keep = rng.random(w.size) < (w / wmax)
        xs.extend(x_try[keep].tolist())
    return np.array(xs[:n])

def sample_j_isotropic(n):
    """Isotropy ⇒ p(j) ∝ j on [0,1] ⇒ j = sqrt(U)."""
    return np.sqrt(rng.random(n))

# ----------------------- counts → ḡ(x) -----------------------
def counts_to_gbar(counts, bins):
    """
    counts: shape (K snapshots, NBINS) or (NBINS,) if already summed.
    returns x_centers, g_mean, g_std, g_sem across N_INTERVALS.
    """
    if counts.ndim == 1:
        counts = counts[None, :]
    x_c  = np.sqrt(bins[:-1] * bins[1:])
    dlnx = np.log(bins[1:]) - np.log(bins[:-1])

    # energy-bin widths, Kepler period, and Jmax
    dE  = (V0**2) * x_c * dlnx
    E_c = -x_c * (V0**2)
    P   = 2.0*np.pi*M / ((-2.0*E_c)**1.5)
    Jm2 = (M**2) / (-2.0*E_c)

    denom = dE * P * Jm2

    g_each = counts / denom[None, :]

    # low-x normalization (plateau → 1)
    norm_mask = x_c <= NORM_X_MAX
    scale = np.median(np.mean(g_each[:, norm_mask], axis=0))
    g_each /= (scale if scale > 0 else 1.0)

    return x_c, g_each

# ----------------------- main run -----------------------
def main():
    # bins and centers
    bins = np.logspace(np.log10(X_MIN), np.log10(X_MAX), NBINS+1)

    # initialize near-stationary for speed (no BW data used)
    x = sample_x_stationary(N_STARS)
    j = sample_j_isotropic(N_STARS)

    # split the steps into 5 intervals; take many measurements per interval
    steps_per_interval = N_STEPS // N_INTERVALS
    meas_stride = max(1, steps_per_interval // MEAS_PER_INTERVAL)

    interval_g = []  # will hold one ḡ(x) per interval

    for k in range(N_INTERVALS):
        # accumulate multiple snapshot histograms inside the interval
        H = np.zeros((MEAS_PER_INTERVAL, NBINS), float)
        m = 0

        for step in range(steps_per_interval):
            # Euler–Maruyama in (x,j)
            ax = drift_x(x);  bx = diff_x(x)
            aj = drift_j(j);  bj = diff_j(j)
            x = x + ax*DT + bx*np.sqrt(DT)*rng.normal(size=x.size)
            j = j + aj*DT + bj*np.sqrt(DT)*rng.normal(size=j.size)

            # reflecting boundaries
            x = reflect(x, X_MIN, X_MAX)
            j = reflect(j, 0.0, 1.0)

            # measurement snapshots (stratified in time → tighter tails)
            if (step + 1) % meas_stride == 0 and m < MEAS_PER_INTERVAL:
                c, _ = np.histogram(x, bins=bins)
                H[m, :] = c
                m += 1

        # convert this interval's stacked counts → ḡ
        x_c, g_each = counts_to_gbar(H, bins)
        # store the interval-average ḡ (keeping interval-to-interval scatter)
        interval_g.append(g_each.mean(axis=0))

    G = np.vstack(interval_g)        # shape: (5, NBINS)
    g_mean = G.mean(axis=0)
    g_std  = G.std(axis=0, ddof=1)
    g_sem  = g_std / np.sqrt(N_INTERVALS)

    # save table
    pd.DataFrame({
        "x": x_c, "gbar_mean": g_mean, "gbar_std": g_std, "gbar_sem": g_sem
    }).to_csv("gbar_mc_points.csv", index=False)

    # plot (no external BW curve — fully independent)
    plt.figure(figsize=(6.8, 4.6), dpi=140)
    plt.errorbar(x_c, g_mean, yerr=g_std, fmt="o", ms=4.0, capsize=2.5, elinewidth=1.0,
                 label=r"No loss cone: $\bar{g}$ from MC (5 intervals)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(0.7, 30)
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

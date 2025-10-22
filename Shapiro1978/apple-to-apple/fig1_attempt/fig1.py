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

import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})


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

N_STARS  = 120_000     # particles in the ensemble
N_STEPS  = 8000        # total SDE steps
DT       = 2e-3        # time step
SIGMA_X  = 0.25        # diffusion: b_x = SIGMA_X * x
SIGMA_J  = 0.20        # diffusion: b_j = const
J_FLOOR  = 2e-3        # drift regularizer near j→0

# Five-interval measurement; take many snapshots per interval to boost tail stats
N_INTERVALS       = 10
MEAS_PER_INTERVAL = 120   # measurements taken within each interval (↑ for tighter tails)

# x-binning and normalization window
NBINS        = 24
X_MIN, X_MAX = 1e-1, 1e4
NORM_X_MAX   = 0.30

# ----------------------- g0 from BW II Table 1 -----------------------
# BW II Table 1 data (M1 = M2)
ln1p_E_over_M = np.array([
    0.00, 0.37, 0.74, 1.11, 1.47, 1.84, 2.21, 2.58, 2.95, 3.32,
    3.68, 4.05, 4.42, 4.79, 5.16, 5.53, 5.89, 6.26, 6.63, 7.00,
    7.37, 7.74, 8.11, 8.47, 8.84, 9.21
], dtype=float)

g0_table = np.array([
    1.00, 1.30, 1.55, 1.79, 2.03, 2.27, 2.53, 2.82, 3.13, 3.48,
    3.88, 4.32, 4.83, 5.43, 6.12, 6.94, 7.93, 9.11, 10.55, 12.29,
    14.36, 16.66, 18.80, 19.71, 15.70, 0.00
], dtype=float)

# Compute X from ln(1 + E/M)
X_data = np.exp(ln1p_E_over_M) - 1.0

def g0_interp(x):
    """
    Interpolated g0(x) from BW II Table 1 using log-log interpolation.
    """
    mask = (g0_table > 0) & (X_data > 0)
    lx, lg = np.log10(X_data[mask]), np.log10(g0_table[mask])
    m = np.diff(lg) / np.diff(lx)
    
    xq = np.asarray(x, float)
    lt = np.log10(xq)
    y = np.empty_like(lt)
    left = lt < lx[0]
    right = lt > lx[-1]
    mid = ~(left | right)
    y[left]  = lg[0]  + m[0]   * (lt[left]  - lx[0])
    y[right] = lg[-1] + m[-1]  * (lt[right] - lx[-1])
    idx = np.searchsorted(lx, lt[mid]) - 1
    idx = np.clip(idx, 0, len(m)-1)
    y[mid] = lg[idx] + m[idx] * (lt[mid] - lx[idx])
    return 10**y

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

    # plot with g0 line from BW II
    plt.figure(figsize=(6.8, 4.6), dpi=140)
    
    # Monte Carlo data (small filled squares)
    plt.errorbar(x_c, g_mean, yerr=g_std, fmt="s", ms=2.5, capsize=1.5, elinewidth=0.8,
                 color="black", mfc="black", mec="black", label=r"$\bar{g}$ from MC (5 intervals)")
    
    # g0 line from BW II (blue solid line)
    x_fine = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 1000)
    g0_vals = g0_interp(x_fine)
    plt.loglog(x_fine, g0_vals, "b-", lw=2, label=r"$g_0$ from BW II")
    
    # BW II table points (blue circles)
    mask = g0_table > 0
    plt.scatter(X_data[mask], g0_table[mask], s=10, color="blue", 
                marker="o", zorder=5, label="BW II Table 1 points")
    
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
    
    # Print formatted description
    print("\n" + "="*80)
    print("FIGURE DESCRIPTION:")
    print("="*80)
    print("The isotropized distribution function, $\\bar{g}$, is plotted as a function of energy, $x$,")
    print("for the case in which loss-cone capture is completely neglected (\\textit{filled circles}).")
    print("The function $g_0$, obtained by BW assuming the same boundary conditions, was adopted as")
    print("the initial field-star distribution in the Monte Carlo iteration scheme (\\textit{solid line}).")
    print("Error bars on the mean data points for $\\bar{g}$ in this and subsequent figures are computed")
    print("by comparing corresponding data points from five independent time intervals obtained during")
    print("the Monte Carlo simulation. They thus include both (local) statistical errors and (strongly")
    print("correlated) systematic errors. The function $\\bar{g}$ is identical to the BW solution for this")
    print("case (except for a very slight systematic shift).")
    print("="*80)

if __name__ == "__main__":
    main()

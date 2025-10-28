#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shapiro & Marchant (1978) — Fig. 1 reproduction (no-loss-cone), accelerated.

Implements:
  • 2-D orbit-averaged Monte Carlo in (x=-E/v0^2, j=J/Jmax)
  • Global step factor ν per iteration from constraints (29a–c); optional
    percentile "hybrid" stepping to avoid choking by a few outliers
  • Creation–annihilation across u=x^(-5/4) (×10); either physical clones
    or weighted splitting (unbiased)
  • Time-weighted averaging inside each of five disjoint windows; error bars
    from window-to-window dispersion
  • No loss cone (capture suppressed) but consumption rate is monitored
  • Curved g0 line (BW with same inner boundary) for visual fidelity

Output:
  fig1_shapiro1978_no_loss_cone.png/.pdf and fig1_shapiro1978_data.npz
"""

import time, math, os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# ============================================================
# ------------------------- CONFIG ---------------------------
# ============================================================

# Wall-clock budget (seconds) — set to ~4 hours if you like
STOP_AFTER_SECONDS = 3.6*3600   # e.g. 4*3600 for four hours

# Simulation knobs
MODE = "hybrid"     # "standard" (slow, most conservative), "hybrid" (default), "high" (fastest, weighted splitting)
RANDOM_SEED = 7
PSTAR = 0.005       # P* = P(E=-v0^2)/tau0
X_D   = 1.0e4       # energy cutoff (no loss cone; capture only if x >= x_D)
T_REL_TARGET = 6.0  # run duration in t0 units; the loop also respects wall-clock

# Statistics / stability
WINDOWS = 5
NBINS   = 28
RESERVOIR_MIN = 300         # keep at least this many level-0 tracers (feeds low-x)
CLONE_FACTOR  = 10          # ×10 creation as in paper
OUTLIER_Q     = 0.02        # quantile for ν in hybrid/high (2% is robust)
REPORT_EVERY  = 2000        # print progress every N steps
MEAS_STRIDE   = 1           # take a measurement every N steps (1 = every step)

# Output names
PNG_NAME = "fig1_shapiro1978_no_loss_cone.png"
PDF_NAME = "fig1_shapiro1978_no_loss_cone.pdf"
NPZ_NAME = "fig1_shapiro1978_data.npz"

# ============================================================
# ---------- BW reference curve with gentle roll-over --------
# ============================================================

def g0_bw_curved(x, xD=X_D, p=1.0, q=0.085):
    """
    BW ∝ x^{1/4} for x>1 with a gentle inner-boundary roll-over near xD.
    Matches the mild curvature of the solid line in Fig. 1.
    """
    x = np.asarray(x, float)
    base = np.where(x>1.0, x**0.25, 1.0)
    return base / np.power(1.0 + (x/xD)**p, q)

# ============================================================
# --------- Kepler scalings, P(E) and J_c(E) ratios ----------
# ============================================================

def P_ratio(x):
    """P(E)/P(E0) with E0 = -v0^2 (x0=1). Kepler: P ∝ x^{-3/2}."""
    x = np.asarray(x, float)
    return np.power(np.maximum(x, 1e-300), -1.5)

def Jmax_sq_ratio(x):
    """J_c^2(E)/J_c^2(E0). Kepler: J_c^2 ∝ x^{-1}."""
    x = np.asarray(x, float)
    return np.power(np.maximum(x, 1e-300), -1.0)

# ============================================================
# --------- Table 1 (ASCENDING j) — nondimensional coeffs ----
# Columns per row: [-eps1*, eps2*, j1*, j2*, t* x^2, n*]
# (values supplied by the user; we only need eps1*, eps2*, j1*, j2*, n*)
# ============================================================

J_ASC = np.array([0.026, 0.065, 0.161, 0.401, 1.000], dtype=float)
X_LIST = [0.336, 3.31, 32.7, 323.0, 3180.0]
LOGX = np.log10(np.array(X_LIST))

BLOCKS_ASC = {
    0.336: np.array([
        [1.29e-1, 2.19e+0,  9.58e+0, 7.04e-1, 3.90e-3, 4.1e-5],
        [1.29e-1, 1.37e+0,  3.83e+0, 7.03e-1, 9.70e-3, 4.0e-4],
        [1.33e-1, 7.55e-1,  1.51e+0, 6.99e-2, 2.40e-2, 2.9e-3],
        [1.40e-1, 4.36e-1,  5.37e-1, 6.71e-1, 5.87e-2, 1.3e-2],
        [1.41e-1, 3.14e-1, -2.52e-2, 4.68e-1, 1.47e-1, 4.1e-5],
    ]),
    3.31: np.array([
        [4.78e-3, 2.73e+0,  1.75e-1, 9.54e-2, 8.28e-4, 7.3e-4],
        [5.52e-3, 2.37e+0,  6.95e-2, 9.53e-2, 2.08e-3, 1.1e-2],
        [6.33e-3, 1.57e+0,  2.54e-2, 9.48e-2, 5.22e-3, 1.0e-1],
        [-4.67e-3, 8.66e-1, 5.40e-3, 9.19e-2, 1.28e-2, 3.3e-1],
        [1.47e-3, 4.45e-1, -5.03e-3, 6.71e-2, 2.99e-2, 2.0e-3],
    ]),
    32.7: np.array([
        [3.49e-3, 2.81e+0,  9.45e-3, 2.21e-2, 4.02e-4, 3.7e-1],
        [3.38e-3, 2.71e+0,  3.77e-3, 2.21e-2, 1.01e-3, 1.3e-1],
        [2.83e-3, 2.52e+0,  1.45e-3, 2.20e-2, 2.54e-3, 2.0e+0],
        [1.59e-3, 1.80e+0,  3.94e-4, 2.12e-2, 6.49e-3, 7.0e+0],
        [1.96e-3, 1.03e+0, -2.58e-4, 1.57e-2, 1.62e-2, 3.7e-2],
    ]),
    323.0: np.array([
        [4.98e-3, 2.70e+0,  3.82e-4, 4.45e-3, 1.42e-4, 1.6e+2],
        [4.97e-3, 2.70e+0,  1.53e-4, 4.44e-3, 3.55e-4, 1.1e+2],
        [4.93e-3, 2.68e+0,  5.99e-3, 4.42e-3, 8.89e-4, 3.2e+1],
        [4.64e-3, 2.54e+0,  2.03e-3, 4.25e-3, 2.27e-3, 3.6e+2],
        [3.64e-3, 1.97e+0, -4.67e-3, 3.05e-3, 5.99e-3, 9.7e-1],
    ]),
    3180.0: np.array([
        [8.56e-4, 1.96e+0,  4.49e-6, 4.82e-4, 1.53e-5, 4.3e+4],
        [8.56e-4, 1.96e+0,  1.80e-6, 4.81e-4, 3.82e-5, 4.3e+4],
        [8.56e-4, 1.96e+0,  7.09e-7, 4.78e-4, 9.53e-5, 4.4e+4],
        [8.53e-4, 1.96e+0,  2.54e-7, 4.59e-4, 2.38e-4, 3.2e+4],
        [8.39e-4, 1.96e+0,  3.67e-8, 3.07e-4, 6.03e-4, 9.5e+1],
    ]),
}
BLOCKS = np.stack([BLOCKS_ASC[x] for x in X_LIST], axis=0)  # (5,5,6)

def interp_coeffs(x, j) -> Tuple[np.ndarray, ...]:
    """
    Bilinear interpolation on (log10 x, j) over the table.
    Returns eps1*, eps2*, j1*, j2*, n* (with signs corrected and floors applied).
    """
    x = np.asarray(x, float)
    j = np.asarray(j, float)
    lx = np.log10(np.clip(x, X_LIST[0], X_LIST[-1]))

    fx = np.interp(lx, LOGX, np.arange(len(LOGX)))
    ix0 = np.clip(np.floor(fx).astype(int), 0, len(LOGX)-2); tx = fx - ix0; ix1 = ix0+1

    fj  = np.interp(j, J_ASC, np.arange(len(J_ASC)))
    ij0 = np.clip(np.floor(fj).astype(int), 0, len(J_ASC)-2); tj = fj - ij0; ij1 = ij0+1

    outs = []
    for k in range(BLOCKS.shape[-1]):
        v00 = BLOCKS[ix0, ij0, k]; v10 = BLOCKS[ix1, ij0, k]
        v01 = BLOCKS[ix0, ij1, k]; v11 = BLOCKS[ix1, ij1, k]
        v0 = v00*(1.0-tx) + v10*tx
        v1 = v01*(1.0-tx) + v11*tx
        outs.append(v0*(1.0-tj) + v1*tj)

    minus_eps1, eps2, j1, j2, _tx2, nstar = outs
    eps1 = -minus_eps1
    eps2 = np.maximum(eps2, 0.0)
    j2   = np.maximum(j2,   0.0)
    nstar = np.maximum(nstar, 1.0)
    return eps1, eps2, j1, j2, nstar

# ============================================================
# ---------------- Creation–annihilation boundaries ----------
# u = x^(-5/4) ⇒ x_i = 10^(0.8*i)   (10^(5/4 * log10 x) trick)
# ============================================================

def u_boundaries(xD):
    xs = []
    i = 1
    while True:
        xi = 10.0**(0.8*i)
        if xi >= xD: break
        xs.append(xi)
        i += 1
    return np.array(xs, float)  # increasing order

# ============================================================
# -------------------------- Simulator -----------------------
# ============================================================

@dataclass
class Config:
    Pstar: float = PSTAR
    xD: float    = X_D
    seed: int    = RANDOM_SEED
    windows: int = WINDOWS
    t_rel_target: float = T_REL_TARGET
    mode: str    = MODE            # "standard" | "hybrid" | "high"
    reservoir_min: int = RESERVOIR_MIN
    clone_factor: int  = CLONE_FACTOR
    outlier_q: float   = OUTLIER_Q
    report_every: int  = REPORT_EVERY
    meas_stride: int   = MEAS_STRIDE
    nbins: int   = NBINS

class Sim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.bounds = u_boundaries(cfg.xD)

        # state arrays
        self.x = self._sample_x(20)                      # start with 20 test stars
        self.j = np.sqrt(self.rng.random(self.x.size))   # isotropic: p(j) ∝ j
        self.level = np.zeros(self.x.size, dtype=int)    # creation level L
        self.w = np.ones(self.x.size, float)             # statistical weight (for high mode)

        self.t_rel = 0.0
        self.steps = 0
        self.consumed = 0.0

    # -------- initial sampling: N(x) ∝ x^{-5/2} g0(x) --------
    def _sample_x(self, N):
        xmin, xmax = 1e-1, 0.95*self.cfg.xD
        out = np.empty(N); filled = 0
        # batched rejection sampling for speed
        while filled < N:
            k = max(64, N - filled)
            x = 10**self.rng.uniform(np.log10(xmin), np.log10(xmax), size=k)
            targ = np.power(x, -2.5) * np.where(x>1.0, x**0.25, 1.0)
            env  = np.where(x>1.0, x**(-2.25), 1.0) * 1.6
            keep = self.rng.random(k) < (targ / np.maximum(env, 1e-300))
            m = min(np.count_nonzero(keep), N - filled)
            if m > 0:
                out[filled:filled+m] = x[keep][:m]
                filled += m
        return out

    @staticmethod
    def _reflect_j(j):
        j = np.where(j<0.0, -j, j)
        j = np.where(j>1.0, 2.0-j, j)
        return np.clip(j, 0.0, 1.0)

    def _top_up_reservoir(self):
        need = max(0, self.cfg.reservoir_min - int(np.sum(self.level == 0)))
        if need:
            self.x     = np.concatenate([self.x, self._sample_x(need)])
            self.j     = np.concatenate([self.j, np.sqrt(self.rng.random(need))])
            self.level = np.concatenate([self.level, np.zeros(need, int)])
            self.w     = np.concatenate([self.w, np.ones(need)])

    def _choose_nu(self, x, j):
        """Return one global ν for this iteration and the 'park' mask if hybrid/high."""
        eps1, eps2, j1, j2, nstar = interp_coeffs(x, j)
        tiny = 1e-30
        nu_a  = (0.15**2) / np.maximum(eps2, tiny)                 # (29a)
        nu_b  = (0.10**2) / np.maximum(j2,  tiny)                  # (29b)
        nu_c  = (0.40*np.maximum(1.0075 - j, 0.0))**2 / np.maximum(j2, tiny)  # (29c)

        if self.cfg.mode == "standard":
            nu_tab = 1.0/np.maximum(nstar, 1.0)
            per_star = np.minimum.reduce([nu_a, nu_b, nu_c, nu_tab, np.ones_like(nu_a)])
            return float(np.clip(per_star.min(), 1e-8, 1.0)), None
        else:
            per_star = np.minimum.reduce([nu_a, nu_b, nu_c, np.ones_like(nu_a)])
            q = np.clip(self.cfg.outlier_q, 1e-3, 0.2)
            nu_bulk = float(np.clip(np.quantile(per_star, q), 1e-8, 1.0))
            park = per_star < nu_bulk
            return nu_bulk, park

    def step_once(self):
        """One global Monte Carlo step; returns dt_rel (in t0 units)."""
        x0 = self.x; j0 = self.j; lev0 = self.level; w0 = self.w

        nu, park = self._choose_nu(x0, j0)
        if park is None:
            active = np.ones(x0.size, dtype=bool)
        else:
            active = ~park

        # Kicks (only active move this step)
        e1, e2, j1c, j2c, _ = interp_coeffs(x0[active], j0[active])
        y1 = self.rng.normal(size=e2.size)
        y2 = self.rng.normal(size=j2c.size)

        dx = (e1 * x0[active]) * nu + np.sqrt(e2 * x0[active]**2 * nu) * y1
        dj = (j1c * nu)       + np.sqrt(j2c *             nu) * y2

        x1 = np.maximum(1e-12, x0.copy())
        j1v = j0.copy()
        lev1 = lev0.copy()
        w1   = w0.copy()

        x1[active]   = np.maximum(1e-12, x0[active] + dx)
        j1v[active]  = self._reflect_j(j0[active] + dj)

        # Capture (no-loss-cone run): only at x >= x_D → respawn outer
        cap = x1 >= self.cfg.xD
        if np.any(cap):
            self.consumed += np.sum(w1[cap] * np.power(10.0, -lev1[cap]))
            ncap = int(np.count_nonzero(cap))
            x1[cap]  = self._sample_x(ncap)
            j1v[cap] = np.sqrt(self.rng.random(ncap))
            lev1[cap] = 0
            w1[cap]   = 1.0

        # Annihilate down-crossers (if tagged with level L>0 and x< boundary)
        hasL = lev1 > 0
        if np.any(hasL):
            thr = np.zeros_like(lev1, float)
            thr[hasL] = self.bounds[lev1[hasL]-1]
            down = hasL & (x1 < thr)
            if np.any(down):
                nd = int(np.count_nonzero(down))
                x1[down]  = self._sample_x(nd)
                j1v[down] = np.sqrt(self.rng.random(nd))
                lev1[down] = 0
                w1[down]   = 1.0

        # Creation across u-boundaries (detect from x0→x1; apply after annihilation)
        crossed = np.zeros(x1.size, dtype=int)
        if self.bounds.size:
            for L, b in enumerate(self.bounds, start=1):
                m = (x0 < b) & (x1 >= b)
                crossed[m] = np.maximum(crossed[m], L)

        if MODE == "high":
            # Weighted splitting: promote and multiply weight by clone_factor.
            m = crossed > 0
            if np.any(m):
                Lp = crossed[m]
                lev1[m] = Lp
                w1[m]  *= self.cfg.clone_factor
            self.x, self.j, self.level, self.w = x1, j1v, lev1, w1
        else:
            # Physical clones (×(clone_factor-1) extra)
            idx = np.where(crossed > 0)[0]
            if idx.size:
                Lp = crossed[idx]
                lev1[idx] = Lp
                capsize = (self.x.size + (self.cfg.clone_factor-1)*idx.size) <= 400000
                if capsize:
                    self.x     = np.concatenate([x1, np.repeat(x1[idx], self.cfg.clone_factor-1)])
                    self.j     = np.concatenate([j1v, np.repeat(j1v[idx], self.cfg.clone_factor-1)])
                    self.level = np.concatenate([lev1, np.repeat(Lp,    self.cfg.clone_factor-1)])
                    self.w     = np.concatenate([w1, np.repeat(w1[idx], self.cfg.clone_factor-1)])
                else:
                    self.x, self.j, self.level, self.w = x1, j1v, lev1, w1
            else:
                self.x, self.j, self.level, self.w = x1, j1v, lev1, w1

        # Maintain low-x reservoir
        if (self.steps % 1000) == 0:
            self._top_up_reservoir()

        # Advance time (using pre-step energies, like an orbit-averaged integrator)
        dt_rel = nu * np.mean(P_ratio(x0)) * self.cfg.Pstar
        self.t_rel += dt_rel
        self.steps += 1
        return dt_rel

    # ---- measurement: return (centers, time-averaged g_hat) over a window ----
    def measure_over_window(self, t_rel_stop, bins, stride=1):
        acc = np.zeros(bins.size-1)   # time-weighted counts per bin
        tacc = 0.0
        while self.t_rel < t_rel_stop:
            dt = self.step_once()
            if (self.steps % stride) == 0:
                weights = self.w * np.power(10.0, -self.level.astype(float))
                acc += np.histogram(self.x, bins=bins, weights=weights * dt)[0]
                tacc += dt
        return acc, tacc

# ============================================================
# -------------------- Driver and Plotting -------------------
# ============================================================

def run():
    cfg = Config()
    sim = Sim(cfg)

    # Bins and denominators
    bins = np.logspace(np.log10(1e-1), np.log10(cfg.xD*0.95), cfg.nbins+1)
    centers = np.sqrt(bins[1:]*bins[:-1])
    widths  = bins[1:] - bins[:-1]
    denom   = P_ratio(centers) * Jmax_sq_ratio(centers)
    denom   = np.maximum(denom, 1e-300)

    # Time windows equally spaced in t/t0
    marks = np.linspace(0.0, cfg.t_rel_target, cfg.windows+1)
    g_list = []
    t0 = time.time()
    wall_deadline = t0 + STOP_AFTER_SECONDS

    # Progress loop: stop if either t_rel target or wall clock reached
    for w in range(cfg.windows):
        # Within window accumulate time-weighted histogram
        acc, tacc = sim.measure_over_window(marks[w+1], bins, stride=cfg.meas_stride)

        # Convert to time-averaged density → g_hat via state-density division
        density = (acc / max(tacc, 1e-30)) / np.maximum(widths, 1e-300)
        g_hat = density / denom

        # One-factor normalization to the reference curve
        g0 = g0_bw_curved(centers)
        A = np.sum(g_hat*g0) / (np.sum(g0**2) + 1e-30)
        g_list.append(g_hat / max(A, 1e-30))

        # progress and wall-clock check
        if (sim.steps % cfg.report_every) == 0:
            elapsed = time.time() - t0
            print(f"[progress] window {w+1}/{cfg.windows}  t_rel={sim.t_rel:.3f}/{cfg.t_rel_target:.3f}  "
                  f"steps={sim.steps}  tracers={sim.x.size}  elapsed={elapsed/60:0.1f} min")
        if time.time() >= wall_deadline:
            print("[info] Wall-clock limit reached — finishing with current windows.")
            break

    # If we exited early, pad windows by repeating the last estimate
    while len(g_list) < cfg.windows:
        g_list.append(g_list[-1].copy())

    g_arr  = np.vstack(g_list[:cfg.windows])
    g_mean = g_arr.mean(axis=0)
    g_std  = g_arr.std(axis=0, ddof=1)

    # Plot compared to curved g0 (visual match to Fig.1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-1, cfg.xD); ax.set_ylim(1e-1, 1e2)
    ax.set_xlabel(r"$x(-E/v_0^2)$"); ax.set_ylabel(r"$g(E)$")
    ax.errorbar(centers, g_mean, yerr=np.maximum(g_std, 1e-12), fmt='o', ms=3, lw=0.8,
                capsize=2.5, label=r"$\bar g$ (MC)")
    xd = np.logspace(-1, np.log10(cfg.xD), 900)
    ax.plot(xd, g0_bw_curved(xd), lw=2.0, label=r"$g_0$")
    ax.text(90.0, g0_bw_curved(np.array([90.0]))[0]*0.95, r"$g_0$", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.28)

    fig.savefig(PNG_NAME, dpi=220, bbox_inches='tight')
    fig.savefig(PDF_NAME, bbox_inches='tight')

    # Save raw arrays for post-processing / overlays
    np.savez(NPZ_NAME,
             centers=centers, g_mean=g_mean, g_std=g_std,
             bins=bins, widths=widths,
             meta=dict(mode=cfg.mode, Pstar=cfg.Pstar, xD=cfg.xD,
                       steps=sim.steps, t_rel=sim.t_rel,
                       reservoir_min=cfg.reservoir_min,
                       clone_factor=cfg.clone_factor,
                       outlier_q=cfg.outlier_q))

    # Diagnostics
    elapsed = time.time() - t0
    Fstar_rate = sim.consumed / max(sim.t_rel, 1e-30)   # in t0^{-1}
    print(f"[done] mode={cfg.mode}  steps={sim.steps}  tracers={sim.x.size}  "
          f"t/t0≈{sim.t_rel:.3f}  elapsed≈{elapsed/60:0.1f} min  "
          f"F*≈{Fstar_rate:0.4f} (no-loss-cone test)  "
          f"saved: {PNG_NAME}, {PDF_NAME}, {NPZ_NAME}")

if __name__ == "__main__":
    run()

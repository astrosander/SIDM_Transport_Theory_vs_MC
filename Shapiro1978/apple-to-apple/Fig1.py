"""
Shapiro & Marchant (1978) — Fig. 1 (no-loss-cone) reproduction
P* = 0.005, x_D = 1e4, x_crit = 10

Implements:
  • Orbit-averaged 2-D Monte Carlo in (x=-E/v0^2, j=J/Jmax)
  • Kicks from the Table-1 nondimensional coefficients you provided
  • Step factor ν chosen by eqs. (29a–c) and ν = 1/n* (take the minimum)
  • Creation–annihilation cloning across u = x^(-5/4) boundaries (×10 per level)
  • No-loss-cone boundary: remove only when x >= x_D
  • Estimator: g(x) ∝ [counts / Δx] / [P(x) Jmax(x)^2], then 1-factor LSQ normalization
  • Five time windows for mean and error bars
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import multiprocessing as mp
from functools import partial

# -------------------------
# Bahcall–Wolf reference g0
# -------------------------
def g0_bw(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = x > 1.0
    out[m] = np.power(x[m], 0.25)
    return out

# -------------------------
# Kepler scalings (x = -E/v0^2)
# -------------------------
def P_ratio(x):
    """P(E)/P(E0) with E0=-v0^2 (x0=1): P ∝ x^{-3/2}."""
    x = np.asarray(x, float)
    return np.power(np.maximum(x, 1e-300), -1.5)

def Jmax_sq_ratio(x):
    """J_c^2(E)/J_c^2(E0): J_c^2 ∝ x^{-1}."""
    x = np.asarray(x, float)
    return np.power(np.maximum(x, 1e-300), -1.0)

# ------------------------------------------------------------
# Table of nondimensional orbital perturbations (your data)
# Columns per row (after j): [-eps1*, eps2*, j1*, j2*, t* x^2, n*]
#
# IMPORTANT: The printed table lists j in DESCENDING order.
# We store each 5-row block in ASCENDING j order so interpolation is correct.
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
    Bilinear interpolation over (log10 x, j) returning:
      eps1*, eps2*, j1*, j2*, t*x^2, n*
    Note: first column is -eps1* in the table; we flip the sign.
    """
    x = np.asarray(x, float)
    j = np.asarray(j, float)
    lx = np.log10(np.clip(x, X_LIST[0], X_LIST[-1]))

    fx = np.interp(lx, LOGX, np.arange(len(LOGX)))
    i0x = np.clip(np.floor(fx).astype(int), 0, len(LOGX)-2)
    tx = fx - i0x
    i1x = i0x + 1

    fj = np.interp(j, J_ASC, np.arange(len(J_ASC)))
    i0j = np.clip(np.floor(fj).astype(int), 0, len(J_ASC)-2)
    tj = fj - i0j
    i1j = i0j + 1

    outs = []
    for k in range(BLOCKS.shape[-1]):
        v00 = BLOCKS[i0x, i0j, k]
        v10 = BLOCKS[i1x, i0j, k]
        v01 = BLOCKS[i0x, i1j, k]
        v11 = BLOCKS[i1x, i1j, k]
        v0 = v00*(1.0-tx) + v10*tx
        v1 = v01*(1.0-tx) + v11*tx
        vk = v0*(1.0-tj) + v1*tj
        outs.append(vk)

    minus_eps1, eps2, j1, j2, tstar_x2, nstar = outs
    eps1 = -minus_eps1
    j2 = np.maximum(j2, 0.0)     # variance coefficient must be ≥0
    nstar = np.maximum(nstar, 1.0)
    return eps1, eps2, j1, j2, tstar_x2, nstar

# ------------------------------------------------------------
# Creation–annihilation (u = x^(-5/4) ⇒ x_i = 10^(0.8 i))
# ------------------------------------------------------------
class CloneMgr:
    def __init__(self, xD, max_levels=None):
        xs = []
        i = 1
        while True:
            xi = 10.0**(0.8*i)
            if xi >= xD: break
            xs.append(xi); i += 1
        self.bounds = np.array(xs, float)
        self.levels = len(self.bounds) if max_levels is None else min(max_levels, len(self.bounds))

# ------------------------------------------------------------
# Monte Carlo integrator
# ------------------------------------------------------------
@dataclass
class Params:
    Pstar: float = 0.005
    xD: float = 1.0e4
    N0: int = 20
    seed: int = 7
    windows: int = 5
    t_total_relax: float = 6.0
    max_tracers: int = 400000
    max_levels: int = None

class SM78_NoLossCone:
    def __init__(self, prm: Params):
        self.prm = prm
        self.rng = np.random.default_rng(prm.seed)
        self.clone = CloneMgr(prm.xD, prm.max_levels)

        # state
        self.x = self.sample_x(prm.N0)
        self.j = np.sqrt(self.rng.random(prm.N0))   # isotropy: p(j) ∝ j
        self.level = np.zeros(prm.N0, dtype=int)    # creation–annihilation tag
        self.t_rel = 0.0
        self.consumed = 0.0

    # ----- initialization: N(x) ∝ P Jmax^2 g0 ∝ x^{-5/2} g0 -----
    def sample_x(self, N):
        xmin, xmax = 1e-1, 0.9*self.prm.xD
        out = np.empty(N)
        i = 0
        while i < N:
            x = 10**self.rng.uniform(np.log10(xmin), np.log10(xmax))
            w = x**(-2.5) * (x**0.25 if x>1 else 1.0)
            env = (x**(-2.25) if x>1 else 1.0)  # easy envelope
            if self.rng.random() < w/(1.5*env):
                out[i] = x; i += 1
        return out

    def respawn_outer(self, idx):
        n = idx.size
        if n == 0: return
        self.x[idx] = self.sample_x(n)
        self.j[idx] = np.sqrt(self.rng.random(n))
        self.level[idx] = 0

    def reflect_j(self, j):
        j = np.where(j<0.0, -j, j)
        j = np.where(j>1.0, 2.0-j, j)
        return np.clip(j, 0.0, 1.0)

    # ----- one orbit-averaged step -----
    def step_once(self):
        x, j, lev = self.x, self.j, self.level
        eps1, eps2, j1, j2, _, nstar = interp_coeffs(x, j)

        # ν from step-adjustment constraints (29a–c) + tabulated 1/n*
        # (a) sqrt(ν ε2*) < 0.15    → ν < (0.15)^2 / ε2*
        # (b) sqrt(ν j2*) < 0.10    → ν < (0.10)^2 / j2*
        # (c) sqrt(ν j2*) < 0.40(1.0075 - j)
        tiny = 1e-30
        nu_a = (0.15**2) / np.maximum(eps2, tiny)
        nu_b = (0.10**2) / np.maximum(j2,  tiny)
        nu_c = np.square(0.40*np.maximum(1.0075 - j, 0.0)) / np.maximum(j2, tiny)
        nu_tab = 1.0/np.maximum(nstar, 1.0)

        nu = np.minimum.reduce([nu_a, nu_b, nu_c, nu_tab, np.ones_like(nu_tab)])

        # orbit-averaged kicks (uncorrelated here; ℓ* not in table)
        y1 = self.rng.normal(size=x.size)
        y2 = self.rng.normal(size=x.size)
        dx = (eps1 * x) * nu + np.sqrt(np.maximum(eps2, 0.0) * x**2 * nu) * y1
        dj = (j1 * nu)       + np.sqrt(np.maximum(j2,   0.0) *      nu) * y2

        x_old = x.copy()
        x_new = np.maximum(1e-12, x + dx)
        j_new = self.reflect_j(j + dj)

        # consume only at x >= xD (loss cone suppressed)
        cap = x_new >= self.prm.xD
        if np.any(cap):
            self.consumed += np.sum(10.0**(-lev[cap]))
            self.respawn_outer(np.where(cap)[0])
            x_new[cap] = self.x[cap]; j_new[cap] = self.j[cap]; lev[cap] = self.level[cap]

        # annihilate clones that down-cross their tagged boundary
        hasL = lev > 0
        if np.any(hasL):
            thr = np.zeros_like(lev, float)
            thr[hasL] = self.clone.bounds[lev[hasL]-1]
            down = hasL & (x_new < thr)
            self.respawn_outer(np.where(down)[0])
            x_new[down] = self.x[down]; j_new[down] = self.j[down]; lev[down] = self.level[down]

        # creation across u-boundaries (at most one per step; with ν above, multi-crossings are rare)
        crossed = np.zeros(x.size, dtype=int)
        for L, b in enumerate(self.clone.bounds, start=1):
            mask = (x_old < b) & (x_new >= b)
            crossed[mask] = np.maximum(crossed[mask], L)

        idx = np.where(crossed > 0)[0]
        if idx.size > 0 and (self.x.size + 9*idx.size) <= self.prm.max_tracers:
            Lp = crossed[idx]
            lev[idx] = Lp  # parent promoted
            # add nine clones at same state
            self.x = np.concatenate([x_new, np.repeat(x_new[idx], 9)])
            self.j = np.concatenate([j_new, np.repeat(j_new[idx], 9)])
            self.level = np.concatenate([lev, np.repeat(Lp, 9)])
        else:
            self.x = x_new; self.j = j_new; self.level = lev

        # advance time (relaxation units): dt/t0 = ν * [P/P0] * P*
        # Recalculate nu for the final particle array after cloning
        eps1_final, eps2_final, j1_final, j2_final, _, nstar_final = interp_coeffs(self.x, self.j)
        nu_a_final = (0.15**2) / np.maximum(eps2_final, tiny)
        nu_b_final = (0.10**2) / np.maximum(j2_final,  tiny)
        nu_c_final = np.square(0.40*np.maximum(1.0075 - self.j, 0.0)) / np.maximum(j2_final, tiny)
        nu_tab_final = 1.0/np.maximum(nstar_final, 1.0)
        nu_final = np.minimum.reduce([nu_a_final, nu_b_final, nu_c_final, nu_tab_final, np.ones_like(nu_tab_final)])
        
        self.t_rel += np.mean(nu_final * P_ratio(self.x)) * self.prm.Pstar

    # ----- observable: g(x) -----
    def measure_g(self, nbins=40, xmin=1e-1, xmax=None):
        if xmax is None: xmax = self.prm.xD
        bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
        centers = np.sqrt(bins[1:]*bins[:-1])
        widths = bins[1:] - bins[:-1]          # **critical** for log bins
        w = 10.0**(-self.level.astype(float))  # creation–annihilation renorm

        counts, _ = np.histogram(self.x, bins=bins, weights=w)
        density = counts / np.maximum(widths, 1e-300)     # convert counts to N(x)

        # g(x) ∝ N(x) / [P Jmax^2] up to a global factor
        denom = P_ratio(centers) * Jmax_sq_ratio(centers)  # ∝ x^{-5/2}
        g_hat = density / np.maximum(denom, 1e-300)

        # one overall normalization by LSQ against the BW curve
        g0 = g0_bw(centers)
        A = np.sum(g_hat*g0) / np.sum(g0**2 + 1e-30)
        g_hat /= max(A, 1e-30)
        return centers, g_hat

# -------------------------
# Single simulation function for multiprocessing
# -------------------------
def run_single_simulation(seed=7, verbose=False):
    """Run a single simulation and return the results."""
    import time
    prm = Params(seed=seed)
    sim = SM78_NoLossCone(prm)

    # take five windows uniform in t/t0
    T_goal = prm.t_total_relax
    marks = np.linspace(0.0, T_goal, prm.windows+1)[1:]
    g_list = []; m = 0; steps = 0
    
    start_time = time.time()
    last_progress_time = start_time
    last_progress_steps = 0

    while sim.t_rel < T_goal and steps < 2_000_000:
        sim.step_once(); steps += 1
        
        # Progress reporting every 1000 steps or 10 seconds (only if verbose)
        if verbose:
            current_time = time.time()
            if steps % 1000 == 0 or (current_time - last_progress_time) > 10:
                elapsed_time = current_time - start_time
                progress = sim.t_rel / T_goal
                
                # Time estimator based on current progress rate
                if progress > 0.01:  # Only estimate after 1% progress
                    steps_per_second = (steps - last_progress_steps) / (current_time - last_progress_time)
                    remaining_steps = (T_goal - sim.t_rel) * steps / sim.t_rel if sim.t_rel > 0 else 0
                    eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                    eta_str = f"ETA: {eta_seconds/60:.1f}min" if eta_seconds > 60 else f"ETA: {eta_seconds:.1f}s"
                else:
                    eta_str = "ETA: calculating..."
                
                print(f"[Sim {seed}] t_rel: {sim.t_rel:.3f}/{T_goal:.3f} ({progress*100:.1f}%), "
                      f"steps: {steps}, time: {elapsed_time:.1f}s, {eta_str}")
                
                last_progress_time = current_time
                last_progress_steps = steps
        
        while m < prm.windows and sim.t_rel >= marks[m]:
            xc, gc = sim.measure_g(nbins=40, xmin=1e-1, xmax=prm.xD*0.95)
            g_list.append(gc); m += 1

    while len(g_list) < prm.windows:
        xc, gc = sim.measure_g(nbins=40, xmin=1e-1, xmax=prm.xD*0.95)
        g_list.append(gc)

    g_arr = np.vstack(g_list)
    g_mean = g_arr.mean(axis=0)
    g_std = g_arr.std(axis=0, ddof=1)
    
    return {
        'seed': seed,
        'g_mean': g_mean,
        'g_std': g_std,
        'x_centers': xc,
        't_rel_final': sim.t_rel,
        'steps': steps,
        'tracers_final': sim.x.size,
        'consumed': sim.consumed
    }

# -------------------------
# Multiprocessing wrapper
# -------------------------
def run_multiprocess(n_simulations=4, n_processes=None, base_seed=7):
    """Run multiple simulations in parallel and combine results."""
    if n_processes is None:
        n_processes = min(n_simulations, mp.cpu_count())
    
    print(f"Running {n_simulations} simulations using {n_processes} processes...")
    
    # Create seeds for each simulation
    seeds = [base_seed + i for i in range(n_simulations)]
    
    # Run simulations in parallel
    with mp.Pool(processes=n_processes) as pool:
        # Only the first simulation shows progress
        results = []
        for i, seed in enumerate(seeds):
            verbose = (i == 0)  # Only first simulation shows progress
            result = pool.apply_async(run_single_simulation, (seed, verbose))
            results.append(result)
        
        # Collect results
        simulation_results = [result.get() for result in results]
    
    # Combine results from all simulations
    g_means = np.array([r['g_mean'] for r in simulation_results])
    g_stds = np.array([r['g_std'] for r in simulation_results])
    x_centers = simulation_results[0]['x_centers']  # Same for all simulations
    
    # Calculate ensemble statistics
    g_ensemble_mean = g_means.mean(axis=0)
    g_ensemble_std = g_means.std(axis=0, ddof=1)  # Standard deviation across simulations
    g_ensemble_std_mean = g_stds.mean(axis=0)     # Average of individual standard deviations
    
    # Print summary
    print(f"\nSimulation Summary:")
    for i, r in enumerate(simulation_results):
        print(f"  Sim {r['seed']}: t_rel={r['t_rel_final']:.3f}, steps={r['steps']}, "
              f"tracers={r['tracers_final']}, consumed={r['consumed']:.2f}")
    
    return {
        'x_centers': x_centers,
        'g_ensemble_mean': g_ensemble_mean,
        'g_ensemble_std': g_ensemble_std,
        'g_ensemble_std_mean': g_ensemble_std_mean,
        'individual_results': simulation_results
    }

# -------------------------
# Driver + plotting
# -------------------------
def run(seed=7):
    import time
    prm = Params(seed=seed)
    sim = SM78_NoLossCone(prm)

    # take five windows uniform in t/t0
    T_goal = prm.t_total_relax
    marks = np.linspace(0.0, T_goal, prm.windows+1)[1:]
    g_list = []; m = 0; steps = 0
    
    start_time = time.time()
    last_progress_time = start_time
    last_progress_steps = 0

    while sim.t_rel < T_goal and steps < 2_000_000:
        sim.step_once(); steps += 1
        
        # Progress reporting every 1000 steps or 10 seconds
        current_time = time.time()
        if steps % 1000 == 0 or (current_time - last_progress_time) > 10:
            elapsed_time = current_time - start_time
            progress = sim.t_rel / T_goal
            
            # Time estimator based on current progress rate
            if progress > 0.01:  # Only estimate after 1% progress
                steps_per_second = (steps - last_progress_steps) / (current_time - last_progress_time)
                remaining_steps = (T_goal - sim.t_rel) * steps / sim.t_rel if sim.t_rel > 0 else 0
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                eta_str = f"ETA: {eta_seconds/60:.1f}min" if eta_seconds > 60 else f"ETA: {eta_seconds:.1f}s"
            else:
                eta_str = "ETA: calculating..."
            
            print(f"[progress] t_rel: {sim.t_rel:.3f}/{T_goal:.3f} ({progress*100:.1f}%), "
                  f"steps: {steps}, time: {elapsed_time:.1f}s, {eta_str}")
            
            last_progress_time = current_time
            last_progress_steps = steps
        while m < prm.windows and sim.t_rel >= marks[m]:
            xc, gc = sim.measure_g(nbins=40, xmin=1e-1, xmax=prm.xD*0.95)
            g_list.append(gc); m += 1

    while len(g_list) < prm.windows:
        xc, gc = sim.measure_g(nbins=40, xmin=1e-1, xmax=prm.xD*0.95)
        g_list.append(gc)

    g_arr = np.vstack(g_list)
    g_mean = g_arr.mean(axis=0)
    g_std  = g_arr.std(axis=0, ddof=1)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-1, prm.xD); ax.set_ylim(1e-1, 1e2)
    ax.set_xlabel(r"$x(-E/v_0^2)$"); ax.set_ylabel(r"$g(E)$")

    ax.errorbar(xc, g_mean, yerr=g_std, fmt='o', ms=3, lw=0.8, capsize=2.5, label=r"$\bar g$ (MC)")
    xd = np.logspace(-1, np.log10(prm.xD), 700)
    ax.plot(xd, g0_bw(xd), lw=2.0, label=r"$g_0$")
    ax.text(80.0, g0_bw([80.0])[0]*0.95, r"$g_0$", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.25)

    fig.savefig("fig1_shapiro1978_reproduction.png", dpi=300, bbox_inches='tight')
    fig.savefig("fig1_shapiro1978_reproduction.pdf", bbox_inches='tight')
    print(f"t/t0 ≈ {sim.t_rel:.3f} | tracers (after cloning): {sim.x.size} | steps: {steps}")
    print("Saved: fig1_shapiro1978_reproduction.[png|pdf]")

def plot_results(xc, g_mean, g_std, prm, filename_prefix="fig1_shapiro1978_reproduction"):
    """Plot the results with error bars."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-1, prm.xD); ax.set_ylim(1e-1, 1e2)
    ax.set_xlabel(r"$x(-E/v_0^2)$"); ax.set_ylabel(r"$g(E)$")

    ax.errorbar(xc, g_mean, yerr=g_std, fmt='o', ms=3, lw=0.8, capsize=2.5, label=r"$\bar g$ (MC)")
    xd = np.logspace(-1, np.log10(prm.xD), 700)
    ax.plot(xd, g0_bw(xd), lw=2.0, label=r"$g_0$")
    ax.text(80.0, g0_bw([80.0])[0]*0.95, r"$g_0$", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.25)

    fig.savefig(f"{filename_prefix}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{filename_prefix}.pdf", bbox_inches='tight')
    print(f"Saved: {filename_prefix}.[png|pdf]")

def run_multiprocess_with_plotting(n_simulations=4, n_processes=None, base_seed=7):
    """Run multiprocess simulations and create plots."""
    import time
    start_time = time.time()
    
    # Run simulations
    results = run_multiprocess(n_simulations, n_processes, base_seed)
    
    # Create plots
    prm = Params(seed=base_seed)
    
    # Plot 1: Ensemble mean with ensemble std
    plot_results(
        results['x_centers'], 
        results['g_ensemble_mean'], 
        results['g_ensemble_std'],
        prm,
        "fig1_shapiro1978_ensemble"
    )
    
    # Plot 2: Ensemble mean with individual std mean
    plot_results(
        results['x_centers'], 
        results['g_ensemble_mean'], 
        results['g_ensemble_std_mean'],
        prm,
        "fig1_shapiro1978_individual_std"
    )
    
    # Plot 3: All individual simulations
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-1, prm.xD); ax.set_ylim(1e-1, 1e2)
    ax.set_xlabel(r"$x(-E/v_0^2)$"); ax.set_ylabel(r"$g(E)$")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['individual_results'])))
    for i, r in enumerate(results['individual_results']):
        ax.plot(r['x_centers'], r['g_mean'], 'o-', ms=2, alpha=0.7, 
                color=colors[i], label=f"Sim {r['seed']}")
    
    xd = np.logspace(-1, np.log10(prm.xD), 700)
    ax.plot(xd, g0_bw(xd), 'k-', lw=2.0, label=r"$g_0$")
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.25)
    
    fig.savefig("fig1_shapiro1978_all_simulations.png", dpi=300, bbox_inches='tight')
    fig.savefig("fig1_shapiro1978_all_simulations.pdf", bbox_inches='tight')
    print("Saved: fig1_shapiro1978_all_simulations.[png|pdf]")
    
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"Average time per simulation: {total_time/n_simulations:.1f}s")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multiprocess":
        n_sim = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_multiprocess_with_plotting(n_sim, n_proc)
    else:
        run()

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

SEED               = 1
# VELOCITY_VALS      = np.linspace(0.0, 10_000_000.0, 100)

N_VELOCITY = 100
V_MIN = 10_000.0
V_MAX = 10000_000_000.0
VELOCITY_VALS = np.geomspace(V_MIN, V_MAX, N_VELOCITY)

S_BATH             = 150_000.0
M_PARTICLE         = 1.0
M_BATH             = 1.0
MU                 = (M_PARTICLE * M_BATH) / (M_PARTICLE + M_BATH)

USE_ABSOLUTE_RATES = False
N_F                = 1.0

SIGMA0             = 2.0e-28
W                   = 1.0e3

N_BATH_SAMPLES     = 4_000_000

N_WORKERS          = min(os.cpu_count() or 1, len(VELOCITY_VALS))

def sigma_tot_rutherford(V: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return sigma0 / (1.0 + (V*V)/(w*w))

def _safe_log_ratio(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r = (A - B) / (A + B)
    r = np.clip(r, 1e-300, 1.0)
    return np.log(r)

def C1C2C3(V: np.ndarray, w: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = np.asarray(V, dtype=float)
    A = w*w + 0.5*V*V
    B = 0.5*V*V

    C1 = np.empty_like(V)
    C2 = np.empty_like(V)
    C3 = np.empty_like(V)

    mask = B > 1e-12
    if np.any(mask):
        Am = A[mask]; Bm = B[mask]
        fac1 = (Am*Am - Bm*Bm) / 2.0
        logt = _safe_log_ratio(Am, Bm)

        t1 = (Am/(Bm*Bm)) * (1.0/(Am - Bm) - 1.0/(Am + Bm))
        t2 = (1.0/(Bm*Bm)) * logt
        C1[mask] = fac1 * (t1 + t2)

        t1 = ((Am*Am)/(Bm**3)) * (1.0/(Am - Bm) - 1.0/(Am + Bm))
        t2 = (2.0*Am/(Bm**3)) * logt
        t3 = -2.0/(Bm*Bm)
        C2[mask] = fac1 * (t1 + t2 + t3)

        C3[mask] = (Am/(Bm**3))*(3.0*Am*Am - 2.0*Bm*Bm) \
                   + (3.0*Am*Am*(Am*Am - Bm*Bm)/(2.0*Bm**4)) * logt

    if np.any(~mask):
        C1[~mask] = 0.0
        C2[~mask] = 1.0/3.0
        C3[~mask] = 0.0

    C1 = np.clip(C1, -1.0, 1.0)
    C2 = np.clip(C2,  0.0, 1.0)
    C3 = np.clip(C3, -1.0, 1.0)
    return C1, C2, C3

def theory_rates_for_v_parallel(v_mag: float, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    v_f = rng.normal(0.0, S_BATH, size=(N_BATH_SAMPLES, 3))

    v1 = np.array([v_mag, 0.0, 0.0])
    g  = v1 - v_f
    V  = np.linalg.norm(g, axis=1)
    ok = V > 0.0
    if not np.any(ok):
        return dict(v=v_mag, drift=0, Dpar=0, Dperp=0, Mmix=0, M3=0)

    g    = g[ok]; V = V[ok]
    ghat = g / V[:, None]
    mu_e = ghat[:, 0]

    C1, C2, C3 = C1C2C3(V, W)
    a_perp   = 0.5*(1.0 - C2)
    par_coef = C2 - 2.0*C1 + 1.0
    T1 = C3 - 3.0*C2 + 3.0*C1 - 1.0
    T2 = -1.0 + C1 + C2 - C3
    y2 = 1.0 - 2.0*C1 + C2

    sig_tot = sigma_tot_rutherford(V, SIGMA0, W)
    weight  = sig_tot * V * (N_F if USE_ABSOLUTE_RATES else 1.0)

    F  = (MU / M_PARTICLE) * V
    F2 = F*F
    F3 = F2*F

    drift_par = np.mean(weight * (F * (C1 - 1.0) * mu_e))
    D_par     = np.mean(weight * (F2 * ( a_perp*(1.0 - mu_e**2) + par_coef*(mu_e**2) )))
    total2    = np.mean(weight * (2.0 * F2 * (1.0 - C1)))
    D_perp    = total2 - D_par
    M3_par    = np.mean(weight * ( F3 * ( mu_e**3 * T1 + 1.5 * mu_e * (1.0 - mu_e**2) * T2 ) ))
    Mixed_first = np.mean(weight * ( F3 * mu_e * (-2.0 * y2) ))
    M_mix       = Mixed_first - M3_par

    return dict(
        v      = float(v_mag),
        drift  = abs(drift_par / (S_BATH**1)),
        Dpar   = abs(D_par     / (S_BATH**2)),
        Dperp  = abs(D_perp    / (S_BATH**2)),
        Mmix   = abs(M_mix     / (S_BATH**3)),
        M3     = abs(M3_par    / (S_BATH**3)),
    )

def add_line(ax, x, y, label):
    y = np.clip(np.asarray(y, float), 1e-300, None)
    ax.loglog(x, y, '-', lw=2.0, label=label)

def main():
    print(f"Computing {len(VELOCITY_VALS)} analytical bins with {N_WORKERS} workers ...")
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = []
        rng_master = np.random.default_rng(SEED)
        for v in VELOCITY_VALS:
            seed_i = int(rng_master.integers(1, 2**31-1))
            futs.append(ex.submit(theory_rates_for_v_parallel, float(v), seed_i))
        for fut in as_completed(futs):
            out = fut.result()
            results[out['v']] = out

    xs = sorted(results.keys())
    drift = [results[v]['drift'] for v in xs]
    Dpar  = [results[v]['Dpar']  for v in xs]
    Dperp = [results[v]['Dperp'] for v in xs]
    M3    = [results[v]['M3']    for v in xs]
    Mmix  = [results[v]['Mmix']  for v in xs]

    fig, ax = plt.subplots(figsize=(9,6))
    add_line(ax, xs, drift, r"$\langle \Delta v_\parallel\rangle$")
    add_line(ax, xs, Dpar,  r"$\langle \Delta v_\parallel^2\rangle$")
    add_line(ax, xs, Dperp, r"$\langle \Delta v_\perp^2\rangle$")
    add_line(ax, xs, M3,    r"$\langle (\Delta v_\parallel)^3\rangle$")
    add_line(ax, xs, Mmix,  r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    ax.set_xlabel(r'$v_{\rm particle}$')
    ax.set_ylabel(r'analytical rates (normalized by $s_{\rm bath}^n$)')
    ax.set_title('Analytical transport (Rutherford/Yukawa): bath-averaged, normalized (parallel)')
    ax.grid(True, which='both', ls=':')
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_analytical_only_parallel.png", dpi=160)
    fig.savefig("fig3_analytical_only_parallel.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

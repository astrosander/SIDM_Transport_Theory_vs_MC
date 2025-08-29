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

N_HERMITE          = 100

N_WORKERS          = min(os.cpu_count() or 1, len(VELOCITY_VALS))

def sigma_tot_rutherford(V: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    V = np.asarray(V, float)
    return sigma0 / (1.0 + (V*V)/(w*w))

def C1C2C3_closed_form_stable(V: np.ndarray, w: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = np.asarray(V, float)
    A = w*w + 0.5*V*V
    B = 0.5*V*V

    C1 = np.empty_like(V)
    C2 = np.empty_like(V)
    C3 = np.empty_like(V)

    small = B < 1e-12
    big   = ~small

    if np.any(big):
        Ab = A[big]; Bb = B[big]
        R = np.log(np.clip((Ab - Bb) / (Ab + Bb), 1e-300, 1.0))

        C1[big] = (Ab / Bb) + ((Ab*Ab - Bb*Bb) / (2.0*Bb*Bb)) * R
        C2[big] = 1.0       + (Ab * (Ab*Ab - Bb*Bb) / (Bb**3)) * R
        C3[big] = (Ab / (Bb**3)) * (3.0*Ab*Ab - 2.0*Bb*Bb) \
                  + (3.0*Ab*Ab * (Ab*Ab - Bb*Bb) / (2.0*Bb**4)) * R

    if np.any(small):
        C1[small] = 0.0
        C2[small] = 1.0/3.0
        C3[small] = 0.0

    C1 = np.clip(C1, -1.0, 1.0)
    C2 = np.clip(C2,  0.0, 1.0)
    C3 = np.clip(C3, -1.0, 1.0)
    return C1, C2, C3

def bath_quadrature_nodes_weights(n_hermite: int, s_bath: float):
    t, w = np.polynomial.hermite.hermgauss(n_hermite)
    x = math.sqrt(2.0) * s_bath * t
    w1 = w / math.sqrt(math.pi)

    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    W3 = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :])
    return X.reshape(-1), Y.reshape(-1), Z.reshape(-1), W3.reshape(-1)

VX_BATH, VY_BATH, VZ_BATH, W_BATH = bath_quadrature_nodes_weights(N_HERMITE, S_BATH)

def theory_rates_for_v_quadrature(v_mag: float) -> Dict[str, float]:
    vb = np.stack([VX_BATH, VY_BATH, VZ_BATH], axis=1)
    w_b = W_BATH

    v1 = np.array([v_mag, 0.0, 0.0])
    g  = v1[None, :] - vb
    V  = np.linalg.norm(g, axis=1)
    ok = V > 0.0
    if not np.any(ok):
        return dict(v=v_mag, drift=0.0, Dpar=0.0, Dperp=0.0, Mmix=0.0, M3=0.0)

    g   = g[ok];  V = V[ok]; w_b = w_b[ok]
    ghat = g / V[:, None]
    mu_e = ghat[:, 0]

    C1, C2, C3 = C1C2C3_closed_form_stable(V, W)

    a_perp   = np.maximum(0.0, 0.5*(1.0 - C2))
    p_para   = np.maximum(0.0, 1.0 - 2.0*C1 + C2)

    T1 = C3 - 3.0*C2 + 3.0*C1 - 1.0
    y2 = 1.0 - 2.0*C1 + C2
    T2 = -2.0*y2 - T1

    sig_tot = sigma_tot_rutherford(V, SIGMA0, W)
    weight  = sig_tot * V * (N_F if USE_ABSOLUTE_RATES else 1.0)

    F  = (MU / M_PARTICLE) * V
    F2 = F*F
    F3 = F2*F

    drift_par = np.sum(w_b * weight * (F * (C1 - 1.0) * mu_e))

    D_par     = np.sum(w_b * weight * (F2 * ( a_perp*(1.0 - mu_e**2) + p_para*(mu_e**2) )))

    total2    = np.sum(w_b * weight * (2.0 * F2 * (1.0 - C1)))
    D_perp    = total2 - D_par

    M3_par    = np.sum(w_b * weight * ( F3 * ( mu_e**3 * T1 + 1.5 * mu_e * (1.0 - mu_e**2) * T2 ) ))

    Mixed_first = np.sum(w_b * weight * ( -2.0 * F3 * mu_e * y2 ))
    M_mix       = Mixed_first - M3_par

    return dict(
        v      = float(v_mag),
        drift  = drift_par  / (S_BATH**1),
        Dpar   = D_par      / (S_BATH**2),
        Dperp  = D_perp     / (S_BATH**2),
        M3     = M3_par     / (S_BATH**3),
        Mmix   = M_mix      / (S_BATH**3),
    )

def add_line_abslog(ax, x, y, label):
    ax.loglog(x, np.clip(np.abs(y), 1e-300, None), '-', lw=2.0, label=label)

def main():
    print(f"Computing {len(VELOCITY_VALS)} bins with {N_WORKERS} workers using 3D Gauss–Hermite ...")
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(theory_rates_for_v_quadrature, float(v)) for v in VELOCITY_VALS]
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
    add_line_abslog(ax, xs, drift, r"$\langle \Delta v_\parallel\rangle$")
    add_line_abslog(ax, xs, Dpar,  r"$\langle \Delta v_\parallel^2\rangle$")
    add_line_abslog(ax, xs, Dperp, r"$\langle \Delta v_\perp^2\rangle$")
    add_line_abslog(ax, xs, M3,    r"$\langle (\Delta v_\parallel)^3\rangle$")
    add_line_abslog(ax, xs, Mmix,  r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    ax.set_xlabel(r'$v_{\rm particle}$')
    ax.set_ylabel(r'analytical rates (normalized by $s_{\rm bath}^n$); $|\,\cdot\,|$ on log-y')
    ax.set_title('Rutherford/Yukawa transport — closed-form Cn + 3D Gauss–Hermite bath (no dips)')
    ax.grid(True, which='both', ls=':')
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_analytical_no_dips.png", dpi=160)
    fig.savefig("fig3_analytical_no_dips.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

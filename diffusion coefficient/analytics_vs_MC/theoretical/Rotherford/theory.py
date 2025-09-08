#!/usr/bin/env python3
import os, math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------ controls ------------------------
SEED               = 1
N_VELOCITY         = 100
V_MIN, V_MAX       = 1.0e4, 1.0e8
VELOCITY_VALS      = np.geomspace(V_MIN, V_MAX, N_VELOCITY)

S_BATH             = 1.5e5
M_PARTICLE         = 1.0
M_BATH             = 1.0
MU                 = (M_PARTICLE*M_BATH)/(M_PARTICLE+M_BATH)

USE_ABSOLUTE_RATES = False
N_F                = 1.0

SIGMA0             = 2.0e-28
W                  = 1.0e3

N_HERMITE          = 100
N_WORKERS          = min(os.cpu_count() or 1, len(VELOCITY_VALS))

# use extended precision where available
ftype = np.longdouble

# ------------------------ cross section ------------------------
def sigma_tot_rutherford(V: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    V = np.asarray(V, dtype=ftype)
    return (ftype(sigma0) /
            (1.0 + (V*V)/(ftype(w)*ftype(w))))

# ------------------------ stable angular integrals ------------------------
# We integrate over x = cosθ the kernel (A - B x)^{-2}, with
# A = w^2 + V^2/2,  B = V^2/2,  and return exact closed forms.
# Use log1p for ln((A-B)/(A+B)) = ln(1 - 2B/(A+B)).

def _A_B(V, w):
    V = np.asarray(V, dtype=ftype)
    A = ftype(w)*ftype(w) + 0.5*V*V
    B = 0.5*V*V
    return A, B

def I0_closed(V, w):
    A, B = _A_B(V, w)
    return 2.0 / (A*A - B*B)

def I1_closed(V, w):
    A, B = _A_B(V, w)
    # I1 = A/B^2*(1/(A-B)-1/(A+B)) + (1/B^2)*ln((A-B)/(A+B))
    logR = np.log1p(-2.0*B/(A+B))
    return (A/(B*B))*(1.0/(A-B) - 1.0/(A+B)) + (logR/(B*B))

def I2_closed(V, w):
    A, B = _A_B(V, w)
    logR = np.log1p(-2.0*B/(A+B))
    return (A*A/(B**3))*(1.0/(A-B) - 1.0/(A+B)) + (2.0*A/(B**3))*logR + 2.0/(B*B)

def I3_closed(V, w):
    A, B = _A_B(V, w)
    logR = np.log1p(-2.0*B/(A+B))
    # Unnormalized I3 (from exact integration):
    # I3 = [ 2A^3 B/(A^2 - B^2) + 3A^2 ln((A-B)/(A+B)) + 4AB ] / B^4
    return ( (2.0*A*A*A*B)/(A*A - B*B) + 3.0*A*A*logR + 4.0*A*B ) / (B**4)

def combos_T1_K1(V, w):
    """Return T1 = C3-3C2+3C1-1 and K1 = C1+C2-C3-1
       computed as single, cancellation-free ratios."""
    I0 = I0_closed(V, w)
    I1 = I1_closed(V, w)
    I2 = I2_closed(V, w)
    I3 = I3_closed(V, w)
    # combined numerators before division
    NT1 = I3 - 3.0*I2 + 3.0*I1 - I0
    NK1 = I1 + I2 - I3 - I0
    return NT1 / I0, NK1 / I0

def C1C2C3_closed(V, w):
    """Stable C1, C2, C3 from closed I_n/I_0; with small-V limits."""
    V = np.asarray(V, dtype=ftype)
    A, B = _A_B(V, w)
    I0 = I0_closed(V, w)

    small = (B < ftype(1e-20))
    big   = ~small

    C1 = np.empty_like(V, dtype=ftype)
    C2 = np.empty_like(V, dtype=ftype)
    C3 = np.empty_like(V, dtype=ftype)

    if np.any(big):
        Ib = I0[big]
        C1[big] = I1_closed(V[big], w) / Ib
        C2[big] = I2_closed(V[big], w) / Ib
        # C3 via I3/I0 (ok), but we’ll also use combos for differences
        C3[big] = I3_closed(V[big], w) / Ib

    if np.any(small):
        # isotropic limit for dσ/dcosθ ≈ const:
        C1[small] = 0.0
        C2[small] = 1.0/3.0
        C3[small] = 0.0

    return C1, C2, C3

# ------------------------ bath quadrature ------------------------
def bath_quadrature_nodes_weights(n_hermite: int, s_bath: float):
    t, w = np.polynomial.hermite.hermgauss(n_hermite)
    x = math.sqrt(2.0)*s_bath*t
    w1 = w/math.sqrt(math.pi)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    W3 = (w1[:,None,None]*w1[None,:,None]*w1[None,None,:])
    return X.reshape(-1).astype(ftype), Y.reshape(-1).astype(ftype), \
           Z.reshape(-1).astype(ftype), W3.reshape(-1).astype(ftype)

VX_BATH, VY_BATH, VZ_BATH, W_BATH = bath_quadrature_nodes_weights(N_HERMITE, S_BATH)

# ------------------------ main per-v evaluator ------------------------
def theory_rates_for_v(v_mag: float) -> Dict[str, float]:
    vb = np.stack([VX_BATH, VY_BATH, VZ_BATH], axis=1)
    wb = W_BATH.copy()

    v1 = np.array([v_mag, 0.0, 0.0], dtype=ftype)
    g  = v1[None,:] - vb
    V  = np.linalg.norm(g, axis=1).astype(ftype)
    ok = V > 0.0
    if not np.any(ok):
        z = 0.0
        return dict(v=v_mag, drift=z, Dpar=z, Dperp=z, M3=z, Mmix=z)

    g   = g[ok];  V = V[ok]; wb = wb[ok]
    ghat = g / V[:,None]
    mu_e = ghat[:,0]

    # robust angular pieces
    C1, C2, C3 = C1C2C3_closed(V, W)
    T1, K1     = combos_T1_K1(V, W)   # cancellation-free differences

    a_perp = 0.5*(1.0 - C2)            # ≥0
    p_para = (1.0 - 2.0*C1 + C2)       # ≥0

    sig_tot = sigma_tot_rutherford(V, SIGMA0, W)
    weight  = sig_tot * V * (ftype(N_F) if USE_ABSOLUTE_RATES else 1.0)

    F  = (MU/M_PARTICLE) * V
    F2 = F*F
    F3 = F2*F

    # 1st/2nd order (already well behaved)
    drift_par = np.sum(wb*weight*(F*(C1 - 1.0)*mu_e))
    D_par     = np.sum(wb*weight*(F2*(a_perp*(1.0 - mu_e*mu_e) + p_para*(mu_e*mu_e))))
    total2    = np.sum(wb*weight*(2.0*F2*(1.0 - C1)))
    D_perp    = total2 - D_par

    # 3rd-order, built from robust combos:
    # <(Δv_||)^3>_θ = (F^3/8) * 8 * [μ^3 T1 + (3/2) μ (1-μ^2) * (−2 y2 − T1)]
    # but y2 = 1 − 2C1 + C2 = p_para. Using K1 = C1+C2−C3−1 = −(T1 + 2 y2)
    # ⇒ the bracket becomes [ μ^3 T1 + (3/2) μ (1-μ^2) * (K1) ]  (no tiny subtractions)
    M3_par = np.sum(wb*weight*(F3*(mu_e**3 * T1 + 1.5*mu_e*(1.0 - mu_e*mu_e)*K1)))

    # <Δv_|| Δv_⊥^2>_θ = (F^3/2) * (C1+C2−C3−1) * [μ − μ^3] = (F^3/2)*K1*μ(1−μ^2)
    M_mix  = np.sum(wb*weight*(0.5*F3*K1*mu_e*(1.0 - mu_e*mu_e)))

    inv_s = ftype(1.0)/S_BATH
    return dict(
        v      = float(v_mag),
        drift  = float(drift_par * inv_s),
        Dpar   = float(D_par     * inv_s*inv_s),
        Dperp  = float(D_perp    * inv_s*inv_s),
        M3     = float(M3_par    * inv_s*inv_s*inv_s),
        Mmix   = float(M_mix     * inv_s*inv_s*inv_s),
    )

# ------------------------ plotting helpers ------------------------
def add_line_abslog(ax, x, y, label):
    eps = 1e-320
    ax.loglog(x, np.maximum(np.abs(y), eps), '-', lw=2.0, label=label)

# ------------------------ run ------------------------
def main():
    np.random.seed(SEED)

    print(f"Computing {len(VELOCITY_VALS)} bins with {N_WORKERS} workers …")
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(theory_rates_for_v, float(v)) for v in VELOCITY_VALS]
        for fut in as_completed(futs):
            out = fut.result()
            results[out['v']] = out

    xs   = np.array(sorted(results.keys()))
    drift = np.array([results[v]['drift'] for v in xs])
    Dpar  = np.array([results[v]['Dpar']  for v in xs])
    Dperp = np.array([results[v]['Dperp'] for v in xs])
    M3    = np.array([results[v]['M3']    for v in xs])
    Mmix  = np.array([results[v]['Mmix']  for v in xs])

    fig, ax = plt.subplots(figsize=(9.2,6.0))
    add_line_abslog(ax, xs, drift, r"$\langle \Delta v_\parallel\rangle$")
    add_line_abslog(ax, xs, Dpar,  r"$\langle \Delta v_\parallel^2\rangle$")
    add_line_abslog(ax, xs, Dperp, r"$\langle \Delta v_\perp^2\rangle$")
    add_line_abslog(ax, xs, M3,    r"$\langle (\Delta v_\parallel)^3\rangle$")
    add_line_abslog(ax, xs, Mmix,  r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    ax.set_xlabel(r'$v_{\rm particle}$')
    ax.set_ylabel(r'analytical rates (normalized by $s_{\rm bath}^n$); $|\,\cdot\,|$ on log-y')
    ax.set_title('Rutherford/Yukawa transport — robust 3rd-order (no dips/cancellation)')
    ax.grid(True, which='both', ls=':')
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_analytical_no_dips.png", dpi=160)
    fig.savefig("fig3_analytical_no_dips.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

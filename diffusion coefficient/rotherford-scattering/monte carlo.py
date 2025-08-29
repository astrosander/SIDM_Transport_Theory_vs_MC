import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Tuple
import os

rng = np.random.default_rng(20250812)

m_t = 1.78266192e-26
m_f = 1.78266192e-26
mu = (m_t * m_f) / (m_t + m_f)

s_bath = 1.5e5
rho = 5.35e-22
n_f = rho / m_f

sigma0 = 2.0e-28
w = 1.0e3

N_theory_samples = 7000
N_accept_per_v = 50000
block_accepts = 250
rate_boost = 5.0e11
safety_factor = 1.4
v_max_factor = 4.0
N_v = 10

prefix = f"img/aniso_fix_rutherford"
output_format = "png"

def sigma_tot_rutherford_vec(v: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return sigma0 / (1.0 + (v*v)/(w*w))

def sigma_tot_rutherford_scalar(v: float, sigma0: float, w: float) -> float:
    return float(sigma0 / (1.0 + (v*v)/(w*w)))

GL_N = 128
GL_X, GL_W = np.polynomial.legendre.leggauss(GL_N)

def rutherford_C1C2(v: np.ndarray, w: float) -> Tuple[np.ndarray, np.ndarray]:
    A = w*w + 0.5*v*v
    B = 0.5*v*v
    ApB = A + B
    AmB = np.maximum(A - B, 1e-300)
    Cnorm = (A*A - B*B)/2.0

    I1 = (A/(B*B)) * (1.0/AmB - 1.0/ApB) + (1.0/(B*B)) * (np.log(AmB) - np.log(ApB))
    C1 = Cnorm * I1

    I2 = (A*A/(B**3)) * (1.0/AmB - 1.0/ApB) + (2.0*A/(B**3)) * (np.log(AmB) - np.log(ApB)) - (2.0/(B*B))
    C2 = Cnorm * I2

    return C1, C2

def dsigma_dcos_rutherford(v: np.ndarray, cos_theta: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    denom = w*w + (v*v)*(1.0 - cos_theta)/2.0
    return (sigma0 * w**4) / (2.0 * denom**2)

@dataclass
class MCEst:
    v: float
    drift_par: float; drift_par_err: float
    diff_par: float;  diff_par_err: float
    diff_perp: float; diff_perp_err: float
    acceptance: float; blocks: int; accepts: int

def build_lookup(Vmax: float, nV: int = 600):
    Vgrid = np.linspace(1e-6, Vmax, nV)
    C1, C2 = rutherford_C1C2(Vgrid, w)
    f  = dsigma_dcos_rutherford(Vgrid[:, None], GL_X[None, :], sigma0, w)
    sig = sigma_tot_rutherford_vec(Vgrid, sigma0, w)
    C3 = (GL_W[None, :] * (GL_X[None, :]**3) * f).sum(axis=1) / sig
    SIG = sig
    return Vgrid, C1, C2, C3, SIG

Vmax_lookup = v_max_factor * s_bath * 1.5
Vgrid_L, C1_L, C2_L, C3_L, SIG_L = build_lookup(Vmax_lookup, nV=600)
def _interpL(V, arr): return np.interp(V, Vgrid_L, arr, left=arr[0], right=arr[-1])

def theory_rates_at_v_fast(v_mag: float) -> Tuple[float, float, float]:
    v_vec = np.array([v_mag, 0.0, 0.0])
    vf = gaussian_bath_velocities(N_theory_samples, s_bath)
    g  = v_vec - vf
    V  = np.linalg.norm(g, axis=1)
    good = V > 0
    if not np.any(good):
        return 0.0, 0.0, 0.0
    V   = V[good]
    mu_e = g[good, 0] / V

    C1 = _interpL(V, C1_L); C2 = _interpL(V, C2_L); SIG = _interpL(V, SIG_L)
    a_perp = 0.5*(1.0 - C2)
    par_cf = (C2 - 2.0*C1 + 1.0)
    wt = n_f * SIG * V

    drift = np.mean(wt * (mu/m_t)    * (C1 - 1.0) * (V * mu_e))
    Dpar  = np.mean(wt * (mu/m_t)**2 * (V**2) * (a_perp*(1.0 - mu_e**2) + par_cf*(mu_e**2)))
    total = np.mean(wt * 2.0 * (mu/m_t)**2 * (V**2) * (1.0 - C1))
    Dperp = total - Dpar

    s = rate_boost
    return float(drift*s), float(Dpar*s), float(Dperp*s)

def _gamma_max_bound() -> float:
    return safety_factor * rate_boost * n_f * (sigma0 * w * 0.5)

def sample_cos_rutherford_vec(V: np.ndarray) -> np.ndarray:
    A = w*w + 0.5*V*V
    B = 0.5*V*V
    u = rng.random(V.shape[0])
    denom = (1.0/(A + B)) + (2.0*B / ((A + B)*(np.maximum(A - B, 1e-300)))) * u
    inv = 1.0/denom
    x = (A - inv)/B
    return np.clip(x, -1.0, 1.0)

def simulate_fixed_v_fast(v_mag: float, batch: int = 16384) -> MCEst:
    v_vec = np.array([v_mag, 0.0, 0.0])
    Γmax = _gamma_max_bound()

    t_block = 0.0; acc_blk = 0
    s1 = s2 = s3 = 0.0
    blocks = []
    accepts_total = 0

    while accepts_total < N_accept_per_v:
        dt = rng.exponential(1.0/Γmax, size=batch)
        vf = gaussian_bath_velocities(batch, s_bath)
        g  = v_vec - vf
        V  = np.linalg.norm(g, axis=1)
        good = V > 0
        if not np.any(good):
            continue
        dt = dt[good]; g = g[good]; V = V[good]
        ghat = g / V[:, None]

        sig = sigma_tot_rutherford_vec(V, sigma0, w)
        cosθ = sample_cos_rutherford_vec(V)

        p = (rate_boost * n_f * sig * V) / Γmax
        acc = rng.random(V.size) < p

        t_block += dt.sum()

        if not np.any(acc):
            continue

        Va    = V[acc]
        ghata = ghat[acc]
        cosa  = cosθ[acc]
        sina  = np.sqrt(1.0 - np.clip(cosa*cosa, 0.0, 1.0))
        phi   = rng.uniform(0.0, 2.0*np.pi, size=Va.size)

        for i in range(Va.size):
            e1, e2 = ortho_basis_from(ghata[i])
            n = sina[i]*np.cos(phi[i])*e1 + sina[i]*np.sin(phi[i])*e2 + cosa[i]*ghata[i]
            dv = (mu/m_t) * Va[i] * (n - ghata[i])
            dv_par   = dv[0]
            dv_perp2 = float(np.dot(dv, dv) - dv_par**2)
            s1 += dv_par
            s2 += dv_par**2
            s3 += dv_perp2

        acc_blk       += Va.size
        accepts_total += Va.size

        while acc_blk >= block_accepts:
            blocks.append((s1/t_block, s2/t_block, s3/t_block))
            t_block = 0.0; acc_blk = 0; s1 = s2 = s3 = 0.0

    arr = np.array(blocks, float)
    means = arr.mean(axis=0) if len(arr) else np.zeros(3)
    errs  = arr.std(axis=0, ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(3)
    return MCEst(
        v=v_mag,
        drift_par=float(means[0]), drift_par_err=float(errs[0]),
        diff_par=float(means[1]),  diff_par_err=float(errs[1]),
        diff_perp=float(means[2]), diff_perp_err=float(errs[2]),
        acceptance=accepts_total / max(batch*len(blocks), 1),
        blocks=len(arr), accepts=accepts_total
    )

def gaussian_bath_velocities(n: int, s: float) -> np.ndarray:
    return rng.normal(0.0, s, size=(n, 3))

def ortho_basis_from(g_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if abs(g_hat[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(g_hat, ref)
    n1 = np.linalg.norm(e1)
    if n1 < 1e-15:
        e1 = np.array([1.0, 0.0, 0.0])
        n1 = 1.0
    e1 /= n1
    e2 = np.cross(g_hat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2

# Main execution
v_grid = np.linspace(0.0, v_max_factor*s_bath, N_v)

theory_rows, mc_rows = [], []
for vmag in v_grid:
    th = theory_rates_at_v_fast(vmag)
    theory_rows.append((vmag, *th))
    mc_rows.append(simulate_fixed_v_fast(vmag, batch=16384))

theory_df = pd.DataFrame(theory_rows, columns=["v","drift_par","diff_par","diff_perp"])
mc_df = pd.DataFrame([
    dict(v=r.v,
         drift_par=r.drift_par, drift_par_err=r.drift_par_err,
         diff_par=r.diff_par,   diff_par_err=r.diff_par_err,
         diff_perp=r.diff_perp, diff_perp_err=r.diff_perp_err,
         acceptance=r.acceptance, blocks=r.blocks, accepts=r.accepts)
    for r in mc_rows
])

th_csv = f"{prefix}_theory.csv"
mc_csv = f"{prefix}_mc.csv"
theory_df.to_csv(th_csv, index=False)
mc_df.to_csv(mc_csv, index=False)

def overlay(x, y_th, y_mc, yerr, ylabel, title, path):
    plt.figure()
    plt.plot(x, y_th, label="Theory (angular)")
    plt.errorbar(x, y_mc, yerr=yerr, fmt="o", capsize=3, label="MC")
    plt.xlabel("speed v")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()

p1 = f"{prefix}_drift.png"
overlay(theory_df["v"], theory_df["drift_par"], mc_df["drift_par"], mc_df["drift_par_err"],
        r"$\langle \Delta v_\parallel \rangle / dt$",
        f"Rutherford: Drift along $\\hat v$", p1)

p2 = f"{prefix}_Dpar.png"
overlay(theory_df["v"], theory_df["diff_par"], mc_df["diff_par"], mc_df["diff_par_err"],
        r"$\langle (\Delta v_\parallel)^2 \rangle / dt$",
        f"Rutherford: $D_\\parallel$", p2)

p3 = f"{prefix}_Dperp.png"
overlay(theory_df["v"], theory_df["diff_perp"], mc_df["diff_perp"], mc_df["diff_perp_err"],
        r"$\langle \Delta v_\perp^2 \rangle / dt$",
        f"Rutherford: $D_\\perp$ (sum of 2)", p3)

print("Saved files:")
print(th_csv)
print(mc_csv)
print(p1)
print(p2)
print(p3)

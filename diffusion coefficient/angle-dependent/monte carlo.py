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

scatt_kind = "rutherford"

N_theory_samples = 7000

N_accept_per_v = 50000
block_accepts = 250
rate_boost = 5.0e11
safety_factor = 1.4

v_max_factor = 4.0
N_v = 10

prefix = f"img/aniso_fix_{scatt_kind}"
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

def build_lookup(kind: str, Vmax: float, nV: int = 600):
    Vgrid = np.linspace(1e-6, Vmax, nV)
    if kind == "rutherford":
        C1, C2 = rutherford_C1C2(Vgrid, w)
        f  = dsigma_dcos_rutherford(Vgrid[:, None], GL_X[None, :], sigma0, w)
        sig = sigma_tot_rutherford_vec(Vgrid, sigma0, w)
        C3 = (GL_W[None, :] * (GL_X[None, :]**3) * f).sum(axis=1) / sig
        SIG = sig
    else:
        f  = dsigma_dcos_moller(Vgrid[:, None], GL_X[None, :], sigma0, w)
        sig = sigma_tot_moller(Vgrid, sigma0, w)
        C1 = (GL_W[None, :]*GL_X[None, :]*f).sum(axis=1)     / sig
        C2 = (GL_W[None, :]*(GL_X[None, :]**2)*f).sum(axis=1)/ sig
        C3 = (GL_W[None, :]*(GL_X[None, :]**3)*f).sum(axis=1)/ sig
        SIG = sig
    return Vgrid, C1, C2, C3, SIG

Vmax_lookup = v_max_factor * s_bath * 1.5
Vgrid_L, C1_L, C2_L, C3_L, SIG_L = build_lookup(scatt_kind, Vmax_lookup, nV=600)
def _interpL(V, arr): return np.interp(V, Vgrid_L, arr, left=arr[0], right=arr[-1])

def theory_rates_at_v_fast(v_mag: float, kind: str) -> Tuple[float, float, float]:
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

def theory_rates_third_at_v_fast(v_mag: float, kind: str, N_bath: int = 30000):
    v_vec = np.array([v_mag, 0.0, 0.0])
    vf = gaussian_bath_velocities(N_bath, s_bath)
    g  = v_vec - vf
    V  = np.linalg.norm(g, axis=1)
    good = V > 0
    if not np.any(good):
        return 0.0, 0.0
    V   = V[good]
    mu_e = g[good, 0] / V
    C1 = _interpL(V, C1_L); C2 = _interpL(V, C2_L); C3 = _interpL(V, C3_L); SIG = _interpL(V, SIG_L)
    wt = n_f * SIG * V
    pref = (mu/m_t)**3 * (V**3)
    Tpar3 = np.mean(wt * pref * _S_third_raw(mu_e, C1, C2, C3))
    Tmix3 = np.mean(wt * pref * _S_mix_raw (mu_e, C1, C2, C3))
    s = rate_boost
    return float(Tpar3*s), float(Tmix3*s)

def _gamma_max_bound(kind: str) -> float:
    if kind == "rutherford":
        return safety_factor * rate_boost * n_f * (sigma0 * w * 0.5)
    return safety_factor * rate_boost * n_f * float(np.max(SIG_L * Vgrid_L))

def sample_cos_rutherford_vec(V: np.ndarray) -> np.ndarray:
    A = w*w + 0.5*V*V
    B = 0.5*V*V
    u = rng.random(V.shape[0])
    denom = (1.0/(A + B)) + (2.0*B / ((A + B)*(np.maximum(A - B, 1e-300)))) * u
    inv = 1.0/denom
    x = (A - inv)/B
    return np.clip(x, -1.0, 1.0)

def sample_cos_moller_vec(V: np.ndarray) -> np.ndarray:
    fmax = (dsigma_dcos_moller(V[:,None], GL_X[None,:], sigma0, w)).max(axis=1) * 1.05 + 1e-30
    out = np.empty_like(V)
    filled = np.zeros(V.shape[0], dtype=bool)
    while not np.all(filled):
        need = np.where(~filled)[0]
        x_try = rng.uniform(-1.0, 1.0, size=need.size)
        y_try = rng.random(need.size) * fmax[need]
        ok = y_try <= dsigma_dcos_moller(V[need], x_try, sigma0, w)
        out[need[ok]] = x_try[ok]
        filled[need[ok]] = True
    return out

def simulate_fixed_v_fast(v_mag: float, kind: str, batch: int = 16384) -> MCEst:
    v_vec = np.array([v_mag, 0.0, 0.0])
    Γmax = _gamma_max_bound(kind)

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

        if kind == "rutherford":
            sig = sigma_tot_rutherford_vec(V, sigma0, w)
            cosθ = sample_cos_rutherford_vec(V)
        else:
            sig = sigma_tot_moller(V, sigma0, w)
            cosθ = sample_cos_moller_vec(V)

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

def sigma_tot_rutherford(v: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    return sigma0 / (1.0 + (v*v)/(w*w))

def dsigma_dcos_moller(v: np.ndarray, cos_theta: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    sin2 = 1.0 - cos_theta**2
    top = (3.0*cos_theta**2 + 1.0)*v**4 + 4.0*v*v*w*w + 4.0*w**4
    bot = (sin2 * v**4 + 4.0*v*v*w*w + 4.0*w**4)**2
    return sigma0 * w**4 * top / bot

def sigma_tot_moller(v: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    vw2 = v*v*w*w
    w4 = w**4
    v4 = v**4
    logarg = np.maximum(w*w/(v*v + w*w), 1e-300)
    term1 = 1.0/(v*v*w*w + w4)
    term2 = 1.0/(v4 + 2.0*vw2) * np.log(logarg)
    return sigma0 * w**4 * ( term1 + term2 )


def moller_C1C2(v: np.ndarray, sigma0: float, w: float, npts: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    x, wgt = np.polynomial.legendre.leggauss(npts)
    V = v[:, None]
    X = x[None, :]
    W = wgt[None, :]
    f = dsigma_dcos_moller(V, X, sigma0, w)
    sig = sigma_tot_moller(v, sigma0, w)
    num1 = np.sum(W * X * f, axis=1)
    num2 = np.sum(W * (X*X) * f, axis=1)
    C1 = num1 / sig
    C2 = num2 / sig
    return C1, C2

def theory_rates_at_v(v_mag: float, kind: str) -> Tuple[float, float, float]:
    v_vec = np.array([v_mag, 0.0, 0.0])
    v_f = gaussian_bath_velocities(N_theory_samples, s_bath)
    g = v_vec - v_f
    V = np.linalg.norm(g, axis=1)
    ok = V > 0
    g = g[ok]; V = V[ok]
    g_hat = g / V[:, None]
    mu_e = g_hat[:, 0]

    if kind == "rutherford":
        C1, C2 = rutherford_C1C2(V, w)
        sig_tot = sigma_tot_rutherford(V, sigma0, w)
    else:
        C1, C2 = moller_C1C2(V, sigma0, w, npts=256)
        sig_tot = sigma_tot_moller(V, sigma0, w)

    a_perp = 0.5*(1.0 - C2)
    par_coef = (C2 - 2.0*C1 + 1.0)

    weight = n_f * sig_tot * V
    drift_par = np.mean(weight * (mu/m_t) * (C1 - 1.0) * (V * mu_e))
    diff_par  = np.mean(weight * (mu/m_t)**2 * (V**2) * ( a_perp*(1.0 - mu_e**2) + par_coef*(mu_e**2) ))
    total2    = np.mean(weight * 2.0 * (mu/m_t)**2 * (V**2) * (1.0 - C1))
    diff_perp = total2 - diff_par

    scale = rate_boost
    return float(drift_par*scale), float(diff_par*scale), float(diff_perp*scale)


def sigmaV_max_bound(v_mag: float, kind: str) -> float:
    Vmax = v_mag + 8.0*s_bath
    Vs = np.linspace(0.0, Vmax, 400)
    if kind == "rutherford":
        sig = sigma_tot_rutherford(Vs, sigma0, w)
    else:
        sig = sigma_tot_moller(Vs, sigma0, w)
    return float(np.nanmax(sig * Vs))

def sample_cos_rutherford(v: float, n: int, w: float) -> np.ndarray:
    A = w*w + 0.5*v*v
    B = 0.5*v*v
    u = rng.random(n)
    denom = (1.0/(A + B)) + (2.0*B / ((A + B)*(A - B))) * u
    inv = 1.0/denom
    x = (A - inv)/B
    return np.clip(x, -1.0, 1.0)

def sample_cos_moller(v: float, n: int) -> np.ndarray:
    x_grid = np.linspace(-1.0, 1.0, 2001)
    f_grid = dsigma_dcos_moller(np.array([v]), x_grid, sigma0, w)[0]
    fmax = float(np.max(f_grid)) * 1.05 + 1e-30
    out = np.empty(n)
    k = 0
    while k < n:
        x_try = rng.uniform(-1.0, 1.0)
        y = rng.random() * fmax
        if y <= dsigma_dcos_moller(np.array([v]), np.array([x_try]), sigma0, w)[0]:
            out[k] = x_try
            k += 1
    return out

def simulate_fixed_v(v_mag: float, kind: str) -> MCEst:
    v_vec = np.array([v_mag, 0.0, 0.0])

    if kind == "rutherford":
        sig_fun = lambda V: sigma_tot_rutherford(V, sigma0, w)
        cos_sampler = lambda V, n: sample_cos_rutherford(V, n, w)
    else:
        sig_fun = lambda V: sigma_tot_moller(V, sigma0, w)
        cos_sampler = lambda V, n: sample_cos_moller(V, n)

    sigVmax = sigmaV_max_bound(v_mag, kind)
    Gamma_max = safety_factor * rate_boost * n_f * sigVmax

    t_block = 0.0; acc_blk = 0
    s1 = s2 = s3 = 0.0
    blocks = []
    attempts = 0; accepts = 0

    while accepts < N_accept_per_v:
        dt = rng.exponential(1.0 / Gamma_max)
        t_block += dt
        attempts += 1

        v_f = gaussian_bath_velocities(1, s_bath)[0]
        g = v_vec - v_f
        V = float(np.linalg.norm(g))
        if V < 1e-15:
            continue
        g_hat = g / V

        sig_tot = float(sig_fun(V))
        p = (rate_boost * n_f * sig_tot * V) / Gamma_max

        if p > 1.0:
            sigVmax = max(sigVmax, sig_tot*V, sigmaV_max_bound(v_mag, kind))
            Gamma_max = safety_factor * rate_boost * n_f * sigVmax
            p = min(1.0, (rate_boost * n_f * sig_tot * V) / Gamma_max)

        if rng.random() < p:
            cos_th = float(cos_sampler(V, 1)[0])
            sin_th = np.sqrt(max(0.0, 1.0 - cos_th**2))
            phi = rng.uniform(0.0, 2.0*np.pi)

            e1, e2 = ortho_basis_from(g_hat)
            n_hat = sin_th*np.cos(phi)*e1 + sin_th*np.sin(phi)*e2 + cos_th*g_hat

            dv = (mu/m_t) * V * (n_hat - g_hat)
            dv_par = dv[0]
            dv_perp2 = float(np.dot(dv, dv) - dv_par**2)

            s1 += dv_par
            s2 += dv_par**2
            s3 += dv_perp2

            acc_blk += 1
            accepts += 1

            if acc_blk >= block_accepts:
                blocks.append((s1/t_block, s2/t_block, s3/t_block))
                t_block = 0.0; acc_blk = 0; s1 = s2 = s3 = 0.0

    if acc_blk > 0 and t_block > 0.0:
        blocks.append((s1/t_block, s2/t_block, s3/t_block))

    arr = np.array(blocks, float)
    means = np.mean(arr, axis=0) if len(arr) > 0 else np.zeros(3)
    errs  = (np.std(arr, axis=0, ddof=1)/np.sqrt(len(arr))) if len(arr) > 1 else np.zeros(3)

    return MCEst(
        v=v_mag,
        drift_par=float(means[0]), drift_par_err=float(errs[0]),
        diff_par=float(means[1]),  diff_par_err=float(errs[1]),
        diff_perp=float(means[2]), diff_perp_err=float(errs[2]),
        acceptance=accepts/max(attempts,1), blocks=len(arr), accepts=accepts
    )

v_grid = np.linspace(0.0, v_max_factor*s_bath, N_v)

theory_rows, mc_rows = [], []
for vmag in v_grid:
    th = theory_rates_at_v_fast(vmag, scatt_kind)
    theory_rows.append((vmag, *th))
    mc_rows.append(simulate_fixed_v_fast(vmag, scatt_kind, batch=16384))

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
        f"{scatt_kind.capitalize()}: Drift along $\\hat v$", p1)

p2 = f"{prefix}_Dpar.png"
overlay(theory_df["v"], theory_df["diff_par"], mc_df["diff_par"], mc_df["diff_par_err"],
        r"$\langle (\Delta v_\parallel)^2 \rangle / dt$",
        f"{scatt_kind.capitalize()}: $D_\\parallel$", p2)

p3 = f"{prefix}_Dperp.png"
overlay(theory_df["v"], theory_df["diff_perp"], mc_df["diff_perp"], mc_df["diff_perp_err"],
        r"$\langle \Delta v_\perp^2 \rangle / dt$",
        f"{scatt_kind.capitalize()}: $D_\\perp$ (sum of 2)", p3)

print("Saved files:")
print(th_csv)
print(mc_csv)
print(p1)
print(p2)
print(p3)

def _ang_moments_C123(v_scalar: float, kind: str, npts: int = 256):
    x, wgt = np.polynomial.legendre.leggauss(npts)
    if kind == "rutherford":
        f = dsigma_dcos_rutherford(np.array([v_scalar]), x, sigma0, w)[0]
        sig = float(sigma_tot_rutherford(np.array([v_scalar]), sigma0, w))
    else:
        f = dsigma_dcos_moller(np.array([v_scalar]), x, sigma0, w)[0]
        sig = float(sigma_tot_moller(np.array([v_scalar]), sigma0, w))
    C1 = float(np.sum(wgt * x      * f) / sig)
    C2 = float(np.sum(wgt * x**2   * f) / sig)
    C3 = float(np.sum(wgt * x**3   * f) / sig)
    return C1, C2, C3, sig

def _S_third_raw(mu_e, C1, C2, C3):
    term1 = mu_e**3 * C3 + 1.5 * mu_e * (1.0 - mu_e**2) * (C1 - C3)
    term2 = -3.0 * mu_e * ( (1.0 - mu_e**2) * (1.0 - C2) / 2.0 + mu_e**2 * C2 )
    term3 = 3.0 * mu_e**3 * C1 - mu_e**3
    return term1 + term2 + term3

def _S_mix_raw(mu_e, C1, C2, C3):
    S = _S_third_raw(mu_e, C1, C2, C3)
    return -2.0 * mu_e * (C1 - 1.0)**2 - S

def theory_rates_third_at_v(v_mag: float, kind: str, N_bath: int = 40000):
    v_vec = np.array([v_mag, 0.0, 0.0])
    v_f  = gaussian_bath_velocities(N_bath, s_bath)
    g    = v_vec - v_f
    V    = np.linalg.norm(g, axis=1)
    good = V > 0
    V    = V[good]
    ghat = g[good] / V[:, None]
    mu_e = ghat[:, 0]

    C1 = np.empty_like(V); C2 = np.empty_like(V); C3 = np.empty_like(V); SIG = np.empty_like(V)
    for i, Vi in enumerate(V):
        c1, c2, c3, sig = _ang_moments_C123(float(Vi), kind)
        C1[i], C2[i], C3[i], SIG[i] = c1, c2, c3, sig

    wt = n_f * SIG * V
    pref = (mu/m_t)**3 * (V**3)

    Tpar3 = np.mean( wt * pref * _S_third_raw(mu_e, C1, C2, C3) )
    Tmix3 = np.mean( wt * pref * _S_mix_raw  (mu_e, C1, C2, C3) )

    return float(Tpar3 * rate_boost), float(Tmix3 * rate_boost)

Tpar3_vals = []
Tmix3_vals = []
for vmag in v_grid:
    t3, tm = theory_rates_third_at_v_fast(vmag, scatt_kind, N_bath=30000)
    Tpar3_vals.append(t3)
    Tmix3_vals.append(tm)
Tpar3_vals = np.array(Tpar3_vals)
Tmix3_vals = np.array(Tmix3_vals)

eps = 1e-300
def _norm(y): 
    m = np.max(np.abs(y))
    return y / (m + eps) if m > 0 else y

y1 = _norm(theory_df["drift_par"].values)
y2 = _norm(theory_df["diff_par"].values)
y3 = _norm(theory_df["diff_perp"].values)
y4 = _norm(Tpar3_vals)
y5 = _norm(Tmix3_vals)

plt.figure()
plt.plot(v_grid, y1, label="$\\langle\\Delta v_\\parallel\\rangle/dt$")
plt.plot(v_grid, y2, label="$\\langle\\Delta v_\\parallel^2\\rangle/dt$")
plt.plot(v_grid, y3, label="$\\langle\\Delta v_\\perp^2\\rangle/dt$")
plt.plot(v_grid, y4, label="$\\langle\\Delta v_\\parallel^3\\rangle/dt$")
plt.plot(v_grid, y5, label="$\\langle\\Delta v_\\parallel\\Delta v_\\perp^2\\rangle/dt$")
plt.xlabel("speed $v$")
plt.ylabel("normalized value")
plt.title(f"{scatt_kind.capitalize()}")
plt.legend()
one_fig = f"{prefix}_1ALL_normalized.{output_format}"
os.makedirs(os.path.dirname(prefix), exist_ok=True)
plt.tight_layout()
plt.savefig(one_fig, dpi=160)
plt.show()

print("Saved unified normalized figure:", one_fig)

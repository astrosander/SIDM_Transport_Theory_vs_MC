import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple
import os

rng = np.random.default_rng(7)

m_t = 1.78266192e-26
m_f = 1.78266192e-26
s_bath = 1.5e5
rho_dm = 5.35e-22
n_f = rho_dm / m_f

sigma_model = "powerlaw"
sigma0 = 1.0e-27
V0 = 150e3
alpha_pow = -2.0
V_min_clip = 1.0

N_accept_per_v = 3000
block_accepts = 300
rate_boost = 5.0e11
safety_factor = 1.4
probe_v_max = 4.0
N_v_points = 12

N_bath_samples_theory = 120000 

out_prefix = "img/powerlaw/powerlaw_"
output_format = "pdf"
isPreview = False

def sigma_constant(V: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    return np.full_like(V, p["sigma0"], dtype=float)

def sigma_powerlaw(V: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Vc = np.maximum(V, p["V_min_clip"])
    return p["sigma0"] * (Vc / p["V0"]) ** p["alpha_pow"]

def sigma_yukawa_like(V: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Vc = np.maximum(V, p["V_min_clip"])
    return p["sigma0"] / (1.0 + (Vc / p["V0"]) ** 4)

def get_sigma(name: str) -> Callable[[np.ndarray, Dict[str, Any]], np.ndarray]:
    return {"constant": sigma_constant, "powerlaw": sigma_powerlaw, "yukawa": sigma_yukawa_like}[name]

sigma_params = dict(sigma0=sigma0, V0=V0, alpha_pow=alpha_pow, V_min_clip=V_min_clip)
sigma_fn = get_sigma(sigma_model)


def gaussian_bath_velocities(n: int, s: float) -> np.ndarray:
    return rng.normal(0.0, s, size=(n, 3))

def random_unit_vectors(n: int) -> np.ndarray:
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2*np.pi, size=n)
    s = np.sqrt(1.0 - u*u)
    return np.stack((s*np.cos(phi), s*np.sin(phi), u), axis=1)

def sigmaV_max_bound(v_mag: float, s: float, sigma_fn: Callable, sigma_p: Dict[str, Any]) -> float:
    V_min = sigma_p["V_min_clip"]
    V_max = max(V_min + 10.0, v_mag + 8.0*s)
    xs1 = np.geomspace(max(V_min, 1e-2), max(V_max, V_min+1e-2), num=300)
    xs2 = np.linspace(max(V_min, 0.0), V_max, num=120)
    V_grid = np.unique(np.concatenate([xs1, xs2]))
    prod = sigma_fn(V_grid, sigma_p) * V_grid
    finite = np.isfinite(prod) & (prod >= 0.0)
    return float(np.max(prod[finite]))


def theory_rates_at_v(v_mag: float) -> Tuple[float, float, float]:
    v_vec = np.array([v_mag, 0.0, 0.0])
    mu = (m_t * m_f) / (m_t + m_f)

    v_f = gaussian_bath_velocities(N_bath_samples_theory, s_bath)
    g = v_vec - v_f
    V = np.linalg.norm(g, axis=1)
    ex_g = g[:, 0]
    sig = sigma_fn(V, sigma_params)
    wt = n_f * sig * V

    drift_par_coll = -(mu/m_t) * ex_g
    diff_par_coll = (mu/m_t)**2 * ((V**2)/3.0 + ex_g**2)
    diff_perp_coll = (mu/m_t)**2 * ((5.0/3.0) * (V**2) - ex_g**2)

    drift_rate = float(np.mean(wt * drift_par_coll))
    diff_par_rate = float(np.mean(wt * diff_par_coll))
    diff_perp_rate = float(np.mean(wt * diff_perp_coll))

    return (drift_rate * rate_boost,
            diff_par_rate * rate_boost,
            diff_perp_rate * rate_boost)


@dataclass
class MCEstimate:
    v: float
    drift_par: float
    drift_par_err: float
    diff_par: float
    diff_par_err: float
    diff_perp: float
    diff_perp_err: float
    acceptance: float
    blocks: int
    accepts: int

def simulate_fixed_v(v_mag: float) -> MCEstimate:
    v_vec = np.array([v_mag, 0.0, 0.0])
    mu = (m_t * m_f) / (m_t + m_f)

    sigVmax = sigmaV_max_bound(v_mag, s_bath, sigma_fn, sigma_params)
    Gamma_max = safety_factor * rate_boost * n_f * sigVmax

    t_block = 0.0
    acc_in_block = 0
    sum_drift = sum_dpar = sum_dperp = 0.0
    blocks = []
    attempts_total = 0
    accepts_total = 0

    while accepts_total < N_accept_per_v:
        dt = rng.exponential(1.0 / Gamma_max)
        t_block += dt
        attempts_total += 1

        v_f = gaussian_bath_velocities(1, s_bath)[0]
        g = v_vec - v_f
        V = float(np.linalg.norm(g))
        sig = float(sigma_fn(np.array([V]), sigma_params)[0])
        p = (rate_boost * n_f * sig * V) / Gamma_max

        if p > 1.0:
            sigVmax = max(sigVmax, sig * V, sigmaV_max_bound(v_mag, s_bath, sigma_fn, sigma_params))
            Gamma_max = safety_factor * rate_boost * n_f * sigVmax
            p = min(1.0, (rate_boost * n_f * sig * V) / Gamma_max)

        if rng.random() < p:
            n_hat = random_unit_vectors(1)[0]
            g_new = V * n_hat
            dv = (mu/m_t) * (g_new - g)

            dv_par = dv[0]
            dv_perp2 = float(np.dot(dv, dv) - dv_par**2)

            sum_drift += dv_par
            sum_dpar += dv_par**2
            sum_dperp += dv_perp2
            acc_in_block += 1
            accepts_total += 1

            if acc_in_block >= block_accepts:
                blocks.append((sum_drift/t_block, sum_dpar/t_block, sum_dperp/t_block, t_block))
                t_block = 0.0
                acc_in_block = 0
                sum_drift = sum_dpar = sum_dperp = 0.0

    if t_block > 0.0 and acc_in_block > 0:
        blocks.append((sum_drift/t_block, sum_dpar/t_block, sum_dperp/t_block, t_block))

    arr = np.array([[b[0], b[1], b[2]] for b in blocks], dtype=float)
    nblk = max(len(blocks), 1)
    means = np.mean(arr, axis=0) if nblk > 0 else np.zeros(3)
    errs = np.std(arr, axis=0, ddof=1)/np.sqrt(nblk) if nblk > 1 else np.zeros(3)

    return MCEstimate(
        v=v_mag,
        drift_par=float(means[0]), drift_par_err=float(errs[0]),
        diff_par=float(means[1]), diff_par_err=float(errs[1]),
        diff_perp=float(means[2]), diff_perp_err=float(errs[2]),
        acceptance=accepts_total/max(attempts_total,1),
        blocks=nblk, accepts=accepts_total
    )


v_vals = np.linspace(0.0, probe_v_max*s_bath, N_v_points)
theory = []
mc_pts = []

for vmag in v_vals:
    theory.append((vmag, *theory_rates_at_v(vmag)))
    mc_pts.append(simulate_fixed_v(vmag))

theory_df = pd.DataFrame(theory, columns=["v","drift_par","diff_par","diff_perp"])
mc_df = pd.DataFrame([
    dict(v=pt.v,
         drift_par=pt.drift_par, drift_par_err=pt.drift_par_err,
         diff_par=pt.diff_par, diff_par_err=pt.diff_par_err,
         diff_perp=pt.diff_perp, diff_perp_err=pt.diff_perp_err,
         acceptance=pt.acceptance, blocks=pt.blocks, accepts=pt.accepts)
    for pt in mc_pts
])


# th_path = f"{out_prefix}theory.csv"
# mc_path = f"{out_prefix}mc.csv"
# theory_df.to_csv(th_path, index=False)
# mc_df.to_csv(mc_path, index=False)

fig1 = f"{out_prefix}drift_vs_v.{output_format}"
fig2 = f"{out_prefix}diff_par_vs_v.{output_format}"
fig3 = f"{out_prefix}diff_perp_vs_v.{output_format}"

plt.figure()
plt.plot(theory_df["v"], theory_df["drift_par"], label="Theory")
plt.errorbar(mc_df["v"], mc_df["drift_par"], yerr=mc_df["drift_par_err"], fmt="o", capsize=3, label="MC")
plt.xlabel("speed v")
plt.ylabel(r"$\langle \Delta v_\parallel \rangle / dt$")
plt.title(r"Dependence of drift along $\hat v$ from speed")
plt.legend(); plt.tight_layout(); plt.savefig(fig1, dpi=160); 

if isPreview:
    plt.show()

plt.figure()
plt.plot(theory_df["v"], theory_df["diff_par"], label="Theory")
plt.errorbar(mc_df["v"], mc_df["diff_par"], yerr=mc_df["diff_par_err"], fmt="o", capsize=3, label="MC")
plt.xlabel("speed v")
plt.ylabel(r"$\langle (\Delta v_\parallel)^2 \rangle / dt$")
plt.title("Parallel diffusion (raw 2nd moment)")
plt.legend(); plt.tight_layout(); plt.savefig(fig2, dpi=160); 

if isPreview:
    plt.show()

plt.figure()
plt.plot(theory_df["v"], theory_df["diff_perp"], label="Theory")
plt.errorbar(mc_df["v"], mc_df["diff_perp"], yerr=mc_df["diff_perp_err"], fmt="o", capsize=3, label="MC")
plt.xlabel("speed v")
plt.ylabel(r"$\langle \Delta v_\perp^2 \rangle / dt$")
plt.title("Perpendicular diffusion (sum of 2 components)")
plt.legend(); plt.tight_layout(); plt.savefig(fig3, dpi=160); 

if isPreview:
    plt.show()

# zero_energy_path = f"{out_prefix}energy_rate_zero.csv"
# pd.DataFrame({"v": v_vals, "energy_rate": np.zeros_like(v_vals)}).to_csv(zero_energy_path, index=False)


# sample_idx = np.linspace(0, len(v_vals)-1, 6, dtype=int)
# compare = pd.DataFrame({
#     "v": theory_df["v"].values[sample_idx],
#     "th_drift_par": theory_df["drift_par"].values[sample_idx],
#     "mc_drift_par": mc_df["drift_par"].values[sample_idx],
#     "th_diff_par": theory_df["diff_par"].values[sample_idx],
#     "mc_diff_par": mc_df["diff_par"].values[sample_idx],
#     "th_diff_perp": theory_df["diff_perp"].values[sample_idx],
#     "mc_diff_perp": mc_df["diff_perp"].values[sample_idx],
# })
# print(("Theory vs MC — sample speeds", compare))
# # display_dataframe_to_user("Theory vs MC — sample speeds", compare)

# print("Saved:")
# print(th_path)
# print(mc_path)
# print(fig1)
# print(fig2)
# print(fig3)
# print(zero_energy_path)
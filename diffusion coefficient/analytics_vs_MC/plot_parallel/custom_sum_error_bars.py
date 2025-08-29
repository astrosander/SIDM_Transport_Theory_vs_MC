import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from numba import njit, prange, set_num_threads

try:
    import multiprocessing as _mp
    set_num_threads(_mp.cpu_count())
except Exception:
    pass

rng = np.random.default_rng(1)

num_bath = 1_000_000
velocity_vals = np.linspace(0.0, 1_000_000.0, 100)
s_bath = 150_000.0
number_encounters = 10_000_000

m_particle = 1.0
m_bath = 1.0
mu = (m_particle * m_bath) / (m_particle + m_bath)

KAPPA = 1.0

USE_VDEP_THETA_MIN = True
THETA_MIN_FIXED = 1.0e-3

sigma0 = 2.0e-28
w = 1.0e3

USE_ABSOLUTE_RATES = False
n_f = 1.0

v_bath = rng.normal(0.0, s_bath, size=(num_bath, 3)).astype(np.float64)

np.random.seed(1)

@njit(fastmath=True, cache=True)
def _norm3(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@njit(fastmath=True, cache=True)
def _theta_min_from(V: float, w_: float) -> float:
    eps = 1e-15
    if USE_VDEP_THETA_MIN:
        Veff = V if V > eps else eps
        th = 2.0 * w_ / Veff
    else:
        th = THETA_MIN_FIXED
    if th < 1e-12:
        th = 1e-12
    if th > math.pi - 1e-12:
        th = math.pi - 1e-12
    return th

@njit(fastmath=True, cache=True)
def sigma_tot_rutherford_scalar(V: float, sigma0_: float, w_: float) -> float:
    # return sigma0_ / (1.0 + (V*V)/(w_*w_))
    # return sigma0_
    # return sigma0_ / (1.0 + (V*V*V*V)/(w_*w_*w_*w_))
    thmin = _theta_min_from(V, w_)
    half = 0.5 * thmin
    cot_half = math.cos(half) / math.sin(half)
    pref = KAPPA / (mu * V * V) if V > 0.0 else 0.0
    return math.pi * (pref * pref) * (cot_half * cot_half)

@njit(fastmath=True, cache=True)
def sample_cos_rutherford_scalar(V: float, w_: float) -> float:
    # A = w_*w_ + 0.5*V*V
    # B = 0.5*V*V
    # u = np.random.random()
    # denom = (1.0/(A + B)) + (2.0*B / ((A + B)*max(A - B, 1e-300))) * u
    # inv = 1.0/denom
    # x = (A - inv)/B
    # if x < -1.0:
    #     x = -1.0
    # elif x > 1.0:
    #     x = 1.0
    # return x

    # return 2.0 * np.random.random() - 1.0

    thmin = _theta_min_from(V, w_)
    u = np.random.random()
    tmin = math.tan(0.5 * thmin)
    denom = math.sqrt(max(1.0 - u, 1e-300))
    t = tmin / denom
    theta = 2.0 * math.atan(t)
    if theta < 0.0:
        theta = 0.0
    if theta > math.pi:
        theta = math.pi
    return math.cos(theta)

@njit(fastmath=True, cache=True)
def ortho_basis_from(g_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.empty(3, dtype=np.float64)
    if abs(g_hat[2]) < 0.9:
        ref[0] = 0.0; ref[1] = 0.0; ref[2] = 1.0
    else:
        ref[0] = 0.0; ref[1] = 1.0; ref[2] = 0.0

    e1 = np.empty(3, dtype=np.float64)
    e1[0] = g_hat[1]*ref[2] - g_hat[2]*ref[1]
    e1[1] = g_hat[2]*ref[0] - g_hat[0]*ref[2]
    e1[2] = g_hat[0]*ref[1] - g_hat[1]*ref[0]
    n1 = _norm3(e1)
    if n1 < 1e-15:
        e1[0] = 1.0; e1[1] = 0.0; e1[2] = 0.0
        n1 = 1.0
    e1[0] /= n1; e1[1] /= n1; e1[2] /= n1

    e2 = np.empty(3, dtype=np.float64)
    e2[0] = g_hat[1]*e1[2] - g_hat[2]*e1[1]
    e2[1] = g_hat[2]*e1[0] - g_hat[0]*e1[2]
    e2[2] = g_hat[0]*e1[1] - g_hat[1]*e1[0]
    n2 = _norm3(e2)
    e2[0] /= n2; e2[1] /= n2; e2[2] /= n2
    return e1, e2

@njit(fastmath=True, cache=True, parallel=True)
def compute_moments_for_velocity(particle_velocity: float,
                                 v_bath_: np.ndarray,
                                 number_encounters_: int,
                                 sigma0_: float, w_: float,
                                 use_abs_rates: bool, n_f_: float):
    S1 = 0.0
    S2 = 0.0
    S3 = 0.0
    S4 = 0.0
    S5 = 0.0

    for _ in prange(number_encounters_):
        idx = np.random.randint(0, v_bath_.shape[0])
        vb0 = v_bath_[idx, 0]
        vb1 = v_bath_[idx, 1]
        vb2 = v_bath_[idx, 2]

        g0 = particle_velocity - vb0
        g1 = -vb1
        g2 = -vb2
        g_mag = math.sqrt(g0*g0 + g1*g1 + g2*g2)
        if g_mag <= 0.0:
            continue
        inv_g = 1.0 / g_mag
        gh0 = g0 * inv_g
        gh1 = g1 * inv_g
        gh2 = g2 * inv_g

        cos_theta = sample_cos_rutherford_scalar(g_mag, w_)
        s2 = 1.0 - cos_theta*cos_theta
        sin_theta = math.sqrt(s2) if s2 > 0.0 else 0.0
        phi = 2.0 * math.pi * np.random.random()
        cphi = math.cos(phi)
        sphi = math.sin(phi)

        g_hat = np.array([gh0, gh1, gh2], dtype=np.float64)
        e1, e2 = ortho_basis_from(g_hat)

        nh0 = sin_theta * cphi * e1[0] + sin_theta * sphi * e2[0] + cos_theta * gh0
        nh1 = sin_theta * cphi * e1[1] + sin_theta * sphi * e2[1] + cos_theta * gh1
        nh2 = sin_theta * cphi * e1[2] + sin_theta * sphi * e2[2] + cos_theta * gh2

        dv0 = 0.5 * (g_mag * nh0 - g0)
        dv1 = 0.5 * (g_mag * nh1 - g1)
        dv2 = 0.5 * (g_mag * nh2 - g2)

        dv_par = dv0
        dv_perp2 = dv1*dv1 + dv2*dv2

        sigma_total = sigma_tot_rutherford_scalar(g_mag, sigma0_, w_)
        weight = g_mag * sigma_total * (n_f_ if use_abs_rates else 1.0)

        S1 += dv_par * weight
        S2 += (dv_par*dv_par) * weight
        S3 += dv_perp2 * weight
        S4 += (dv_par*dv_par*dv_par) * weight
        S5 += dv_par * dv_perp2 * weight

    invN = 1.0 / number_encounters_
    return (S1*invN, S2*invN, S3*invN, S4*invN, S5*invN)

dv_parallels          = []
dv_parallels2         = []
dv_perps2             = []
dv_parallels3         = []
dv_parallels_perps2   = []

for particle_velocity in velocity_vals:
    S1, S2, S3, S4, S5 = compute_moments_for_velocity(
        float(particle_velocity),
        v_bath,
        int(number_encounters),
        float(sigma0), float(w),
        bool(USE_ABSOLUTE_RATES), float(n_f),
    )

    dv_parallels.append(        abs(S1 / (s_bath**1)) )
    dv_parallels2.append(       abs(S2 / (s_bath**2)) )
    dv_perps2.append(           abs(S3 / (s_bath**2)) )
    dv_parallels3.append(       abs(S4 / (s_bath**3)) )
    dv_parallels_perps2.append( abs(S5 / (s_bath**3)) )

def normalize_identity(arr):
    return arr

dv_parallels            = normalize_identity(dv_parallels)
dv_parallels2           = normalize_identity(dv_parallels2)
dv_perps2               = normalize_identity(dv_perps2)
dv_parallels3           = normalize_identity(dv_parallels3)
dv_parallels_perps2     = normalize_identity(dv_parallels_perps2)

plt.figure(figsize=(8,6))
plt.loglog(velocity_vals, dv_parallels,           label = r"$\langle \Delta v_\parallel\rangle$")
plt.loglog(velocity_vals, dv_parallels2,          label = r"$\langle \Delta v_\parallel^2\rangle$")
plt.loglog(velocity_vals, dv_perps2,              label = r"$\langle \Delta v_\perp^2\rangle$")
plt.loglog(velocity_vals, dv_parallels3,          label = r"$\langle \Delta v_\parallel^3\rangle$")
plt.loglog(velocity_vals, dv_parallels_perps2,    label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

plt.xlabel(r'$v_{\rm particle}$')
plt.ylabel(r'velocity increments')
mode_text = r"$\theta_{\min}(V)=2w/V$" if USE_VDEP_THETA_MIN else r"$\theta_{\min}=\mathrm{const}$"
plt.title(r'Velocity increments with Rutherford (Coulomb/grav.) scattering, ' + mode_text)
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("velocity_changes_rutherford.png", dpi=160)
plt.savefig("velocity_changes_rutherford.pdf", dpi=160)
plt.show()

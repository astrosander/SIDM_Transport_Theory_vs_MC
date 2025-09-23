#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads

# ------------------ controls ------------------
try:
    import multiprocessing as _mp
    set_num_threads(_mp.cpu_count())
except Exception:
    pass

SEED = 1
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# sweep of the bath 1D velocity dispersion (Maxwellian components ~ N(0, s_bath))
N_S = 50
S_MIN, S_MAX = 5.0e3, 5.0e6
sweep_s = np.geomspace(S_MIN, S_MAX, N_S)

# per-s Monte Carlo workload
N_ENC = 100_000_000  # heavy; reduce if needed

# masses
m_particle = 1.0
m_bath     = 1.0
mu_red     = (m_particle * m_bath) / (m_particle + m_bath)

# screened Rutherford kernel params (you can change)
sigma0 = 2.0e-28
w      = 1.0e3

USE_ABSOLUTE_RATES = False
n_f = 1.0

# ------------------ helpers ------------------
@njit(fastmath=True, cache=True)
def _norm3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

@njit(fastmath=True, cache=True)
def ortho_basis_from_unit(gx, gy, gz):
    """Given a *unit* vector g_hat=(gx,gy,gz), build two orthonormal
       vectors e1,e2 s.t. {e1,e2,g_hat} is right-handed."""
    # ref not parallel to g_hat
    if abs(gz) < 0.9:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = 0.0, 1.0, 0.0

    e10 = gy*rz - gz*ry
    e11 = gz*rx - gx*rz
    e12 = gx*ry - gy*rx
    n1  = _norm3(e10, e11, e12)
    if n1 < 1e-15:
        e10, e11, e12 = 1.0, 0.0, 0.0
        n1 = 1.0
    e10, e11, e12 = e10/n1, e11/n1, e12/n1

    e20 = gy*e12 - gz*e11
    e21 = gz*e10 - gx*e12
    e22 = gx*e11 - gy*e10
    n2  = _norm3(e20, e21, e22)
    e20, e21, e22 = e20/n2, e21/n2, e22/n2
    return e10, e11, e12, e20, e21, e22

@njit(fastmath=True, cache=True)
def _A_B(V, w_):
    A = w_*w_ + 0.5*V*V
    B = 0.5*V*V
    return A, B

@njit(fastmath=True, cache=True)
def sample_cos_rutherford_scalar(V: float, w_: float) -> float:
    """Inverse-CDF for dσ/dcosθ ∝ 1/(A - B cosθ)^2 with A=w^2+V^2/2, B=V^2/2."""
    A, B = _A_B(V, w_)
    u = np.random.random()
    ABp = A + B
    ABm = A - B if (A - B) > 1e-300 else 1e-300
    inv_target = (1.0/ABp) + u*((1.0/ABm) - (1.0/ABp))
    inv = 1.0 / inv_target
    x = (A - inv)/B
    if x < -1.0: x = -1.0
    elif x > 1.0: x = 1.0
    return x

@njit(fastmath=True, cache=True)
def sigma_tot_rutherford_scalar(V: float, sigma0_: float, w_: float) -> float:
    # simple screened Rutherford total cross section ∝ 1/(1+V^2/w^2)
    return sigma0_ / (1.0 + (V*V)/(w_*w_))

# ------------------ core MC (both v1 and vb ~ Maxwellian(s_bath)) ------------------
@njit(fastmath=True, cache=True, parallel=True)
def moments_vs_s_bath(s_bath: float,
                      n_enc: int,
                      sigma0_: float, w_: float,
                      use_abs_rates: bool, n_f_: float,
                      m1_: float, mu_: float):
    """Return per-time transport moments averaged over both target and bath
       velocities drawn from the same Maxwellian (component σ = s_bath)."""
    S1 = 0.0
    S2 = 0.0
    S3 = 0.0
    S4 = 0.0
    S5 = 0.0

    for _ in prange(n_enc):
        # draw target velocity (v1) and bath velocity (vb) from same distribution
        v1x = np.random.normal(0.0, s_bath)
        v1y = np.random.normal(0.0, s_bath)
        v1z = np.random.normal(0.0, s_bath)

        vb0 = np.random.normal(0.0, s_bath)
        vb1 = np.random.normal(0.0, s_bath)
        vb2 = np.random.normal(0.0, s_bath)

        # relative velocity g = v1 - vb
        g0 = v1x - vb0
        g1 = v1y - vb1
        g2 = v1z - vb2
        V  = _norm3(g0, g1, g2)
        if V <= 0.0:
            continue

        invV = 1.0 / V
        gh0, gh1, gh2 = g0*invV, g1*invV, g2*invV

        # scattering angles
        c = sample_cos_rutherford_scalar(V, w_)
        s2 = 1.0 - c*c
        s = math.sqrt(s2) if s2 > 0.0 else 0.0
        phi = 2.0 * math.pi * np.random.random()
        cphi = math.cos(phi)
        sphi = math.sin(phi)

        # orthonormal basis around g_hat
        e10, e11, e12, e20, e21, e22 = ortho_basis_from_unit(gh0, gh1, gh2)

        # new direction of relative velocity after scattering
        nh0 = s*cphi*e10 + s*sphi*e20 + c*gh0
        nh1 = s*cphi*e11 + s*sphi*e21 + c*gh1
        nh2 = s*cphi*e12 + s*sphi*e22 + c*gh2

        # particle velocity change (lab) for general masses
        fac = (mu_ / m1_)
        dvx = fac * (V*nh0 - g0)
        dvy = fac * (V*nh1 - g1)
        dvz = fac * (V*nh2 - g2)

        # define "parallel" relative to the particle's initial velocity v1
        v1n = _norm3(v1x, v1y, v1z)
        if v1n <= 0.0:
            continue
        v1hx, v1hy, v1hz = v1x/v1n, v1y/v1n, v1z/v1n

        dv_par = dvx*v1hx + dvy*v1hy + dvz*v1hz
        dv2    = dvx*dvx + dvy*dvy + dvz*dvz
        dv_perp2 = dv2 - dv_par*dv_par

        # rate weight ~ n σ_tot V   (n_f_=1 if using normalized units)
        sigma_total = sigma_tot_rutherford_scalar(V, sigma0_, w_)
        weight = V * sigma_total * (n_f_ if use_abs_rates else 1.0)

        S1 += dv_par * weight
        S2 += (dv_par*dv_par) * weight
        S3 += dv_perp2 * weight
        S4 += (dv_par*dv_par*dv_par) * weight
        S5 += dv_par * dv_perp2 * weight

    invN = 1.0 / n_enc
    return S1*invN, S2*invN, S3*invN, S4*invN, S5*invN

# ------------------ run sweep and plot ------------------
def main():
    drift   = []
    Dpar    = []
    Dperp   = []
    M3      = []
    Mmix    = []

    # compute MC moments for each s_bath
    for s in sweep_s:
        S1, S2, S3, S4, S5 = moments_vs_s_bath(
            float(s), int(N_ENC),
            float(sigma0), float(w),
            bool(USE_ABSOLUTE_RATES), float(n_f),
            float(m_particle), float(mu_red)
        )
        # normalize by appropriate powers of s to compare across s
        drift.append(   S1 / s)       # 1st order ~ s^1
        Dpar.append(    S2 / (s*s))   # 2nd order ~ s^2
        Dperp.append(   S3 / (s*s))
        M3.append(      S4 / (s*s*s)) # 3rd order ~ s^3
        Mmix.append(    S5 / (s*s*s))

    plt.figure(figsize=(9, 6.4))
    plt.loglog(sweep_s, np.abs(drift),   '-', lw=2.0, label=r"$\langle\Delta v_\parallel\rangle/s$")
    plt.loglog(sweep_s, np.abs(Dpar),    '-', lw=2.0, label=r"$\langle\Delta v_\parallel^2\rangle/s^2$")
    plt.loglog(sweep_s, np.abs(Dperp),   '-', lw=2.0, label=r"$\langle\Delta v_\perp^2\rangle/s^2$")
    plt.loglog(sweep_s, np.abs(M3),      '-', lw=2.0, label=r"$\langle(\Delta v_\parallel)^3\rangle/s^3$")
    plt.loglog(sweep_s, np.abs(Mmix),    '-', lw=2.0, label=r"$\langle\Delta v_\parallel \Delta v_\perp^2\rangle/s^3$")

    plt.xlabel(r"$s_{\rm bath}$")
    plt.ylabel("per-time moments")
    plt.title("Transport moments vs $s_{\\rm bath}$")
    plt.grid(True, which="both", ls=":")
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    plt.savefig("moments_vs_sbath.png", dpi=160)
    plt.savefig("moments_vs_sbath.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

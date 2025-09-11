#!/usr/bin/env python3
import os, math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads

# ───────────────── config ─────────────────
try:
    import multiprocessing as _mp
    set_num_threads(_mp.cpu_count())
except Exception:
    pass

rng = np.random.default_rng(1)
np.random.seed(1)

# Bath and velocity grid
num_bath = 1_000_000
N_VELOCITY = 60
V_MIN, V_MAX = 1.0e4, 1.0e8
velocity_vals = np.geomspace(V_MIN, V_MAX, N_VELOCITY)

s_bath = 1.5e5
number_encounters = 1_000_000  # MC scatters per grid point

# Masses and reduced mass
m_particle = 1.0
m_bath = 1.0
mu_red = (m_particle*m_bath)/(m_particle+m_bath)

# Coulomb/grav kernel params
KAPPA = 1.0                 # "G m1 m2" or "q1 q2 / (4πϵ0)"-like factor → set 1 here
w = 1.0e3                   # screening parameter so that θ_min(V)=2w/V
USE_VDEP_THETA_MIN = True
THETA_MIN_FIXED = 1.0e-3    # only used if USE_VDEP_THETA_MIN=False

# Rate options
USE_ABSOLUTE_RATES = False
n_f = 1.0

# Bath velocities (Maxwellian sample)
v_bath = rng.normal(0.0, s_bath, size=(num_bath, 3)).astype(np.float64)

# ───────────────── Coulomb helpers ─────────────────

@njit(fastmath=True, cache=True)
def _theta_min_from(V: float, w_: float) -> float:
    # θ_min(V) = 2 w / V (with small/large guards)
    eps = 1e-15
    if USE_VDEP_THETA_MIN:
        Veff = V if V > eps else eps
        th = 2.0 * w_ / Veff
    else:
        th = THETA_MIN_FIXED
    if th < 1e-12: th = 1e-12
    if th > math.pi - 1e-12: th = math.pi - 1e-12
    return th

@njit(fastmath=True, cache=True)
def tau2_from_V(V: float, w_: float) -> float:
    # τ^2 = cot^2(θ_min/2)
    tmin = 0.5 * _theta_min_from(V, w_)
    tanh = math.tan(tmin)
    if tanh < 1e-16:
        # small-angle limit: tan ~ angle
        z = tmin
        tanh = z + (z*z*z)/3.0
    return 1.0 / (tanh*tanh)

@njit(fastmath=True, cache=True)
def sigma_tot_coulomb_scalar(V: float, kappa_: float, mu_red_: float, w_: float) -> float:
    # σ_tot = π (κ/(μ V^2))^2 cot^2(θ_min/2) = π pref^2 τ^2
    if V <= 0.0:
        return 0.0
    pref = kappa_ / (mu_red_ * V * V)
    tau2 = tau2_from_V(V, w_)
    return math.pi * (pref*pref) * tau2

@njit(fastmath=True, cache=True)
def sample_cos_coulomb_scalar(V: float, w_: float) -> float:
    """
    Draw θ with PDF ∝ csc^4(θ/2) for θ ∈ [θ_min, π], via exact inversion:
      tan(θ/2) = tan(θ_min/2) / sqrt(1-u),  u∈[0,1)
    """
    thmin = _theta_min_from(V, w_)
    u = np.random.random()
    tmin = math.tan(0.5 * thmin)
    denom = math.sqrt(max(1.0 - u, 1e-300))
    t = tmin / denom
    # cosθ from tan(θ/2)
    t2 = t*t
    return (1.0 - t2) / (1.0 + t2)

# ───────────────── angular averages (Coulomb, closed-form & stable) ─────────────────

@njit(fastmath=True, cache=True)
def coulomb_C1_C2_Y2_T1_K1_from_tau2(t2: float):
    """
    τ^2 = t2 = cot^2(θ_min/2)

    C1 = 1 - 2*ln(1+t2)/t2
    C2 = 1 - 4*ln(1+t2)/t2 + 4/[t2(1+t2)]
    Y2 = 4/(1+t2)
    T1 = -4 + 4/(1+t2)^2
    K1 = -4*t2/(1+t2)^2
    """
    ln1p = math.log1p(t2)
    inv_t2 = 1.0/t2
    one_pt2 = 1.0 + t2
    inv_one_pt2 = 1.0/one_pt2
    inv_one_pt2_sq = inv_one_pt2*inv_one_pt2

    C1 = 1.0 - 2.0*ln1p*inv_t2
    C2 = 1.0 - 4.0*ln1p*inv_t2 + 4.0*(inv_t2*inv_one_pt2)
    Y2 = 4.0*inv_one_pt2
    T1 = -4.0 + 4.0*inv_one_pt2_sq
    K1 = -4.0 * t2 * inv_one_pt2_sq
    return C1, C2, Y2, T1, K1

# ───────────────── frames & basis ─────────────────

@njit(fastmath=True, cache=True)
def _norm3(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@njit(fastmath=True, cache=True)
def ortho_basis_from(g_hat):
    ref = np.empty(3, dtype=np.float64)
    if abs(g_hat[2]) < 0.9: ref[:] = (0.0, 0.0, 1.0)
    else:                  ref[:] = (0.0, 1.0, 0.0)

    e1 = np.empty(3, dtype=np.float64)
    e1[0] = g_hat[1]*ref[2] - g_hat[2]*ref[1]
    e1[1] = g_hat[2]*ref[0] - g_hat[0]*ref[2]
    e1[2] = g_hat[0]*ref[1] - g_hat[1]*ref[0]
    n1 = _norm3(e1)
    if n1 < 1e-15:
        e1[:] = (1.0, 0.0, 0.0); n1 = 1.0
    e1 /= n1

    e2 = np.empty(3, dtype=np.float64)
    e2[0] = g_hat[1]*e1[2] - g_hat[2]*e1[1]
    e2[1] = g_hat[2]*e1[0] - g_hat[0]*e1[2]
    e2[2] = g_hat[0]*e1[1] - g_hat[1]*e1[0]
    e2 /= _norm3(e2)
    return e1, e2

# ───────────────── MC estimator (Coulomb) ─────────────────

@njit(fastmath=True, cache=True, parallel=True)
def compute_moments_MC(particle_velocity: float,
                       v_bath_: np.ndarray,
                       number_encounters_: int,
                       kappa_: float, mu_red_: float, w_: float,
                       use_abs_rates: bool, n_f_: float):
    S1=S2=S3=S4=S5 = 0.0

    for _ in prange(number_encounters_):
        idx = np.random.randint(0, v_bath_.shape[0])
        vb0, vb1, vb2 = v_bath_[idx, 0], v_bath_[idx, 1], v_bath_[idx, 2]

        g0 = particle_velocity - vb0
        g1 = -vb1; g2 = -vb2
        V = math.sqrt(g0*g0 + g1*g1 + g2*g2)
        if V <= 0.0:
            continue
        gh0 = g0 / V; gh1 = g1 / V; gh2 = g2 / V

        x = sample_cos_coulomb_scalar(V, w_)
        s2 = max(0.0, 1.0 - x*x)
        sin_theta = math.sqrt(s2)
        phi = 2.0 * math.pi * np.random.random()
        cphi = math.cos(phi); sphi = math.sin(phi)

        g_hat = np.array([gh0, gh1, gh2], dtype=np.float64)
        e1, e2 = ortho_basis_from(g_hat)

        nh0 = sin_theta*cphi*e1[0] + sin_theta*sphi*e2[0] + x*gh0
        nh1 = sin_theta*cphi*e1[1] + sin_theta*sphi*e2[1] + x*gh1
        nh2 = sin_theta*cphi*e1[2] + sin_theta*sphi*e2[2] + x*gh2

        dv0 = 0.5 * (V*nh0 - g0)
        dv1 = 0.5 * (V*nh1 - g1)
        dv2 = 0.5 * (V*nh2 - g2)

        dv_par = dv0
        dv_perp2 = dv1*dv1 + dv2*dv2

        sigma_total = sigma_tot_coulomb_scalar(V, kappa_, mu_red_, w_)
        weight = V * sigma_total * (n_f_ if use_abs_rates else 1.0)

        S1 += dv_par * weight
        S2 += (dv_par*dv_par) * weight
        S3 += dv_perp2 * weight
        S4 += (dv_par*dv_par*dv_par) * weight
        S5 += dv_par * dv_perp2 * weight

    invN = 1.0/number_encounters_
    return S1*invN, S2*invN, S3*invN, S4*invN, S5*invN

# ── THEORY estimator (Coulomb/grav kernel, sequential + Kahan) ─────────
from numba import njit

@njit(fastmath=True, cache=True)   # ← no parallel here
def compute_moments_THEORY(particle_velocity: float,
                           v_bath_: np.ndarray,
                           kappa_: float, mu_red_: float, w_: float,
                           use_abs_rates: bool, n_f_: float):
    S1=S2=S3=S4=S5 = 0.0
    c1=c2=c3=c4=c5 = 0.0   # Kahan compensations
    N = v_bath_.shape[0]

    for i in range(N):
        vb0, vb1, vb2 = v_bath_[i,0], v_bath_[i,1], v_bath_[i,2]
        g0 = particle_velocity - vb0
        g1 = -vb1; g2 = -vb2
        V  = math.sqrt(g0*g0 + g1*g1 + g2*g2)
        if V <= 0.0:
            continue
        mu_e = g0 / V

        # Coulomb small-angle averages via τ^2 = cot^2(θ_min/2)
        t2 = tau2_from_V(V, w_)
        C1, C2, Y2, T1, K1 = coulomb_C1_C2_Y2_T1_K1_from_tau2(t2)
        a_perp = 0.5*(1.0 - C2)   # ≥ 0
        p_para = Y2               # = 4/(1+t2) ≥ 0

        sigma_total = sigma_tot_coulomb_scalar(V, kappa_, mu_red_, w_)
        weight = V * sigma_total * (n_f_ if use_abs_rates else 1.0)

        F  = (mu_red/m_particle) * V
        F2 = F*F
        F3 = F2*F

        # targets
        s1 = weight * (F*(C1 - 1.0)*mu_e)
        s2 = weight * (F2*( a_perp*(1.0 - mu_e*mu_e) + p_para*(mu_e*mu_e) ))
        total2 = 2.0*F2*(1.0 - C1)
        s3 = weight * ( total2 - (F2*( a_perp*(1.0 - mu_e*mu_e) + p_para*(mu_e*mu_e) )) )
        m3 = weight * ( F3*( mu_e*mu_e*mu_e*T1 + 1.5*mu_e*(1.0 - mu_e*mu_e)*K1 ) )
        mix_first = weight * (-2.0*F3*mu_e*p_para)
        s5 = mix_first - m3

        # Kahan for S1..S5
        y = s1 - c1; t = S1 + y; c1 = (t - S1) - y; S1 = t
        y = s2 - c2; t = S2 + y; c2 = (t - S2) - y; S2 = t
        y = s3 - c3; t = S3 + y; c3 = (t - S3) - y; S3 = t
        y = m3 - c4; t = S4 + y; c4 = (t - S4) - y; S4 = t
        y = s5 - c5; t = S5 + y; c5 = (t - S5) - y; S5 = t

    invN = 1.0 / N
    return S1*invN, S2*invN, S3*invN, S4*invN, S5*invN


# ───────────────── run & plot ─────────────────

def main():
    dv_par_MC=[]; dv_par2_MC=[]; dv_perp2_MC=[]; dv_par3_MC=[]; dv_par_perp2_MC=[]
    dv_par_TH=[]; dv_par2_TH=[]; dv_perp2_TH=[]; dv_par3_TH=[]; dv_par_perp2_TH=[]

    for vp in velocity_vals:
        S1,S2,S3,S4,S5 = compute_moments_MC(
            float(vp), v_bath, int(number_encounters),
            float(KAPPA), float(mu_red), float(w),
            bool(USE_ABSOLUTE_RATES), float(n_f)
        )
        T1,T2,T3,T4,T5 = compute_moments_THEORY(
            float(vp), v_bath,
            float(KAPPA), float(mu_red), float(w),
            bool(USE_ABSOLUTE_RATES), float(n_f)
        )

        # normalize by s_bath^n (mimics your plotting convention)
        s1 = s_bath
        s2 = s_bath**2
        s3 = s_bath**3

        dv_par_MC.append(       abs(S1/s1) )
        dv_par2_MC.append(      abs(S2/s2) )
        dv_perp2_MC.append(     abs(S3/s2) )
        dv_par3_MC.append(      abs(S4/s3) )
        dv_par_perp2_MC.append( abs(S5/s3) )

        dv_par_TH.append(       abs(T1/s1) )
        dv_par2_TH.append(      abs(T2/s2) )
        dv_perp2_TH.append(     abs(T3/s2) )
        dv_par3_TH.append(      abs(T4/s3) )
        dv_par_perp2_TH.append( abs(T5/s3) )

    # quick sanity: show that theory 3rd-order is not identically zero
    # print(dv_par3_TH[:5], dv_par_perp2_TH[:5])

    plt.figure(figsize=(9.6,6.8))
    # theory (solid)
    plt.loglog(velocity_vals, dv_par_TH,       '-', lw=2.1, label=r"theory  $\langle \Delta v_\parallel\rangle$")
    plt.loglog(velocity_vals, dv_par2_TH,      '-', lw=2.1, label=r"theory  $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(velocity_vals, dv_perp2_TH,     '-', lw=2.1, label=r"theory  $\langle \Delta v_\perp^2\rangle$")
    # plt.loglog(velocity_vals, dv_par3_TH,      '-', lw=2.1, label=r"theory  $\langle (\Delta v_\parallel)^3\rangle$")
    # plt.loglog(velocity_vals, dv_par_perp2_TH, '-', lw=2.1, label=r"theory  $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")
    print("dv_par3_TH=", dv_par3_TH)
    print("dv_par_perp2_TH=", dv_par_perp2_TH)

    # MC (markers)
    plt.loglog(velocity_vals, dv_par_MC,       'o', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\parallel\rangle$")
    plt.loglog(velocity_vals, dv_par2_MC,      's', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(velocity_vals, dv_perp2_MC,     'D', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\perp^2\rangle$")
    plt.loglog(velocity_vals, dv_par3_MC,      '^', ms=4, alpha=0.75, label=r"MC      $\langle (\Delta v_\parallel)^3\rangle$")
    plt.loglog(velocity_vals, dv_par_perp2_MC, 'v', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    mode_text = r"$\theta_{\min}(V)=2w/V$"
    plt.xlabel(r'$v_{\rm particle}$')
    plt.ylabel(r'normalized rates ($/s_{\rm bath}^n$), absolute value')
    plt.title(r'Coulomb/grav. scattering — MC vs analytic, ' + mode_text)
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("mc_vs_theory_coulomb.png", dpi=160)
    plt.savefig("mc_vs_theory_coulomb.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

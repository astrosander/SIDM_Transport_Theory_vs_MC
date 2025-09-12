#!/usr/bin/env python3
import os, math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads

try:
    import multiprocessing as _mp
    set_num_threads(_mp.cpu_count())
except Exception:
    pass

SEED = 1
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

num_bath = 1_000_000
s_bath   = 1.5e5
v_bath   = rng.normal(0.0, s_bath, size=(num_bath, 3)).astype(np.float64)

N_VELOCITY = 60
V_MIN, V_MAX = 1.0e4, 1.0e8
velocity_vals = np.geomspace(V_MIN, V_MAX, N_VELOCITY)

m_particle = 1.0
m_bath     = 1.0
mu_red     = (m_particle*m_bath)/(m_particle+m_bath)

sigma0 = 2.0e-28
w      = 1.0e3

USE_ABSOLUTE_RATES = False
n_f = 1.0

number_encounters = 1_000_000

USE_SHORT_RANGE_KERNEL = False

@njit(fastmath=True, cache=True)
def _norm3(v0, v1, v2):
    return math.sqrt(v0*v0 + v1*v1 + v2*v2)

@njit(fastmath=True, cache=True)
def _A_B(V: float, w_: float):
    A = w_*w_ + 0.5*V*V
    B = 0.5*V*V
    return A, B

@njit(fastmath=True, cache=True)
def _logR(A: float, B: float) -> float:
    t = -2.0*B/(A + B)
    if t < -0.9999999999999998:
        t = -0.9999999999999998
    return math.log1p(t)

@njit(fastmath=True, cache=True)
def I0(V: float, w_: float) -> float:
    A,B = _A_B(V, w_)
    return 2.0/(A*A - B*B)

@njit(fastmath=True, cache=True)
def I1(V: float, w_: float) -> float:
    A,B = _A_B(V, w_)
    R = _logR(A,B)
    return (A/(B*B))*(1.0/(A-B) - 1.0/(A+B)) + R/(B*B)

@njit(fastmath=True, cache=True)
def I2(V: float, w_: float) -> float:
    A,B = _A_B(V, w_)
    R = _logR(A,B)
    return (A*A/(B**3))*(1.0/(A-B) - 1.0/(A+B)) + (2.0*A/(B**3))*R + 2.0/(B*B)

@njit(fastmath=True, cache=True)
def I3(V: float, w_: float) -> float:
    A,B = _A_B(V, w_)
    R = _logR(A,B)

    return ( (2.0*A*A*A*B)/(A*A - B*B) + 3.0*A*A*R + 4.0*A*B ) / (B**4)

@njit(fastmath=True, cache=True)
def C1C2C3(V: float, w_: float):
    if 0.5*V*V < 1e-20:
        return 0.0, 1.0/3.0, 0.0
    I0v = I0(V, w_)
    return I1(V, w_)/I0v, I2(V, w_)/I0v, I3(V, w_)/I0v

@njit(fastmath=True, cache=True)
def Y2_T1_K1(V: float, w_: float):
    I0v = I0(V, w_)
    I1v = I1(V, w_)
    I2v = I2(V, w_)
    I3v = I3(V, w_)
    Y2 = (I2v - 2.0*I1v + I0v) / I0v
    T1 = (I3v - 3.0*I2v + 3.0*I1v - I0v) / I0v
    K1 = (I1v + I2v - I3v - I0v) / I0v
    return Y2, T1, K1

@njit(fastmath=True, cache=True)
def sigma_tot_coulomb(V: float, sigma0_: float, w_: float) -> float:
    if USE_SHORT_RANGE_KERNEL:
        z = V/w_
        return sigma0_ / ((1.0 + z*z)*(1.0 + z*z))
    else:
        return sigma0_ * I0(V, w_)

@njit(fastmath=True, cache=True)
def sample_cos_from_coulomb(V: float, w_: float) -> float:
    A,B = _A_B(V, w_)
    u = np.random.random()
    ABp = (A + B)
    ABm = (A - B) if (A - B) > 1e-300 else 1e-300
    inv_target = (1.0/ABp) + u*((1.0/ABm) - (1.0/ABp))
    inv = 1.0 / inv_target
    x = (A - inv)/B
    if x < -1.0: x = -1.0
    elif x > 1.0: x = 1.0
    return x

@njit(fastmath=True, cache=True, parallel=True)
def compute_moments_MC(particle_velocity: float,
                       v_bath_: np.ndarray,
                       number_encounters_: int,
                       sigma0_: float, w_: float,
                       use_abs_rates: bool, n_f_: float,
                       m_part_: float, mu_red_: float):
    S1=S2=S3=S4=S5 = 0.0

    for _ in prange(number_encounters_):
        idx = np.random.randint(0, v_bath_.shape[0])
        vb0 = v_bath_[idx, 0]
        vb1 = v_bath_[idx, 1]
        vb2 = v_bath_[idx, 2]

        g0 = particle_velocity - vb0
        g1 = -vb1
        g2 = -vb2
        V  = math.sqrt(g0*g0 + g1*g1 + g2*g2)
        if V <= 0.0:
            continue
        gh0 = g0 / V; gh1 = g1 / V; gh2 = g2 / V

        cos_theta = sample_cos_from_coulomb(V, w_)
        s2 = 1.0 - cos_theta*cos_theta
        sin_theta = math.sqrt(s2) if s2 > 0.0 else 0.0
        phi = 2.0 * math.pi * np.random.random()
        cphi = math.cos(phi); sphi = math.sin(phi)

        if abs(gh2) < 0.9:
            ref0,ref1,ref2 = 0.0, 0.0, 1.0
        else:
            ref0,ref1,ref2 = 0.0, 1.0, 0.0
        e10 = gh1*ref2 - gh2*ref1
        e11 = gh2*ref0 - gh0*ref2
        e12 = gh0*ref1 - gh1*ref0
        n1 = math.sqrt(e10*e10 + e11*e11 + e12*e12)
        if n1 < 1e-15:
            e10,e11,e12 = 1.0,0.0,0.0
            n1 = 1.0
        e10,e11,e12 = e10/n1, e11/n1, e12/n1
        e20 = gh1*e12 - gh2*e11
        e21 = gh2*e10 - gh0*e12
        e22 = gh0*e11 - gh1*e10
        n2 = math.sqrt(e20*e20 + e21*e21 + e22*e22)
        e20,e21,e22 = e20/n2, e21/n2, e22/n2

        nh0 = sin_theta*cphi*e10 + sin_theta*sphi*e20 + cos_theta*gh0
        nh1 = sin_theta*cphi*e11 + sin_theta*sphi*e21 + cos_theta*gh1
        nh2 = sin_theta*cphi*e12 + sin_theta*sphi*e22 + cos_theta*gh2

        fac  = (mu_red_/m_part_)
        dv0  = fac * (V*nh0 - g0)
        dv1  = fac * (V*nh1 - g1)
        dv2  = fac * (V*nh2 - g2)
        dv_par   = dv0
        dv_perp2 = dv1*dv1 + dv2*dv2

        sigma_total = sigma_tot_coulomb(V, sigma0_, w_)
        weight = V * sigma_total * (n_f_ if use_abs_rates else 1.0)

        S1 += dv_par * weight
        S2 += (dv_par*dv_par) * weight
        S3 += dv_perp2 * weight
        S4 += (dv_par*dv_par*dv_par) * weight
        S5 += dv_par * dv_perp2 * weight

    invN = 1.0/number_encounters_
    return S1*invN, S2*invN, S3*invN, S4*invN, S5*invN

@njit(fastmath=True, cache=True)
def compute_moments_THEORY(particle_velocity: float,
                           v_bath_: np.ndarray,
                           sigma0_: float, w_: float,
                           use_abs_rates: bool, n_f_: float,
                           m_part_: float, mu_red_: float):
    S1=S2=S3=S4=S5 = 0.0
    c1=c2=c3=c4=c5 = 0.0  
    N = v_bath_.shape[0]

    for i in range(N):
        vb0 = v_bath_[i,0]; vb1 = v_bath_[i,1]; vb2 = v_bath_[i,2]
        g0 = particle_velocity - vb0
        g1 = -vb1; g2 = -vb2
        V  = math.sqrt(g0*g0 + g1*g1 + g2*g2)
        if V <= 0.0:
            continue
        mu_e = g0 / V

        C1, C2, C3 = C1C2C3(V, w_)
        Y2, T1, K1 = Y2_T1_K1(V, w_)

        a_perp = 0.5*(1.0 - C2)      
        p_para = Y2                  

        sigma_total = sigma_tot_coulomb(V, sigma0_, w_)
        weight = V * sigma_total * (n_f_ if use_abs_rates else 1.0)

        F  = (mu_red_/m_part_) * V
        F2 = F*F
        F3 = F2*F

        s1 = weight * (F*(C1 - 1.0)*mu_e)
        s2 = weight * (F2*( a_perp*(1.0 - mu_e*mu_e) + p_para*(mu_e*mu_e) ))
        total2 = 2.0*F2*(1.0 - C1)
        s3 = weight * ( total2 - (F2*( a_perp*(1.0 - mu_e*mu_e) + p_para*(mu_e*mu_e) )) )
        s4 = weight * ( F3*( mu_e*mu_e*mu_e*T1 + 1.5*mu_e*(1.0 - mu_e*mu_e)*K1 ) )
        mix_first = weight * (-2.0*F3*mu_e*p_para)
        s5 = mix_first - s4

        y = s1 - c1; t = S1 + y; c1 = (t - S1) - y; S1 = t
        y = s2 - c2; t = S2 + y; c2 = (t - S2) - y; S2 = t
        y = s3 - c3; t = S3 + y; c3 = (t - S3) - y; S3 = t
        y = s4 - c4; t = S4 + y; c4 = (t - S4) - y; S4 = t
        y = s5 - c5; t = S5 + y; c5 = (t - S5) - y; S5 = t

    invN = 1.0/N
    return S1*invN, S2*invN, S3*invN, S4*invN, S5*invN

def main():
    dv_par_MC=[]; dv_par2_MC=[]; dv_perp2_MC=[]; dv_par3_MC=[]; dv_par_perp2_MC=[]
    dv_par_TH=[]; dv_par2_TH=[]; dv_perp2_TH=[]; dv_par3_TH=[]; dv_par_perp2_TH=[]

    print("Kernel:", "short-range" if USE_SHORT_RANGE_KERNEL else "screened Coulomb/grav")
    print(f"Computing {len(velocity_vals)} bins …")

    for vp in velocity_vals:
        S1,S2,S3,S4,S5 = compute_moments_MC(
            float(vp), v_bath, int(number_encounters),
            float(sigma0), float(w),
            bool(USE_ABSOLUTE_RATES), float(n_f),
            float(m_particle), float(mu_red)
        )
        T1,T2,T3,T4,T5 = compute_moments_THEORY(
            float(vp), v_bath,
            float(sigma0), float(w),
            bool(USE_ABSOLUTE_RATES), float(n_f),
            float(m_particle), float(mu_red)
        )

        s1 = s_bath
        s2 = s_bath**2
        s3 = s_bath**3

        dv_par_MC.append(       S1/s1 )
        dv_par2_MC.append(      S2/s2 )
        dv_perp2_MC.append(     S3/s2 )
        dv_par3_MC.append(      S4/s3 )
        dv_par_perp2_MC.append( S5/s3 )

        dv_par_TH.append(       T1/s1 )
        dv_par2_TH.append(      T2/s2 )
        dv_perp2_TH.append(     T3/s2 )
        dv_par3_TH.append(      T4/s3 )
        dv_par_perp2_TH.append( T5/s3 )

    # Sanity: show whether the two 3rd-order curves line up (they should be close)
    print("dv_par3_TH[0:5]      =", dv_par3_TH[:5])
    print("dv_par_perp2_TH[0:5] =", dv_par_perp2_TH[:5])

    plt.figure(figsize=(9.0,6.6))

    # theory
    plt.loglog(velocity_vals, np.abs(dv_par_TH),       '-', lw=2.0, label=r"theory  $\langle \Delta v_\parallel\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_par2_TH),      '-', lw=2.0, label=r"theory  $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_perp2_TH),     '-', lw=2.0, label=r"theory  $\langle \Delta v_\perp^2\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_par3_TH),      '-', lw=2.0, label=r"theory  $\langle (\Delta v_\parallel)^3\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_par_perp2_TH), '-', lw=2.0, label=r"theory  $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    # MC
    plt.loglog(velocity_vals, np.abs(dv_par_MC),       'o', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\parallel\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_par2_MC),      's', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_perp2_MC),     'D', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\perp^2\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_par3_MC),      '^', ms=4, alpha=0.75, label=r"MC      $\langle (\Delta v_\parallel)^3\rangle$")
    plt.loglog(velocity_vals, np.abs(dv_par_perp2_MC), 'v', ms=4, alpha=0.75, label=r"MC      $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    plt.xlabel(r'$v_{\rm particle}$')
    plt.ylabel(r'per-time moments (normalized by $s_{\rm bath}^n$)')
    title = 'short-range' if USE_SHORT_RANGE_KERNEL else 'screened Coulomb/grav'
    plt.title(f'MC vs theory — {title} kernel (consistent angle + σ_tot)')
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("mc_vs_theory_coulomblike.png", dpi=160)
    plt.savefig("mc_vs_theory_coulomblike.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

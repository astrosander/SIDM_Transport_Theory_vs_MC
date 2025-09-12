#!/usr/bin/env python3
import math, os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from numba import njit, prange, set_num_threads

try:
    import multiprocessing as _mp
    set_num_threads(_mp.cpu_count())
except Exception:
    pass

SEED = 1
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

num_bath   = 1_000_000
s_bath     = 1.5e5
v_bath     = rng.normal(0.0, s_bath, size=(num_bath, 3)).astype(np.float64)
speed_bath = np.sqrt((v_bath*v_bath).sum(axis=1))

N_VELOCITY = 60
V_MIN, V_MAX = 1.0e4, 1.0e8
velocity_vals = np.geomspace(V_MIN, V_MAX, N_VELOCITY)

G          = 1.0
M_test     = 1.0
m_field    = 1.0
mu_red     = (M_test*m_field)/(M_test+m_field)

n_density  = 1.0e-6
T_window   = 1.0

number_encounters = 20_000_000

USE_ABS = True
def NORM(n):
    return (s_bath**n)

@njit(fastmath=True, cache=True)
def _norm3(x0, x1, x2):
    return math.sqrt(x0*x0 + x1*x1 + x2*x2)

@njit(fastmath=True, cache=True)
def _orthonormal_basis_from(vx, vy, vz):
    vmag = _norm3(vx, vy, vz)
    if vmag < 1e-300:
        return 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
    hx, hy, hz = vx/vmag, vy/vmag, vz/vmag
    if abs(hz) < 0.9:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = 0.0, 1.0, 0.0
    e1x = hy*rz - hz*ry
    e1y = hz*rx - hx*rz
    e1z = hx*ry - hy*rx
    n1 = _norm3(e1x, e1y, e1z)
    if n1 < 1e-300:
        e1x, e1y, e1z = 1.0, 0.0, 0.0
        n1 = 1.0
    e1x, e1y, e1z = e1x/n1, e1y/n1, e1z/n1
    e2x = hy*e1z - hz*e1y
    e2y = hz*e1x - hx*e1z
    e2z = hx*e1y - hy*e1x
    n2 = _norm3(e2x, e2y, e2z)
    e2x, e2y, e2z = e2x/n2, e2y/n2, e2z/n2
    return e1x, e1y, e1z, e2x, e2y, e2z

def henon_third_order_from_sample(V: float,
                                  speeds: np.ndarray,
                                  n_dens: float,
                                  G: float,
                                  M: float,
                                  m: float,
                                  T: float) -> Tuple[float, float]:
    v = speeds
    N = v.size
    if N == 0:
        return 0.0, 0.0

    mask_in  = (v < V)
    mask_out = ~mask_in

    I2 = n_dens * np.mean((v[mask_in]**2)) if mask_in.any() else 0.0
    I4 = n_dens * np.mean((v[mask_in]**4)) if mask_in.any() else 0.0
    I6 = n_dens * np.mean((v[mask_in]**6)) if mask_in.any() else 0.0
    J1 = n_dens * np.mean((v[mask_out]))   if mask_out.any() else 0.0

    C13 = (32.0 * math.pi**2 * G*G * m*m*m * T) / (M + m)
    inner_13 = 0.5*I2 - (I6 / (10.0*V**4) if V > 0 else 0.0)
    tail_13  = (2.0*V/5.0) * J1
    dVx3 = - C13 * (inner_13 + tail_13)

    C14 = (16.0 * math.pi**2 * G*G * m*m*m * T) / (M + m)
    inner_14 = 0.5*I2 - (I4 / (3.0*V*V) if V > 0 else 0.0) + (I6 / (10.0*V**4) if V > 0 else 0.0)
    tail_14  = (4.0*V/15.0) * J1
    dVxVy2 = - C14 * (inner_14 + tail_14)

    return dVx3, dVxVy2

@njit(fastmath=True, cache=True, parallel=True)
def mc_moments_gravity(V_particle: float,
                       v_bath_: np.ndarray,
                       n_dens: float,
                       G_: float,
                       M_: float,
                       m_: float,
                       T_: float,
                       n_enc: int):
    S1 = 0.0
    S2 = 0.0
    S3 = 0.0
    S4 = 0.0
    S5 = 0.0

    ex0, ex1, ex2 = 1.0, 0.0, 0.0

    Nbath = v_bath_.shape[0]

    for _ in prange(n_enc):
        idx = np.random.randint(0, Nbath)
        vb0 = v_bath_[idx, 0]
        vb1 = v_bath_[idx, 1]
        vb2 = v_bath_[idx, 2]

        w0 = vb0 - V_particle
        w1 = vb1
        w2 = vb2
        w  = _norm3(w0, w1, w2)
        if w <= 0.0:
            continue

        b90 = G_*(M_+m_)/(w*w)
        bmin = b90
        bmax = w * T_
        if bmax <= bmin*(1.0 + 1e-12):
            continue

        u = np.random.random()
        b2 = bmin*bmin + (bmax*bmax - bmin*bmin)*u
        b  = math.sqrt(b2)
        phi = 2.0*math.pi*np.random.random()

        e1x,e1y,e1z, e2x,e2y,e2z = _orthonormal_basis_from(w0, w1, w2)
        bx = math.cos(phi)*e1x + math.sin(phi)*e2x
        by = math.cos(phi)*e1y + math.sin(phi)*e2y
        bz = math.cos(phi)*e1z + math.sin(phi)*e2z
        wx = w0/w; wy = w1/w; wz = w2/w

        beta = 2.0*math.atan(b90 / b)
        cb   = math.cos(beta)
        sb   = math.sin(beta)

        wpx = w*(cb*wx + sb*bx)
        wpy = w*(cb*wy + sb*by)
        wpz = w*(cb*wz + sb*bz)

        dWx = wpx - w0
        dWy = wpy - w1
        dWz = wpz - w2

        fac = - (m_ / (M_ + m_))
        dVx = fac * dWx
        dVy = fac * dWy
        dVz = fac * dWz

        dv_par   = dVx
        dv_perp2 = dVy*dVy + dVz*dVz

        area = math.pi*(bmax*bmax - bmin*bmin)
        weight = n_dens * w * area / n_enc

        S1 += weight * (dv_par)
        S2 += weight * (dv_par*dv_par)
        S3 += weight * (dv_perp2)
        S4 += weight * (dv_par*dv_par*dv_par)
        S5 += weight * (dv_par*dv_perp2)

    return S1, S2, S3, S4, S5

def main():
    print("Hénon (1960) gravitational model — MC (exact two-body) vs 3rd-order analytics")
    print(f"Bath: N={num_bath:,}, s_bath={s_bath:g}, density n={n_density:g}, T={T_window:g}")
    print(f"Encounters per V: {number_encounters:,}")

    dv1_MC=[]; dv2par_MC=[]; dv2perp_MC=[]; dv3par_MC=[]; dv1perp2_MC=[]
    dv3par_TH=[]; dv1perp2_TH=[]

    for V in velocity_vals:
        S1,S2,S3,S4,S5 = mc_moments_gravity(float(V), v_bath,
                                            float(n_density),
                                            float(G),
                                            float(M_test),
                                            float(m_field),
                                            float(T_window),
                                            int(number_encounters))

        s1 = NORM(1); s2 = NORM(2); s3 = NORM(3)
        dv1_MC.append(       (abs(S1) if USE_ABS else S1) / s1 )
        dv2par_MC.append(    (abs(S2) if USE_ABS else S2) / s2 )
        dv2perp_MC.append(   (abs(S3) if USE_ABS else S3) / s2 )
        dv3par_MC.append(    (abs(S4) if USE_ABS else S4) / s3 )
        dv1perp2_MC.append(  (abs(S5) if USE_ABS else S5) / s3 )

        T13, T14 = henon_third_order_from_sample(float(V),
                                                 speed_bath,
                                                 float(n_density),
                                                 float(G),
                                                 float(M_test),
                                                 float(m_field),
                                                 float(T_window))
        dv3par_TH.append(    (abs(T13) if USE_ABS else T13) / s3 )
        dv1perp2_TH.append(  (abs(T14) if USE_ABS else T14) / s3 )

    plt.figure(figsize=(9.5,6.8))

    plt.loglog(velocity_vals, dv3par_TH,    '-', lw=2.2, label=r"theory  $\langle(\Delta v_\parallel)^3\rangle$ (Hénon)")
    plt.loglog(velocity_vals, dv1perp2_TH,  '-', lw=2.2, label=r"theory  $\langle\Delta v_\parallel\,\Delta v_\perp^2\rangle$ (Hénon)")

    plt.loglog(velocity_vals, dv1_MC,       'o', ms=4, alpha=0.80, label=r"MC      $\langle \Delta v_\parallel\rangle$")
    plt.loglog(velocity_vals, dv2par_MC,    's', ms=4, alpha=0.80, label=r"MC      $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(velocity_vals, dv2perp_MC,   'D', ms=4, alpha=0.80, label=r"MC      $\langle \Delta v_\perp^2\rangle$")
    plt.loglog(velocity_vals, dv3par_MC,    '^', ms=4, alpha=0.85, label=r"MC      $\langle (\Delta v_\parallel)^3\rangle$")
    plt.loglog(velocity_vals, dv1perp2_MC,  'v', ms=4, alpha=0.85, label=r"MC      $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    plt.xlabel(r'$V_{\rm test}$')
    plt.ylabel(r'per-time moments (normalized by $s_{\rm bath}^n$)')
    plt.title('Gravitational (unscreened) scattering — MC (exact 2-body) vs Hénon (1960) 3rd-order')
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("henon_gravity_mc_vs_theory.png", dpi=160)
    plt.savefig("henon_gravity_mc_vs_theory.pdf", dpi=160)
    plt.show()

    print("Sample theory values (3rd order):")
    print("  dv3par_TH[:5]   =", [float(x) for x in dv3par_TH[:5]])
    print("  dv1perp2_TH[:5] =", [float(x) for x in dv1perp2_TH[:5]])

if __name__ == "__main__":
    main()

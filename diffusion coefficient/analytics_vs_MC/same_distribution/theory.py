#!/usr/bin/env python3
import numpy as np
import math
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

num_bath   = 1_000_000
s_bath     = 1.5e5
v_bath     = rng.normal(0.0, s_bath, size=(num_bath, 3)).astype(np.float64)

num_targets = 4_000
v_targets   = rng.normal(0.0, s_bath, size=(num_targets, 3)).astype(np.float64)

encounters_per_target = 200_000

sigma0 = 2.0e-28
w      = 1.0e3

m_particle = 1.0
m_bath     = 1.0
mu_red     = (m_particle*m_bath)/(m_particle+m_bath)

USE_ABSOLUTE_RATES = False
n_f = 1.0

N_VELOCITY = 50
V_MIN, V_MAX = 1.0e4, 1.0e8
velocity_bin_edges = np.geomspace(V_MIN, V_MAX, N_VELOCITY+1)
velocity_bin_centers = np.sqrt(velocity_bin_edges[:-1]*velocity_bin_edges[1:])

@njit(fastmath=True, cache=True)
def _norm3(vx, vy, vz):
    return math.sqrt(vx*vx + vy*vy + vz*vz)

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
    n1  = _norm3(e1x, e1y, e1z)
    if n1 < 1e-300:
        e1x, e1y, e1z = 1.0, 0.0, 0.0
        n1 = 1.0
    e1x, e1y, e1z = e1x/n1, e1y/n1, e1z/n1
    e2x = hy*e1z - hz*e1y
    e2y = hz*e1x - hx*e1z
    e2z = hx*e1y - hy*e1x
    n2  = _norm3(e2x, e2y, e2z)
    e2x, e2y, e2z = e2x/n2, e2y/n2, e2z/n2
    return e1x, e1y, e1z, e2x, e2y, e2z

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
def sigma_tot_rutherford_scalar(V: float, sigma0_: float, w_: float) -> float:
    return sigma0_ / (1.0 + (V*V)/(w_*w_))

@njit(fastmath=True, cache=True)
def sample_cos_rutherford_scalar(V: float, w_: float) -> float:
    A = w_*w_ + 0.5*V*V
    B = 0.5*V*V
    u = np.random.random()
    ABp = (A + B)
    ABm = (A - B) if (A - B) > 1e-300 else 1e-300
    inv = 1.0 / ((1.0/ABp) + (2.0*B/(ABp*ABm))*u)
    x = (A - inv)/B
    if x < -1.0: x = -1.0
    elif x > 1.0: x = 1.0
    return x

@njit(fastmath=True, cache=True, parallel=True)
def mc_moments_rutherford_same_dist(v_target: np.ndarray,
                                    v_bath_: np.ndarray,
                                    n_enc: int,
                                    sigma0_: float, w_: float,
                                    use_abs_rates: bool, n_f_: float,
                                    m_part_: float, mu_red_: float):
    S1=S2=S3=S4=S5 = 0.0

    vt0, vt1, vt2 = v_target[0], v_target[1], v_target[2]
    Vt = _norm3(vt0, vt1, vt2)
    if Vt > 0.0:
        ep0, ep1, ep2 = vt0/Vt, vt1/Vt, vt2/Vt
    else:
        ep0, ep1, ep2 = 1.0, 0.0, 0.0

    Nbath = v_bath_.shape[0]

    for _ in prange(n_enc):
        j = np.random.randint(0, Nbath)
        vb0 = v_bath_[j,0]
        vb1 = v_bath_[j,1]
        vb2 = v_bath_[j,2]

        g0 = vt0 - vb0
        g1 = vt1 - vb1
        g2 = vt2 - vb2
        V  = _norm3(g0, g1, g2)
        if V <= 0.0:
            continue

        x = sample_cos_rutherford_scalar(V, w_)
        s2 = 1.0 - x*x
        s  = math.sqrt(s2) if s2 > 0.0 else 0.0
        phi = 2.0 * math.pi * np.random.random()
        cphi = math.cos(phi)
        sphi = math.sin(phi)

        e1x,e1y,e1z, e2x,e2y,e2z = _orthonormal_basis_from(g0, g1, g2)
        ghx, ghy, ghz = g0/V, g1/V, g2/V

        nhx = s*cphi*e1x + s*sphi*e2x + x*ghx
        nhy = s*cphi*e1y + s*sphi*e2y + x*ghy
        nhz = s*cphi*e1z + s*sphi*e2z + x*ghz

        fac = (mu_red_/m_part_)
        dv0 = fac * (V*nhx - g0)
        dv1 = fac * (V*nhy - g1)
        dv2 = fac * (V*nhz - g2)

        dv_par  = dv0*ep0 + dv1*ep1 + dv2*ep2
        dv2_all = dv0*dv0 + dv1*dv1 + dv2*dv2
        dv_perp2 = dv2_all - dv_par*dv_par
        if dv_perp2 < 0.0:
            dv_perp2 = 0.0

        sigma_tot = sigma_tot_rutherford_scalar(V, sigma0_, w_)
        weight = V * sigma_tot * (n_f_ if use_abs_rates else 1.0)

        S1 += weight * (dv_par)
        S2 += weight * (dv_par*dv_par)
        S3 += weight * (dv_perp2)
        S4 += weight * (dv_par*dv_par*dv_par)
        S5 += weight * (dv_par*dv_perp2)

    invN = 1.0/max(n_enc,1)
    return S1*invN, S2*invN, S3*invN, S4*invN, S5*invN

def main():
    print("Targets and bath drawn from the SAME Maxwellian; 'parallel' = along each target's velocity.")
    print(f"num_targets={len(v_targets)}, encounters/target={encounters_per_target}")

    U1=U2=U3=U4=U5 = 0.0

    B1 = np.zeros_like(velocity_bin_centers)
    B2 = np.zeros_like(velocity_bin_centers)
    B3 = np.zeros_like(velocity_bin_centers)
    B4 = np.zeros_like(velocity_bin_centers)
    B5 = np.zeros_like(velocity_bin_centers)
    Bn = np.zeros_like(velocity_bin_centers)

    for i in range(v_targets.shape[0]):
        vt = v_targets[i]
        S1,S2,S3,S4,S5 = mc_moments_rutherford_same_dist(
            vt, v_bath,
            int(encounters_per_target),
            float(sigma0), float(w),
            bool(USE_ABSOLUTE_RATES), float(n_f),
            float(m_particle), float(mu_red)
        )
        U1 += S1; U2 += S2; U3 += S3; U4 += S4; U5 += S5

        Vmag = np.linalg.norm(vt)
        if Vmag >= V_MIN and Vmag <= V_MAX:
            j = np.searchsorted(velocity_bin_edges, Vmag) - 1
            if 0 <= j < velocity_bin_centers.size:
                B1[j] += S1; B2[j] += S2; B3[j] += S3; B4[j] += S4; B5[j] += S5; Bn[j] += 1.0

    s1 = s_bath
    s2 = s_bath**2
    s3 = s_bath**3

    U1 /= len(v_targets); U2 /= len(v_targets); U3 /= len(v_targets); U4 /= len(v_targets); U5 /= len(v_targets)
    print("\nUNCONDITIONAL ensemble (targets averaged over same Maxwellian):")
    print("  <Δv_∥>   / s =", U1/s1)
    print("  <Δv_∥^2> / s^2 =", U2/s2)
    print("  <Δv_⊥^2> / s^2 =", U3/s2)
    print("  <Δv_∥^3> / s^3 =", U4/s3)
    print("  <Δv_∥Δv_⊥^2> / s^3 =", U5/s3)

    mask = Bn > 0.5
    dv_par_bin      = np.zeros_like(B1); dv_par2_bin     = np.zeros_like(B1)
    dv_perp2_bin    = np.zeros_like(B1); dv_par3_bin     = np.zeros_like(B1)
    dv_par_perp2bin = np.zeros_like(B1)

    dv_par_bin[mask]       = (B1[mask]/Bn[mask]) / s1
    dv_par2_bin[mask]      = (B2[mask]/Bn[mask]) / s2
    dv_perp2_bin[mask]     = (B3[mask]/Bn[mask]) / s2
    dv_par3_bin[mask]      = (B4[mask]/Bn[mask]) / s3
    dv_par_perp2bin[mask]  = (B5[mask]/Bn[mask]) / s3

    plt.figure(figsize=(9.0,6.6))
    vplot = velocity_bin_centers[mask]
    plt.loglog(vplot, np.abs(dv_par_bin[mask]),         '-', lw=2.0, label=r"cond. $\langle \Delta v_\parallel\rangle$")
    plt.loglog(vplot, np.abs(dv_par2_bin[mask]),        '-', lw=2.0, label=r"cond. $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(vplot, np.abs(dv_perp2_bin[mask]),       '-', lw=2.0, label=r"cond. $\langle \Delta v_\perp^2\rangle$")
    plt.loglog(vplot, np.abs(dv_par3_bin[mask]),        '-', lw=2.0, label=r"cond. $\langle (\Delta v_\parallel)^3\rangle$")
    plt.loglog(vplot, np.abs(dv_par_perp2bin[mask]),    '-', lw=2.0, label=r"cond. $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    plt.xlabel(r'$|V_{\rm target}|$')
    plt.ylabel(r'per-time moments (normalized by $s_{\rm bath}^n$)')
    plt.title('Same-distribution target & bath (Rutherford kernel): conditional moments vs |V|')
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("same_dist_rutherford_conditional.png", dpi=160)
    plt.savefig("same_dist_rutherford_conditional.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt

SEED        = 1
np.random.seed(SEED)

G           = 1.0
m_particle  = 1.0
M_field     = 1.0
T_window    = 1.0
n_density   = 1.0

s_bath      = 1.5e5
V_MIN, V_MAX = 1.0e4, 1.0e10
N_VELOCITY  = 60
velocity_vals = np.geomspace(V_MIN, V_MAX, N_VELOCITY)

N_MC_SPEED  = 1_000_000

USE_ABS = True

def sample_maxwell_speeds(nsamples: int, s: float, rng=np.random):
    v = rng.normal(0.0, s, size=(nsamples, 3))
    return np.sqrt(np.sum(v*v, axis=1))

def a_of_v(v: np.ndarray, s: float, n: float) -> np.ndarray:
    c = n * 4.0 * math.pi / ((2.0 * math.pi * s * s) ** 1.5)
    return c * np.exp(-(v*v) / (2.0 * s * s))

def henon_prefactors(G, m, M, T):
    Cx3 = - (32.0 * (math.pi**2) * (G**2) * (m**3) * T) / (M + m)
    Cx_yy = - (16.0 * (math.pi**2) * (G**2) * (m**3) * T) / (M + m)
    return Cx3, Cx_yy

def make_cumulants_for_a(s: float, n: float, vmax: float, ngrid: int = 40000):
    v = np.linspace(0.0, vmax, ngrid)
    dv = v[1] - v[0]
    av = a_of_v(v, s, n)

    v2 = v*v
    v4 = v2*v2
    v6 = v4*v2

    def cumtrap(y):
        c = np.empty_like(y)
        c[0] = 0.0
        c[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * dv)
        return c

    J2 = cumtrap(av * v2)
    J4 = cumtrap(av * v4)
    J6 = cumtrap(av * v6)
    L1 = cumtrap(av * v)

    J2_tot = J2[-1]
    J4_tot = J4[-1]
    J6_tot = J6[-1]
    L1_tot = L1[-1]

    return dict(v=v, J2=J2, J4=J4, J6=J6, L1=L1,
                J2_tot=J2_tot, J4_tot=J4_tot, J6_tot=J6_tot, L1_tot=L1_tot)

def interp_cum(cum_v, cum_y, x):
    v = cum_v
    y = cum_y
    if x <= v[0]:
        return 0.0
    if x >= v[-1]:
        return float(y[-1])
    i = np.searchsorted(v, x)
    x0, x1 = v[i-1], v[i]
    y0, y1 = y[i-1], y[i]
    t = (x - x0)/(x1 - x0)
    return float((1.0 - t)*y0 + t*y1)

def theory_curves_henon(Vs, s, n, G, m, M, T):
    vmax = max(10.0*s, Vs.max()*0.0 + 10.0*s)
    C = make_cumulants_for_a(s, n, vmax=vmax, ngrid=50000)
    vgrid = C['v']
    J2, J4, J6, L1 = C['J2'], C['J4'], C['J6'], C['L1']
    L1_tot = C['L1_tot']

    Cx3, Cx_yy = henon_prefactors(G, m, M, T)

    th_x3 = []
    th_x_yy = []
    for V in Vs:
        J2_V = interp_cum(vgrid, J2, V)
        J4_V = interp_cum(vgrid, J4, V)
        J6_V = interp_cum(vgrid, J6, V)
        L1_V = interp_cum(vgrid, L1, V)

        tail_L1 = L1_tot - L1_V

        term_inner_13 = 0.5*J2_V - (1.0/(10.0*V**4))*J6_V
        term_tail_13  = (2.0*V/5.0)*tail_L1
        M3 = Cx3 * (term_inner_13 + term_tail_13)

        term_inner_14 = (0.5*J2_V) - (1.0/(3.0*V*V))*J4_V + (1.0/(10.0*V**4))*J6_V
        term_tail_14  = (4.0*V/15.0)*tail_L1
        Mx_yy = Cx_yy * (term_inner_14 + term_tail_14)

        th_x3.append(M3)
        th_x_yy.append(Mx_yy)

    return np.array(th_x3), np.array(th_x_yy)

def mc_curves_henon(Vs, s, n, G, m, M, T, Nmc):
    v = sample_maxwell_speeds(Nmc, s)
    v2 = v*v
    v4 = v2*v2

    Cx3, Cx_yy = henon_prefactors(G, m, M, T)

    mc_x3 = []
    mc_x_yy = []

    for V in Vs:
        m_le = (v <= V)
        m_gt = ~m_le

        f13 = np.empty_like(v)
        f13[m_le] = 0.5 - (v4[m_le] / (10.0 * V**4))
        f13[m_gt] = (2.0*V) / (5.0 * v[m_gt])

        f14 = np.empty_like(v)
        f14[m_le] = 0.5 - (v2[m_le] / (3.0 * V*V)) + (v4[m_le] / (10.0 * V**4))
        f14[m_gt] = (4.0*V) / (15.0 * v[m_gt])

        Ef13 = np.mean(f13)
        Ef14 = np.mean(f14)

        I13 = n * Ef13
        I14 = n * Ef14

        mc_x3.append(Cx3 * I13)
        mc_x_yy.append(Cx_yy * I14)

    return np.array(mc_x3), np.array(mc_x_yy)

def main():
    print("Hénon (1960) test: 3rd-order moments vs MC (speed sampling)")

    th_x3, th_x_yy = theory_curves_henon(velocity_vals, s_bath, n_density,
                                         G, m_particle, M_field, T_window)
    mc_x3, mc_x_yy = mc_curves_henon(velocity_vals, s_bath, n_density,
                                     G, m_particle, M_field, T_window, N_MC_SPEED)

    rel_err_x3  = np.max(np.abs(mc_x3 - th_x3) / np.maximum(1e-300, np.abs(th_x3)))
    rel_err_xy2 = np.max(np.abs(mc_x_yy - th_x_yy) / np.maximum(1e-300, np.abs(th_x_yy)))
    print(f"max relative error  <ΔVx^3>         : {rel_err_x3:.3e}")
    print(f"max relative error  <ΔVx ΔVy^2>     : {rel_err_xy2:.3e}")

    y1 = np.abs(th_x3) if USE_ABS else th_x3
    y2 = np.abs(th_x_yy) if USE_ABS else th_x_yy
    z1 = np.abs(mc_x3)  if USE_ABS else mc_x3
    z2 = np.abs(mc_x_yy)if USE_ABS else mc_x_yy

    plt.figure(figsize=(9.2,6.2))
    plt.loglog(velocity_vals, y1, '-',  lw=2.2, label=r"theory  $\langle \Delta V_x^3\rangle$")
    plt.loglog(velocity_vals, y2, '-',  lw=2.2, label=r"theory  $\langle \Delta V_x\,\Delta V_y^2\rangle$")
    plt.loglog(velocity_vals, z1, 'o',  ms=4,  alpha=0.75, label=r"MC      $\langle \Delta V_x^3\rangle$")
    plt.loglog(velocity_vals, z2, 's',  ms=4,  alpha=0.75, label=r"MC      $\langle \Delta V_x\,\Delta V_y^2\rangle$")
    plt.xlabel(r'$V$  (test-star speed)')
    plt.ylabel(r'moment over time $T$')
    plt.title(r"Hénon (1960): third-order moments — MC (speed sampling) vs analytics")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("henon_mc_vs_theory_third_order.png", dpi=160)
    plt.savefig("henon_mc_vs_theory_third_order.pdf", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()

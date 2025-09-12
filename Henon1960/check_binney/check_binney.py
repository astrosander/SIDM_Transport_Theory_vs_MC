#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

G        = 1.0
M_test   = 1.0
m_field  = 1.0
s_bath   = 1.5e5
sigma0   = 2.0e-28
w        = 1.0e3

N_V      = 120
V_MIN, V_MAX = 1.0e4, 1.0e8
v_vals   = np.geomspace(V_MIN, V_MAX, N_V)

va_max   = 12.0 * s_bath
N_INT    = 20000
va       = np.linspace(0.0, va_max, N_INT)

norm = 4.0 * np.pi / ((2.0*np.pi*s_bath**2)**1.5)
fa   = norm * va**2 * np.exp(-0.5*(va/s_bath)**2)

cum_v1 = cumtrapz(va     * fa, va, initial=0.0)
cum_v2 = cumtrapz(va**2 * fa, va, initial=0.0)
cum_v4 = cumtrapz(va**4 * fa, va, initial=0.0)
cum_v6 = cumtrapz(va**6 * fa, va, initial=0.0)
I1_tot, I2_tot, I4_tot, I6_tot = cum_v1[-1], cum_v2[-1], cum_v4[-1], cum_v6[-1]

def cum_at(x, xgrid, cum):
    xx = np.clip(x, xgrid[0], xgrid[-1])
    return np.interp(xx, xgrid, cum)

def Lambda(v):
    Lam = (2.0 * w**3 / (v**2)) * np.sqrt(sigma0 / (4.0*np.pi))
    return np.maximum(Lam, 1.0 + 1e-12)

def lnLambda(v): return np.log(Lambda(v))

A1 = -16.0 * np.pi**2 * G**2 * m_field * (M_test + m_field)
A2 =  32.0 * np.pi**2 * G**2 * m_field**2 / 3.0

P13 = -32.0 * np.pi**2 * G**2 * m_field**3 / (M_test + m_field)
P14 = -16.0 * np.pi**2 * G**2 * m_field**3 / (M_test + m_field)

D_dvpar   = np.zeros_like(v_vals)
D_par2    = np.zeros_like(v_vals)
D_perp2   = np.zeros_like(v_vals)
M3_x      = np.zeros_like(v_vals)
M_mix     = np.zeros_like(v_vals)

for i, v in enumerate(v_vals):
    lnL = lnLambda(v)

    Iv1_lo = cum_at(v, va, cum_v1)
    Iv2_lo = cum_at(v, va, cum_v2)
    Iv4_lo = cum_at(v, va, cum_v4)
    Iv6_lo = cum_at(v, va, cum_v6)
    Iv1_hi = I1_tot - Iv1_lo

    D_dvpar[i] = A1 * lnL * Iv2_lo / (v**2)

    D_par2[i]  = A2 * lnL * (Iv4_lo / (v**3) + Iv1_hi)

    D_perp2[i] = A2 * lnL * ((3.0 * Iv2_lo / v) - (Iv4_lo / (v**3)) + 2.0 * Iv1_hi)

    bracket13  = (0.5 * Iv2_lo) - (Iv6_lo / (10.0 * v**4)) + (2.0 * v / 5.0) * Iv1_hi
    M3_x[i]    = P13 * bracket13

    bracket14  = (0.5 * Iv2_lo) - (Iv4_lo / (3.0 * v**2)) + (Iv6_lo / (10.0 * v**4)) \
                 + (4.0 * v / 15.0) * Iv1_hi
    M_mix[i]   = P14 * bracket14

y1 = np.abs(D_dvpar   / s_bath)
y2 = np.abs(D_par2    / (s_bath**2))
y3 = np.abs(D_perp2   / (s_bath**2))
y4 = np.abs(M3_x      / (s_bath**3))
y5 = np.abs(M_mix     / (s_bath**3))

plt.figure(figsize=(9.2,6.4))
plt.loglog(v_vals, y1, lw=2.2, label=r'$|D[\Delta v_\parallel]|/s_{\rm bath}$')
plt.loglog(v_vals, y2, lw=2.2, label=r'$D[(\Delta v_\parallel)^2]/s_{\rm bath}^2$')
plt.loglog(v_vals, y3, lw=2.2, label=r'$D[(\Delta \mathbf{v}_\perp)^2]/s_{\rm bath}^2$')
plt.loglog(v_vals, y4, lw=2.2, label=r'$|\langle \Delta V_x^3\rangle|/s_{\rm bath}^3$')
plt.loglog(v_vals, y5, lw=2.2, label=r'$|\langle \Delta V_x \Delta V_y^2\rangle|/s_{\rm bath}^3$')

plt.xlabel(r'$v$')
plt.ylabel('normalized rates')
# plt.title(r'Gravitational (Coulomb) transport: drift, variances, and 3rd-order (Hénon)')
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()

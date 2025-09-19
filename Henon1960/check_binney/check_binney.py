import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

G        = 1.0
M_test   = 1.0
m_field  = 1.0
s_bath   = 1.5e5
w        = 1.0e3

N_V      = 140
V_MIN, V_MAX = 1.0e4, 1.0e8
V_vals   = np.geomspace(V_MIN, V_MAX, N_V)

va_max   = 12.0 * s_bath
N_INT    = 20000
va       = np.linspace(0.0, va_max, N_INT)

norm = 4.0 * np.pi / ((2.0*np.pi*s_bath**2)**1.5)
a_of_v = norm * np.exp(-0.5*(va/s_bath)**2)

cum_I1 = cumtrapz(va**1 * a_of_v, va, initial=0.0)
cum_I2 = cumtrapz(va**2 * a_of_v, va, initial=0.0)
cum_I4 = cumtrapz(va**4 * a_of_v, va, initial=0.0)
cum_I6 = cumtrapz(va**6 * a_of_v, va, initial=0.0)
I1_tot, I2_tot, I4_tot, I6_tot = cum_I1[-1], cum_I2[-1], cum_I4[-1], cum_I6[-1]

def cum_at(x, xgrid, cum):
    xx = np.clip(x, xgrid[0], xgrid[-1])
    return np.interp(xx, xgrid, cum)

def lnLambda(V):
    Lam = 1.0 + (2.0 * w**3) * (V**2 + (1e-30))
    return np.log(Lam)

A1 = -16.0 * np.pi**2 * G**2 * m_field * (M_test + m_field)
A2 =  32.0 * np.pi**2 * G**2 * m_field**2 / 3.0

P13 = -32.0 * np.pi**2 * G**2 * m_field**3 / (M_test + m_field)
P14 = -16.0 * np.pi**2 * G**2 * m_field**3 / (M_test + m_field)

D_dvpar   = np.zeros_like(V_vals)
D_par2    = np.zeros_like(V_vals)
D_perp2   = np.zeros_like(V_vals)
M3_x      = np.zeros_like(V_vals)
M_mix     = np.zeros_like(V_vals)

for i, V in enumerate(V_vals):
    lnL = lnLambda(V)

    I1_lo = cum_at(V, va, cum_I1)
    I2_lo = cum_at(V, va, cum_I2)
    I4_lo = cum_at(V, va, cum_I4)
    I6_lo = cum_at(V, va, cum_I6)

    I1_hi = I1_tot - I1_lo

    D_dvpar[i] = A1 * lnL * (I2_lo / (V**2))

    D_par2[i]  = A2 * lnL * ( (I4_lo / (V**3)) + I1_hi )

    D_perp2[i] = A2 * lnL * ( (3.0 * I2_lo / V) - (I4_lo / (V**3)) + 2.0 * I1_hi )

    bracket_M3x  = (0.5 * I2_lo) - (I6_lo / (10.0 * V**4)) + (2.0 * V / 5.0) * I1_hi
    M3_x[i]      = P13 * bracket_M3x

    bracket_Mmix = (0.5 * I2_lo) - (I4_lo / (3.0 * V**2)) + (I6_lo / (10.0 * V**4)) \
                   + (4.0 * V / 15.0) * I1_hi
    M_mix[i]     = P14 * bracket_Mmix

y1 = np.abs(D_dvpar   / s_bath)        
y2 = np.abs(D_par2    / (s_bath**2))   
y3 = np.abs(D_perp2   / (s_bath**2))   
y4 = np.abs(M3_x      / (s_bath**3))   
y5 = np.abs(M_mix     / (s_bath**3))

plt.figure(figsize=(9.2, 6.4))
plt.loglog(V_vals, y1, lw=2.0, label=r'$|\,\langle \Delta v_\parallel\rangle|/s_{\rm bath}$')
plt.loglog(V_vals, y2, lw=2.0, label=r'$\langle (\Delta v_\parallel)^2\rangle/s_{\rm bath}^2$')
plt.loglog(V_vals, y3, lw=2.0, label=r'$\langle (\Delta \mathbf{v}_\perp)^2\rangle/s_{\rm bath}^2$')
plt.loglog(V_vals, y4, lw=2.0, label=r'$|\,\langle \Delta V_x^3\rangle|/s_{\rm bath}^3$')
plt.loglog(V_vals, y5, lw=2.0, label=r'$|\,\langle \Delta V_x \Delta V_y^2\rangle|/s_{\rm bath}^3$')
plt.xlabel(r'$V$')
plt.ylabel('normalized rates')
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("binney_henon.png", dpi=150)
plt.savefig("binney_henon.pdf")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

# --- Constants and base parameters (kept symbolic/normalized like the user's script) ---
G        = 1.0
M_test   = 1.0
m_field  = 1.0
w        = 1.0e3  # Coulomb log proxy parameterization (kept identical to the original code)

# --- Speed grids ---
N_V      = 600
V_MIN, V_MAX = 1.0e3, 5.0e8                  # Slightly wider range to capture more of the MB tail when s_bath grows
V_vals   = np.geomspace(V_MIN, V_MAX, N_V)

# We'll scan s_bath over a decade or so
N_SB     = 24
S_MIN, S_MAX = 5.0e4, 8.0e5
S_vals   = np.geomspace(S_MIN, S_MAX, N_SB)

# --- Helper: Coulomb log proxy ---
def lnLambda(V):
    Lam = 1.0 + (2.0 * w**3) * (V**2 + (1e-30))
    return np.log(Lam)

# --- Prefactors from the user's code (Binney/Hénon-style normalizations) ---
A1  = -16.0 * np.pi**2 * G**2 * m_field * (M_test + m_field)
A2  =  32.0 * np.pi**2 * G**2 * m_field**2 / 3.0
P13 = -32.0 * np.pi**2 * G**2 * m_field**3 / (M_test + m_field)
P14 = -16.0 * np.pi**2 * G**2 * m_field**3 / (M_test + m_field)

# --- Utility: cumulative integral interpolator ---
def cum_at(x, xgrid, cum):
    xx = np.clip(x, xgrid[0], xgrid[-1])
    return np.interp(xx, xgrid, cum)

# --- Core routine: for a given s_bath, compute field-star integrals, then V-dependent transport coeffs,
#     then Maxwell–Boltzmann-average over test-star speeds with the same s_bath. ---
def compute_mb_averaged_moments_vs_s(s_bath, V_vals):
    # Field-star speed grid and Maxwell–Boltzmann a(v)
    va_max = 12.0 * s_bath
    N_INT  = 40000
    va     = np.linspace(0.0, va_max, N_INT)

    # 3D speed PDF for field particles
    norm = 4.0 * np.pi / ((2.0*np.pi*s_bath**2)**1.5)
    a_of_v = norm * np.exp(-0.5*(va/s_bath)**2)

    # Precompute cumulative integrals over field-star PDF
    cum_I1 = cumtrapz(va**1 * a_of_v, va, initial=0.0)
    cum_I2 = cumtrapz(va**2 * a_of_v, va, initial=0.0)
    cum_I4 = cumtrapz(va**4 * a_of_v, va, initial=0.0)
    cum_I6 = cumtrapz(va**6 * a_of_v, va, initial=0.0)
    I1_tot, I2_tot, I4_tot, I6_tot = cum_I1[-1], cum_I2[-1], cum_I4[-1], cum_I6[-1]

    # V-dependent transport coefficients (same formulas as user code)
    D_dvpar = np.zeros_like(V_vals)   # <Δv_parallel>
    D_par2  = np.zeros_like(V_vals)   # <(Δv_parallel)^2>
    D_perp2 = np.zeros_like(V_vals)   # <(Δv_perp)^2>
    M3_x    = np.zeros_like(V_vals)   # <ΔV_x^3>
    M_mix   = np.zeros_like(V_vals)   # <ΔV_x ΔV_y^2>

    for i, V in enumerate(V_vals):
        lnL = lnLambda(V)

        I1_lo = cum_at(V, va, cum_I1)
        I2_lo = cum_at(V, va, cum_I2)
        I4_lo = cum_at(V, va, cum_I4)
        I6_lo = cum_at(V, va, cum_I6)

        I1_hi = I1_tot - I1_lo

        # First and second order
        D_dvpar[i] = A1 * lnL * (I2_lo / (V**2))
        D_par2[i]  = A2 * lnL * ( (I4_lo / (V**3)) + I1_hi )
        D_perp2[i] = A2 * lnL * ( (3.0 * I2_lo / V) - (I4_lo / (V**3)) + 2.0 * I1_hi )

        # Third order
        bracket_M3x  = (0.5 * I2_lo) - (I6_lo / (10.0 * V**4)) + (2.0 * V / 5.0) * I1_hi
        M3_x[i]      = P13 * bracket_M3x

        bracket_Mmix = (0.5 * I2_lo) - (I4_lo / (3.0 * V**2)) + (I6_lo / (10.0 * V**4)) + (4.0 * V / 15.0) * I1_hi
        M_mix[i]     = P14 * bracket_Mmix

    # Maxwell–Boltzmann PDF for *test* speeds, same dispersion s_bath
    # f_MB(V) dV = 4π V^2 (2π s^2)^(-3/2) exp(-V^2 / (2 s^2)) dV
    fV = 4.0 * np.pi * V_vals**2 / ((2.0*np.pi*s_bath**2)**1.5) * np.exp(-0.5*(V_vals/s_bath)**2)

    # Normalize numerically over the V grid (geom space → integrate in log-space by trapezoid on V)
    # Use standard trapezoid on V (not logV), as fV is defined in V.
    dV = np.gradient(V_vals)
    Z  = np.trapz(fV, V_vals)  # should be ~1 over large enough V-range

    # Ensemble-averaged *signed* moments (third-order should → 0)
    avg_D_dvpar = np.trapz(D_dvpar * fV, V_vals) / Z
    avg_D_par2  = np.trapz(D_par2  * fV, V_vals) / Z
    avg_D_perp2 = np.trapz(D_perp2 * fV, V_vals) / Z
    avg_M3_x    = np.trapz(M3_x    * fV, V_vals) / Z
    avg_M_mix   = np.trapz(M_mix   * fV, V_vals) / Z

    # Return also the normalized (dimensionless) versions
    y1 = np.abs(avg_D_dvpar / s_bath)       # |<Δv_∥>| / s_bath
    y2 =       avg_D_par2  / (s_bath**2)    # <(Δv_∥)^2> / s_bath^2
    y3 =       avg_D_perp2 / (s_bath**2)    # <(Δv_⊥)^2> / s_bath^2
    # For third-order, show magnitude of the *signed* mean (should be ~0)
    y4 = np.abs(avg_M3_x   / (s_bath**3))   # |<ΔV_x^3>| / s_bath^3
    y5 = np.abs(avg_M_mix  / (s_bath**3))   # |<ΔV_x ΔV_y^2>| / s_bath^3

    return {
        "avg": (avg_D_dvpar, avg_D_par2, avg_D_perp2, avg_M3_x, avg_M_mix),
        "y":   (y1, y2, y3, y4, y5),
        "Z":   Z
    }

# --- Sweep s_bath and collect Maxwell–Boltzmann-averaged, normalized moments ---
Y1, Y2, Y3, Y4, Y5 = [], [], [], [], []
signed_M3x, signed_Mmix = [], []

for s in S_vals:
    out = compute_mb_averaged_moments_vs_s(s, V_vals)
    (avg1, avg2, avg3, avg4, avg5) = out["avg"]
    (y1, y2, y3, y4, y5) = out["y"]

    Y1.append(y1);  Y2.append(y2);  Y3.append(y3);  Y4.append(y4);  Y5.append(y5)
    signed_M3x.append(avg4 / (s**3))
    signed_Mmix.append(avg5 / (s**3))

Y1 = np.array(Y1);  Y2 = np.array(Y2);  Y3 = np.array(Y3);  Y4 = np.array(Y4);  Y5 = np.array(Y5)
signed_M3x = np.array(signed_M3x)
signed_Mmix = np.array(signed_Mmix)

# --- Plot: MB-averaged normalized moments vs s_bath ---
plt.figure(figsize=(9.6, 6.6))
plt.loglog(S_vals, Y1, lw=2.0, label=r'$|\,\langle \Delta v_\parallel\rangle|/s_{\rm bath}$')
plt.loglog(S_vals, Y2, lw=2.0, label=r'$\langle (\Delta v_\parallel)^2\rangle/s_{\rm bath}^2$')
plt.loglog(S_vals, Y3, lw=2.0, label=r'$\langle (\Delta v_\perp)^2\rangle/s_{\rm bath}^2$')
plt.loglog(S_vals, Y4, lw=2.0, label=r'$|\,\langle \Delta v_\parallel^3\rangle|/s_{\rm bath}^3$ (MB avg)')
plt.loglog(S_vals, Y5, lw=2.0, label=r'$|\,\langle \Delta v_\parallel \Delta v_\perp^2\rangle|/s_{\rm bath}^3$ (MB avg)')
plt.xlabel(r'$s_{\rm bath}$')
plt.ylabel('MB-averaged normalized rates')
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.legend(ncol=1, fontsize=9)
plt.tight_layout()
plt.savefig("mb_averaged_moments_vs_sbath.png", dpi=150)
plt.savefig("mb_averaged_moments_vs_sbath.pdf")

# --- Plot: signed third-order MB averages (should ~0) ---
plt.figure(figsize=(9.6, 6.6))
plt.semilogx(S_vals, signed_M3x, lw=2.0, label=r'$\langle \Delta v_\parallel^3\rangle/s_{\rm bath}^3$ (signed)')
plt.semilogx(S_vals, signed_Mmix, lw=2.0, label=r'$\langle \Delta v_\parallel \Delta v_\perp^2\rangle/s_{\rm bath}^3$ (signed)')
plt.axhline(0.0, lw=1.2)
plt.xlabel(r'$s_{\rm bath}$')
plt.ylabel('MB-averaged signed third-order (dimensionless)')
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("mb_signed_third_order_vs_sbath.png", dpi=150)
plt.savefig("mb_signed_third_order_vs_sbath.pdf")

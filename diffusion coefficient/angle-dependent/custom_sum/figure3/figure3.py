import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# Global RNG and parameters
# =========================
rng = np.random.default_rng(1)

NUM_BATH            = 1_000_000
NUMBER_ENCOUNTERS   = 100_000
VELOCITY_VALS       = np.linspace(0.0, 100_000_000.0, 10)
S_BATH              = 150_000.0

M_PARTICLE          = 1.0
M_BATH              = 1.0
MU                  = (M_PARTICLE * M_BATH) / (M_PARTICLE + M_BATH)

USE_ABSOLUTE_RATES  = False   # keep False to match your normalization-by s_bath^n
N_F                 = 1.0

# Yukawa (screened Rutherford) parameters
SIGMA0              = 2.0e-28
W                   = 1.0e3

# =========================
# Bath velocities
# =========================
V_BATH = rng.normal(0.0, S_BATH, size=(NUM_BATH, 3))

# =========================
# Helpers: geometry & CM update
# =========================
def ortho_basis_from(g_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if abs(g_hat[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(g_hat, ref)
    n1 = np.linalg.norm(e1)
    if n1 < 1e-15:
        e1 = np.array([1.0, 0.0, 0.0]); n1 = 1.0
    e1 /= n1
    e2 = np.cross(g_hat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2

def post_collision_equal_masses(v1: np.ndarray, v2: np.ndarray, n_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    CM-rotation update for equal masses:
      g' = |g| n_hat,  Vcm = (v1+v2)/2,  v1' = Vcm + g'/2,  v2' = Vcm - g'/2
    """
    g = v1 - v2
    Vcm = 0.5 * (v1 + v2)
    g_after = np.linalg.norm(g) * n_hat
    v1_after = Vcm + 0.5 * g_after
    v2_after = Vcm - 0.5 * g_after
    return v1_after, v2_after

# =========================
# Cross section (vectorized & scalar)
# =========================
def sigma_tot_rutherford(V: np.ndarray, sigma0: float, w: float) -> np.ndarray:
    """Vectorized total cross-section σ_tot(V) = σ0 / (1 + V^2/w^2)."""
    V = np.asarray(V, dtype=float)
    return sigma0 / (1.0 + (V*V)/(w*w))

def sigma_tot_rutherford_scalar(V: float, sigma0: float, w: float) -> float:
    """Scalar wrapper for MC loop."""
    return float(sigma_tot_rutherford(np.array([V]), sigma0, w)[0])

# =========================
# Analytic angular moments C1, C2, C3 of Yukawa kernel
# Kernel: dσ/dx ∝ 1 / (A - B x)^2,  with A = w^2 + V^2/2,  B = V^2/2,  x = cosθ
# We compute: Cn = (∫_{-1}^1 x^n/(A - Bx)^2 dx) / (∫_{-1}^1 1/(A - Bx)^2 dx)
# by change of variable t = A - Bx and binomial expansion; closed-form with logs.
# =========================
def _Cn_from_AB(n: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute C_n for given arrays A,B (same shape) with t1=A-B, t2=A+B.
    Handles B→0 by returning the uniform-angle limits.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    shape = np.broadcast(A, B).shape
    A = np.broadcast_to(A, shape)
    B = np.broadcast_to(B, shape)

    t1 = A - B
    t2 = A + B
    # Safe guards
    eps = 1e-300
    t1 = np.maximum(t1, eps)
    t2 = np.maximum(t2, eps)
    Bp = np.maximum(B, eps)

    # J_k = ∫ t^{k-2} dt between [t1,t2]
    def Jk(k: int):
        if k == 0:
            return (1.0/t1 - 1.0/t2)
        elif k == 1:
            return np.log(t2) - np.log(t1)
        else:
            return (t2**(k-1) - t1**(k-1)) / (k-1)

    # I_n = (1/B^{n+1}) ∑_{k=0..n} (-1)^k C(n,k) A^{n-k} J_k
    In = np.zeros_like(A)
    from math import comb
    for k in range(n+1):
        coef = ((-1)**k) * comb(n, k) * (A**(n-k))
        In += coef * Jk(k)
    In = In / (Bp**(n+1))

    # I0 (denominator)
    I0 = (1.0/Bp) * (1.0/t1 - 1.0/t2)

    Cn = In / I0

    # For V→0 (B→0), the kernel becomes isotropic in x: C1=0, C2=1/3, C3=0
    small = (B < 1e-12)
    if np.any(small):
        if n == 1 or n == 3:
            Cn = np.where(small, 0.0, Cn)
        elif n == 2:
            Cn = np.where(small, 1.0/3.0, Cn)
        elif n == 0:
            Cn = np.where(small, 1.0, Cn)
    # Physical clipping (tiny numerical excursions)
    if n >= 1:
        Cn = np.clip(Cn, -1.0, 1.0)
    return Cn

def rutherford_C123(V: np.ndarray, w: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (C1, C2, C3) arrays for given V (vectorized)."""
    V = np.asarray(V, dtype=float)
    A = w*w + 0.5*V*V
    B = 0.5*V*V
    C1 = _Cn_from_AB(1, A, B)
    C2 = _Cn_from_AB(2, A, B)
    C3 = _Cn_from_AB(3, A, B)
    return C1, C2, C3

# =========================
# Theory curves (strict): drift, D∥, D⊥, and 3rd-order moments
# =========================
def theory_curves_fig3(v_particle: float,
                       v_bath_samples: np.ndarray,
                       mu: float,
                       m_t: float,
                       sigma0: float,
                       w: float) -> Tuple[float, float, float, float, float]:
    """
    Strict theory for (drift_||, D_||, D_⊥, M3_par, M_mixed) at a given test speed,
    averaged over the Maxwellian bath with weight σ_tot(V)*V (n_f omitted
    to match MC normalization).
      M3_par   = ⟨(Δv_||)^3⟩
      M_mixed  = ⟨Δv_|| Δv_⊥^2⟩
    """
    v1 = np.array([v_particle, 0.0, 0.0])
    g  = v1 - v_bath_samples
    V  = np.linalg.norm(g, axis=1)
    ok = V > 0.0
    if not np.any(ok):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    g    = g[ok]; V = V[ok]
    ghat = g / V[:,None]
    mu_e = ghat[:,0]  # cos between test axis (x) and relative direction

    sig_tot = sigma_tot_rutherford(V, sigma0, w)      # vectorized
    C1, C2, C3 = rutherford_C123(V, w)

    fac = (mu / m_t)
    F   = fac * V                # useful factor
    F2  = F*F
    F3  = F2*F

    # 1st & 2nd order (as earlier)
    a_perp   = 0.5*(1.0 - C2)
    par_coef = (C2 - 2.0*C1 + 1.0)
    weight   = sig_tot * V

    drift_par = np.mean(weight * (C1 - 1.0) * (F * mu_e))
    D_par     = np.mean(weight * (F2) * ( a_perp*(1.0 - mu_e**2) + par_coef*(mu_e**2) ))
    total2    = np.mean(weight * 2.0 * (F2) * (1.0 - C1))
    D_perp    = total2 - D_par

    # 3rd-order along test axis:
    # ⟨(Δv_||)^3⟩ = F^3 [ μ_e^3 (C3 - 3C2 + 3C1 - 1) + (3/2) μ_e (1-μ_e^2) (-1 + C1 + C2 - C3) ]
    T1 = (C3 - 3.0*C2 + 3.0*C1 - 1.0)
    T2 = (-1.0 + C1 + C2 - C3)
    M3_par = np.mean(weight * ( F3 * ( mu_e**3 * T1 + 1.5 * mu_e * (1.0 - mu_e**2) * T2 ) ))

    # Mixed 3rd: ⟨Δv_|| Δv_⊥^2⟩ = ⟨Δv_|| |Δv|^2⟩ - ⟨(Δv_||)^3⟩,
    # where |Δv|^2 = 2 F^2 (1 - cosθ). Averaging over φ kills the transverse linear term:
    # ⟨Δv_|| |Δv|^2⟩ = F^3 * μ_e * (-2 + 4C1 - 2C2).
    Mixed_first = np.mean(weight * ( F3 * mu_e * (-2.0 + 4.0*C1 - 2.0*C2) ))
    M_mixed = Mixed_first - M3_par

    return float(drift_par), float(D_par), float(D_perp), float(M3_par), float(M_mixed)

# =========================
# Figure 3: MC + theory overlay (now with 3rd order too)
# =========================
def function1_plot_fig3():
    # MC curves
    dv_parallels_MC        = []
    dv_parallels2_MC       = []
    dv_perps2_MC           = []
    dv_parallels3_MC       = []
    dv_parallels_perps2_MC = []

    # Theory arrays
    drift_th = []
    Dpar_th  = []
    Dperp_th = []
    M3_th    = []
    Mmix_th  = []

    # Reuse a large random subset of bath for theory averaging (fast & unbiased)
    N_THEORY_SAMPLES = min(NUM_BATH, 200_000)
    idx_th = rng.integers(0, NUM_BATH, size=N_THEORY_SAMPLES)
    V_BATH_TH = V_BATH[idx_th]

    for v_particle in VELOCITY_VALS:
        # ---------- MC accumulation ----------
        S1=S2=S3=S4=S5 = 0.0
        for _ in range(NUMBER_ENCOUNTERS):
            idx = rng.integers(0, NUM_BATH)
            vb  = V_BATH[idx]

            v1_before = np.array([v_particle, 0.0, 0.0])
            v2_before = vb

            g = v1_before - v2_before
            g_mag = np.linalg.norm(g)
            if g_mag <= 0.0:
                continue
            g_hat = g / g_mag

            # Sample Yukawa CM scattering angle
            # Inverse-CDF for x = cosθ with A = w^2 + V^2/2, B = V^2/2
            A = W*W + 0.5*g_mag*g_mag
            B = 0.5*g_mag*g_mag
            u = rng.random()
            denom = (1.0/(A + B)) + (2.0*B / ((A + B)*max(A - B, 1e-300))) * u
            inv = 1.0/denom
            cos_theta = float(np.clip((A - inv)/B, -1.0, 1.0))
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
            phi = rng.uniform(0.0, 2.0*np.pi)

            e1, e2 = ortho_basis_from(g_hat)
            n_hat = sin_theta*math.cos(phi)*e1 + sin_theta*math.sin(phi)*e2 + cos_theta*g_hat

            v1_after, _ = post_collision_equal_masses(v1_before, v2_before, n_hat)

            dv_vec   = v1_after - v1_before
            dv_par   = dv_vec[0]
            dv_perp2 = dv_vec[1]*dv_vec[1] + dv_vec[2]*dv_vec[2]

            sigma_total = sigma_tot_rutherford_scalar(g_mag, SIGMA0, W)
            weight = g_mag * sigma_total * (N_F if USE_ABSOLUTE_RATES else 1.0)

            S1 += dv_par * weight
            S2 += (dv_par*dv_par) * weight
            S3 += dv_perp2 * weight
            S4 += (dv_par*dv_par*dv_par) * weight
            S5 += dv_par * dv_perp2 * weight

        S1 /= NUMBER_ENCOUNTERS; S2 /= NUMBER_ENCOUNTERS; S3 /= NUMBER_ENCOUNTERS
        S4 /= NUMBER_ENCOUNTERS; S5 /= NUMBER_ENCOUNTERS

        dv_parallels_MC.append(        abs(S1 / (S_BATH**1)) )
        dv_parallels2_MC.append(       abs(S2 / (S_BATH**2)) )
        dv_perps2_MC.append(           abs(S3 / (S_BATH**2)) )
        dv_parallels3_MC.append(       abs(S4 / (S_BATH**3)) )
        dv_parallels_perps2_MC.append( abs(S5 / (S_BATH**3)) )

        # ---------- THEORY at this v (strict) ----------
        th_drift, th_Dpar, th_Dperp, th_M3, th_Mmix = theory_curves_fig3(
            v_particle=v_particle,
            v_bath_samples=V_BATH_TH,
            mu=MU,
            m_t=M_PARTICLE,
            sigma0=SIGMA0,
            w=W
        )
        drift_th.append( abs(th_drift / (S_BATH**1)) )
        Dpar_th.append(  abs(th_Dpar  / (S_BATH**2)) )
        Dperp_th.append( abs(th_Dperp / (S_BATH**2)) )
        M3_th.append(    abs(th_M3    / (S_BATH**3)) )
        Mmix_th.append(  abs(th_Mmix  / (S_BATH**3)) )

    # ---------- PLOT: MC vs Theory (5 moments) ----------
    plt.figure(figsize=(10,7))

    # MC curves (solid)
    plt.loglog(VELOCITY_VALS, dv_parallels_MC,        '-',  label = r"MC: $\langle \Delta v_\parallel\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels2_MC,       '-',  label = r"MC: $\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(VELOCITY_VALS, dv_perps2_MC,           '-',  label = r"MC: $\langle \Delta v_\perp^2\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels3_MC,       '-',  label = r"MC: $\langle \Delta v_\parallel^3\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels_perps2_MC, '-',  label = r"MC: $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    # Theory overlays (dashed)
    plt.loglog(VELOCITY_VALS, drift_th,   '--', label=r"Theory: $\langle \Delta v_\parallel\rangle$")
    plt.loglog(VELOCITY_VALS, Dpar_th,    '--', label=r"Theory: $D_\parallel$")
    plt.loglog(VELOCITY_VALS, Dperp_th,   '--', label=r"Theory: $D_\perp$")
    plt.loglog(VELOCITY_VALS, M3_th,      '--', label=r"Theory: $\langle (\Delta v_\parallel)^3\rangle$")
    plt.loglog(VELOCITY_VALS, Mmix_th,    '--', label=r"Theory: $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    plt.xlabel(r'$v_{\rm particle}$')
    plt.ylabel(r'velocity increments (normalized by $s_{\rm bath}^n$)')
    plt.title(r'Figure 3: Yukawa/screened Rutherford — MC vs analytic theory (incl. 3rd order)')
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig("velocity_changes_fig3_with_theory.png", dpi=160)
    plt.savefig("velocity_changes_fig3_with_theory.pdf", dpi=160)
    # plt.show()

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    function1_plot_fig3()

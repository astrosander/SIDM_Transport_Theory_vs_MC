import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple

rng = np.random.default_rng(1)

NUM_BATH            = 1_000_000
NUMBER_ENCOUNTERS   = 100_000
VELOCITY_VALS       = np.linspace(0.0, 1_000_000.0, 10)
S_BATH              = 150_000.0
M_PARTICLE          = 1.0
M_BATH              = 1.0
MU                  = (M_PARTICLE * M_BATH) / (M_PARTICLE + M_BATH)
USE_ABSOLUTE_RATES  = False
N_F                 = 1.0

SIGMA0              = 2.0e-28
W                   = 1.0e3

V_BATH = rng.normal(0.0, S_BATH, size=(NUM_BATH, 3))

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
    g = v1 - v2
    Vcm = 0.5 * (v1 + v2)
    g_after = np.linalg.norm(g) * n_hat
    v1_after = Vcm + 0.5 * g_after
    v2_after = Vcm - 0.5 * g_after
    return v1_after, v2_after

def sigma_tot_rutherford_scalar(V: float, sigma0: float, w: float) -> float:
    return float(sigma0 / (1.0 + (V*V)/(w*w)))

def sample_cos_rutherford_scalar(V: float, w: float, sigma0: float) -> float:
    A = w*w + 0.5*V*V
    B = 0.5*V*V
    u = rng.random()
    denom = (1.0/(A + B)) + (2.0*B / ((A + B)*max(A - B, 1e-300))) * u
    inv = 1.0/denom
    x = (A - inv)/B
    return float(np.clip(x, -1.0, 1.0))

def function1_plot_fig3():
    dv_parallels        = []
    dv_parallels2       = []
    dv_perps2           = []
    dv_parallels3       = []
    dv_parallels_perps2 = []

    for v_particle in VELOCITY_VALS:
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

            cos_theta = sample_cos_rutherford_scalar(g_mag, W, SIGMA0)
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

        dv_parallels.append(        abs(S1 / (S_BATH**1)) )
        dv_parallels2.append(       abs(S2 / (S_BATH**2)) )
        dv_perps2.append(           abs(S3 / (S_BATH**2)) )
        dv_parallels3.append(       abs(S4 / (S_BATH**3)) )
        dv_parallels_perps2.append( abs(S5 / (S_BATH**3)) )

    plt.figure(figsize=(8,6))
    plt.loglog(VELOCITY_VALS, dv_parallels,           label = r"$\langle \Delta v_\parallel\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels2,          label = r"$\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(VELOCITY_VALS, dv_perps2,              label = r"$\langle \Delta v_\perp^2\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels3,          label = r"$\langle \Delta v_\parallel^3\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels_perps2,    label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    plt.xlabel(r'$v_{\rm particle}$')
    plt.ylabel(r'velocity increments (normalized by $s_{\rm bath}^n$)')
    plt.title(r'Figure 3: Yukawa/screened Rutherford')
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig("velocity_changes_fig3.png", dpi=160)
    plt.savefig("velocity_changes_fig3.pdf", dpi=160)
    # plt.show()

KAPPA_EFF = 1.0

def sample_cos_grav_trunc(V: float, theta_min: float) -> Tuple[float, float]:
    x_max = float(np.clip(np.cos(theta_min), -1.0, 0.999999999))
    denom = (1.0/(1.0 - x_max) - 0.5)
    u = rng.random()
    inv = 0.5 + u * denom
    x = 1.0 - 1.0/inv
    return float(np.clip(x, -1.0, x_max)), x_max

def sigma_tot_grav_trunc(V: float, x_max: float) -> float:
    A = 2.0*np.pi * (KAPPA_EFF**2) / (MU**2 * V**4)
    return float( A * (1.0/(1.0 - x_max) - 0.5) )

def function2_plot_fig4(THETA_MODE: str = "mapping", THETA_FIXED: float = 1e-4):
    dv_parallels        = []
    dv_parallels2       = []
    dv_perps2           = []
    dv_parallels3       = []
    dv_parallels_perps2 = []

    for v_particle in VELOCITY_VALS:
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

            if THETA_MODE == "mapping":
                theta_min = 2.0 * W / max(g_mag, 1e-30)
                theta_min = float(np.clip(theta_min, 1e-12, np.pi-1e-12))
            elif THETA_MODE == "fixed":
                theta_min = float(np.clip(THETA_FIXED, 1e-12, np.pi-1e-12))
            else:
                theta_min = 1e-12

            cos_theta, x_max = sample_cos_grav_trunc(g_mag, theta_min)
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
            phi = rng.uniform(0.0, 2.0*np.pi)

            e1, e2 = ortho_basis_from(g_hat)
            n_hat = sin_theta*math.cos(phi)*e1 + sin_theta*math.sin(phi)*e2 + cos_theta*g_hat

            v1_after, _ = post_collision_equal_masses(v1_before, v2_before, n_hat)

            dv_vec   = v1_after - v1_before
            dv_par   = dv_vec[0]
            dv_perp2 = dv_vec[1]*dv_vec[1] + dv_vec[2]*dv_vec[2]

            sigma_total = sigma_tot_grav_trunc(g_mag, x_max)
            weight = g_mag * sigma_total * (N_F if USE_ABSOLUTE_RATES else 1.0)

            S1 += dv_par * weight
            S2 += (dv_par*dv_par) * weight
            S3 += dv_perp2 * weight
            S4 += (dv_par*dv_par*dv_par) * weight
            S5 += dv_par * dv_perp2 * weight

        S1 /= NUMBER_ENCOUNTERS; S2 /= NUMBER_ENCOUNTERS; S3 /= NUMBER_ENCOUNTERS
        S4 /= NUMBER_ENCOUNTERS; S5 /= NUMBER_ENCOUNTERS

        dv_parallels.append(        abs(S1 / (S_BATH**1)) )
        dv_parallels2.append(       abs(S2 / (S_BATH**2)) )
        dv_perps2.append(           abs(S3 / (S_BATH**2)) )
        dv_parallels3.append(       abs(S4 / (S_BATH**3)) )
        dv_parallels_perps2.append( abs(S5 / (S_BATH**3)) )

    plt.figure(figsize=(8,6))
    plt.loglog(VELOCITY_VALS, dv_parallels,           label = r"$\langle \Delta v_\parallel\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels2,          label = r"$\langle \Delta v_\parallel^2\rangle$")
    plt.loglog(VELOCITY_VALS, dv_perps2,              label = r"$\langle \Delta v_\perp^2\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels3,          label = r"$\langle \Delta v_\parallel^3\rangle$")
    plt.loglog(VELOCITY_VALS, dv_parallels_perps2,    label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

    lbl = {"mapping":"(θ_min=2w/V mapping → matches Fig. 3 shapes)",
           "fixed"  :f"(θ_min={THETA_FIXED:g} fixed)",
           "zero"   :"(θ_min→0: logarithmic growth)"}[THETA_MODE]
    plt.xlabel(r'$v_{\rm particle}$')
    plt.ylabel(r'velocity increments (normalized by $s_{\rm bath}^n$)')
    plt.title(rf'Figure 4: gravitational/Coulomb with cutoff {lbl}')
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(f"velocity_changes_fig4_{THETA_MODE}.png", dpi=160)
    plt.savefig(f"velocity_changes_fig4_{THETA_MODE}.pdf", dpi=160)
    # plt.show()

if __name__ == "__main__":
    function1_plot_fig3()
    # function2_plot_fig4("mapping")
    # function2_plot_fig4("fixed", 1e-3)
    # function2_plot_fig4("zero")

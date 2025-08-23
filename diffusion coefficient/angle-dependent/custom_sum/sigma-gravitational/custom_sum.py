import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple

rng = np.random.default_rng(1)

num_bath = 1_000_000
velocity_vals = np.linspace(0.0, 1_000_000.0, 10)
s_bath = 150_000.0
number_encounters = 10_000

m_particle = 1.0
m_bath = 1.0

sigma0 = 2.0e-28
w = 1.0e3

USE_ABSOLUTE_RATES = False
n_f = 1.0

v_bath = rng.normal(0.0, s_bath, size=(num_bath, 3))

def sigma_tot_rutherford_scalar(V: float, sigma0: float, w: float) -> float:
    return float(sigma0 / (1.0 + (V*V)/(w*w)))

def dsigma_dcos_rutherford(V: float, cos_theta: float, sigma0: float, w: float) -> float:
    denom = w*w + (V*V)*(1.0 - cos_theta)/2.0
    return (sigma0 * w**4) / (2.0 * denom**2)

def sample_cos_rutherford_scalar(V: float) -> float:
    A = w*w + 0.5*V*V
    B = 0.5*V*V
    u = rng.random()
    denom = (1.0/(A + B)) + (2.0*B / ((A + B)*max(A - B, 1e-300))) * u
    inv = 1.0/denom
    x = (A - inv)/B
    return float(np.clip(x, -1.0, 1.0))

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

dv_parallels          = []
dv_parallels2         = []
dv_perps2             = []
dv_parallels3         = []
dv_parallels_perps2   = []

for particle_velocity in velocity_vals:
    S1 = 0.0
    S2 = 0.0
    S3 = 0.0
    S4 = 0.0
    S5 = 0.0

    for _ in range(number_encounters):
        idx = rng.integers(0, num_bath)
        vb = v_bath[idx]

        v1_before = np.array([particle_velocity, 0.0, 0.0])
        v2_before = vb

        g = v1_before - v2_before
        g_mag = np.linalg.norm(g)
        if g_mag <= 0.0:
            continue
        g_hat = g / g_mag

        cos_theta = sample_cos_rutherford_scalar(g_mag)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
        phi = rng.uniform(0.0, 2.0*np.pi)

        e1, e2 = ortho_basis_from(g_hat)
        n_hat = sin_theta*math.cos(phi)*e1 + sin_theta*math.sin(phi)*e2 + cos_theta*g_hat

        v1_after, _ = post_collision_equal_masses(v1_before, v2_before, n_hat)

        dv_vec = v1_after - v1_before
        dv_par = dv_vec[0]
        dv_perp2 = dv_vec[1]*dv_vec[1] + dv_vec[2]*dv_vec[2]

        sigma_total = sigma_tot_rutherford_scalar(g_mag, sigma0, w)
        weight = g_mag * sigma_total * (n_f if USE_ABSOLUTE_RATES else 1.0)

        S1 += dv_par * weight
        S2 += (dv_par*dv_par) * weight
        S3 += dv_perp2 * weight
        S4 += (dv_par*dv_par*dv_par) * weight
        S5 += dv_par * dv_perp2 * weight

    S1 /= number_encounters
    S2 /= number_encounters
    S3 /= number_encounters
    S4 /= number_encounters
    S5 /= number_encounters

    dv_parallels.append(          abs(S1 / (s_bath**1)) )
    dv_parallels2.append(         abs(S2 / (s_bath**2)) )
    dv_perps2.append(             abs(S3 / (s_bath**2)) )
    dv_parallels3.append(         abs(S4 / (s_bath**3)) )
    dv_parallels_perps2.append(   abs(S5 / (s_bath**3)) )

def normalize_identity(arr):
    return arr

dv_parallels            = normalize_identity(dv_parallels)
dv_parallels2           = normalize_identity(dv_parallels2)
dv_perps2               = normalize_identity(dv_perps2)
dv_parallels3           = normalize_identity(dv_parallels3)
dv_parallels_perps2     = normalize_identity(dv_parallels_perps2)

plt.figure(figsize=(8,6))
plt.loglog(velocity_vals, dv_parallels,           label = r"$\langle \Delta v_\parallel\rangle$")
plt.loglog(velocity_vals, dv_parallels2,          label = r"$\langle \Delta v_\parallel^2\rangle$")
plt.loglog(velocity_vals, dv_perps2,              label = r"$\langle \Delta v_\perp^2\rangle$")
plt.loglog(velocity_vals, dv_parallels3,          label = r"$\langle \Delta v_\parallel^3\rangle$")
plt.loglog(velocity_vals, dv_parallels_perps2,    label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

plt.xlabel(r'$v_{\rm particle}$')
plt.ylabel(r'velocity increments')
plt.title(r'velocity increments with rutherford scattering')
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("velocity_changes_rutherford.png", dpi=160)
plt.savefig("velocity_changes_rutherford.pdf", dpi=160)
plt.show()

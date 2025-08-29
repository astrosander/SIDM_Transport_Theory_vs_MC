import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
num_bath = 1_000_000
velocity_vals = np.linspace(0, 500_000, 100)
s_bath = 150_000
number_encounters = 10_000
m_particle = 1.0
m_bath = 1.0

assert abs(m_particle - m_bath) < 1e-12, "Analytical overlay assumes equal masses."

v_bath = rng.normal(0.0, s_bath, size=(num_bath, 3))

def normalize(vec):
    vmax = max(abs(np.max(vec)), abs(np.min(vec)))
    return vec / vmax if vmax != 0 else vec

# def sigma(V):
#     return np.ones_like(V)
    # return 1 / (1 + (V / 1000) ** 4)


num_dv_par, num_dv_par2, num_dv_perp2, num_dv_par3, num_dv_par_perp2 = [], [], [], [], []
ana_dv_par, ana_dv_par2, ana_dv_perp2, ana_dv_par3, ana_dv_par_perp2 = [], [], [], [], []

for V in velocity_vals:
    idx = rng.integers(0, num_bath, size=number_encounters)
    vb = v_bath[idx]  # (N,3)

    g = np.empty_like(vb)
    g[:, 0] = V - vb[:, 0]
    g[:, 1] = -vb[:, 1]
    g[:, 2] = -vb[:, 2]
    g_norm = np.linalg.norm(g, axis=1)

    w = g_norm * sigma(g_norm)
    n = rng.normal(size=(number_encounters, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    g_dot_n = np.einsum("ij,ij->i", g, n)

    flip = g_dot_n >= 0
    n[flip] *= -1.0
    g_dot_n[flip] *= -1.0  # now g·n <= 0

    dv = -g_dot_n[:, None] * n
    dv_par = dv[:, 0]
    dv_perp = dv[:, 1]

    num_dv_par.append(np.mean(dv_par * w))
    num_dv_par2.append(np.mean((dv_par**2) * w))
    num_dv_perp2.append(np.mean((dv_perp**2) * w))
    num_dv_par3.append(np.mean((dv_par**3) * w))
    num_dv_par_perp2.append(np.mean(dv_par * (dv_perp**2) * w))

    safe = g_norm > 0
    c = np.zeros_like(g_norm)
    c[safe] = g[safe, 0] / g_norm[safe]

    A1 = -(1.0 / 3.0) * g_norm * g[:, 0]
    A2 = (g_norm**3 / 15.0) * (1.0 + 2.0 * c**2)
    A_perp2 = (g_norm**3 / 15.0) * (2.0 - c**2)
    A3 = -(g_norm**4 / 35.0) * (3.0 * c + 2.0 * c**3)
    A_mix = -(g_norm**4 / 35.0) * c * (3.0 - 2.0 * c**2)

    ana_dv_par.append(np.mean(A1))
    ana_dv_par2.append(np.mean(A2))
    ana_dv_perp2.append(np.mean(A_perp2))
    ana_dv_par3.append(np.mean(A3))
    ana_dv_par_perp2.append(np.mean(A_mix))

# ----------------------
# Normalize each curve by its own max |value| (shape comparison)
# ----------------------
num_dv_par      = normalize(np.array(num_dv_par))
num_dv_par2     = normalize(np.array(num_dv_par2))
num_dv_perp2    = normalize(np.array(num_dv_perp2))
num_dv_par3     = normalize(np.array(num_dv_par3))
num_dv_par_perp2= normalize(np.array(num_dv_par_perp2))

ana_dv_par      = normalize(np.array(ana_dv_par))
ana_dv_par2     = normalize(np.array(ana_dv_par2))
ana_dv_perp2    = normalize(np.array(ana_dv_perp2))
ana_dv_par3     = normalize(np.array(ana_dv_par3))
ana_dv_par_perp2= normalize(np.array(ana_dv_par_perp2))

# ----------------------
# Plot: numerical (solid) + analytical (dashed)
# ----------------------
plt.figure(figsize=(9, 6))

# Numerical (MC)
plt.plot(velocity_vals, num_dv_par,          label=r"MC $\langle \Delta v_\parallel\rangle$")
plt.plot(velocity_vals, num_dv_par2,         label=r"MC $\langle \Delta v_\parallel^2\rangle$")
plt.plot(velocity_vals, num_dv_perp2,        label=r"MC $\langle \Delta v_\perp^2\rangle$")
plt.plot(velocity_vals, num_dv_par3,         label=r"MC $\langle \Delta v_\parallel^3\rangle$")
plt.plot(velocity_vals, num_dv_par_perp2,    label=r"MC $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

# Analytical overlays
plt.plot(velocity_vals, ana_dv_par,          linestyle="--", label=r"Analytic $\langle \Delta v_\parallel\rangle$")
plt.plot(velocity_vals, ana_dv_par2,         linestyle="--", label=r"Analytic $\langle \Delta v_\parallel^2\rangle$")
plt.plot(velocity_vals, ana_dv_perp2,        linestyle="--", label=r"Analytic $\langle \Delta v_\perp^2\rangle$")
plt.plot(velocity_vals, ana_dv_par3,         linestyle="--", label=r"Analytic $\langle \Delta v_\parallel^3\rangle$")
plt.plot(velocity_vals, ana_dv_par_perp2,    linestyle="--", label=r"Analytic $\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

plt.xlabel(r"$v_{\mathrm{particle}}$")
plt.ylabel("normalized weighted moments")
plt.title(r"Velocity-change moments: Monte Carlo vs. angle-averaged analytic (equal masses)")
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()

plt.savefig("velocity_changes_overlay.png", dpi=160)
plt.savefig("velocity_changes_overlay.pdf", dpi=160)
plt.show()

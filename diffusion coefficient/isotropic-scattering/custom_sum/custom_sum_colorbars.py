import os
import math
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Configuration & parameters
# ----------------------------
MASTER_SEED = 1
rng_master = np.random.default_rng(MASTER_SEED)

num_bath = 1_000_000
velocity_vals = np.linspace(0, 5_000_000, 100)
s_bath = 150_000
number_encounters = 100_000

m_particle = 1.0
m_bath = 1.0
denom = (m_particle + m_bath)

# Parallelism config
MAX_WORKERS = min(32, (os.cpu_count() or 1))
CHUNK_SIZE = 10_000  # tune for your machine; must be <= number_encounters

# ----------------------------
# Bath velocities (3D Gaussian)
# ----------------------------
v_bath = rng_master.normal(0.0, s_bath, size=(num_bath, 3))

# ----------------------------
# Helper functions
# ----------------------------
def sigma(V):
    # Vectorized cross-section; keep as 1 for now (matches original)
    # return 1.0 / (1.0 + (V/150000)**4)  # example alternative
    return np.ones_like(V)

def simulate_chunk(particle_velocity, n_samples, rng):
    """
    Vectorized simulation for a chunk of encounters for a single particle velocity.
    Returns sums needed to compute means & unbiased stds for two series
    (x1 = Δv_parallel*|g|*sigma, x2 = Δv_perp*|g|*sigma),
    plus other moment sums to preserve original outputs.
    """
    # Sample bath indices
    idx = rng.integers(0, num_bath, size=n_samples)
    vb = v_bath[idx]  # shape (n, 3)

    # Relative velocity g = v_before - v_bath
    # particle v_before = [v, 0, 0]
    g = np.empty_like(vb)
    g[:, 0] = particle_velocity - vb[:, 0]
    g[:, 1] = -vb[:, 1]
    g[:, 2] = -vb[:, 2]

    # Random isotropic unit vectors n
    n = rng.normal(size=vb.shape)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    # Ensure g•n <= 0; if positive, flip n
    g_dot_n = np.einsum('ij,ij->i', g, n)
    mask = g_dot_n >= 0
    if np.any(mask):
        n[mask] *= -1.0
    g_dot_n = np.einsum('ij,ij->i', g, n)  # now non-positive

    gn = n * g_dot_n[:, None]
    gt = g - gn
    g_after = gt - gn

    # v_particle_after = (m_p * v_before + m_b * (v_bath + g_after)) / (m_p + m_b)
    v_after_x = (m_particle * particle_velocity + m_bath * (vb[:, 0] + g_after[:, 0])) / denom
    v_after_y = (m_bath * (vb[:, 1] + g_after[:, 1])) / denom  # v_before_y = 0

    dv_parallel = v_after_x - particle_velocity
    dv_perp = v_after_y  # since initial y-component = 0
    dv = np.sqrt(v_after_y**2 + (v_after_x - particle_velocity)**2)

    g_norm = np.linalg.norm(g, axis=1)
    w = g_norm * sigma(g_norm)

    # Series used in plots (matching original script semantics)
    x1 = dv_parallel**3#2*dv
    x2 = dv_perp**2*dv_parallel  # NOTE: original code labeled this as <Δv_parallel^2>, but used perp

    # Other moments (kept for completeness; not plotted by default)
    sum_perp2 = (dv_perp**2 * w).sum()
    sum_par3 = (dv_parallel**3 * w).sum()
    sum_mix = (dv_parallel * (dv_perp**2) * w).sum()

    # Sufficient statistics for means & stds
    s1 = x1.sum()
    s2 = x2.sum()
    ss1 = (x1**2).sum()
    ss2 = (x2**2).sum()

    return {
        "N": n_samples,
        "s1": s1, "ss1": ss1,
        "s2": s2, "ss2": ss2,
        "perp2": sum_perp2,
        "par3": sum_par3,
        "mix": sum_mix,
    }

def combine_stats(stats_list):
    """Combine chunk statistics into global means and unbiased stds."""
    N = sum(s["N"] for s in stats_list)
    s1 = sum(s["s1"] for s in stats_list)
    ss1 = sum(s["ss1"] for s in stats_list)
    s2 = sum(s["s2"] for s in stats_list)
    ss2 = sum(s["ss2"] for s in stats_list)

    mean1 = s1 / N
    mean2 = s2 / N

    # Unbiased sample variance: Var = (E[x^2] - (E[x])^2) * N/(N-1)
    if N > 1:
        var1 = (ss1 / N - mean1**2) * N / (N - 1)
        var2 = (ss2 / N - mean2**2) * N / (N - 1)
    else:
        var1 = var2 = 0.0

    std1 = math.sqrt(max(var1, 0.0))
    std2 = math.sqrt(max(var2, 0.0))

    # Also combine other moment sums (means)
    perp2_mean = sum(s["perp2"] for s in stats_list) / N
    par3_mean = sum(s["par3"] for s in stats_list) / N
    mix_mean = sum(s["mix"] for s in stats_list) / N

    return mean1, std1, mean2, std2, perp2_mean, par3_mean, mix_mean

def normalize_with_scale(vals):
    vals = np.asarray(vals, dtype=float)
    scale = np.max(np.abs(vals)) if len(vals) else 1.0
    scale = scale if scale != 0.0 else 1.0
    return vals / scale, z

# ----------------------------
# Main computation (multithreaded over encounters)
# ----------------------------
dv_parallels = []           # means for series 1   (<Δv_parallel>*|g|*sigma) / N
dv_parallels2 = []          # means for series 2   (<Δv_perp>*|g|*sigma) / N
dv_perps2 = []              # other moment means
dv_parallels3 = []
dv_parallels_perps2 = []

stds_par = []               # std for series 1 (pre-normalization)
stds_par2 = []              # std for series 2 (pre-normalization)

# Seed hierarchy: one SeedSequence per velocity, which spawns per-chunk RNGs
top_ss = np.random.SeedSequence(MASTER_SEED)
vel_ss_list = top_ss.spawn(len(velocity_vals))

for vi, v in enumerate(velocity_vals):
    # Determine chunk sizes
    n_chunks = (number_encounters + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunk_sizes = [CHUNK_SIZE] * n_chunks
    remainder = number_encounters - CHUNK_SIZE * (n_chunks - 1)
    if remainder > 0:
        chunk_sizes[-1] = remainder

    # Spawn independent RNGs for each chunk
    chunk_ss = vel_ss_list[vi].spawn(n_chunks)
    rngs = [np.random.default_rng(cs) for cs in chunk_ss]

    stats = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(simulate_chunk, v, n, rg)
            for n, rg in zip(chunk_sizes, rngs)
        ]
        for fut in as_completed(futures):
            stats.append(fut.result())

    mean1, std1, mean2, std2, perp2_mean, par3_mean, mix_mean = combine_stats(stats)

    dv_parallels.append(mean1)
    dv_parallels2.append(mean2)
    dv_perps2.append(perp2_mean)
    dv_parallels3.append(par3_mean)
    dv_parallels_perps2.append(mix_mean)

    stds_par.append(std1)
    stds_par2.append(std2)

# ----------------------------
# Normalize series (and their stds with same scales)
# ----------------------------
dv_parallels, s1 = normalize_with_scale(dv_parallels)
dv_parallels2, s2 = normalize_with_scale(dv_parallels2)

stds_par = np.asarray(stds_par) / s1
stds_par2 = np.asarray(stds_par2) / s2

# ----------------------------
# Plot with rectangular 1σ shading per velocity bin
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

h1, = ax.plot(velocity_vals, dv_parallels,  label=r"$\langle \Delta v_\parallel\rangle$")
h2, = ax.plot(velocity_vals, dv_parallels2, label=r"$\langle \Delta v_\parallel^2\rangle$")  # kept label to match your original

# Handle non-uniform grids robustly
if len(velocity_vals) > 1:
    mids = velocity_vals
    edges = np.empty(len(mids) + 1, dtype=float)
    edges[1:-1] = 0.5 * (mids[1:] + mids[:-1])
    edges[0] = mids[0] - (edges[1] - mids[0])
    edges[-1] = mids[-1] + (mids[-1] - edges[-2])
else:
    mids = velocity_vals
    edges = np.array([mids[0] - 0.5, mids[0] + 0.5])

from matplotlib.patches import Rectangle

# Series 1 rectangles
for i, (v, mu, sd) in enumerate(zip(velocity_vals, dv_parallels, stds_par)):
    width = edges[i+1] - edges[i]
    x0 = edges[i]
    y0 = mu - sd
    ax.add_patch(Rectangle((x0, y0), width, 2.0*sd,
                           facecolor=h1.get_color(), alpha=0.2, linewidth=0))

# Series 2 rectangles (optional; comment out if too busy)
for i, (v, mu, sd) in enumerate(zip(velocity_vals, dv_parallels2, stds_par2)):
    width = edges[i+1] - edges[i]
    x0 = edges[i]
    y0 = mu - sd
    ax.add_patch(Rectangle((x0, y0), width, 2.0*sd,
                           facecolor=h2.get_color(), alpha=0.2, linewidth=0))

ax.set_xlabel(r"$v_\text{particle}$")
ax.set_ylabel(r"$\langle \Delta v\rangle$ (normalized)")
ax.set_title("Velocity changes (Constant)")
ax.legend()
ax.set_yscale('symlog')
ax.grid(True)

plt.tight_layout()
plt.savefig("velocity_changes_constant.png", dpi=160)
plt.savefig("velocity_changes_constant.pdf", dpi=160)
plt.show()
plt.clf()



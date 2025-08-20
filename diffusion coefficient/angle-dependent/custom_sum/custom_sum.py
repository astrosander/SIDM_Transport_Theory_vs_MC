import random
import numpy as np
import math
import matplotlib.pyplot as plt

# random.seed(1)
rng = np.random.default_rng(1)

# v_bath = []
num_bath = 1000_000
velocity_vals = np.linspace(0, 1000_000, 10)
s_bath = 150_000
number_encounters = 10_000
m_particle = 1
m_bath = 1

# Rutherford scattering parameters (from monte carlo.py)
sigma0 = 2.0e-28
w = 1.0e3

# for i in range(num_bath):
# 	v_bath.append((random.gauss(0, s_bath),random.gauss(0, s_bath),random.gauss(0, s_bath)))

v_bath = rng.normal(0.0, s_bath, size=(num_bath, 3))  

# Rutherford scattering functions (from monte carlo.py)
def sigma_tot_rutherford_scalar(v: float, sigma0: float, w: float) -> float:
    return float(sigma0 / (1.0 + (v*v)/(w*w)))

def dsigma_dcos_rutherford(v: float, cos_theta: float, sigma0: float, w: float) -> float:
    denom = w*w + (v*v)*(1.0 - cos_theta)/2.0
    return (sigma0 * w**4) / (2.0 * denom**2)

def sample_cos_rutherford_scalar(V: float) -> float:
    """Sample cos(theta) for Rutherford scattering given relative velocity magnitude V"""
    A = w*w + 0.5*V*V
    B = 0.5*V*V
    u = rng.random()
    denom = (1.0/(A + B)) + (2.0*B / ((A + B)*(max(A - B, 1e-300)))) * u
    inv = 1.0/denom
    x = (A - inv)/B
    return np.clip(x, -1.0, 1.0)

def ortho_basis_from(g_hat: np.ndarray):
    """Generate orthogonal basis vectors from a unit vector"""
    if abs(g_hat[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(g_hat, ref)
    n1 = np.linalg.norm(e1)
    if n1 < 1e-15:
        e1 = np.array([1.0, 0.0, 0.0])
        n1 = 1.0
    e1 /= n1
    e2 = np.cross(g_hat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2

relative_magnitudes = []
dv_parallels = []
dv_parallels2 = []
dv_parallels3 = []
dv_perps2 = []
dv_parallels3 = []
dv_parallels_perps2 = []

plt.figure(figsize=(8,6))

for particle_velocity in velocity_vals:
	mean_relative_magnitude = 0
	dv_parallel = 0
	dv_parallel2 = 0
	dv_perp2 = 0
	dv_parallel3 = 0
	dv_parallel_perp2 = 0

	for i in range(number_encounters):
		# bath_index = random.randint(0, num_bath-1)
		bath_index = rng.integers(0, num_bath)
		vx_bath, vy_bath, vz_bath = v_bath[bath_index]
		
		#before collision:
		vx_particle_before = particle_velocity
		vy_particle_before = 0
		vz_particle_before = 0

		v_particle_before = np.array([particle_velocity, 0, 0])

		#relative velocity
		gx = vx_particle_before - vx_bath
		gy = -vy_bath
		gz = -vz_bath

		g = np.array([gx, gy, gz])
		g_mag = np.linalg.norm(g)
		
		if g_mag <= 0:
			continue  # Skip if no relative velocity
			
		g_hat = g / g_mag  # Unit vector in relative velocity direction
		
		# Sample scattering angle using Rutherford differential cross-section
		cos_theta = sample_cos_rutherford_scalar(g_mag)
		sin_theta = np.sqrt(1.0 - np.clip(cos_theta*cos_theta, 0.0, 1.0))
		phi = rng.uniform(0.0, 2.0*np.pi)
		
		# Create scattering direction in g_hat coordinate system
		e1, e2 = ortho_basis_from(g_hat)
		n = sin_theta*np.cos(phi)*e1 + sin_theta*np.sin(phi)*e2 + cos_theta*g_hat
		
		# Ensure scattering is backwards (g_dot_n < 0 for collision)
		g_dot_n = np.dot(g, n)
		if g_dot_n >= 0:
		    n = -n
		    g_dot_n = -g_dot_n

		gn = n*g_dot_n
		gt = g - gn
		g_after = gt - gn


		v_particle_after = (m_particle * v_particle_before + m_bath * (v_bath[bath_index] + g_after)) / (m_particle + m_bath)
		dv_particle_parrallel = v_particle_after[0] - v_particle_before[0]
		dv_particle_perp = v_particle_after[1] - v_particle_before[1]

		# Use Rutherford total cross-section
		sigma_total = sigma_tot_rutherford_scalar(g_mag, sigma0, w)

		dv_parallel += (dv_particle_parrallel) * g_mag * sigma_total
		dv_parallel2 += (dv_particle_parrallel)**2 * g_mag * sigma_total
		dv_perp2 += (dv_particle_perp)**2 * g_mag * sigma_total
		dv_parallel3 += (dv_particle_parrallel)**3 * g_mag * sigma_total
		dv_parallel_perp2 += dv_particle_parrallel*(dv_particle_perp)**2 * g_mag * sigma_total	

		mean_relative_magnitude += (dv_particle_perp)**2 * np.linalg.norm(g) #vx_particle_after-vx_particle_before

	dv_parallel =  dv_parallel / number_encounters
	dv_parallel2 = dv_parallel2 / number_encounters
	dv_perp2 = dv_perp2 / number_encounters
	dv_parallel3 = dv_parallel3 / number_encounters
	dv_parallel_perp2 = dv_parallel_perp2 / number_encounters
	
	# mean_relative_magnitude = mean_relative_magnitude / number_encounters

	# relative_magnitudes.append(mean_relative_magnitude)
	dv_parallels.append(abs(dv_parallel / s_bath))
	dv_parallels2.append(abs(dv_parallel2 / s_bath**2))
	dv_perps2.append(abs(dv_perp2 / s_bath**2))
	dv_parallels3.append(abs(dv_parallel3 / s_bath**3))
	dv_parallels_perps2.append(abs(dv_parallel_perp2 / s_bath**3))

def normalize(dv):
	return dv
	# return dv / max(abs(max(dv)), abs(min(dv)))

dv_parallels = normalize(dv_parallels)
dv_parallels2 = normalize(dv_parallels2)
dv_perps2 = normalize(dv_perps2)
dv_parallels3 = normalize(dv_parallels3)
dv_parallels_perps2 = normalize(dv_parallels_perps2)

plt.loglog(velocity_vals, dv_parallels, label = r"$\langle \Delta v_\parallel\rangle$")
plt.loglog(velocity_vals, dv_parallels2, label = r"$\langle \Delta v_\parallel^2\rangle$")
plt.loglog(velocity_vals, dv_perps2, label = r"$\langle \Delta v_\perp^2\rangle$")
plt.loglog(velocity_vals, dv_parallels3, label = r"$\langle \Delta v_\parallel^3\rangle$")
plt.loglog(velocity_vals, dv_parallels_perps2, label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")

plt.xlabel(r'$v_\text{particle}$')
plt.ylabel(r'$\langle \Delta v_\text{relative}\rangle$')
plt.title(r'Velocity changes (Rutherford)')
plt.legend()
plt.grid(True)
plt.savefig("velocity changes rutherford.png", dpi=160)
plt.savefig("velocity changes rutherford.pdf", dpi=160)
plt.show()
plt.clf()

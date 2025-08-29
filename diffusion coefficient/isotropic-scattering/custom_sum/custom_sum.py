import random
import numpy as np
import math
import matplotlib.pyplot as plt

# random.seed(1)
rng = np.random.default_rng(1)

# v_bath = []
num_bath = 1000_000
# velocity_vals = np.logspace(0, 1000_000, num=10, base=10.0)
# np.logspace(0, 1000_000, 5)

velocity_vals = np.linspace(0, 1000_0000, 10)
s_bath = 150_000
number_encounters = 10_000
m_particle = 1
m_bath = 1

# for i in range(num_bath):
# 	v_bath.append((random.gauss(0, s_bath),random.gauss(0, s_bath),random.gauss(0, s_bath)))

v_bath = rng.normal(0.0, s_bath, size=(num_bath, 3))  

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
		
		# print(np.dot(g, g))

		# nx = 2.0*random.random()-1.0
		# ny = 2.0*random.random()-1.0
		# nz = 2.0*random.random()-1.0
		# n = np.array([nx,ny,nz])
		# n_mag = np.linalg.norm(n)#np.sqrt(nx*nx+ny*ny+nz*nz)
		# n/=n_mag

		# u = rng.uniform(-1.0, 1.0, size=1)
		# phi = rng.uniform(0.0, 2*np.pi, size=1)
		# s = np.sqrt(1.0 - u*u)
		# n = np.stack((s*np.cos(phi), s*np.sin(phi), u), axis=1)[0]
		# print(n[0])

		n = np.random.normal(size=3); n /= np.linalg.norm(n)

		g_dot_n = np.dot(g, n)
		if g_dot_n >= 0:
		    n = -n
		    g_dot_n = -g_dot_n

		gn = n*g_dot_n	#[nx*g_dot_n, ny*g_dot_n, nz*g_dot_n]
		# print(np.linalg.norm(g)/np.linalg.norm(gn))
		gt = g - gn

		g_after = gt-gn


		# g_norm = np.linalg.norm(g, axis=1)
	    # A1 = -(1.0 / 3.0) * g_norm * g[:, 0]
	    # A2 = (g_norm**3 / 15.0) * (1.0 + 2.0 * c**2)
	    # A_perp2 = (g_norm**3 / 15.0) * (2.0 - c**2)
	    # A3 = -(g_norm**4 / 35.0) * (3.0 * c + 2.0 * c**3)
	    # A_mix = -(g_norm**4 / 35.0) * c * (3.0 - 2.0 * c**2)
	    
		v_particle_after = (m_particle * v_particle_before + m_bath * (v_bath[bath_index] + g_after)) / (m_particle + m_bath)
		dv_particle_parrallel = v_particle_after[0] - v_particle_before[0]
		dv_particle_perp = v_particle_after[1] - v_particle_before[1]

		def sigma(V):
			# return 1.0
			return 1.0 / (1.0 + (V/150000)**4)

		dv_parallel += (dv_particle_parrallel) * np.linalg.norm(g)	*sigma(np.linalg.norm(g))
		dv_parallel2 += (dv_particle_parrallel)**2 * np.linalg.norm(g)	*sigma(np.linalg.norm(g))
		dv_perp2 += (dv_particle_perp)**2 * np.linalg.norm(g) *sigma(np.linalg.norm(g))
		dv_parallel3 += (dv_particle_parrallel)**3 * np.linalg.norm(g)	*sigma(np.linalg.norm(g))
		dv_parallel_perp2 += dv_particle_parrallel*(dv_particle_perp)**2 * np.linalg.norm(g) *sigma(np.linalg.norm(g))	

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
	return dv / max(abs(max(dv)), abs(min(dv)))

dv_parallels = dv_parallels
dv_parallels2 = dv_parallels2 
dv_perps2 = dv_perps2 
dv_parallels3 = dv_parallels3
dv_parallels_perps2 = dv_parallels_perps2

# plt.plot(velocity_vals, dv_parallels, label = r"$\langle \Delta v_\parallel\rangle$")
# plt.plot(velocity_vals, dv_parallels2, label = r"$\langle \Delta v_\parallel^2\rangle$")
# plt.plot(velocity_vals, dv_perps2, label = r"$\langle \Delta v_\perp^2\rangle$")
# plt.plot(velocity_vals, dv_parallels3, label = r"$\langle \Delta v_\parallel^3\rangle$")
# plt.plot(velocity_vals, dv_parallels_perps2, label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")


plt.loglog(velocity_vals, dv_parallels, label = r"$\langle \Delta v_\parallel\rangle$")
plt.loglog(velocity_vals, dv_parallels2, label = r"$\langle \Delta v_\parallel^2\rangle$")
plt.loglog(velocity_vals, dv_perps2, label = r"$\langle \Delta v_\perp^2\rangle$")
plt.loglog(velocity_vals, dv_parallels3, label = r"$\langle \Delta v_\parallel^3\rangle$")
plt.loglog(velocity_vals, dv_parallels_perps2, label = r"$\langle \Delta v_\parallel \Delta v_\perp^2\rangle$")


plt.xlabel(r'$v_\text{particle}$')
plt.ylabel(r'$\langle \Delta v_\text{relative}\rangle$')
plt.title(r'Velocity changes (Constant)')
plt.legend()
plt.grid(True)
plt.savefig("velocity changes constant.png", dpi=160)
plt.savefig("velocity changes constant.pdf", dpi=160)
plt.show()
plt.clf()

import random
import numpy as np
import math
import matplotlib.pyplot as plt

random.seed(1)

v_bath = []
num_bath = 1000_000
velocity_vals = np.linspace(0, 1, 100)
bath_velocity = 1
number_encounters = 10_000
m_particle = 1
m_bath = 1

for i in range(num_bath):
	v_bath.append(bath_velocity*(2*random.random()-1,2*random.random()-1,2*random.random()-1))

relative_magnitudes = []

plt.figure(figsize=(8,6))

for particle_velocity in velocity_vals:
	mean_relative_magnitude = 0
	for i in range(number_encounters):
		vx_bath, vy_bath, vz_bath = v_bath[random.randint(0, num_bath-1)]
		
		#before collision:
		vx_particle_before = particle_velocity
		vy_particle_before = 0
		vz_particle_before = 0

		#relative velocity
		vx_relative = vx_particle_before - vx_bath
		vy_relative = -vy_bath
		vz_relative = -vz_bath

		g = (vx_relative, vy_relative, vz_relative)
		g = (vx_relative, vy_relative, vz_relative)

		# print(np.dot(g, g))

		#after collision:
		vx_particle_after = vx_particle_before * (m_particle - m_bath) / (m_particle + m_bath) + vx_bath * 2 * m_bath / (m_particle + m_bath)
		vy_particle_after = vy_particle_before * (m_particle - m_bath) / (m_particle + m_bath) + vy_bath * 2 * m_bath / (m_particle + m_bath)
		vz_particle_after = vz_particle_before * (m_particle - m_bath) / (m_particle + m_bath) + vz_bath * 2 * m_bath / (m_particle + m_bath)
		
		mean_relative_magnitude += np.dot(g, g)#vx_particle_after-vx_particle_before

		n = [2.0*random.random()-1.0,2.0*random.random()-1,2.0*random.random()-1]

		# print(np.dot(g, n))
		# print(n)
		
		# t1 = n
		# for i in range(3):
		# 	t1[i] = t1[i] * np.dot(g, n)

		t2 = np.dot(g, n)

		#math.sqrt((particle_velocity-vx)**2+vy**2+vz**2)


	mean_relative_magnitude = mean_relative_magnitude / number_encounters

	relative_magnitudes.append(mean_relative_magnitude)


plt.plot(velocity_vals, relative_magnitudes)

plt.xlabel(r'$v_\text{particle}$')
plt.ylabel(r'$v_\text{relative}$')
plt.title(r'Relative velocity vs particle velocity')
plt.legend()
plt.grid(True)
plt.savefig("relative velocity vs particle velocity.png", dpi=160)
plt.show()
#!/usr/bin/env python3

import numpy as np
import h5py
import os
from scipy import integrate

print("Galactic Nuclear Cluster Simulation - Python Version")
print("="*60)

base_dir = 'examples/1comp'
model_file = f'{base_dir}/model.in'
mfrac_file = f'{base_dir}/mfrac.in'
output_dir = f'{base_dir}/output'

print(f"\nReading configuration from: {base_dir}")

with open(model_file, 'r') as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

n_threads = int(lines[0])
jmin, jmax = map(float, lines[1].split())
mbh = float(lines[2].split()[0].replace('d', 'e'))
emin_factor = float(lines[3].split()[0].replace('d', 'e'))
emax_factor = float(lines[3].split()[1].replace('d', 'e'))
eboundary = float(lines[4].split()[0].replace('d', 'e'))
seed_value = int(lines[5])
same_ini_seed = int(lines[6])
gx_bins, dc_bins = map(int, lines[7].split())
same_evl_seed = int(lines[8])
num_update_per_snap = int(lines[9])
timestep_snapshot = float(lines[10])
num_snapshots = int(lines[11])
alpha_ini = float(lines[12])

print(f"MBH = {mbh:.2e} Msun")
print(f"gx_bins = {gx_bins}, dc_bins = {dc_bins}")
print(f"Snapshots = {num_snapshots}, Updates/snap = {num_update_per_snap}")

with open(mfrac_file, 'r') as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

parts = lines[1].split()
mc = float(parts[1])
asymptot = float(parts[3])
weight_n = float(parts[4])

print(f"Particle mass mc = {mc} Msun")
print(f"Initial particles = {int(asymptot * weight_n)}")

rh = 1.0
v0 = np.sqrt(mbh / rh)
n0 = 2e4 / (206264.0**3)
energy0 = v0**2
x_boundary = eboundary
trlx = 0.34 * np.sqrt(rh**3 / mbh) / np.log(mbh / mc)

print(f"Relaxation time = {trlx:.2e}")

log10_emin = np.log10(emin_factor)
log10_emax = np.log10(emax_factor)

x_centers_log = np.linspace(log10_emin, log10_emax, gx_bins)
x_centers = 10**x_centers_log

log_j_centers = np.linspace(np.log10(jmin), np.log10(jmax), dc_bins)
j_centers = 10**log_j_centers

print("\nInitializing particles...")
np.random.seed(seed_value if same_ini_seed else None)

n_particles = int(asymptot * weight_n)
particles = []

for i in range(n_particles):
    x_init = x_boundary * (np.random.rand()**(1.0 / (1.0 + alpha_ini)))
    e_init = x_init * energy0
    j_init = np.random.uniform(jmin, jmax)
    
    particles.append({
        'id': i,
        'en': e_init,
        'jm': j_init,
        'x': x_init,
        'weight': weight_n,
        'exit_flag': 0
    })

print(f"Created {len(particles)} particles")

def compute_gx(particles, x_centers, x_boundary):
    weights = np.array([p['weight'] for p in particles if p['exit_flag'] == 0])
    x_values = np.array([p['x'] for p in particles if p['exit_flag'] == 0])
    
    gx = np.zeros(len(x_centers))
    
    for i in range(len(x_centers)):
        if i == 0:
            xmin = x_centers[0] / 1.5
        else:
            xmin = np.sqrt(x_centers[i-1] * x_centers[i])
        
        if i == len(x_centers) - 1:
            xmax = x_centers[-1] * 1.5
        else:
            xmax = np.sqrt(x_centers[i] * x_centers[i+1])
        
        mask = (x_values >= xmin) & (x_values < xmax)
        dx = np.log10(xmax) - np.log10(xmin)
        
        if np.sum(mask) > 0 and dx > 0:
            gx[i] = np.sum(weights[mask]) / dx
    
    x_boundary_idx = np.argmin(np.abs(x_centers - x_boundary))
    if gx[x_boundary_idx] > 0:
        gx = gx / gx[x_boundary_idx]
    
    return gx

def compute_density(particles, mbh):
    r_min = np.log10(0.05)
    r_max = 5.2
    n_r_bins = 24
    
    r_centers_log = np.linspace(r_min, r_max, n_r_bins)
    r_centers = 10**r_centers_log
    
    density = np.zeros(len(r_centers))
    
    for p in particles:
        if p['exit_flag'] == 0:
            a = mbh / (2.0 * p['en'])
            e = np.sqrt(max(0, 1.0 - p['jm']**2))
            
            for j, r in enumerate(r_centers):
                r_peri = a * (1.0 - e)
                r_apo = a * (1.0 + e)
                
                if r_peri < r < r_apo:
                    e_orb = -1.0 / (2.0 * a)
                    l_sq = mbh * a * (1.0 - e**2)
                    
                    v_r_sq = 2.0 * (e_orb + mbh / r) - l_sq / r**2
                    
                    if v_r_sq > 0:
                        v_r_inv = 1.0 / np.sqrt(v_r_sq)
                        density[j] += p['weight'] * v_r_inv
    
    for j, r in enumerate(r_centers):
        if density[j] > 0:
            density[j] = density[j] / (np.pi * 4.0 * np.pi * r**2)
    
    return r_centers, density

def save_hdf5(filename, x_centers, gx, r_centers, density):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        grp = f.create_group('1')
        subgrp = grp.create_group('star')
        
        fgx_grp = subgrp.create_group('fgx')
        fgx_grp.create_dataset('   X', data=np.log10(x_centers))
        fgx_grp.create_dataset('  FX', data=gx)
        
        fden_grp = subgrp.create_group('fden')
        fden_grp.create_dataset('   X', data=np.log10(r_centers))
        fden_grp.create_dataset('  FX', data=density)
    
    print(f"  Saved: {filename}")

print("\nComputing initial distribution...")
gx_initial = compute_gx(particles, x_centers, x_boundary)
r_centers, density = compute_density(particles, mbh)

os.makedirs(f'{output_dir}/ini/hdf5', exist_ok=True)
save_hdf5(f'{output_dir}/ini/hdf5/dms_0_0.hdf5', x_centers, gx_initial, r_centers, density)

print(f"\nActive particles: {len([p for p in particles if p['exit_flag'] == 0])}")

print("\nRunning time evolution...")

for snap in range(1, num_snapshots + 1):
    for update in range(1, num_update_per_snap + 1):
        gx_current = compute_gx(particles, x_centers, x_boundary)
        
        dt_update = timestep_snapshot * trlx / num_update_per_snap
        
        for p in particles:
            if p['exit_flag'] == 0:
                x = p['x']
                jm = p['jm']
                
                x_idx = np.argmin(np.abs(x_centers - x))
                gx_here = gx_current[x_idx] if x_idx < len(gx_current) else gx_current[-1]
                
                kappa = (4.0 * np.pi * mc)**2 * np.log(mbh / mc)
                sigma32 = (2.0 * np.pi * v0**2)**(-1.5)
                
                sigma0 = max(gx_here / x, 1e-20)
                
                de_drift = -sigma0 * sigma32 * n0 * kappa * dt_update
                dee = 4.0 / 3.0 * sigma0 * sigma32 * n0 * kappa
                djj = max((10.0 / 3.0) * (jm**2 - jm**4) * sigma0 * sigma32 * n0 * kappa, 1e-20)
                
                y1 = np.clip(np.random.randn(), -6.0, 6.0)
                y2 = np.clip(np.random.randn(), -6.0, 6.0)
                
                de = de_drift + y1 * np.sqrt(dee * dt_update)
                dj = y2 * np.sqrt(djj * dt_update)
                
                a_old = mbh / (2.0 * p['en'])
                j_old = p['jm'] * np.sqrt(mbh * a_old)
                
                e_new = p['en'] + de
                j_new = j_old + dj
                
                if e_new >= x_boundary * energy0:
                    p['exit_flag'] = 1
                    continue
                
                if e_new < emin_factor * energy0:
                    e_new = emin_factor * energy0
                
                a_new = mbh / (2.0 * e_new)
                jm_new = j_new / np.sqrt(mbh * a_new)
                
                jm_new = np.clip(jm_new, jmin, jmax)
                
                p['en'] = e_new
                p['jm'] = jm_new
                p['x'] = e_new / energy0
        
        gx_final = compute_gx(particles, x_centers, x_boundary)
        r_centers, density = compute_density(particles, mbh)
        
        fname = f'{output_dir}/ecev/dms/dms_{snap}_{update}.hdf5'
        save_hdf5(fname, x_centers, gx_final, r_centers, density)
        
        n_active = len([p for p in particles if p['exit_flag'] == 0])
        print(f"Snapshot {snap}/{num_snapshots}, Update {update}/{num_update_per_snap}: {n_active} active particles")

print("\n" + "="*60)
print("Simulation complete!")
print("="*60)
print(f"\nOutput files saved to: {output_dir}/ecev/dms/")
print(f"Initial state saved to: {output_dir}/ini/hdf5/")


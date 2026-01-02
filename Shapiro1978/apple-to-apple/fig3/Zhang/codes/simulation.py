import numpy as np
import h5py
from scipy import integrate, interpolate, special
import os

class GalacticNuclearCluster:
    def __init__(self, model_file='model.in', mfrac_file='mfrac.in'):
        self.read_inputs(model_file, mfrac_file)
        self.setup_physics()
        
    def read_inputs(self, model_file, mfrac_file):
        with open(model_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        self.n_threads = int(lines[0])
        self.jmin = float(lines[1].split()[0])
        self.jmax = float(lines[1].split()[1])
        self.mbh = float(lines[2].split()[0].replace('d', 'e'))
        self.emin_factor = float(lines[3].split()[0].replace('d', 'e'))
        self.emax_factor = float(lines[3].split()[1].replace('d', 'e'))
        self.eboundary = float(lines[4].split()[0].replace('d', 'e'))
        self.seed_value = int(lines[5])
        self.same_ini_seed = int(lines[6])
        bins_line = lines[7].split()
        self.gx_bins = int(bins_line[0])
        self.dc_bins = int(bins_line[1])
        self.same_evl_seed = int(lines[8])
        self.num_update_per_snap = int(lines[9])
        self.timestep_snapshot = float(lines[10])
        self.num_snapshots = int(lines[11])
        self.alpha_ini = float(lines[12])
        self.loss_cone = int(lines[13])
        self.clone_scheme = int(lines[14])
        self.clone_x0 = float(lines[15])
        
        with open(mfrac_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        parts = lines[1].split()
        self.m1 = float(parts[0])
        self.mc = float(parts[1])
        self.m2 = float(parts[2])
        self.asymptot = float(parts[3])
        self.weight_n = float(parts[4])
        self.clone_factor = int(parts[5])
        
        parts2 = lines[2].split()
        self.frac_star = float(parts2[0])
        self.frac_sbh = float(parts2[1])
        
    def setup_physics(self):
        self.rh = 1.0
        self.v0 = np.sqrt(self.mbh / self.rh)
        self.n0 = 2e4 / (206264.0**3)
        self.energy0 = self.v0**2
        self.x_boundary = self.eboundary
        self.trlx = 0.34 * np.sqrt(self.rh**3 / self.mbh) / (np.log(self.mbh / self.mc))
        
        self.log10_emin = np.log10(self.emin_factor)
        self.log10_emax = np.log10(self.emax_factor)
        self.log10_xboundary = np.log10(self.x_boundary)
        
        self.x_bins = np.logspace(self.log10_emin, self.log10_emax, self.gx_bins)
        self.x_centers = 0.5 * (self.x_bins[:-1] + self.x_bins[1:])
        
        self.log_j_bins = np.linspace(np.log10(self.jmin), np.log10(self.jmax), self.dc_bins)
        self.j_bins = 10**self.log_j_bins
        self.j_centers = 10**(0.5 * (self.log_j_bins[:-1] + self.log_j_bins[1:]))
        
    def initialize_particles(self, n_particles=1000):
        np.random.seed(self.seed_value if self.same_ini_seed else None)
        
        particles = []
        for i in range(n_particles):
            x_init = self.x_boundary * (np.random.rand()**(1.0 / (1.0 + self.alpha_ini)))
            
            e_init = x_init * self.energy0
            
            j_init = np.random.uniform(self.jmin, self.jmax)
            
            obtype = 1 if np.random.rand() < self.frac_star else 2
            
            particle = {
                'id': i,
                'en': e_init,
                'jm': j_init,
                'x': x_init,
                'obtype': obtype,
                'weight': self.weight_n,
                'exit_flag': 0
            }
            particles.append(particle)
        
        return particles
    
    def compute_gx(self, particles):
        weights = np.array([p['weight'] for p in particles if p['exit_flag'] == 0])
        x_values = np.array([p['x'] for p in particles if p['exit_flag'] == 0])
        
        gx = np.zeros(len(self.x_centers))
        
        for i, xc in enumerate(self.x_centers):
            if i == 0:
                xmin = self.x_bins[0]
            else:
                xmin = 0.5 * (self.x_centers[i-1] + xc)
            
            if i == len(self.x_centers) - 1:
                xmax = self.x_bins[-1]
            else:
                xmax = 0.5 * (xc + self.x_centers[i+1])
            
            mask = (x_values >= xmin) & (x_values < xmax)
            dx = np.log10(xmax) - np.log10(xmin)
            
            if np.sum(mask) > 0:
                gx[i] = np.sum(weights[mask]) / dx
        
        norm_factor = self.get_normalization_factor(gx)
        gx = gx / norm_factor
        
        return gx
    
    def get_normalization_factor(self, gx):
        x_boundary_idx = np.argmin(np.abs(self.x_centers - self.x_boundary))
        
        if x_boundary_idx > 0:
            g_boundary = gx[x_boundary_idx]
        else:
            g_boundary = gx[0]
        
        if g_boundary > 0:
            return g_boundary
        else:
            return 1.0
    
    def extend_gx_to_boundary(self, gx):
        gx_extended = np.zeros_like(self.x_centers)
        
        for i, x in enumerate(self.x_centers):
            if x <= self.x_boundary:
                gx_extended[i] = np.exp(np.log10(x))
            else:
                gx_extended[i] = gx[i]
        
        return gx_extended
    
    def compute_density(self, particles):
        r_min = np.log10(0.05)
        r_max = 5.2
        n_r_bins = self.gx_bins
        
        r_bins = np.logspace(r_min, r_max, n_r_bins)
        r_centers = 10**(0.5 * (np.log10(r_bins[:-1]) + np.log10(r_bins[1:])))
        
        density = np.zeros(len(r_centers))
        
        for p in particles:
            if p['exit_flag'] == 0:
                a = self.mbh / (2.0 * p['en'])
                e = np.sqrt(1.0 - p['jm']**2)
                
                for j, r in enumerate(r_centers):
                    if a * (1.0 - e) < r < a * (1.0 + e):
                        v_r = self.get_inverse_radial_velocity(r, a, e)
                        if v_r > 0:
                            density[j] += p['weight'] * v_r
        
        for j, r in enumerate(r_centers):
            density[j] = density[j] / (np.pi * 4.0 * np.pi * r**2)
        
        return r_centers, density
    
    def get_inverse_radial_velocity(self, r, a, e):
        if r < a * (1.0 - e) or r > a * (1.0 + e):
            return 0.0
        
        e_orb = -1.0 / (2.0 * a)
        l = np.sqrt(self.mbh * a * (1.0 - e**2))
        
        v_r_sq = 2.0 * (e_orb + self.mbh / r) - l**2 / r**2
        
        if v_r_sq > 0:
            return 1.0 / np.sqrt(v_r_sq)
        else:
            return 0.0
    
    def compute_diffusion_coefficients(self, particles, gx):
        alpha = 7.0 / 4.0
        
        gx_func = interpolate.interp1d(
            np.log10(self.x_centers), 
            gx, 
            kind='linear', 
            fill_value=(gx[0], gx[-1]), 
            bounds_error=False
        )
        
        def extended_gx(x_val):
            if x_val <= 0:
                g0 = gx[np.argmin(np.abs(self.x_centers - self.x_boundary))]
                return g0 * np.exp(x_val)
            elif x_val <= self.x_boundary:
                g0 = gx[np.argmin(np.abs(self.x_centers - self.x_boundary))]
                return g0 * np.exp(x_val)
            else:
                return gx_func(np.log10(x_val))
        
        coeff_grid = {}
        
        for i, x in enumerate(self.x_centers[::2]):
            for j, jm in enumerate(self.j_centers[::2]):
                kappa = (4.0 * np.pi * self.mc)**2 * np.log(self.mbh / self.mc)
                sigma32 = (2.0 * np.pi * self.v0**2)**(-1.5)
                
                def integrand(x_prime):
                    return extended_gx(x_prime)
                
                sigma0, _ = integrate.quad(integrand, self.emin_factor / x, 1.0, limit=50, epsabs=1e-6, epsrel=1e-6)
                sigma0 += extended_gx(0) / x
                
                de_dt = -sigma0 * sigma32 * self.n0 * kappa
                djj_dt = (10.0 / 3.0) * (jm**2 - jm**4) * sigma0 * sigma32 * self.n0 * kappa
                
                coeff_grid[(i, j)] = {
                    'de': de_dt,
                    'djj': djj_dt if djj_dt > 0 else 1e-20
                }
        
        return coeff_grid
    
    def evolve_particle(self, particle, gx, dt):
        x = particle['x']
        jm = particle['jm']
        
        x_idx = np.argmin(np.abs(self.x_centers[::2] - x))
        j_idx = np.argmin(np.abs(self.j_centers[::2] - jm))
        
        alpha = 7.0 / 4.0
        
        x_idx_full = np.argmin(np.abs(self.x_centers - x))
        if x_idx_full < len(gx):
            gx_here = gx[x_idx_full]
        else:
            gx_here = gx[-1]
        
        kappa = (4.0 * np.pi * self.mc)**2 * np.log(self.mbh / self.mc)
        sigma32 = (2.0 * np.pi * self.v0**2)**(-1.5)
        
        sigma0 = gx_here / x
        
        de_drift = -sigma0 * sigma32 * self.n0 * kappa * dt
        dee = 4.0 / 3.0 * sigma0 * sigma32 * self.n0 * kappa
        djj = (10.0 / 3.0) * (jm**2 - jm**4) * sigma0 * sigma32 * self.n0 * kappa
        
        if djj < 0:
            djj = 1e-20
        
        y1 = np.random.randn()
        y2 = np.random.randn()
        
        y1 = np.clip(y1, -6.0, 6.0)
        y2 = np.clip(y2, -6.0, 6.0)
        
        de = de_drift + y1 * np.sqrt(dee * dt)
        dj = y2 * np.sqrt(djj * dt)
        
        a_old = self.mbh / (2.0 * particle['en'])
        j_old = particle['jm'] * np.sqrt(self.mbh * a_old)
        
        e_new = particle['en'] + de
        j_new = j_old + dj
        
        if e_new >= self.x_boundary * self.energy0:
            particle['exit_flag'] = 1
            return
        
        a_new = self.mbh / (2.0 * e_new)
        jm_new = j_new / np.sqrt(self.mbh * a_new)
        
        if jm_new < self.jmin:
            jm_new = 2.0 * self.jmin - jm_new
        if jm_new > self.jmax:
            jm_new = self.jmax
        if jm_new < self.jmin:
            jm_new = self.jmin
        
        particle['en'] = e_new
        particle['jm'] = jm_new
        particle['x'] = e_new / self.energy0
    
    def run_simulation(self, output_dir='output'):
        os.makedirs(f'{output_dir}/ecev/dms', exist_ok=True)
        os.makedirs(f'{output_dir}/ini/hdf5', exist_ok=True)
        
        particles = self.initialize_particles(n_particles=int(self.asymptot * self.weight_n))
        
        gx_initial = self.compute_gx(particles)
        r_centers, density = self.compute_density(particles)
        
        self.save_hdf5(f'{output_dir}/ini/hdf5/dms_0_0.hdf5', 
                       self.x_centers, gx_initial, r_centers, density)
        
        print(f"Initial particles: {len([p for p in particles if p['exit_flag'] == 0])}")
        
        for snap in range(1, self.num_snapshots + 1):
            for update in range(1, self.num_update_per_snap + 1):
                gx_current = self.compute_gx(particles)
                
                dt_update = self.timestep_snapshot * self.trlx / self.num_update_per_snap
                
                for p in particles:
                    if p['exit_flag'] == 0:
                        self.evolve_particle(p, gx_current, dt_update)
                
                gx_final = self.compute_gx(particles)
                r_centers, density = self.compute_density(particles)
                
                fname = f'{output_dir}/ecev/dms/dms_{snap}_{update}.hdf5'
                self.save_hdf5(fname, self.x_centers, gx_final, r_centers, density)
                
                n_active = len([p for p in particles if p['exit_flag'] == 0])
                print(f"Snapshot {snap}, Update {update}: {n_active} active particles")
        
        print("Simulation complete!")
    
    def save_hdf5(self, filename, x_centers, gx, r_centers, density):
        with h5py.File(filename, 'w') as f:
            grp = f.create_group('1')
            subgrp = grp.create_group('star')
            
            fgx_grp = subgrp.create_group('fgx')
            fgx_grp.create_dataset('   X', data=np.log10(x_centers))
            fgx_grp.create_dataset('  FX', data=gx)
            
            fden_grp = subgrp.create_group('fden')
            fden_grp.create_dataset('   X', data=np.log10(r_centers))
            fden_grp.create_dataset('  FX', data=density)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = 'examples/1comp'
    
    model_file = f'{base_dir}/model.in'
    mfrac_file = f'{base_dir}/mfrac.in'
    output_dir = f'{base_dir}/output'
    
    print(f"Reading configuration from {base_dir}")
    print(f"Model file: {model_file}")
    print(f"Mass fraction file: {mfrac_file}")
    
    try:
        gnc = GalacticNuclearCluster(model_file=model_file, mfrac_file=mfrac_file)
        print(f"Configuration loaded successfully")
        print(f"MBH = {gnc.mbh:.2e} Msun")
        print(f"Number of snapshots = {gnc.num_snapshots}")
        print(f"Updates per snapshot = {gnc.num_update_per_snap}")
        gnc.run_simulation(output_dir=output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


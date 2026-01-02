#!/usr/bin/env python
"""
Monte Carlo Fokker-Planck simulation for stellar dynamics around a massive black hole.
This is a Python port of the Fortran GNC code, implementing the same physics for
two-body relaxation-driven evolution of stars in a galactic nucleus.

The code simulates the evolution of particle distribution g(x) where x = E/E_0
is the dimensionless energy, evolving according to diffusion coefficients
computed self-consistently from the distribution function.
"""

import numpy as np
import h5py
import os
from dataclasses import dataclass
from typing import List, Tuple

PC_IN_AU = 206264.806
G_CONST = 4 * np.pi**2


@dataclass
class ModelParams:
    mbh: float = 4e6
    m_star: float = 1.0
    n_particles: int = 800
    clone_factor: int = 30
    jmin: float = 5e-4
    jmax: float = 0.99999
    emin_factor: float = 0.03
    emax_factor: float = 1e5
    x_boundary: float = 0.05
    gx_bins: int = 24
    dc_bins: int = 72
    alpha_ini: float = 0.25
    n_snapshots: int = 10
    n_updates_per_snap: int = 10
    dt_snapshot_tnr: float = 0.1
    seed: int = 100
    loss_cone: bool = False
    clone_scheme: int = 1


def compute_rh_vh_nh(mbh: float) -> Tuple[float, float, float]:
    r0 = 3.1 * PC_IN_AU
    n0 = 2e4 / PC_IN_AU**3
    rh = r0 * (mbh / 4e6)**0.55
    vh = np.sqrt(mbh / rh)
    nh = n0 * (mbh / 4e6)**(-0.65)
    return rh, vh, nh


@dataclass
class Particle:
    mass: float
    energy: float
    j: float
    weight_n: float = 1.0
    weight_clone: float = 1.0
    exit_flag: int = 1


class Simulation:
    def __init__(self, params: ModelParams):
        self.p = params
        np.random.seed(params.seed)
        
        self.rh, self.vh, self.nh = compute_rh_vh_nh(params.mbh)
        self.energy0 = -params.mbh / self.rh
        self.energy_min = self.energy0 * params.emin_factor
        self.energy_max = self.energy0 * params.emax_factor
        self.energy_boundary = params.x_boundary * self.energy0
        
        self.log10_emin = np.log10(params.emin_factor)
        self.log10_emax = np.log10(params.emax_factor)
        self.log10_jmin = np.log10(params.jmin)
        self.log10_jmax = np.log10(params.jmax)
        
        self.clone_e0 = (params.x_boundary / 10) * self.energy0
        
        self.particles: List[Particle] = []
        self.gx = None
        self.gx_x = None
        self.fden = None
        self.fden_r = None
        
        self.n0 = self.nh
        self.v0 = self.vh
        
        sigma32 = (2 * np.pi * self.v0**2)**(-1.5)
        kappa = (4 * np.pi * params.m_star)**2 * np.log(params.mbh / params.m_star)
        self.coeff_scale = sigma32 * self.n0 * kappa
        
        self.tnr = self._compute_tnr()
        self.dt_update = params.dt_snapshot_tnr * self.tnr / params.n_updates_per_snap
        
    def _compute_tnr(self) -> float:
        ln_lambda = np.log(self.p.mbh / self.p.m_star)
        tnr_years = 0.34 * (self.vh * 29.79)**3 / (self.nh * PC_IN_AU**3 * self.p.m_star * ln_lambda) * 1e6
        return tnr_years / 1e6
    
    def initialize_particles(self):
        self.particles = []
        x_bd = self.p.x_boundary
        n_per_task = self.p.n_particles
        
        for i in range(n_per_task):
            x = self._sample_initial_x(x_bd)
            j = self._sample_initial_j()
            en = x * self.energy0
            
            p = Particle(
                mass=self.p.m_star,
                energy=en,
                j=j,
                weight_n=self.p.n_particles / n_per_task,
                weight_clone=self._get_clone_weight(en),
            )
            self.particles.append(p)
    
    def _sample_initial_x(self, x_bd: float) -> float:
        alpha = self.p.alpha_ini
        u = np.random.random()
        x_min = x_bd
        x_max = self.p.emin_factor
        
        if alpha != 1.0:
            x = (u * (x_min**(1 - alpha) - x_max**(1 - alpha)) + x_max**(1 - alpha))**(1 / (1 - alpha))
        else:
            x = x_min * (x_max / x_min)**u
        return x
    
    def _sample_initial_j(self) -> float:
        u = np.random.random()
        j = np.sqrt(u) * (self.p.jmax - self.p.jmin) + self.p.jmin
        return max(self.p.jmin, min(j, self.p.jmax))
    
    def _get_clone_weight(self, en: float) -> float:
        if self.p.clone_scheme < 1:
            return 1.0
        lvl = int(np.log10(en / self.clone_e0))
        if lvl < 0:
            lvl = 0
        return float(self.p.clone_factor ** lvl)
    
    def compute_gx(self):
        x_centers = np.linspace(self.log10_emin, np.log10(self.p.x_boundary), self.p.gx_bins)
        gx = np.zeros(len(x_centers))
        
        for p in self.particles:
            if p.exit_flag != 1:
                continue
            x = p.energy / self.energy0
            log_x = np.log10(x)
            if log_x < self.log10_emin or x > self.p.x_boundary:
                continue
            idx = np.searchsorted(x_centers, log_x)
            idx = min(max(idx, 0), len(gx) - 1)
            weight = p.weight_n * p.weight_clone
            gx[idx] += weight
        
        dx = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1.0
        normalization = np.sum(gx) * dx
        if normalization > 0:
            gx = gx / normalization
        gx_asymp = gx[-1] if len(gx) > 0 and gx[-1] > 0 else 1.0
        gx = gx / gx_asymp if gx_asymp > 0 else gx
        
        self.gx_x = x_centers
        self.gx = gx
        return x_centers, gx
    
    def compute_density(self):
        r_min = np.log10(0.5 * self.rh / self.p.emax_factor)
        r_max = np.log10(0.5 * self.rh / self.p.emin_factor)
        n_bins = self.p.gx_bins
        r_centers = np.linspace(r_min, r_max, n_bins)
        fden = np.zeros(n_bins)
        
        for p in self.particles:
            if p.exit_flag != 1:
                continue
            a = self.p.mbh / (-2 * p.energy)
            ecc = np.sqrt(1 - p.j**2)
            weight = p.weight_n * p.weight_clone
            
            for i, log_r in enumerate(r_centers):
                r = 10**log_r
                rp = a * (1 - ecc)
                ra = a * (1 + ecc)
                if rp <= r <= ra:
                    vr_inv = self._compute_vr_inv(r, a, ecc)
                    fden[i] += weight * vr_inv / (4 * np.pi * r**2)
        
        self.fden_r = r_centers
        self.fden = fden
        return r_centers, fden
    
    def _compute_vr_inv(self, r: float, a: float, ecc: float) -> float:
        if a <= 0:
            return 0.0
        term = 2 / r - 1 / a
        if term <= 0:
            return 0.0
        v2 = self.p.mbh * term
        if v2 <= 0:
            return 0.0
        vt2 = self.p.mbh * a * (1 - ecc**2) / r**2
        vr2 = v2 - vt2
        if vr2 <= 0:
            return 0.0
        return 1.0 / np.sqrt(vr2)
    
    def get_diffusion_coeffs(self, en: float, j: float) -> Tuple[float, float, float, float, float]:
        x = en / self.energy0
        log_x = np.log10(x)
        
        if self.gx is None or len(self.gx) == 0:
            g0 = 1.0
        else:
            idx = np.searchsorted(self.gx_x, log_x)
            idx = min(max(idx, 0), len(self.gx) - 1)
            g0 = self.gx[idx] if self.gx[idx] > 0 else 1.0
        
        alpha = 7.0 / 4.0
        sigma0 = self._compute_sigma0(x, g0, alpha)
        sigma_rest = sigma0 * 0.2
        jc = np.sqrt(self.p.mbh * self.p.mbh / (-2 * en))
        
        de_drift = -sigma0 * en * self.coeff_scale
        de_diff = (4.0 / 3.0) * sigma0 * en**2 * self.coeff_scale
        j2 = j**2
        dj_drift = ((5 - 3 * j2) / 12.0 * sigma0 + sigma_rest) * jc * self.coeff_scale
        dj_diff = ((5 - 3 * j2) / 6.0 * sigma0 + sigma_rest) * jc**2 * self.coeff_scale
        dej = -2.0 / 3.0 * j * sigma0 * en * jc * self.coeff_scale
        
        return de_drift, de_diff, dj_drift, dj_diff, dej
    
    def _compute_sigma0(self, x: float, g0: float, alpha: float) -> float:
        x_min = self.p.emin_factor
        if x <= x_min:
            return g0 / x
        
        integral = 0.0
        n_steps = 100
        ds = (1.0 - x_min / x) / n_steps
        for i in range(n_steps):
            s = x_min / x + (i + 0.5) * ds
            e_prime = s * x
            if e_prime <= 0:
                continue
            g_prime = g0 * (e_prime / x)**(alpha - 1.5)
            integral += g_prime * ds
        
        return integral + g0 / x
    
    def evolve_particle(self, p: Particle, dt: float):
        if p.exit_flag != 1:
            return
        
        en = p.energy
        j = p.j
        de_drift, de_diff, dj_drift, dj_diff, dej = self.get_diffusion_coeffs(en, j)
        
        if de_diff <= 0 or dj_diff <= 0:
            return
        
        rho = dej / np.sqrt(abs(de_diff * dj_diff)) if de_diff * dj_diff > 0 else 0.0
        rho = max(-0.99, min(rho, 0.99))
        
        y1 = np.random.randn()
        y2_raw = np.random.randn()
        y2 = rho * y1 + np.sqrt(1 - rho**2) * y2_raw
        y1 = max(-6, min(y1, 6))
        y2 = max(-6, min(y2, 6))
        
        den = de_drift * dt + y1 * np.sqrt(de_diff * dt)
        djp = dj_drift * dt + y2 * np.sqrt(dj_diff * dt)
        
        a_old = self.p.mbh / (-2 * en)
        jc_old = np.sqrt(self.p.mbh * a_old)
        
        en_new = en + den
        a_new = self.p.mbh / (-2 * en_new) if en_new < 0 else a_old
        jc_new = np.sqrt(self.p.mbh * a_new)
        
        j_dim = j * jc_old + djp
        j_new = j_dim / jc_new if jc_new > 0 else j
        
        if j_new < self.p.jmin:
            j_new = 2 * self.p.jmin - j_new
        if j_new > self.p.jmax:
            j_new = self.p.jmax
        if j_new < self.p.jmin:
            j_new = self.p.jmin
        
        p.energy = en_new
        p.j = j_new
        
        x_new = en_new / self.energy0
        if x_new > self.p.x_boundary:
            p.exit_flag = 4
        elif x_new < self.p.emax_factor:
            p.exit_flag = 5
        
        p.weight_clone = self._get_clone_weight(en_new)
    
    def run_snapshot(self, t_start: float, t_end: float):
        dt = t_end - t_start
        period_factor = 2 * np.pi * 1e6
        
        for p in self.particles:
            if p.exit_flag != 1:
                continue
            
            a = self.p.mbh / (-2 * p.energy)
            de_drift, de_diff, dj_drift, dj_diff, dej = self.get_diffusion_coeffs(p.energy, p.j)
            
            if de_diff > 0:
                time_scale_e = min((p.energy * 0.15)**2 / de_diff, abs(p.energy * 0.15) / abs(de_drift + 1e-30))
            else:
                time_scale_e = 1e6
            
            jc = np.sqrt(self.p.mbh * a)
            if dj_diff > 0:
                time_scale_j = min((jc * 0.1)**2 / dj_diff, (0.4 * (1 - p.j) * jc)**2 / dj_diff)
            else:
                time_scale_j = 1e6
            
            time_scale = min(time_scale_e, time_scale_j, dt)
            n_steps = max(1, int(dt / time_scale))
            dt_step = dt / n_steps
            
            for _ in range(n_steps):
                self.evolve_particle(p, dt_step)
                if p.exit_flag != 1:
                    break
        
        self._handle_boundary_particles()
        self.compute_gx()
        self.compute_density()
    
    def _handle_boundary_particles(self):
        new_particles = []
        for p in self.particles:
            if p.exit_flag == 4:
                new_p = Particle(
                    mass=p.mass,
                    energy=self._sample_initial_x(self.p.x_boundary) * self.energy0,
                    j=self._sample_initial_j(),
                    weight_n=p.weight_n,
                )
                new_p.weight_clone = self._get_clone_weight(new_p.energy)
                new_particles.append(new_p)
            elif p.exit_flag == 1:
                new_particles.append(p)
        self.particles = new_particles
    
    def output_hdf5(self, filename: str, snap_id: int, update_id: int):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, 'w') as f:
            grp_fgx = f.create_group('1/star/fgx')
            grp_fgx.create_dataset('   X', data=self.gx_x)
            grp_fgx.create_dataset('  FX', data=self.gx)
            
            grp_fden = f.create_group('1/star/fden')
            grp_fden.create_dataset('   X', data=self.fden_r)
            grp_fden.create_dataset('  FX', data=self.fden)
            
            f.attrs['snapshot'] = snap_id
            f.attrs['update'] = update_id
            f.attrs['mbh'] = self.p.mbh
            f.attrs['rh'] = self.rh
    
    def run(self, output_dir: str = 'output/ecev/dms'):
        print("Initializing simulation...")
        print(f"  MBH = {self.p.mbh:.2e} Msun")
        print(f"  rh = {self.rh:.2e} AU = {self.rh/PC_IN_AU:.2f} pc")
        print(f"  n0 = {self.n0 * PC_IN_AU**3:.2e} pc^-3")
        print(f"  T_NR = {self.tnr:.2f} Myr")
        print(f"  dt_update = {self.dt_update:.4f} Myr")
        
        self.initialize_particles()
        print(f"  Initialized {len(self.particles)} particles")
        
        self.compute_gx()
        self.compute_density()
        os.makedirs(output_dir, exist_ok=True)
        
        time = 0.0
        for snap in range(1, self.p.n_snapshots + 1):
            for update in range(1, self.p.n_updates_per_snap + 1):
                t_start = time
                t_end = time + self.dt_update
                
                print(f"Running snapshot {snap}, update {update}/{self.p.n_updates_per_snap}, "
                      f"t = {t_start:.4f} - {t_end:.4f} Myr")
                
                self.run_snapshot(t_start, t_end)
                
                filename = f"{output_dir}/dms_{snap}_{update}.hdf5"
                self.output_hdf5(filename, snap, update)
                
                time = t_end
                n_active = sum(1 for p in self.particles if p.exit_flag == 1)
                print(f"  Active particles: {n_active}/{len(self.particles)}")
        
        print("Simulation complete!")


def main():
    params = ModelParams(
        mbh=4e6,
        m_star=1.0,
        n_particles=800,
        clone_factor=30,
        jmin=5e-4,
        jmax=0.99999,
        emin_factor=0.03,
        emax_factor=1e5,
        x_boundary=0.05,
        gx_bins=24,
        dc_bins=72,
        alpha_ini=0.25,
        n_snapshots=10,
        n_updates_per_snap=10,
        dt_snapshot_tnr=0.1,
        seed=100,
        loss_cone=False,
        clone_scheme=1,
    )
    
    sim = Simulation(params)
    sim.run(output_dir='output/ecev/dms')


if __name__ == '__main__':
    main()


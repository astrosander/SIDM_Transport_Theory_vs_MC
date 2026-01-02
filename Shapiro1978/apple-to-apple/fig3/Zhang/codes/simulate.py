#!/usr/bin/env python
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

def compute_rh_vh_nh(mbh):
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
    def __init__(self, params):
        self.p = params
        np.random.seed(params.seed)
        self.rh, self.vh, self.nh = compute_rh_vh_nh(params.mbh)
        self.energy0 = params.mbh / self.rh
        self.energy_min = self.energy0 * params.emin_factor
        self.energy_max = self.energy0 * params.emax_factor
        self.energy_boundary = params.x_boundary * self.energy0
        self.log10_emin = np.log10(params.emin_factor)
        self.log10_emax = np.log10(params.emax_factor)
        self.log10_jmin = np.log10(params.jmin)
        self.log10_jmax = np.log10(params.jmax)
        self.clone_e0 = params.x_boundary * self.energy0
        self.particles = []
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

    def _compute_tnr(self):
        ln_lambda = np.log(self.p.mbh / self.p.m_star)
        tnr_years = 0.34 * (self.vh * 29.79)**3 / (self.nh * PC_IN_AU**3 * self.p.m_star * ln_lambda) * 1e6
        return tnr_years / 1e6

    def initialize_particles(self):
        self.particles = []
        n_per_task = self.p.n_particles
        for i in range(n_per_task):
            x = self._sample_initial_x()
            j = self._sample_initial_j()
            en = x * self.energy0
            p = Particle(
                mass=self.p.m_star,
                energy=en,
                j=j,
                weight_n=self.p.n_particles / n_per_task,
                weight_clone=self._get_clone_weight(x)
            )
            self.particles.append(p)

    def _sample_initial_x(self):
        alpha = self.p.alpha_ini
        u = np.random.random()
        x_min = self.p.x_boundary
        x_max = 10.0
        if alpha != 1.0:
            exp = 1 - alpha
            x = (u * (x_max**exp - x_min**exp) + x_min**exp)**(1.0/exp)
        else:
            x = x_min * np.exp(u * np.log(x_max / x_min))
        return x

    def _sample_initial_j(self):
        u = np.random.random()
        j = np.sqrt(u) * (self.p.jmax - self.p.jmin) + self.p.jmin
        return max(self.p.jmin, min(j, self.p.jmax))

    def _get_clone_weight(self, x):
        if self.p.clone_scheme < 1:
            return 1.0
        if x <= self.p.x_boundary:
            return 1.0
        lvl = int(np.log10(x / self.p.x_boundary))
        if lvl < 0:
            lvl = 0
        return float(self.p.clone_factor ** lvl)

    def compute_gx(self):
        n_bins = self.p.gx_bins
        x_edges = np.logspace(np.log10(self.p.x_boundary), np.log10(self.p.emax_factor), n_bins + 1)
        x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
        gx = np.zeros(n_bins)
        total_weight = 0.0
        for p in self.particles:
            if p.exit_flag != 1:
                continue
            x = p.energy / self.energy0
            if x < self.p.x_boundary or x > self.p.emax_factor:
                continue
            idx = np.searchsorted(x_edges, x) - 1
            idx = min(max(idx, 0), n_bins - 1)
            weight = p.weight_n * p.weight_clone
            dx = x_edges[idx+1] - x_edges[idx]
            gx[idx] += weight / dx
            total_weight += weight
        if total_weight > 0:
            gx = gx / total_weight * (x_edges[-1] - x_edges[0])
        alpha = self.p.alpha_ini
        gx_theo_boundary = self.p.x_boundary**(0.25 - alpha)
        if gx[0] > 0:
            gx = gx / gx[0] * gx_theo_boundary
        self.gx_x = np.log10(x_centers)
        self.gx = gx
        return self.gx_x, self.gx

    def compute_density(self):
        r_min_log = np.log10(0.5 * self.rh / self.p.emax_factor)
        r_max_log = np.log10(2.0 * self.rh / self.p.x_boundary)
        n_bins = self.p.gx_bins
        r_edges = np.logspace(r_min_log, r_max_log, n_bins + 1)
        r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
        fden = np.zeros(n_bins)
        for p in self.particles:
            if p.exit_flag != 1:
                continue
            x = p.energy / self.energy0
            if x <= 0:
                continue
            a = self.rh / (2 * x)
            ecc_sq = 1 - p.j**2
            if ecc_sq < 0:
                ecc_sq = 0
            ecc = np.sqrt(ecc_sq)
            rp = a * (1 - ecc)
            ra = a * (1 + ecc)
            weight = p.weight_n * p.weight_clone
            for i in range(n_bins):
                r = r_centers[i]
                if rp <= r <= ra:
                    dr = r_edges[i+1] - r_edges[i]
                    shell_vol = 4 * np.pi * r**2 * dr
                    orbit_time = 2 * np.pi * np.sqrt(a**3 / self.p.mbh)
                    vr_inv = self._compute_vr_inv(r, a, ecc)
                    fden[i] += weight * vr_inv / orbit_time / shell_vol if orbit_time > 0 else 0
        self.fden_r = np.log10(r_centers / self.rh)
        self.fden = fden
        return self.fden_r, self.fden

    def _compute_vr_inv(self, r, a, ecc):
        if a <= 0 or r <= 0:
            return 0.0
        term = 2 / r - 1 / a
        if term <= 0:
            return 0.0
        v2 = self.p.mbh * term
        l2 = self.p.mbh * a * (1 - ecc**2)
        vt2 = l2 / r**2
        vr2 = v2 - vt2
        if vr2 <= 1e-30:
            return 0.0
        return 1.0 / np.sqrt(vr2)

    def get_diffusion_coeffs(self, en, j):
        x = en / self.energy0
        if x <= 0:
            return 0, 0, 0, 0, 0
        log_x = np.log10(x)
        if self.gx is None or len(self.gx) == 0:
            g0 = x**(0.25 - self.p.alpha_ini)
        else:
            idx = np.searchsorted(self.gx_x, log_x)
            idx = min(max(idx, 0), len(self.gx) - 1)
            g0 = max(self.gx[idx], 1e-10)
        alpha = self.p.alpha_ini
        sigma0 = self._compute_sigma0(x, g0, alpha)
        a = self.rh / (2 * x)
        jc2 = self.p.mbh * a
        jc = np.sqrt(jc2) if jc2 > 0 else 1.0
        t_nr_local = self.tnr * (x / self.p.x_boundary)**(1.5 - alpha)
        scale = en / t_nr_local if t_nr_local > 0 else 0
        de_drift = -sigma0 * scale
        de_diff = (4.0 / 3.0) * sigma0 * en * scale
        j2 = j**2
        sigma_j = sigma0 * (5 - 3*j2) / 12.0
        dj_drift = sigma_j * jc / t_nr_local if t_nr_local > 0 else 0
        dj_diff = sigma_j * jc2 / t_nr_local * 2 if t_nr_local > 0 else 0
        dej = -2.0/3.0 * j * sigma0 * en * jc / t_nr_local if t_nr_local > 0 else 0
        return de_drift, de_diff, dj_drift, dj_diff, dej

    def _compute_sigma0(self, x, g0, alpha):
        x_min = self.p.x_boundary
        if x <= x_min:
            return g0 / x
        integral = g0 * (1 - (x_min/x)**(1.5-alpha)) / (1.5 - alpha) if abs(1.5 - alpha) > 0.01 else g0 * np.log(x / x_min)
        return integral / x + g0 / x

    def evolve_particle(self, p, dt):
        if p.exit_flag != 1:
            return
        en = p.energy
        j = p.j
        x = en / self.energy0
        de_drift, de_diff, dj_drift, dj_diff, dej = self.get_diffusion_coeffs(en, j)
        if de_diff <= 1e-30 or dj_diff <= 1e-30:
            return
        corr = dej / np.sqrt(de_diff * dj_diff) if de_diff * dj_diff > 0 else 0.0
        corr = max(-0.99, min(corr, 0.99))
        y1 = np.random.randn()
        y2_raw = np.random.randn()
        y2 = corr * y1 + np.sqrt(1 - corr**2) * y2_raw
        y1 = max(-5, min(y1, 5))
        y2 = max(-5, min(y2, 5))
        den = de_drift * dt + y1 * np.sqrt(de_diff * dt)
        a_old = self.rh / (2 * x)
        jc_old = np.sqrt(self.p.mbh * a_old)
        djp = dj_drift * dt + y2 * np.sqrt(dj_diff * dt)
        en_new = en + den
        x_new = en_new / self.energy0
        if x_new <= 0:
            x_new = 0.01 * self.p.x_boundary
            en_new = x_new * self.energy0
        a_new = self.rh / (2 * x_new)
        jc_new = np.sqrt(self.p.mbh * a_new) if a_new > 0 else jc_old
        j_dim = j * jc_old + djp
        j_new = j_dim / jc_new if jc_new > 0 else j
        if j_new < self.p.jmin:
            j_new = 2 * self.p.jmin - j_new
        if j_new > self.p.jmax:
            j_new = 2 * self.p.jmax - j_new
        j_new = max(self.p.jmin, min(j_new, self.p.jmax))
        p.energy = en_new
        p.j = j_new
        if x_new < self.p.x_boundary:
            p.exit_flag = 4
        elif x_new > self.p.emax_factor:
            p.exit_flag = 5
        p.weight_clone = self._get_clone_weight(x_new)

    def run_snapshot(self, t_start, t_end):
        dt_total = t_end - t_start
        for p in self.particles:
            if p.exit_flag != 1:
                continue
            x = p.energy / self.energy0
            de_drift, de_diff, dj_drift, dj_diff, dej = self.get_diffusion_coeffs(p.energy, p.j)
            if de_diff > 0:
                time_scale = min(
                    (0.1 * p.energy)**2 / de_diff,
                    abs(0.1 * p.energy / (de_drift + 1e-30))
                )
            else:
                time_scale = dt_total
            n_steps = max(1, min(int(dt_total / time_scale) + 1, 1000))
            dt_step = dt_total / n_steps
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
                x_new = self._sample_initial_x()
                new_p = Particle(
                    mass=p.mass,
                    energy=x_new * self.energy0,
                    j=self._sample_initial_j(),
                    weight_n=p.weight_n
                )
                new_p.weight_clone = self._get_clone_weight(x_new)
                new_particles.append(new_p)
            elif p.exit_flag == 1:
                new_particles.append(p)
        self.particles = new_particles

    def output_hdf5(self, filename, snap_id, update_id):
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

    def run(self, output_dir='output/ecev/dms'):
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
                print(f"Snapshot {snap}, update {update}/{self.p.n_updates_per_snap}, t = {t_start:.4f} - {t_end:.4f} Myr")
                self.run_snapshot(t_start, t_end)
                filename = f"{output_dir}/dms_{snap}_{update}.hdf5"
                self.output_hdf5(filename, snap, update)
                time = t_end
                n_active = sum(1 for p in self.particles if p.exit_flag == 1)
                print(f"  Active: {n_active}/{len(self.particles)}")
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
        clone_scheme=1
    )
    sim = Simulation(params)
    sim.run(output_dir='output/ecev/dms')

if __name__ == '__main__':
    main()

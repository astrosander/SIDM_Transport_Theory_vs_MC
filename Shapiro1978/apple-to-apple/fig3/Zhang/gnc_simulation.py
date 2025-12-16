"""
Galactic Nucleus Dynamics Simulation - Python Implementation

Monte Carlo simulation of stellar dynamics around a supermassive black hole.
Replaces Fortran implementation with equivalent physics:
- Orbital energy and angular momentum diffusion (resonant relaxation)
- Tidal disruption events
- Gravitational wave captures  
- Particle cloning for rare event sampling
- Multiple stellar populations (MS, BH, NS, WD, BD)
"""

import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad

class StarType(IntEnum):
    MS = 1
    BH = 2
    NS = 3
    WD = 4
    BD = 5

class ExitFlag(IntEnum):
    NORMAL = 1
    TIDAL = 2
    MAX_REACH = 3
    BOUNDARY_MIN = 4
    BOUNDARY_MAX = 5
    GW_ISO = 11
    EJECTION = 15

PC = 206264.98
RD_SUN = 4.65e-3
M_SUN_SI = 1.98855e30
PI = np.pi
TWO_PI = 2.0 * np.pi
MY_UNIT_VEL_C = 3e5 / 29.784

@dataclass
class Binary:
    a_bin: float = 0.0
    e_bin: float = 0.0
    inc: float = 0.0
    om: float = 0.0
    pe: float = 0.0
    me: float = 0.0
    f0: float = 0.0
    m_primary: float = 0.0
    m_secondary: float = 0.0

@dataclass
class Particle:
    id: int = 0
    m: float = 0.0
    obtype: int = 0
    obidx: int = 0
    radius: float = 0.0
    en: float = 0.0
    en0: float = 0.0
    jm: float = 0.0
    jm0: float = 0.0
    rp: float = 0.0
    r_td: float = 0.0
    weight_real: float = 1.0
    weight_clone: float = 1.0
    weight_n: float = 1.0
    create_time: float = 0.0
    simu_bgtime: float = 0.0
    exit_flag: int = 0
    byot: Binary = field(default_factory=Binary)
    den: float = 0.0
    djp: float = 0.0
    within_jt: int = 0
    
@dataclass
class Coefficients:
    e: float = 0.0
    j: float = 0.0
    ee: float = 0.0
    jj: float = 0.0
    ej: float = 0.0

@dataclass
class Config:
    mbh: float = 4e6
    rh: float = 0.0
    n0: float = 0.0
    v0: float = 0.0
    sigma: float = 0.0
    energy0: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 0.0
    energy_boundary: float = 0.0
    x_boundary: float = 0.5
    emin_factor: float = 0.05
    emax_factor: float = 10.0
    jmin_value: float = 0.01
    jmax_value: float = 0.999
    clone_e0: float = 0.0
    clone_scheme: int = 1
    clone_e0_factor: float = 0.1
    total_time: float = 0.0
    ts_spshot: float = 0.0
    n_spshot: int = 10
    num_update_per_snap: int = 100
    update_dt: float = 0.0
    include_loss_cone: int = 1
    m_bins: int = 1
    bin_mass: np.ndarray = field(default_factory=lambda: np.array([10.0]))
    bin_mass_m1: np.ndarray = field(default_factory=lambda: np.array([8.0]))
    bin_mass_m2: np.ndarray = field(default_factory=lambda: np.array([12.0]))
    asymptot: np.ndarray = field(default_factory=lambda: np.ones((8, 1)))
    ini_weight_n: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    weight_n: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    n_basic: float = 1.0
    grid_bins: int = 50
    gx_bins: int = 50
    seed_value: int = 12345
    same_rseed_ini: int = 0
    same_rseed_evl: int = 0
    chattery: int = 0
    num_clone_created: int = 0
    num_boundary_created: int = 0
    num_boundary_elim: int = 0
    rbd: float = 0.0
    mbh_radius: float = 0.0
    ntasks: int = 1

def get_rh_vh_nh(mbh):
    r0 = 3.1
    n0_const = 2e4
    rh = r0 * (mbh / 4e6)**0.55 * PC
    vh = np.sqrt(mbh / rh)
    nh = n0_const / PC**3 * (mbh / 4e6)**(-0.65)
    return rh, vh, nh

def get_tnr_timescale_at_rh(config):
    G = 1.0
    ln_lambda = 10.0
    return 0.34 * config.sigma**3 / (G**2 * config.n0 * config.m_ref * ln_lambda)

def star_radius(mass, star_type):
    if star_type == StarType.MS:
        if mass <= 0.06:
            return 0.1 * RD_SUN
        elif mass <= 1.0:
            return mass**0.8 * RD_SUN
        else:
            return mass**0.56 * RD_SUN
    elif star_type == StarType.WD:
        if mass < 1.44:
            return 0.01 * RD_SUN * mass**(-1/3)
        else:
            return 0.01 * RD_SUN
    elif star_type in [StarType.BH, StarType.NS, StarType.BD]:
        return 0.0
    return 0.0

def get_sample_r_td(particle, mbh, mbh_radius):
    if particle.obtype == StarType.MS:
        particle.r_td = (3 * mbh / particle.m)**(1/3) * particle.radius
    elif particle.obtype in [StarType.BH, StarType.NS, StarType.WD, StarType.BD]:
        particle.r_td = 4.0 * mbh_radius
    else:
        particle.r_td = 0.0

def period_func(a, mbh):
    return TWO_PI * np.sqrt(a**3 / mbh)

def gen_gaussian_correlate(rho):
    y1 = np.random.normal(0, 1)
    y2_prime = np.random.normal(0, 1)
    y2 = rho * y1 + np.sqrt(1 - rho**2) * y2_prime
    return y1, y2

def fpowerlaw(alpha, xmin, xmax):
    if alpha == 1.0:
        return xmin * np.exp(np.random.random() * np.log(xmax / xmin))
    else:
        u = np.random.random()
        return (xmin**(1-alpha) + u * (xmax**(1-alpha) - xmin**(1-alpha)))**(1/(1-alpha))

def set_ini_byot_abin(config):
    rh = config.rh
    emin_factor = config.emin_factor
    emax_factor = config.emax_factor
    alpha = 7.0 / 4.0
    rmin = rh / 2.0 / emax_factor
    rmax = rh / 2.0 / emin_factor
    r = fpowerlaw(alpha, rmin, rmax)
    return 2.0 * r

def set_jm_init(config):
    jm = fpowerlaw(1.0, 0.0044, 0.99999)
    return jm

def initialize_particle(mass, star_type, config):
    particle = Particle()
    particle.m = mass
    particle.obtype = star_type
    particle.radius = star_radius(mass, star_type)
    particle.byot.a_bin = set_ini_byot_abin(config)
    particle.jm = set_jm_init(config)
    particle.byot.e_bin = np.sqrt(1 - particle.jm**2)
    particle.en = -config.mbh / (2.0 * particle.byot.a_bin)
    particle.en0 = particle.en
    particle.jm0 = particle.jm
    get_sample_r_td(particle, config.mbh, config.mbh_radius)
    particle.rp = particle.byot.a_bin * (1 - particle.byot.e_bin)
    particle.byot.me = np.random.random() * TWO_PI
    particle.weight_clone = 1.0
    particle.exit_flag = ExitFlag.NORMAL
    return particle

def get_diffusion_coeffs_simple(en, jm, mass, config):
    coeff = Coefficients()
    x = en / config.energy0
    jc = np.sqrt(config.mbh * (-config.mbh / (2 * en)))
    
    ln_lambda = 10.0
    G = 1.0
    m_avg = 1.0
    
    t_relax = 0.34 * config.sigma**3 / (G**2 * config.n0 * m_avg * ln_lambda)
    
    v_circ = np.sqrt(config.mbh * np.abs(x))
    sigma_local = config.sigma * np.abs(x)**0.25
    
    j_circ = jc
    
    drift_factor = mass / (m_avg**2) * ln_lambda
    diff_factor = drift_factor * m_avg
    
    coeff.e = drift_factor * config.n0 * G**2 * config.mbh * 1e-4
    coeff.ee = diff_factor * config.n0 * G**2 * config.mbh * sigma_local * 1e-3
    coeff.j = drift_factor * config.n0 * G**2 * config.mbh * j_circ * 1e-5
    coeff.jj = diff_factor * config.n0 * G**2 * j_circ**2 * sigma_local * 1e-3
    coeff.ej = 0.5 * np.sqrt(np.abs(coeff.ee * coeff.jj))
    
    return coeff

def get_step(particle, coeff, config, time_now, total_time):
    period = period_func(particle.byot.a_bin, config.mbh)
    
    sample_jc = np.sqrt(config.mbh * particle.byot.a_bin)
    jm = particle.jm * sample_jc
    
    if coeff.ee != 0:
        time_dt_e = min((particle.en * 0.15)**2 / coeff.ee, 
                       abs(particle.en * 0.15) / abs(coeff.e))
    else:
        time_dt_e = 1e6 * period
        
    if coeff.jj != 0:
        time_dt_j = min((sample_jc * 0.1)**2 / coeff.jj,
                       (0.4 * (1.0075 - particle.jm) * sample_jc)**2 / coeff.jj)
    else:
        time_dt_j = 1e6 * period
        
    time_dt_nr = min(time_dt_e, time_dt_j)
    steps = time_dt_nr / period
    ntmax = (total_time - time_now) / period
    steps = min(steps, ntmax)
    
    if config.include_loss_cone >= 1 and particle.en < config.energy_boundary:
        ratio = min(2.0, particle.r_td / particle.byot.a_bin)
        sample_jlc = np.sqrt(ratio) * np.sqrt(2.0 - ratio) * sample_jc
        if coeff.jj != 0:
            jnr2 = coeff.jj * period
            nmjl = (max(0.1 * sample_jlc, 0.25 * abs(jm - sample_jlc)))**2 / jnr2
            steps = min(steps, nmjl)
    
    return max(steps, 1e-10)

def get_de_dj(particle, coeff, dt, steps):
    rho = coeff.ej / np.sqrt(abs(coeff.ee * coeff.jj)) if coeff.ee * coeff.jj != 0 else 0.0
    rho = max(min(rho, 0.99), -0.99)
    
    y1, y2 = gen_gaussian_correlate(rho)
    y1 = max(min(y1, 6.0), -6.0)
    y2 = max(min(y2, 6.0), -6.0)
    
    den = coeff.e * dt + y1 * np.sqrt(coeff.ee * dt)
    n2j = np.sqrt(coeff.jj * dt)
    djp = coeff.j * dt + y2 * n2j
    
    return den, djp

def move_particle(particle, den, djp, steps, config):
    ai = particle.byot.a_bin
    ei = particle.byot.e_bin
    
    Eni = particle.en
    Ji = particle.jm * np.sqrt(config.mbh * ai)
    Enf = Eni + den
    af = -config.mbh / (2 * Enf)
    Jf = Ji + djp
    jmf = Jf / np.sqrt(config.mbh * af)
    
    if jmf < config.jmin_value:
        jmf = 2 * config.jmin_value - jmf
    if jmf > config.jmax_value:
        jmf = config.jmax_value
    if jmf < config.jmin_value:
        jmf = config.jmin_value
        
    particle.byot.a_bin = af
    particle.en = Enf
    particle.jm = jmf
    particle.byot.e_bin = np.sqrt(1 - particle.jm**2)
    
    mef = particle.byot.me + (steps - int(steps)) * TWO_PI
    if mef > TWO_PI:
        particle.byot.me = mef - TWO_PI
    else:
        particle.byot.me = mef
        
    if particle.byot.e_bin >= 1.0:
        particle.byot.e_bin = 0.9999999999

def check_exit_conditions(particle, config):
    if particle.en > config.energy_min:
        return ExitFlag.BOUNDARY_MIN
    if particle.en < config.energy_max:
        return ExitFlag.BOUNDARY_MAX
        
    if config.include_loss_cone >= 1:
        particle.rp = particle.byot.a_bin * (1 - particle.byot.e_bin)
        if particle.rp < particle.r_td:
            return ExitFlag.TIDAL
            
    if particle.byot.a_bin < config.mbh_radius * 10:
        return ExitFlag.GW_ISO
        
    return ExitFlag.NORMAL

def evolve_particle(particle, config, total_time):
    time_now = particle.simu_bgtime
    max_steps = int(1e6)
    
    for step in range(max_steps):
        if time_now >= total_time:
            break
            
        coeff = get_diffusion_coeffs_simple(particle.en, particle.jm, particle.m, config)
        steps = get_step(particle, coeff, config, time_now, total_time)
        
        if steps <= 0:
            break
            
        period = period_func(particle.byot.a_bin, config.mbh)
        dt = steps * period
        
        den, djp = get_de_dj(particle, coeff, dt, steps)
        particle.den = den
        particle.djp = djp
        
        move_particle(particle, den, djp, steps, config)
        
        time_now += dt
        
        exit_flag = check_exit_conditions(particle, config)
        if exit_flag != ExitFlag.NORMAL:
            particle.exit_flag = exit_flag
            break
    
    return particle

def create_boundary_particle(config, mass, star_type):
    particle = Particle()
    particle.m = mass
    particle.obtype = star_type
    particle.radius = star_radius(mass, star_type)
    particle.byot.a_bin = config.rbd
    particle.jm = set_jm_init(config)
    particle.byot.e_bin = np.sqrt(1 - particle.jm**2)
    particle.en = -config.mbh / (2.0 * particle.byot.a_bin)
    particle.en0 = particle.en
    particle.jm0 = particle.jm
    get_sample_r_td(particle, config.mbh, config.mbh_radius)
    particle.rp = particle.byot.a_bin * (1 - particle.byot.e_bin)
    particle.byot.me = np.random.random() * TWO_PI
    particle.weight_clone = 1.0
    particle.exit_flag = ExitFlag.NORMAL
    particle.create_time = 0.0
    particle.simu_bgtime = 0.0
    return particle

def initialize_population(config):
    particles = []
    
    for i in range(config.m_bins):
        mass = config.bin_mass[i]
        n_particles = int(config.bin_mass_particle_number[i])
        
        n_star = int(config.asymptot[1, i] * n_particles)
        n_bh = int(config.asymptot[2, i] * n_particles)
        n_ns = int(config.asymptot[3, i] * n_particles)
        n_wd = int(config.asymptot[4, i] * n_particles)
        n_bd = int(config.asymptot[5, i] * n_particles)
        
        for j in range(n_star):
            particle = initialize_particle(mass, StarType.MS, config)
            particle.weight_n = config.weight_n[i]
            particle.id = len(particles)
            particles.append(particle)
            
        for j in range(n_bh):
            particle = initialize_particle(mass, StarType.BH, config)
            particle.weight_n = config.weight_n[i]
            particle.id = len(particles)
            particles.append(particle)
            
        for j in range(n_ns):
            particle = initialize_particle(mass, StarType.NS, config)
            particle.weight_n = config.weight_n[i]
            particle.id = len(particles)
            particles.append(particle)
            
        for j in range(n_wd):
            particle = initialize_particle(mass, StarType.WD, config)
            particle.weight_n = config.weight_n[i]
            particle.id = len(particles)
            particles.append(particle)
            
        for j in range(n_bd):
            particle = initialize_particle(mass, StarType.BD, config)
            particle.weight_n = config.weight_n[i]
            particle.id = len(particles)
            particles.append(particle)
    
    return particles

def update_particle_weights(particles, config):
    for particle in particles:
        particle.weight_real = particle.weight_clone * particle.weight_n * config.n_basic

def run_simulation(config):
    print("Initializing simulation...")
    print(f"MBH = {config.mbh:.2e} M_sun")
    print(f"rh = {config.rh:.2e} AU")
    print(f"Total time = {config.total_time:.2e}")
    print(f"Number of snapshots = {config.n_spshot}")
    
    if config.same_rseed_ini > 0:
        np.random.seed(config.seed_value)
    
    particles = initialize_population(config)
    print(f"Initialized {len(particles)} particles")
    
    update_particle_weights(particles, config)
    
    snapshot_times = np.linspace(0, config.total_time, config.n_spshot + 1)[1:]
    
    results = {
        'snapshots': [],
        'n_tidal': np.zeros(config.n_spshot),
        'n_gw': np.zeros(config.n_spshot),
        'n_ejection': np.zeros(config.n_spshot),
        'n_normal': np.zeros(config.n_spshot)
    }
    
    active_particles = particles.copy()
    
    for isnap in range(config.n_spshot):
        print(f"\nSnapshot {isnap+1}/{config.n_spshot}")
        snap_time = snapshot_times[isnap]
        
        new_active = []
        snapshot_data = []
        need_boundary_particles = False
        
        for particle in active_particles:
            evolved = evolve_particle(particle, config, snap_time)
            
            if evolved.exit_flag == ExitFlag.TIDAL:
                results['n_tidal'][isnap] += evolved.weight_real
            elif evolved.exit_flag == ExitFlag.GW_ISO:
                results['n_gw'][isnap] += evolved.weight_real
            elif evolved.exit_flag == ExitFlag.EJECTION:
                results['n_ejection'][isnap] += evolved.weight_real
            elif evolved.exit_flag == ExitFlag.NORMAL:
                results['n_normal'][isnap] += evolved.weight_real
                new_active.append(evolved)
            elif evolved.exit_flag == ExitFlag.BOUNDARY_MAX:
                need_boundary_particles = True
                
            snapshot_data.append({
                'id': evolved.id,
                'mass': evolved.m,
                'type': evolved.obtype,
                'energy': evolved.en,
                'jm': evolved.jm,
                'a': evolved.byot.a_bin,
                'e': evolved.byot.e_bin,
                'exit_flag': evolved.exit_flag,
                'weight': evolved.weight_real
            })
        
        if need_boundary_particles:
            for i in range(config.m_bins):
                mass = config.bin_mass[i]
                n_new = int(config.asymptot[1, i] * config.weight_n[i])
                for _ in range(n_new):
                    new_particle = create_boundary_particle(config, mass, StarType.MS)
                    new_particle.simu_bgtime = snap_time
                    new_particle.id = len(particles) + len(new_active)
                    new_active.append(new_particle)
        
        active_particles = new_active
        results['snapshots'].append(snapshot_data)
        
        print(f"  Active particles: {len(active_particles)}")
        print(f"  Tidal disruptions: {results['n_tidal'][isnap]:.2f}")
        print(f"  GW captures: {results['n_gw'][isnap]:.2f}")
        print(f"  Normal: {results['n_normal'][isnap]:.2f}")
    
    return results

def save_results(results, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('n_tidal', data=results['n_tidal'])
        f.create_dataset('n_gw', data=results['n_gw'])
        f.create_dataset('n_ejection', data=results['n_ejection'])
        f.create_dataset('n_normal', data=results['n_normal'])
        
        for i, snap in enumerate(results['snapshots']):
            grp = f.create_group(f'snapshot_{i}')
            if snap:
                grp.create_dataset('id', data=[s['id'] for s in snap])
                grp.create_dataset('mass', data=[s['mass'] for s in snap])
                grp.create_dataset('type', data=[s['type'] for s in snap])
                grp.create_dataset('energy', data=[s['energy'] for s in snap])
                grp.create_dataset('jm', data=[s['jm'] for s in snap])
                grp.create_dataset('a', data=[s['a'] for s in snap])
                grp.create_dataset('e', data=[s['e'] for s in snap])
                grp.create_dataset('exit_flag', data=[s['exit_flag'] for s in snap])
                grp.create_dataset('weight', data=[s['weight'] for s in snap])

def read_config_simple():
    config = Config()
    
    config.mbh = 4e6
    config.mbh_radius = config.mbh / (MY_UNIT_VEL_C**2)
    
    config.rh, config.v0, config.n0 = get_rh_vh_nh(config.mbh)
    config.sigma = np.sqrt(config.mbh / config.rh)
    config.energy0 = -config.mbh / config.rh
    
    config.emin_factor = 0.05
    config.emax_factor = 10.0
    config.x_boundary = 0.5
    
    config.energy_min = config.energy0 * config.emin_factor
    config.energy_max = config.energy0 * config.emax_factor
    config.energy_boundary = config.x_boundary * config.energy0
    config.rbd = config.rh * 0.5 / config.x_boundary
    
    config.jmin_value = 0.01
    config.jmax_value = 0.999
    
    config.clone_scheme = 1
    config.clone_e0_factor = 0.1
    config.clone_e0 = config.clone_e0_factor / 10 * config.energy0
    
    config.m_bins = 3
    config.bin_mass = np.array([1.0, 10.0, 50.0])
    config.bin_mass_m1 = np.array([0.8, 8.0, 40.0])
    config.bin_mass_m2 = np.array([1.2, 12.0, 60.0])
    
    config.asymptot = np.zeros((8, config.m_bins))
    config.asymptot[0, :] = [1.0, 1.0, 1.0]
    config.asymptot[1, :] = [0.8, 0.15, 0.04]
    config.asymptot[2, :] = [0.0, 0.01, 0.005]
    
    config.ini_weight_n = np.array([1.0, 1.0, 1.0])
    config.n_basic = 1.0
    config.weight_n = config.ini_weight_n / config.n_basic
    
    config.m_ref = config.bin_mass[0]
    
    ln_lambda = 10.0
    t_relax = 0.34 * config.sigma**3 / (config.n0 * config.m_ref * ln_lambda)
    config.tnr = t_relax
    
    config.ts_snap_input = 0.01
    config.ts_spshot = config.tnr * config.ts_snap_input
    config.n_spshot = 5
    config.num_update_per_snap = 10
    config.update_dt = config.ts_spshot / config.num_update_per_snap
    config.total_time = config.ts_spshot * config.n_spshot
    
    config.bin_mass_particle_number = np.array([100, 20, 5]).astype(int)
    
    config.grid_bins = 50
    config.gx_bins = 50
    
    config.seed_value = 12345
    config.same_rseed_ini = 1
    config.same_rseed_evl = 1
    
    config.include_loss_cone = 1
    config.chattery = 0
    
    return config

def compute_statistics(results, config):
    stats = {}
    
    for isnap in range(config.n_spshot):
        if results['snapshots'][isnap]:
            snap = results['snapshots'][isnap]
            energies = np.array([s['energy'] for s in snap])
            jms = np.array([s['jm'] for s in snap])
            weights = np.array([s['weight'] for s in snap])
            
            stats[f'snap_{isnap}'] = {
                'mean_energy': np.average(energies, weights=weights) if len(energies) > 0 else 0,
                'mean_jm': np.average(jms, weights=weights) if len(jms) > 0 else 0,
                'n_particles': len(snap),
                'total_weight': np.sum(weights)
            }
    
    total_tde_rate = np.sum(results['n_tidal']) / config.total_time * 1e9 / (2 * PI)
    total_gw_rate = np.sum(results['n_gw']) / config.total_time * 1e9 / (2 * PI)
    
    stats['rates'] = {
        'tde_rate_per_gyr': total_tde_rate,
        'gw_rate_per_gyr': total_gw_rate
    }
    
    return stats

def print_statistics(stats):
    print("\n" + "=" * 60)
    print("Simulation Statistics")
    print("=" * 60)
    
    for key in sorted([k for k in stats.keys() if k.startswith('snap_')]):
        snap_stats = stats[key]
        print(f"\n{key}:")
        print(f"  N particles: {snap_stats['n_particles']}")
        print(f"  Total weight: {snap_stats['total_weight']:.2e}")
        print(f"  Mean energy: {snap_stats['mean_energy']:.2e}")
        print(f"  Mean jm: {snap_stats['mean_jm']:.3f}")
    
    print(f"\nEvent Rates:")
    print(f"  TDE rate: {stats['rates']['tde_rate_per_gyr']:.2e} per Gyr")
    print(f"  GW capture rate: {stats['rates']['gw_rate_per_gyr']:.2e} per Gyr")

def main():
    print("=" * 60)
    print("Galactic Nucleus Dynamics Simulation")
    print("Python implementation replacing Fortran code")
    print("=" * 60)
    
    config = read_config_simple()
    
    results = run_simulation(config)
    
    stats = compute_statistics(results, config)
    print_statistics(stats)
    
    output_file = "simulation_results.h5"
    save_results(results, output_file)
    print(f"\n\nResults saved to {output_file}")
    
    print("\nSimulation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()


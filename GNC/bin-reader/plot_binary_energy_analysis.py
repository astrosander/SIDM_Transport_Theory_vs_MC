#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import glob
import os

output_dir = "single"
snapshot_range = range(1, 11)
sigma_h_kms = 23.6
component = "all"

MSUN_G = 1.98847e33
KM_TO_CM = 1.0e5
SEC_PER_GYR = 3.15576e16

class FortranBinaryReader:
    
    def __init__(self, filename):
        self.f = open(filename, 'rb')
    
    def close(self):
        self.f.close()
    
    def read_real8(self, n=1):
        data = np.fromfile(self.f, dtype=np.float64, count=n)
        if len(data) == 0:
            return None
        return data if n > 1 else data[0]
    
    def read_int4(self, n=1):
        data = np.fromfile(self.f, dtype=np.int32, count=n)
        if len(data) == 0:
            return None
        return data if n > 1 else data[0]
    
    def skip_bytes(self, n):
        self.f.seek(n, 1)

class ParticleSample:
    
    def __init__(self):
        self.exit_time = 0.0
        self.r_td = 0.0
        self.m = 0.0
        self.en0 = 0.0
        self.jm0 = 0.0
        self.pd = 0.0
        self.rp = 0.0
        self.tgw = 0.0
        self.djp0 = 0.0
        self.state_flag_last = 0
        self.obtype = 0
        self.obidx = 0
        self.rid = 0
        self.idx = 0
        self.id = 0
        self.exit_flag = 0
        self.create_time = 0.0
        self.Jm = 0.0
        self.En = 0.0
        self.den = 0.0
        self.weight_real = 0.0
        self.weight_N = 0.0
        self.a_bin = 0.0
        self.e_bin = 0.0

def read_particle_sample(reader):
    try:
        sp = ParticleSample()
        
        record1_reals = reader.read_real8(9)
        if record1_reals is None:
            return None
        record1_reals = np.atleast_1d(record1_reals)
        if len(record1_reals) < 9:
            return None
        record1_reals = np.nan_to_num(record1_reals, nan=0.0, posinf=0.0, neginf=0.0)
        
        sp.exit_time = float(record1_reals[0])
        sp.r_td = float(record1_reals[1])
        sp.m = float(record1_reals[2])
        sp.en0 = float(record1_reals[3])
        sp.jm0 = float(record1_reals[4])
        sp.pd = float(record1_reals[5])
        sp.rp = float(record1_reals[6])
        sp.tgw = float(record1_reals[7])
        sp.djp0 = float(record1_reals[8])
        
        state_flag = reader.read_int4(1)
        if state_flag is None:
            return None
        state_flag = np.atleast_1d(state_flag)
        sp.state_flag_last = int(state_flag[0]) if len(state_flag) > 0 else 0
        
        record2 = reader.read_int4(5)
        if record2 is None:
            return None
        record2 = np.atleast_1d(record2)
        if len(record2) < 5:
            return None
        record2 = np.nan_to_num(record2, nan=0, posinf=0, neginf=0).astype(int)
        sp.obtype = int(record2[0])
        sp.obidx = int(record2[1])
        sp.rid = int(record2[2])
        sp.idx = int(record2[3])
        sp.id = int(record2[4])
        
        reader.skip_bytes(10*4)
        binary_reals = reader.read_real8(2)
        if binary_reals is not None:
            binary_reals = np.atleast_1d(binary_reals)
            if len(binary_reals) >= 2:
                binary_reals = np.nan_to_num(binary_reals, nan=0.0, posinf=0.0, neginf=0.0)
                sp.a_bin = float(binary_reals[0])
                sp.e_bin = float(binary_reals[1])
        
        reader.skip_bytes(39*8 + 100)
        reader.skip_bytes(2 * (10*4 + 41*8 + 100))
        
        length_to_expand = reader.read_int4(1)
        if length_to_expand is None:
            return None
        length_to_expand = np.atleast_1d(length_to_expand)
        
        exit_flag = reader.read_int4(1)
        if exit_flag is None:
            return None
        exit_flag = np.atleast_1d(exit_flag)
        sp.exit_flag = int(exit_flag[0]) if len(exit_flag) > 0 else 0
        
        record4_reals1 = reader.read_real8(4)
        if record4_reals1 is None:
            return None
        record4_reals1 = np.atleast_1d(record4_reals1)
        if len(record4_reals1) < 4:
            return None
        record4_reals1 = np.nan_to_num(record4_reals1, nan=0.0, posinf=0.0, neginf=0.0)
        sp.create_time = float(record4_reals1[0])
        sp.Jm = float(record4_reals1[1])
        sp.En = float(record4_reals1[2])
        sp.den = float(record4_reals1[3])
        
        reader.skip_bytes(2*4 + 3*8)
        
        within_jt = reader.read_int4(1)
        length = reader.read_int4(1)
        
        weights = reader.read_real8(2)
        if weights is not None:
            weights = np.atleast_1d(weights)
            if len(weights) >= 2:
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                sp.weight_real = float(weights[0])
                sp.weight_N = float(weights[1])
        
        if length is not None:
            length = np.atleast_1d(length)
            if len(length) > 0 and int(length[0]) > 0:
                track_len = int(length[0])
                reader.skip_bytes(track_len * 84)
        
        return sp
        
    except Exception as e:
        print(f"Error reading particle: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_chain_binary(filename):
    if not os.path.exists(filename):
        return []
    
    reader = FortranBinaryReader(filename)
    particles = []
    
    try:
        n = reader.read_int4(1)
        if n is None or n <= 0:
            reader.close()
            return particles
        
        print(f"  Reading {n} particles from {os.path.basename(filename)}...")
        
        for i in range(n):
            flag = reader.read_int4(1)
            if flag is None:
                break
            
            sp = read_particle_sample(reader)
            if sp is not None:
                particles.append(sp)
        
    except Exception as e:
        print(f"  Error reading chain: {e}")
    finally:
        reader.close()
    
    return particles

def validate_particle_data(particles, n_examples=5):
    if len(particles) == 0:
        print("  WARNING: No particles loaded!")
        return
    
    print(f"\n  Validation: Loaded {len(particles)} particles")
    print(f"  Showing first {min(n_examples, len(particles))} examples:")
    print("-" * 100)
    print(f"{'ID':>12} {'obtype':>8} {'m':>15} {'En':>18} {'weight_real':>18} {'exit_flag':>10} {'a_bin':>15}")
    print("-" * 100)
    
    for i, sp in enumerate(particles[:n_examples]):
        print(f"{sp.id:>12} {sp.obtype:>8} {sp.m:>15.6e} {sp.En:>18.6e} "
              f"{sp.weight_real:>18.6e} {sp.exit_flag:>10} {sp.a_bin:>15.6e}")
    
    valid_particles = [sp for sp in particles if sp.weight_real > 0 and sp.m > 0 and np.isfinite(sp.En)]
    if len(valid_particles) > 0:
        energies = [abs(sp.En) for sp in valid_particles]
        masses = [sp.m for sp in valid_particles]
        weights = [sp.weight_real for sp in valid_particles]
        
        print("-" * 100)
        print(f"  Statistics (valid particles only):")
        print(f"    Valid particles: {len(valid_particles)}/{len(particles)}")
        if len(energies) > 0:
            print(f"    Energy range: [{min(energies):.6e}, {max(energies):.6e}]")
            print(f"    Mass range: [{min(masses):.6e}, {max(masses):.6e}]")
            print(f"    Weight range: [{min(weights):.6e}, {max(weights):.6e}]")
        
        nan_count = sum(1 for sp in particles if (np.isnan(sp.En) or np.isnan(sp.m) or 
                                                  np.isnan(sp.weight_real) or 
                                                  np.isinf(sp.En) or np.isinf(sp.m)))
        if nan_count > 0:
            print(f"    WARNING: {nan_count} particles with NaN/Inf values!")
        
        component_counts = {}
        for sp in particles:
            comp_name = {1: 'BH', 2: 'NS', 3: 'MS', 4: 'WD', 5: 'BD'}.get(sp.obtype, 'Other')
            component_counts[comp_name] = component_counts.get(comp_name, 0) + 1
        print(f"    Component breakdown: {component_counts}")
        
        exit_counts = {}
        for sp in particles:
            exit_name = {1: 'normal', 4: 'bound_min', 5: 'bound_max', 
                        2: 'tidal', 8: 'td_empty', 9: 'td_full'}.get(sp.exit_flag, f'flag_{sp.exit_flag}')
            exit_counts[exit_name] = exit_counts.get(exit_name, 0) + 1
        print(f"    Exit flags: {exit_counts}")
    else:
        print("  WARNING: No valid particles found (all have zero weight, mass, or invalid energy)!")
    
    print("-" * 100)

def gather_snapshot_data(output_dir, snapshot_num):
    patterns = [
        f"{output_dir}/bin/single/samchn_*_{snapshot_num:4d}.bin",
        f"{output_dir}/samchn*_{snapshot_num}.bin",
        f"{output_dir}/samchn*_{snapshot_num:4d}.bin",
        f"{output_dir}/*/samchn*_{snapshot_num}.bin",
        f"output/samchn10_6.bin"
    ]
    
    files = []
    for pattern in patterns:
        found = glob.glob(pattern)
        if found:
            files.extend(found)
    
    if not files:
        direct_file = f"{output_dir}/samchn10_{snapshot_num}.bin"
        if os.path.exists(direct_file):
            files = [direct_file]
    
    print(f"  Pattern search found {len(files)} files")

    if not files:
        print(f"No files found for snapshot {snapshot_num}")
        return []
    
    all_particles = []
    for fname in files:
        particles = read_chain_binary(fname)
        all_particles.extend(particles)
    
    if len(all_particles) > 0:
        validate_particle_data(all_particles, n_examples=3)
    
    return all_particles

def compute_kinetic_energy(particles):
    K_tot = 0.0
    K_unweighted = 0.0
    
    for sp in particles:
        if sp.weight_real > 0 and sp.m > 0:
            K_i = abs(sp.En)
            K_tot += sp.weight_real * sp.m * K_i
            K_unweighted += sp.m * K_i
    
    return K_tot, K_unweighted

def analyze_energy_distribution(particles, component_filter=None):
    energies = []
    masses = []
    weights = []
    exit_flags = []
    
    component_map = {
        'BH': 1,
        'NS': 2,
        'MS': 3,
        'WD': 4,
        'BD': 5,
    }
    
    for sp in particles:
        if component_filter and component_filter != 'all':
            if component_filter.upper() in component_map:
                if sp.obtype != component_map[component_filter.upper()]:
                    continue
        
        if (sp.weight_real > 0 and sp.m > 0 and 
            np.isfinite(sp.En) and np.isfinite(sp.weight_real) and np.isfinite(sp.m) and
            not np.isnan(sp.En) and not np.isinf(sp.En)):
            energies.append(abs(sp.En))
            masses.append(sp.m)
            weights.append(sp.weight_real)
            exit_flags.append(sp.exit_flag)
    
    if len(energies) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return np.array(energies), np.array(masses), np.array(weights), np.array(exit_flags)

def plot_energy_distributions(snapshot_data, sigma_h_kms):
    n_snapshots = len(snapshot_data)
    if n_snapshots == 0:
        print("No data to plot!")
        return
    
    ncols = min(3, n_snapshots)
    nrows = (n_snapshots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_snapshots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx, (snap_num, particles) in enumerate(snapshot_data.items()):
        ax = axes[idx]
        
        energies, masses, weights, _ = analyze_energy_distribution(particles, component)
        
        if len(energies) == 0:
            ax.text(0.5, 0.5, f'Snapshot {snap_num}\n(no data)', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        valid_mask = (energies > 0) & np.isfinite(energies) & np.isfinite(weights) & np.isfinite(masses)
        energies_valid = energies[valid_mask]
        weights_valid = weights[valid_mask]
        masses_valid = masses[valid_mask]
        
        if len(energies_valid) == 0:
            ax.text(0.5, 0.5, f'Snapshot {snap_num}\n(no valid data)', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        emin = energies_valid.min()
        emax = energies_valid.max()
        if emin > 0 and emax > emin:
            bins = np.logspace(np.log10(emin), np.log10(emax), 30)
            weights_clipped = np.clip(weights_valid * masses_valid, 0, 1e100)
            ax.hist(energies_valid, bins=bins, weights=weights_clipped, 
                   alpha=0.7, edgecolor='black', label='Weighted')
        else:
            bins = np.linspace(emin, emax, 30)
            weights_clipped = np.clip(weights_valid * masses_valid, 0, 1e100)
            ax.hist(energies_valid, bins=bins, weights=weights_clipped, 
                   alpha=0.7, edgecolor='black', label='Weighted')
        
        if emin > 0 and emax > emin:
            ax.set_xscale('log')
        ax.set_xlabel(r'$|E|$ (dimensionless)')
        ax.set_ylabel(r'Weighted count ($M_\odot \times x$)')
        ax.set_title(f'Snapshot {snap_num}')
        ax.grid(alpha=0.3)
        
        K_tot, _ = compute_kinetic_energy(particles)
        ax.text(0.05, 0.95, f'K_tot = {K_tot:.2e}', 
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(n_snapshots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('energy_distributions_by_snapshot.png', dpi=150)
    print("Saved energy_distributions_by_snapshot.png")
    plt.show()

def plot_total_kinetic_energy(snapshot_data, sigma_h_kms):
    snapshots = sorted(snapshot_data.keys())
    K_tots = []
    K_unweighted = []
    
    for snap in snapshots:
        K_tot, K_unwt = compute_kinetic_energy(snapshot_data[snap])
        K_tots.append(K_tot)
        K_unweighted.append(K_unwt)
    
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_E = MSUN_G * sigma2_cgs
    K_tots_erg = np.array(K_tots) * unit_E
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(snapshots, K_tots, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Snapshot number')
    ax1.set_ylabel(r'$K_{tot}$ (dimensionless, $M_\odot \times x$)')
    ax1.set_title(f'Total Kinetic Energy Evolution ({component})')
    ax1.grid(alpha=0.3)
    
    ax2.plot(snapshots, K_tots_erg, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Snapshot number')
    ax2.set_ylabel(r'$K_{tot}$ (erg)')
    ax2.set_title(f'Total Kinetic Energy (σ_h = {sigma_h_kms} km/s)')
    ax2.grid(alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('total_kinetic_energy_evolution.png', dpi=150)
    print("Saved total_kinetic_energy_evolution.png")
    plt.show()

def plot_energy_outflow_rate(snapshot_data, dt_myr, sigma_h_kms):
    snapshots = sorted(snapshot_data.keys())
    outflow_rates = []
    
    for snap in snapshots:
        particles = snapshot_data[snap]
        
        E_out = 0.0
        for sp in particles:
            if sp.exit_flag in [4, 5]:
                E_out += sp.weight_real * sp.m * abs(sp.En)
        
        rate = E_out / dt_myr * 1e3
        outflow_rates.append(rate)
    
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_P = MSUN_G * sigma2_cgs / SEC_PER_GYR
    outflow_rates_ergs = np.array(outflow_rates) * unit_P
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(snapshots, outflow_rates, 'o-', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Snapshot number')
    ax1.set_ylabel(r'$\dot{E}_{out}$ (dimensionless, $M_\odot \times x$/Gyr)')
    ax1.set_title(f'Energy Outflow Rate ({component})')
    ax1.grid(alpha=0.3)
    
    ax2.plot(snapshots, outflow_rates_ergs, 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax2.set_xlabel('Snapshot number')
    ax2.set_ylabel(r'$\dot{E}_{out}$ (erg/s)')
    ax2.set_title(f'Energy Outflow Power (σ_h = {sigma_h_kms} km/s)')
    ax2.grid(alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('energy_outflow_rate.png', dpi=150)
    print("Saved energy_outflow_rate.png")
    plt.show()

def create_summary_plot(snapshot_data, dt_myr, sigma_h_kms):
    snapshots = sorted(snapshot_data.keys())
    K_tots = []
    outflow_rates = []
    n_particles = []
    
    for snap in snapshots:
        particles = snapshot_data[snap]
        K_tot, _ = compute_kinetic_energy(particles)
        K_tots.append(K_tot)
        n_particles.append(len(particles))
        
        E_out = sum(sp.weight_real * sp.m * abs(sp.En) 
                   for sp in particles if sp.exit_flag in [4, 5])
        outflow_rates.append(E_out / dt_myr * 1e3)
    
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_E = MSUN_G * sigma2_cgs
    unit_P = unit_E / SEC_PER_GYR
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    K_tots_erg = np.array(K_tots) * unit_E
    ax.plot(snapshots, K_tots_erg, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel(r'$K_{tot}$ (erg)')
    ax.set_title('Total Kinetic Energy')
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax = axes[0, 1]
    outflow_ergs = np.array(outflow_rates) * unit_P
    ax.plot(snapshots, outflow_ergs, 'o-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel(r'$\dot{E}_{out}$ (erg/s)')
    ax.set_title('Energy Outflow Rate')
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax = axes[1, 0]
    K_arr = np.array(K_tots)
    E_arr = np.array(outflow_rates)
    tdep = np.full_like(K_arr, np.nan)
    nonzero_mask = np.abs(E_arr) > 1e-20
    tdep[nonzero_mask] = K_arr[nonzero_mask] / E_arr[nonzero_mask]
    ax.plot(snapshots, tdep, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel(r'$K/\dot{E}_{out}$ (Gyr)')
    ax.set_title('Energy Depletion Timescale')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(snapshots, n_particles, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Number of particles')
    ax.set_title(f'Particle Count ({component})')
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Energy Budget Summary (σ_h = {sigma_h_kms} km/s)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('energy_summary.png', dpi=150)
    print("Saved energy_summary.png")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("GNC Binary Energy Analysis")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Snapshots: {list(snapshot_range)}")
    print(f"Component filter: {component}")
    print(f"σ_h = {sigma_h_kms} km/s")
    print("="*70)
    
    snapshot_data = {}
    for snap_num in snapshot_range:
        print(f"\nReading snapshot {snap_num}...")
        particles = gather_snapshot_data(output_dir, snap_num)
        if len(particles) > 0:
            snapshot_data[snap_num] = particles
            print(f"  Loaded {len(particles)} particles")
        else:
            print(f"  No particles found")

    if not snapshot_data:
        print("\nNo data found! Check your output directory and snapshot range.")
        exit(1)
    
    print(f"\nSuccessfully loaded {len(snapshot_data)} snapshots")

    dt_myr = 0.1
    
    print("\nGenerating plots...")
    print("-" * 70)
    
    plot_energy_distributions(snapshot_data, sigma_h_kms)
    plot_total_kinetic_energy(snapshot_data, sigma_h_kms)
    plot_energy_outflow_rate(snapshot_data, dt_myr, sigma_h_kms)
    create_summary_plot(snapshot_data, dt_myr, sigma_h_kms)
    
    print("-" * 70)
    print("Saving all plotting data to NPZ file...")
    
    snapshots = sorted(snapshot_data.keys())
    
    K_tots = []
    K_unweighted = []
    outflow_rates = []
    n_particles = []
    energy_distributions = {}
    
    for snap in snapshots:
        particles = snapshot_data[snap]
        K_tot, K_unwt = compute_kinetic_energy(particles)
        K_tots.append(K_tot)
        K_unweighted.append(K_unwt)
        n_particles.append(len(particles))
        
        E_out = sum(sp.weight_real * sp.m * abs(sp.En) 
                   for sp in particles if sp.exit_flag in [4, 5])
        outflow_rates.append(E_out / dt_myr * 1e3)
        
        energies, masses, weights, exit_flags = analyze_energy_distribution(particles, component)
        energy_distributions[snap] = {
            'energies': energies,
            'masses': masses,
            'weights': weights,
            'exit_flags': exit_flags
        }
    
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_E = MSUN_G * sigma2_cgs
    unit_P = unit_E / SEC_PER_GYR
    K_tots_erg = np.array(K_tots) * unit_E
    outflow_rates_ergs = np.array(outflow_rates) * unit_P
    
    K_arr = np.array(K_tots)
    E_arr = np.array(outflow_rates)
    tdep = np.full_like(K_arr, np.nan)
    nonzero_mask = np.abs(E_arr) > 1e-20
    tdep[nonzero_mask] = K_arr[nonzero_mask] / E_arr[nonzero_mask]
    
    npz_data = {
        'snapshots': np.array(snapshots),
        'K_tots': np.array(K_tots),
        'K_unweighted': np.array(K_unweighted),
        'K_tots_erg': K_tots_erg,
        'outflow_rates': np.array(outflow_rates),
        'outflow_rates_ergs': outflow_rates_ergs,
        'n_particles': np.array(n_particles),
        'depletion_timescales': tdep,
        'sigma_h_kms': sigma_h_kms,
        'dt_myr': dt_myr,
        'component': component,
        'output_dir': output_dir,
        'snapshot_range_start': snapshot_range.start if hasattr(snapshot_range, 'start') else snapshot_range[0],
        'snapshot_range_stop': snapshot_range.stop if hasattr(snapshot_range, 'stop') else snapshot_range[-1] + 1,
    }
    
    for snap in snapshots:
        ed = energy_distributions[snap]
        npz_data[f'energies_snap{snap}'] = ed['energies']
        npz_data[f'masses_snap{snap}'] = ed['masses']
        npz_data[f'weights_snap{snap}'] = ed['weights']
        npz_data[f'exit_flags_snap{snap}'] = ed['exit_flags']
    
    npz_filename = 'energy_analysis_data.npz'
    np.savez(npz_filename, **npz_data)
    print(f"Saved all plotting data to {npz_filename}")
    
    print("-" * 70)
    print("Analysis complete!")
    print("\nGenerated files:")
    print("  - energy_distributions_by_snapshot.png")
    print("  - total_kinetic_energy_evolution.png")
    print("  - energy_outflow_rate.png")
    print("  - energy_summary.png")
    print(f"  - {npz_filename}")

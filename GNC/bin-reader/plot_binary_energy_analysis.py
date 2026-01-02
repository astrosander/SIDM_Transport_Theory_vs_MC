#!/usr/bin/env python3
"""
Read GNC binary particle sample files and analyze energy distributions.

This script reads Fortran unformatted binary files containing particle samples
and plots:
1. Energy distribution histograms per snapshot
2. Total kinetic energy evolution
3. Energy outflow rate

No command-line arguments - edit USER INPUTS section below.
"""

import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import glob
import os

# ============================================================================
# USER INPUTS
# ============================================================================
output_dir = "single"           # Directory containing snapshot subdirectories
snapshot_range = range(1, 11)   # Snapshots to analyze (e.g., 1 to 10)
sigma_h_kms = 23.6              # Velocity dispersion at boundary (km/s) - from printed output
component = "all"               # "MS", "BH", "NS", "WD", "BD", or "all"

# Physical constants (CGS)
MSUN_G = 1.98847e33             # g
KM_TO_CM = 1.0e5                # cm/km
SEC_PER_GYR = 3.15576e16        # s/Gyr

# ============================================================================
# Fortran binary reader utilities
# ============================================================================

class FortranBinaryReader:
    """Read Fortran unformatted binary files with stream access."""
    
    def __init__(self, filename):
        self.f = open(filename, 'rb')
    
    def close(self):
        self.f.close()
    
    def read_real8(self, n=1):
        """Read n double precision reals (8 bytes each)."""
        data = np.fromfile(self.f, dtype=np.float64, count=n)
        if len(data) == 0:
            return None
        return data if n > 1 else data[0]
    
    def read_int4(self, n=1):
        """Read n 4-byte integers."""
        data = np.fromfile(self.f, dtype=np.int32, count=n)
        if len(data) == 0:
            return None
        return data if n > 1 else data[0]
    
    def skip_bytes(self, n):
        """Skip n bytes."""
        self.f.seek(n, 1)

class ParticleSample:
    """Container for particle sample data."""
    
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
        self.En = 0.0  # Current orbital energy (dimensionless)
        self.den = 0.0
        self.weight_real = 0.0
        self.weight_N = 0.0
        self.a_bin = 0.0
        self.e_bin = 0.0

def read_particle_sample(reader):
    """
    Read one particle sample from binary file.
    
    Based on read_sample_info in md_particle_sample.f90:
    Record 1: exit_time, r_td, m, en0, jm0, pd, rp, tgw, djp0 (9 reals), state_flag_last (1 int)
    Record 2: obtype, obidx, rid, idx, id (5 ints)
    Record 3: byot, byot_ini, byot_bf (3 binary structures - complex, skip most)
    Record 4: length_to_expand (int), exit_flag (int), create_time, Jm, En, den,
              write_down_track, track_step, weight_clone, djp, elp, within_jt, length, weight_real, weight_N
    """
    try:
        sp = ParticleSample()
        
        # Record 1: 9 reals + 1 integer
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
        
        # state_flag_last is an INTEGER, not real!
        state_flag = reader.read_int4(1)
        if state_flag is None:
            return None
        state_flag = np.atleast_1d(state_flag)
        sp.state_flag_last = int(state_flag[0]) if len(state_flag) > 0 else 0
        
        # Record 2: 5 integers
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
        
        # Record 3: 3 binary structures (byot, byot_ini, byot_bf)
        # Each binary: 10 ints + 41 reals + 100 char string = 10*4 + 41*8 + 100 = 468 bytes
        # Try to read a_bin and e_bin from first binary (they're early in the structure)
        # Skip 10 integers, then read first 2 reals (a_bin, e_bin)
        reader.skip_bytes(10*4)  # Skip 10 integers
        binary_reals = reader.read_real8(2)  # Read a_bin, e_bin
        if binary_reals is not None:
            binary_reals = np.atleast_1d(binary_reals)
            if len(binary_reals) >= 2:
                binary_reals = np.nan_to_num(binary_reals, nan=0.0, posinf=0.0, neginf=0.0)
                sp.a_bin = float(binary_reals[0])
                sp.e_bin = float(binary_reals[1])
        
        # Skip rest of first binary: remaining 39 reals + 100 char string
        reader.skip_bytes(39*8 + 100)
        # Skip other two binaries completely
        reader.skip_bytes(2 * (10*4 + 41*8 + 100))
        
        # Record 4: length_to_expand (int), exit_flag (int), then many reals and ints
        length_to_expand = reader.read_int4(1)
        if length_to_expand is None:
            return None
        length_to_expand = np.atleast_1d(length_to_expand)
        
        exit_flag = reader.read_int4(1)
        if exit_flag is None:
            return None
        exit_flag = np.atleast_1d(exit_flag)
        sp.exit_flag = int(exit_flag[0]) if len(exit_flag) > 0 else 0
        
        # Now read: create_time, Jm, En, den (4 reals)
        record4_reals1 = reader.read_real8(4)
        if record4_reals1 is None:
            return None
        record4_reals1 = np.atleast_1d(record4_reals1)
        if len(record4_reals1) < 4:
            return None
        record4_reals1 = np.nan_to_num(record4_reals1, nan=0.0, posinf=0.0, neginf=0.0)
        sp.create_time = float(record4_reals1[0])
        sp.Jm = float(record4_reals1[1])
        sp.En = float(record4_reals1[2])  # Current orbital energy (dimensionless)
        sp.den = float(record4_reals1[3])
        
        # Skip: write_down_track (int), track_step (int), weight_clone (real), djp (real), elp (real)
        reader.skip_bytes(2*4 + 3*8)  # 2 ints + 3 reals
        
        # Read: within_jt (int), length (int)
        within_jt = reader.read_int4(1)
        length = reader.read_int4(1)
        
        # Read: weight_real, weight_N (2 reals)
        weights = reader.read_real8(2)
        if weights is not None:
            weights = np.atleast_1d(weights)
            if len(weights) >= 2:
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                sp.weight_real = float(weights[0])
                sp.weight_N = float(weights[1])
        
        # Skip track data if present
        if length is not None:
            length = np.atleast_1d(length)
            if len(length) > 0 and int(length[0]) > 0:
                track_len = int(length[0])
                # Each track entry is a track_type: time, ac, ec, ain, ein, Incin, incout, omin, omout (9 reals)
                # + state_flag, ms_star_type, mm_star_type (3 ints) = 9*8 + 3*4 = 84 bytes per track
                reader.skip_bytes(track_len * 84)
        
        return sp
        
    except Exception as e:
        print(f"Error reading particle: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_chain_binary(filename):
    """
    Read a chain binary file containing multiple particle samples.
    Format: n (int4), then n particle samples.
    """
    if not os.path.exists(filename):
        return []
    
    reader = FortranBinaryReader(filename)
    particles = []
    
    try:
        # Read number of particles in chain
        n = reader.read_int4(1)
        if n is None or n <= 0:
            reader.close()
            return particles
        
        print(f"  Reading {n} particles from {os.path.basename(filename)}...")
        
        # Read each particle
        for i in range(n):
            # Read flag indicating particle type
            flag = reader.read_int4(1)
            if flag is None:
                break
            
            # Read particle data
            sp = read_particle_sample(reader)
            if sp is not None:
                particles.append(sp)
        
    except Exception as e:
        print(f"  Error reading chain: {e}")
    finally:
        reader.close()
    
    return particles

def validate_particle_data(particles, n_examples=5):
    """
    Print validation information about loaded particles to verify correct reading.
    """
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
    
    # Statistics
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
        
        # Check for NaN/Inf
        nan_count = sum(1 for sp in particles if (np.isnan(sp.En) or np.isnan(sp.m) or 
                                                  np.isnan(sp.weight_real) or 
                                                  np.isinf(sp.En) or np.isinf(sp.m)))
        if nan_count > 0:
            print(f"    WARNING: {nan_count} particles with NaN/Inf values!")
        
        # Component breakdown
        component_counts = {}
        for sp in particles:
            comp_name = {1: 'BH', 2: 'NS', 3: 'MS', 4: 'WD', 5: 'BD'}.get(sp.obtype, 'Other')
            component_counts[comp_name] = component_counts.get(comp_name, 0) + 1
        print(f"    Component breakdown: {component_counts}")
        
        # Exit flag breakdown
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
    """
    Gather all particle samples from a snapshot.
    Looks for files: output/bin/single/samchn_*_{snapshot_num}.bin
    """
    # Try multiple possible file patterns
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
            # break
    
    if not files:
        # Try direct filename if pattern doesn't work
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
    
    # Validate data if we got any particles
    if len(all_particles) > 0:
        validate_particle_data(all_particles, n_examples=3)
    
    return all_particles

# ============================================================================
# Analysis functions
# ============================================================================

def compute_kinetic_energy(particles):
    """
    Compute total kinetic energy from particles.
    K = sum(w_i * m_i * |En_i|) where En is dimensionless orbital energy.
    """
    K_tot = 0.0
    K_unweighted = 0.0
    
    for sp in particles:
        if sp.weight_real > 0 and sp.m > 0:
            K_i = abs(sp.En)  # Dimensionless kinetic energy
            K_tot += sp.weight_real * sp.m * K_i
            K_unweighted += sp.m * K_i
    
    return K_tot, K_unweighted

def analyze_energy_distribution(particles, component_filter=None):
    """
    Extract energy distribution data from particles.
    Returns: energies, masses, weights, velocities (all as arrays).
    """
    energies = []
    masses = []
    weights = []
    exit_flags = []
    
    # Component type mapping (from md_particle_sample.f90)
    component_map = {
        'BH': 1,   # star_type_BH
        'NS': 2,   # star_type_NS
        'MS': 3,   # star_type_MS
        'WD': 4,   # star_type_WD
        'BD': 5,   # star_type_BD
    }
    
    for sp in particles:
        # Filter by component type if specified
        if component_filter and component_filter != 'all':
            if component_filter.upper() in component_map:
                if sp.obtype != component_map[component_filter.upper()]:
                    continue
        
        # Only include particles with valid, positive values
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

# ============================================================================
# Plotting functions
# ============================================================================

def plot_energy_distributions(snapshot_data, sigma_h_kms):
    """
    Create multi-panel plot showing energy distributions for each snapshot.
    """
    n_snapshots = len(snapshot_data)
    if n_snapshots == 0:
        print("No data to plot!")
        return
    
    # Create figure with subplots
    ncols = min(3, n_snapshots)
    nrows = (n_snapshots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_snapshots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # print(snapshot_data.items())
    for idx, (snap_num, particles) in enumerate(snapshot_data.items()):
        ax = axes[idx]
        # print(idx)
        
        energies, masses, weights, _ = analyze_energy_distribution(particles, component)
        
        if len(energies) == 0:
            ax.text(0.5, 0.5, f'Snapshot {snap_num}\n(no data)', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Filter out zero/negative/invalid energies for log scale
        valid_mask = (energies > 0) & np.isfinite(energies) & np.isfinite(weights) & np.isfinite(masses)
        energies_valid = energies[valid_mask]
        weights_valid = weights[valid_mask]
        masses_valid = masses[valid_mask]
        
        if len(energies_valid) == 0:
            ax.text(0.5, 0.5, f'Snapshot {snap_num}\n(no valid data)', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create weighted histogram with safe log scale
        emin = energies_valid.min()
        emax = energies_valid.max()
        if emin > 0 and emax > emin:
            bins = np.logspace(np.log10(emin), np.log10(emax), 30)
            # Clip weights*masses to prevent overflow
            weights_clipped = np.clip(weights_valid * masses_valid, 0, 1e100)
            ax.hist(energies_valid, bins=bins, weights=weights_clipped, 
                   alpha=0.7, edgecolor='black', label='Weighted')
        else:
            # Fallback to linear scale if log doesn't work
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
        
        # Add total KE as text
        K_tot, _ = compute_kinetic_energy(particles)
        ax.text(0.05, 0.95, f'K_tot = {K_tot:.2e}', 
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_snapshots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('energy_distributions_by_snapshot.png', dpi=150)
    print("Saved energy_distributions_by_snapshot.png")
    plt.show()

def plot_total_kinetic_energy(snapshot_data, sigma_h_kms):
    """
    Plot total kinetic energy evolution over snapshots.
    """
    snapshots = sorted(snapshot_data.keys())
    K_tots = []
    K_unweighted = []
    
    for snap in snapshots:
        K_tot, K_unwt = compute_kinetic_energy(snapshot_data[snap])
        K_tots.append(K_tot)
        K_unweighted.append(K_unwt)
    
    # Convert to physical units
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_E = MSUN_G * sigma2_cgs
    K_tots_erg = np.array(K_tots) * unit_E
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Dimensionless
    ax1.plot(snapshots, K_tots, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Snapshot number')
    ax1.set_ylabel(r'$K_{tot}$ (dimensionless, $M_\odot \times x$)')
    ax1.set_title(f'Total Kinetic Energy Evolution ({component})')
    ax1.grid(alpha=0.3)
    
    # Physical units
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
    """
    Estimate energy outflow rate by looking at particles leaving.
    
    exit_flag meanings:
    1 = exit_normal, 4 = exit_boundary_min, 5 = exit_boundary_max
    2 = exit_tidal, 8 = exit_tidal_empty, 9 = exit_tidal_full
    """
    snapshots = sorted(snapshot_data.keys())
    outflow_rates = []
    
    for snap in snapshots:
        particles = snapshot_data[snap]
        
        # Sum energy of particles that exited (could refine by exit_flag)
        E_out = 0.0
        for sp in particles:
            if sp.exit_flag in [4, 5]:  # boundary exits
                E_out += sp.weight_real * sp.m * abs(sp.En)
        
        # Rate = energy / time interval
        rate = E_out / dt_myr * 1e3  # Convert to per Gyr
        outflow_rates.append(rate)
    
    # Convert to physical units
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_P = MSUN_G * sigma2_cgs / SEC_PER_GYR
    outflow_rates_ergs = np.array(outflow_rates) * unit_P
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Dimensionless
    ax1.plot(snapshots, outflow_rates, 'o-', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Snapshot number')
    ax1.set_ylabel(r'$\dot{E}_{out}$ (dimensionless, $M_\odot \times x$/Gyr)')
    ax1.set_title(f'Energy Outflow Rate ({component})')
    ax1.grid(alpha=0.3)
    
    # Physical units
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
    """
    Create a comprehensive summary plot with all key quantities.
    """
    snapshots = sorted(snapshot_data.keys())
    K_tots = []
    outflow_rates = []
    n_particles = []
    
    for snap in snapshots:
        particles = snapshot_data[snap]
        K_tot, _ = compute_kinetic_energy(particles)
        K_tots.append(K_tot)
        n_particles.append(len(particles))
        
        # Outflow rate
        E_out = sum(sp.weight_real * sp.m * abs(sp.En) 
                   for sp in particles if sp.exit_flag in [4, 5])
        outflow_rates.append(E_out / dt_myr * 1e3)
    
    # Convert to physical units
    sigma2_cgs = (sigma_h_kms * KM_TO_CM) ** 2
    unit_E = MSUN_G * sigma2_cgs
    unit_P = unit_E / SEC_PER_GYR
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Total kinetic energy (physical)
    ax = axes[0, 0]
    K_tots_erg = np.array(K_tots) * unit_E
    ax.plot(snapshots, K_tots_erg, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel(r'$K_{tot}$ (erg)')
    ax.set_title('Total Kinetic Energy')
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Panel 2: Outflow rate (physical)
    ax = axes[0, 1]
    outflow_ergs = np.array(outflow_rates) * unit_P
    ax.plot(snapshots, outflow_ergs, 'o-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel(r'$\dot{E}_{out}$ (erg/s)')
    ax.set_title('Energy Outflow Rate')
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Panel 3: Depletion timescale
    ax = axes[1, 0]
    K_arr = np.array(K_tots)
    E_arr = np.array(outflow_rates)
    # Calculate draining time, avoiding division by zero
    tdep = np.full_like(K_arr, np.nan)
    nonzero_mask = np.abs(E_arr) > 1e-20  # Avoid division by very small numbers
    tdep[nonzero_mask] = K_arr[nonzero_mask] / E_arr[nonzero_mask]
    ax.plot(snapshots, tdep, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Snapshot')
    ax.set_ylabel(r'$K/\dot{E}_{out}$ (Gyr)')
    ax.set_title('Energy Depletion Timescale')
    ax.grid(alpha=0.3)
    
    # Panel 4: Number of particles
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

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GNC Binary Energy Analysis")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Snapshots: {list(snapshot_range)}")
    print(f"Component filter: {component}")
    print(f"σ_h = {sigma_h_kms} km/s")
    print("="*70)
    
    # Gather data from all snapshots
    snapshot_data = {}
    for snap_num in snapshot_range:
        print(f"\nReading snapshot {snap_num}...")
        particles = gather_snapshot_data(output_dir, snap_num)
        if len(particles) > 0:
            snapshot_data[snap_num] = particles
            print(f"  Loaded {len(particles)} particles")
        else:
            print(f"  No particles found")
        # break

    if not snapshot_data:
        print("\nNo data found! Check your output directory and snapshot range.")
        exit(1)
    
    print(f"\nSuccessfully loaded {len(snapshot_data)} snapshots")

    # Estimate time step between snapshots (from model.in or use default)
    dt_myr = 0.1  # Default: 0.1 Myr per snapshot (adjust based on your timestep_snapshot)
    
    # Create plots
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


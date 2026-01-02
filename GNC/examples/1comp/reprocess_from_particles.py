#!/usr/bin/env python
"""
Standalone Python script to reprocess particle samples into distribution functions.

This script replicates the Fortran gen_ge.f90 algorithm without depending on 
the incomplete Python port. It reads particle samples directly and computes
g(x) distributions.
"""
import os
import sys
import pickle
import numpy as np
import h5py
import math

# Add source directory for unpickling
sys.path.insert(0, '../../source')
sys.path.insert(0, '../../main')

# Constants
PI = math.pi

def load_particle_samples(snapshot, update_idx, base_dir="output/ecev/bin/single"):
    """
    Load particle samples from pickle files.
    
    Returns:
        energy, angular_momentum, weights, mass arrays
    """
    # Pattern: samchn{snapshot}_{update}.pkl
    pkl_file = os.path.join(base_dir, f"samchn{snapshot}_{update_idx}.pkl")
    
    if not os.path.exists(pkl_file):
        print(f"  Warning: Particle file not found: {pkl_file}")
        return None, None, None, None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Structure: dict with 'particles' key containing list of ParticleSampleType objects
        if isinstance(data, dict) and 'particles' in data:
            particles = data['particles']
            
            # Filter particles with normal exit flag (bound particles)
            # exit_flag = 0 means normal (bound orbit)
            bound_particles = [p for p in particles if p.exit_flag == 0]
            
            if len(bound_particles) == 0:
                print(f"  Warning: No bound particles (exit_flag=0) found")
                return None, None, None, None
            
            # Extract arrays from particle objects  
            # Particles store en = -MBH/(2*a) in code units where v0^2 = 1
            # So abs(en) is already the dimensionless energy x
            energies = np.array([abs(p.en) for p in bound_particles])  # x in code units
            angular_mom = np.array([p.jm for p in bound_particles])
            masses = np.array([p.m for p in bound_particles])
            
            # Weights: use 1.0 if not present
            if hasattr(bound_particles[0], 'w'):
                weights = np.array([p.w for p in bound_particles])
            else:
                weights = np.ones(len(bound_particles))
            
            print(f"  Loaded {len(energies)} bound particles from {pkl_file}")
            print(f"    Energy range: [{np.min(energies):.2f}, {np.max(energies):.2f}]")
            print(f"    J range: [{np.min(angular_mom):.3f}, {np.max(angular_mom):.3f}]")
            return energies, angular_mom, weights, masses
        else:
            print(f"  Unexpected data structure in {pkl_file}: {type(data)}")
            if isinstance(data, dict):
                print(f"    Keys: {list(data.keys())}")
            return None, None, None, None
        
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def load_dms_parameters(snapshot, update_idx, base_dir="output/ecev/dms"):
    """
    Load DMS parameters from binary file.
    
    Returns:
        dict with: mbh, v0, n0, rh, emin, emax, nbin_gx
    """
    dms_file = os.path.join(base_dir, f"dms_{snapshot}_{update_idx}")
    
    if not os.path.exists(dms_file):
        print(f"Warning: DMS file not found: {dms_file}, using defaults")
        return {
            'mbh': 4e6,
            'v0': 1.0,
            'n0': 1.0,
            'rh': 1.0,
            'emin': math.log10(0.05),
            'emax': 5.0,
            'nbin_gx': 24,
            'jmin': 5e-4,
            'jmax': 0.99999
        }
    
    try:
        with open(dms_file, 'rb') as f:
            params = pickle.load(f)
        
        print(f"  Loaded parameters: mbh={params.get('mbh', 4e6):.2e}, nbin_gx={params.get('nbin_gx', 24)}")
        return params
        
    except Exception as e:
        print(f"Error loading {dms_file}: {e}")
        return None


def bin_particles_2d(energies, angular_mom, weights, emin, emax, jmin, jmax, nbin_e=24, nbin_j=24):
    """
    Bin particles into 2D (energy, angular momentum) histogram.
    
    This replicates: dms_so_get_nxj_from_nejw() from stellar_obj.f90
    
    Returns:
        nxyw: 2D histogram (nbin_e x nbin_j)
        e_centers: energy bin centers
        j_centers: angular momentum bin centers
    """
    # Create bins in log space for energy
    e_edges = np.linspace(emin, emax, nbin_e + 1)
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    
    # Linear bins for angular momentum
    j_edges = np.linspace(jmin, jmax, nbin_j + 1)
    j_centers = 0.5 * (j_edges[:-1] + j_edges[1:])
    
    # Create weighted 2D histogram
    nxyw, _, _ = np.histogram2d(
        energies, angular_mom,
        bins=[e_edges, j_edges],
        weights=weights
    )
    
    return nxyw, e_centers, j_centers


def compute_gxj(nxyw, e_centers, j_centers, n0, mbh, v0):
    """
    Convert weighted histogram to distribution function g(x,j).
    
    This replicates: dms_so_get_fxj() from stellar_obj.f90 lines 145-175
    
    Formula (linear j-bins):
        g(x,j) = nxyw(i,j) / (x * log(10)) / xstep / ystep 
                 * pi^(-1.5) * v0^6 * x^2.5 / jm / n0 / mbh^3
    """
    ne, nj = nxyw.shape
    gxj = np.zeros_like(nxyw)
    
    xstep = e_centers[1] - e_centers[0] if ne > 1 else 1.0
    ystep = j_centers[1] - j_centers[0] if nj > 1 else 1.0
    log10_val = math.log(10.0)
    
    for i in range(ne):
        x = 10.0 ** e_centers[i]  # Convert from log to linear
        
        for j in range(nj):
            jm = j_centers[j]
            
            if jm > 0 and nxyw[i, j] > 0:
                gxj[i, j] = (nxyw[i, j] / (x * log10_val) / xstep / ystep *
                            PI ** (-1.5) * v0 ** 6 * x ** 2.5 / jm / n0 / mbh ** 3)
    
    return gxj


def integrate_over_j(gxj, j_centers):
    """
    Integrate g(x,j) over angular momentum to get g(x).
    
    This replicates: get_barge_stellar() from stellar_obj.f90 lines 193-256
    
    Formula (linear j-bins):
        g(x) = sum over j: g(x,j) * jm * ystep * 2
    """
    ne, nj = gxj.shape
    gx = np.zeros(ne)
    
    ystep = j_centers[1] - j_centers[0] if nj > 1 else 1.0
    
    for i in range(ne):
        integral = 0.0
        for j in range(nj):
            jm = j_centers[j]
            integral += gxj[i, j] * jm * ystep * 2.0
        gx[i] = integral
    
    return gx


def normalize_distribution(gx, e_centers, x_boundary=100.0):
    """
    Normalize g(x) distribution - simplified version.
    
    The Fortran code does asymptotic matching at x_boundary, but for our purposes  
    we keep the distribution as-is from the particle binning (already has correct
    relative normalization from weights).
    
    Only ensure no invalid values.
    """
    # Replace any NaN or Inf with zeros
    gx = np.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure non-negative
    gx = np.maximum(gx, 0.0)
    
    return gx


def compute_density_from_gx(gx, e_centers, n0, v0, rh, nbin_r=24):
    """
    Compute density profile n(r) from g(x).
    
    This replicates: get_fden() from gen_ge.f90 lines 733-770
    
    Formula:
        n(r) = 2/sqrt(pi) * integral of: g(x) * sqrt(|rh/r - x|) dx
    """
    emin = e_centers[0]
    emax = e_centers[-1]
    
    # Radial range
    rmin = math.log10(0.5 * rh / (10.0 ** emax))
    rmax = math.log10(0.5 * rh / (10.0 ** emin))
    
    r_centers = np.linspace(rmin, rmax, nbin_r)
    density = np.zeros(nbin_r)
    
    # Interpolate g(x) for integration
    from scipy import interpolate
    gx_interp = interpolate.interp1d(e_centers, gx, kind='linear', 
                                     bounds_error=False, fill_value=0.0)
    
    for i in range(nbin_r):
        r_val = 10.0 ** r_centers[i]
        phi_max = rh / r_val  # Maximum energy at this radius
        
        # Integrate from 0 to phi_max
        x_vals = np.logspace(emin, min(math.log10(phi_max), emax), 100)
        integrand = gx_interp(np.log10(x_vals)) * np.sqrt(np.abs(phi_max - x_vals))
        
        # Simple trapezoidal integration
        integral = np.trapz(integrand, x_vals)
        density[i] = 2.0 / math.sqrt(PI) * integral * n0
    
    return r_centers, density


def save_to_hdf5(snapshot, update_idx, e_centers, gx, r_centers, density, params, output_dir="output/ecev/dms"):
    """
    Save distributions to HDF5 file.
    """
    fname = os.path.join(output_dir, f"dms_{snapshot}_{update_idx}.hdf5")
    
    with h5py.File(fname, 'w') as f:
        # Global attributes
        f.attrs['mbh'] = params.get('mbh', 4e6)
        f.attrs['v0'] = params.get('v0', 1.0)
        f.attrs['n0'] = params.get('n0', 1.0)
        f.attrs['rh'] = params.get('rh', 1.0)
        
        # Create group structure: 1/star/
        grp = f.create_group('1')
        star_grp = grp.create_group('star')
        
        # Save g(x) distribution
        fgx_grp = star_grp.create_group('fgx')
        fgx_grp.create_dataset('   X', data=e_centers)
        fgx_grp.create_dataset('  FX', data=gx)
        fgx_grp.attrs['xmin'] = float(params.get('emin', e_centers[0]))
        fgx_grp.attrs['xmax'] = float(params.get('emax', e_centers[-1]))
        fgx_grp.attrs['nbin'] = len(e_centers)
        
        # Save density profile
        fden_grp = star_grp.create_group('fden')
        fden_grp.create_dataset('   X', data=r_centers)
        fden_grp.create_dataset('  FX', data=density)
        fden_grp.attrs['xmin'] = float(r_centers[0])
        fden_grp.attrs['xmax'] = float(r_centers[-1])
        fden_grp.attrs['nbin'] = len(r_centers)
    
    print(f"  [OK] Saved to {fname}")
    print(f"    g(x): min={np.min(gx[gx>0]):.3e}, max={np.max(gx):.3e}")
    print(f"    n(r): min={np.min(density[density>0]):.3e}, max={np.max(density):.3e}")


def process_snapshot(snapshot, update_idx):
    """
    Process a single snapshot: load particles, compute distributions, save HDF5.
    """
    print(f"\n--- Processing Snapshot {snapshot}, Update {update_idx} ---")
    
    # Load parameters
    params = load_dms_parameters(snapshot, update_idx)
    if params is None:
        return False
    
    # Load particle samples
    energies, angular_mom, weights, masses = load_particle_samples(snapshot, update_idx)
    
    if energies is None or len(energies) == 0:
        print(f"  No particles found, skipping")
        return False
    
    # Convert energies to dimensionless log space
    # Following Fortran dms.f90 line 530: xstar = abs(estar) / v0**2
    # Then line 537: nejw(i)%e = log10(xstar)
    # So we need: log10(|E| / v0²)
    v0 = params.get('v0', 1.0)
    energies_dimensionless = energies / (v0 ** 2)  # energies already has abs() applied
    energies_log = np.log10(energies_dimensionless)
    
    # Filter valid particles (within energy/angular momentum bounds)
    emin_default = params.get('emin', math.log10(0.05))
    emax_default = params.get('emax', 5.0)
    jmin = params.get('jmin', 5e-4)
    jmax = params.get('jmax', 0.99999)
    
    # Use FIXED binning from DMS parameters (matching Fortran exactly)
    # Do NOT use adaptive binning - must match expected bin centers
    emin = emin_default
    emax = emax_default
    
    emin_particles = np.min(energies_log)
    emax_particles = np.max(energies_log)
    print(f"  Particle energy range (log10): [{emin_particles:.3f}, {emax_particles:.3f}]")
    print(f"  Using fixed bin range (log10): [{emin:.3f}, {emax:.3f}]")
    
    valid_mask = ((energies_log >= emin) & (energies_log <= emax) & 
                  (angular_mom >= jmin) & (angular_mom <= jmax))
    
    energies = energies_log[valid_mask]
    angular_mom = angular_mom[valid_mask]
    weights = weights[valid_mask]
    
    print(f"  Valid particles: {len(energies)}/{len(energies_log)}")
    
    if len(energies) == 0:
        print(f"  No valid particles after filtering, skipping")
        return False
    
    # Step 1: Bin particles into 2D histogram
    nbin_gx = params.get('nbin_gx', 24)
    nxyw, e_centers, j_centers = bin_particles_2d(
        energies, angular_mom, weights,
        emin, emax, jmin, jmax,
        nbin_e=nbin_gx, nbin_j=nbin_gx
    )
    
    # Step 2: Convert to distribution function g(x,j)
    n0 = params.get('n0', 1.0)
    mbh = params.get('mbh', 4e6)
    v0 = params.get('v0', 1.0)  # Use actual v0 from parameters
    
    gxj = compute_gxj(nxyw, e_centers, j_centers, n0, mbh, v0)
    
    # Step 3: Integrate over j to get g(x)
    gx = integrate_over_j(gxj, j_centers)
    
    # Step 3.5: Normalize g(x)
    gx = normalize_distribution(gx, e_centers)
    
    # Step 4: Compute density profile n(r) from g(x)
    rh = params.get('rh', 1.0)
    # Use v0=1.0 for code units
    r_centers, density = compute_density_from_gx(gx, e_centers, n0, 1.0, rh, nbin_r=nbin_gx)
    
    # Step 5: Save to HDF5
    save_to_hdf5(snapshot, update_idx, e_centers, gx, r_centers, density, params)
    
    return True


def main():
    """
    Main function: process all snapshots.
    """
    print("="*60)
    print("Reprocessing Particle Samples to Distribution Functions")
    print("="*60)
    
    # Check for scipy
    try:
        import scipy
        print("[OK] scipy found")
    except ImportError:
        print("ERROR: scipy is required for integration")
        print("Install with: pip install scipy")
        sys.exit(1)
    
    # Process snapshots 1, 2, 3, 5, 10 with update 10
    snapshots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    update_idx = 10
    
    success_count = 0
    for snapshot in snapshots:
        try:
            if process_snapshot(snapshot, update_idx):
                success_count += 1
        except Exception as e:
            print(f"  ERROR processing snapshot {snapshot}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Processing Complete: {success_count}/{len(snapshots)} snapshots")
    print("="*60)
    
    if success_count > 0:
        print("\n[SUCCESS] You can now run ge.py to generate correct plots!")
        print("  python plot/ge.py")
    else:
        print("\n[WARNING] No snapshots were processed successfully.")
        print("  Check the particle file format in output/ecev/bin/single/")


if __name__ == "__main__":
    main()

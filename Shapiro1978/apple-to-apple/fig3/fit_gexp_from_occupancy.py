#!/usr/bin/env python3
"""
Fit the optimal gexp exponent by chi² minimization.

This script takes the occupancy N_x from a --debug-occupancy-norm run
and finds the gexp value that minimizes chi² when comparing gbar_MC_norm
to the paper's gbar_paper values.

Usage:
    1. Run the MC with --debug-occupancy-norm to get N_x:
       python3 gbar_mc_fast_sm78_fig3_v2.py ... --debug-occupancy-norm > occupancy_run.log

    2. Extract X_BINS and N_x from the output (or modify this script to parse the log)

    3. Run this script:
       python3 fit_gexp_from_occupancy.py

    4. Use the gexp_best value in your production run:
       python3 gbar_mc_fast_sm78_fig3_v2.py ... --gexp <gexp_best>
"""

import numpy as np

# Paper's Table 2 data (from gbar_mc_fast_sm78_fig3_v2.py)
X_BINS = np.array([
    0.225, 0.303, 0.495, 1.04, 1.26, 1.62, 2.35, 5.00, 7.20, 8.94,
    12.10, 19.70, 41.60, 50.30, 64.60, 93.60, 198.00, 287.00, 356.00, 480.00,
    784.00, 1650.00, 2000.00, 2570.00, 3730.00
])

gbar_paper = np.array([
    1.0000, 1.0700, 1.1300, 1.6000, 1.3400, 1.3700, 1.5500, 2.1100, 2.2200, 2.2000,
    2.4100, 3.0000, 3.5000, 3.7900, 3.6100, 3.6600, 4.0300, 3.9800, 3.3100, 2.9200,
    2.3500, 1.5700, 0.8500, 0.7400, 0.2000
])

gbar_err_paper = np.array([
    0.0400, 0.0600, 0.0400, 0.1600, 0.1800, 0.1400, 0.1500, 0.3400, 0.4400, 0.3500,
    0.2200, 0.1700, 0.4300, 0.3000, 0.3000, 0.1800, 0.4400, 0.9900, 0.6800, 0.5300,
    0.1800, 0.5400, 0.1200, 0.1200, 0.1400
])


def compute_chi2(gexp, N_x, x_bins, gbar_paper, gbar_err_paper, norm_idx=0):
    """
    Compute chi² for a given gexp value.
    
    Parameters:
        gexp: Jacobian exponent to test
        N_x: Occupancy array from MC (must match x_bins length)
        x_bins: Energy bin centers
        gbar_paper: Paper's ḡ(x) values
        gbar_err_paper: Paper's error bars
        norm_idx: Index of normalization bin (default 0 = x=0.225)
    
    Returns:
        chi2: chi² value
        chi2_per_dof: chi² per degree of freedom
        residuals: array of (g_MC - g_paper) / sigma
    """
    # Convert N(E) → g(E) using the exponent
    g_unscaled = N_x * (x_bins ** gexp)
    
    # Normalize at x=0.225 (norm_idx = 0)
    if g_unscaled[norm_idx] <= 0:
        return np.inf, np.inf, None
    
    gbar_norm = g_unscaled / g_unscaled[norm_idx]
    
    # Compute residuals (only where paper has valid data)
    mask = (gbar_err_paper > 0) & np.isfinite(gbar_paper) & np.isfinite(gbar_norm)
    if mask.sum() == 0:
        return np.inf, np.inf, None
    
    residuals = (gbar_norm[mask] - gbar_paper[mask]) / gbar_err_paper[mask]
    chi2 = np.sum(residuals**2)
    chi2_per_dof = chi2 / mask.sum()
    
    return chi2, chi2_per_dof, residuals


def fit_gexp(N_x, x_bins=None, gbar_paper_vals=None, gbar_err_paper_vals=None, 
             gexp_range=(1.6, 2.4), n_points=81, verbose=True):
    """
    Find the gexp value that minimizes chi².
    
    Parameters:
        N_x: Occupancy array from MC (must match X_BINS length)
        x_bins: Energy bin centers (default: use X_BINS from this script)
        gbar_paper_vals: Paper's ḡ(x) values (default: use gbar_paper from this script)
        gbar_err_paper_vals: Paper's error bars (default: use gbar_err_paper from this script)
        gexp_range: (min, max) range to search
        n_points: Number of points in the grid search
        verbose: Print results
    
    Returns:
        gexp_best: Optimal exponent value
        chi2_best: Minimum chi² value
        chi2_per_dof_best: Minimum chi²/dof
    """
    if x_bins is None:
        x_bins = X_BINS
    if gbar_paper_vals is None:
        gbar_paper_vals = gbar_paper
    if gbar_err_paper_vals is None:
        gbar_err_paper_vals = gbar_err_paper
    
    # Check that arrays match
    if len(N_x) != len(x_bins) or len(x_bins) != len(gbar_paper_vals):
        raise ValueError(f"Array length mismatch: N_x={len(N_x)}, x_bins={len(x_bins)}, gbar_paper={len(gbar_paper_vals)}")
    
    # Grid search over gexp
    gexp_grid = np.linspace(gexp_range[0], gexp_range[1], n_points)
    chi2_values = []
    chi2_per_dof_values = []
    
    if verbose:
        print(f"Scanning gexp from {gexp_range[0]:.2f} to {gexp_range[1]:.2f} ({n_points} points)...")
    
    for gexp in gexp_grid:
        chi2, chi2_per_dof, _ = compute_chi2(gexp, N_x, x_bins, gbar_paper_vals, gbar_err_paper_vals)
        chi2_values.append(chi2)
        chi2_per_dof_values.append(chi2_per_dof)
    
    chi2_values = np.array(chi2_values)
    chi2_per_dof_values = np.array(chi2_per_dof_values)
    
    # Find minimum
    best_idx = np.argmin(chi2_per_dof_values)
    gexp_best = gexp_grid[best_idx]
    chi2_best = chi2_values[best_idx]
    chi2_per_dof_best = chi2_per_dof_values[best_idx]
    
    # Refine with a finer search around the minimum
    if best_idx > 0 and best_idx < len(gexp_grid) - 1:
        # Do a second pass with finer grid around the minimum
        refine_range = (gexp_grid[best_idx - 1], gexp_grid[best_idx + 1])
        refine_grid = np.linspace(refine_range[0], refine_range[1], 51)
        refine_chi2 = []
        for gexp in refine_grid:
            chi2, chi2_per_dof, _ = compute_chi2(gexp, N_x, x_bins, gbar_paper_vals, gbar_err_paper_vals)
            refine_chi2.append(chi2_per_dof)
        refine_best_idx = np.argmin(refine_chi2)
        gexp_best = refine_grid[refine_best_idx]
        chi2_best, chi2_per_dof_best, _ = compute_chi2(gexp_best, N_x, x_bins, gbar_paper_vals, gbar_err_paper_vals)
    
    if verbose:
        print(f"Chi² scan results:")
        print(f"  Optimal gexp: {gexp_best:.4f}")
        print(f"  Minimum chi²: {chi2_best:.2f}")
        print(f"  Minimum chi²/dof: {chi2_per_dof_best:.3f}")
        print(f"  (target: chi²/dof < 2.0 for good agreement)")
        print()
        print(f"Use this in your production run:")
        print(f"  --gexp {gexp_best:.4f}")
    
    return gexp_best, chi2_best, chi2_per_dof_best


def parse_log_file(log_file):
    """
    Parse the MC output log to extract N_x values.
    
    Looks for the table starting with "# x_center   gbar_MC_norm   gbar_MC_raw"
    and extracts the gbar_MC_raw column (which is N_x in debug-occupancy-norm mode).
    """
    N_x_values = []
    x_values = []
    
    with open(log_file, 'r') as f:
        in_table = False
        for line in f:
            if line.startswith("# x_center"):
                in_table = True
                continue
            if in_table:
                if line.strip() == "":
                    continue
                if line.startswith("# chi") or line.startswith("# max"):
                    break
                if line.startswith("#"):
                    continue
                # Parse the line: x_center  gbar_MC_norm  gbar_MC_raw  gbar_paper  gbar_err_paper
                # Handle scientific notation in x values (e.g., "1.65e+03")
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # First column: x_center (may be in scientific notation)
                        x_str = parts[0]
                        x_val = float(x_str)
                        # Third column: gbar_MC_raw (this is N_x in debug-occupancy-norm mode)
                        gbar_raw = float(parts[2])
                        x_values.append(x_val)
                        N_x_values.append(gbar_raw)
                    except (ValueError, IndexError) as e:
                        continue
    
    if len(N_x_values) == 0:
        raise ValueError(f"Could not parse N_x values from {log_file}. Make sure the log contains the output table.")
    
    # Convert to numpy arrays
    x_parsed = np.array(x_values)
    N_x_parsed = np.array(N_x_values)
    
    # Match to X_BINS order by finding closest x values
    N_x = np.zeros_like(X_BINS)
    for i, x_target in enumerate(X_BINS):
        # Find closest match
        idx = np.argmin(np.abs(x_parsed - x_target))
        if np.abs(x_parsed[idx] - x_target) < 0.01:  # Close enough (within 1%)
            N_x[i] = N_x_parsed[idx]
        else:
            # Interpolate if needed (shouldn't happen if bins match)
            if len(x_parsed) > 1:
                # Sort for interpolation
                sort_idx = np.argsort(x_parsed)
                x_sorted = x_parsed[sort_idx]
                N_sorted = N_x_parsed[sort_idx]
                N_x[i] = np.interp(x_target, x_sorted, N_sorted)
            else:
                N_x[i] = N_x_parsed[idx]
    
    return N_x


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("gexp Fitting Script")
    print("=" * 80)
    print()
    
    # Try to parse from command line argument or use default
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "occupancy_baseline.log"
        print(f"No log file specified, using default: {log_file}")
        print("Usage: python3 fit_gexp_from_occupancy.py [log_file]")
        print()
    
    try:
        print(f"Parsing N_x from {log_file}...")
        N_x = parse_log_file(log_file)
        print(f"Successfully extracted {len(N_x)} N_x values")
        print()
        
        # For an apples-to-apples comparison with Shapiro & Marchant (1978),
        # the exponent is fixed by the paper's derivation (eqs. 9, 11, 13):
        # g(x) ∝ N(E) x^2, so gexp = 2.0 exactly.
        # We compute chi² for this fixed value to assess the physics/algorithm
        # match, not to fit a normalization parameter.
        gexp_paper = 2.0
        chi2_best, chi2_per_dof_best, residuals = compute_chi2(
            gexp_paper, N_x, X_BINS, gbar_paper, gbar_err_paper
        )
        
        print()
        print("=" * 80)
        print("Summary:")
        print("  NOTE: For Shapiro & Marchant (1978), the mapping N(E) → g(E)")
        print("        is fixed by eqs. (9), (11) and (13), which imply g(x) ∝ N(E) x^2.")
        print(f"  Using gexp = {gexp_paper:.1f} (paper value)")
        print(f"  Chi²: {chi2_best:.2f}")
        print(f"  Chi²/dof: {chi2_per_dof_best:.3f}")
        print(f"  (target: chi²/dof < 2.0 for good agreement)")
        print()
        if residuals is not None:
            max_residual = np.abs(residuals).max()
            print(f"  Max |residual|: {max_residual:.2f} sigma")
            print()
        print("Next step (paper-literal): run production with")
        print(f"  --gexp {gexp_paper:.1f}")
        print()
        print("If chi²/dof is still large, the discrepancy is in the physics/algorithm")
        print("(loss-cone treatment, boundary conditions, etc.), not normalization.")
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"ERROR: Log file '{log_file}' not found!")
        print()
        print("Please provide the log file from your occupancy run, or")
        print("manually extract N_x values and modify this script.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


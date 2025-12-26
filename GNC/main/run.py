#!/usr/bin/env python
"""
run.py - Unified run script for GNC Python simulation.

This script runs the complete GNC simulation workflow:
1. (Optional) Generate C-functions auxiliary file
2. Run initialization (ini.py)
3. Run main evolution loop (main.py)

Usage:
    python run.py                    # Run with default settings from model.in
    python run.py --cfuns            # Also generate cfuns (if not exists)
    python run.py --cfuns-only       # Only generate cfuns
    python run.py --help             # Show help
    
For MPI parallel execution:
    mpiexec -n <num_cores> python run.py
"""
from __future__ import annotations

import sys
import os
import subprocess
import argparse


def ensure_directories():
    """Create output directories if they don't exist."""
    dirs = [
        "output/ini/bin/single",
        "output/ini/bin/by",
        "output/ini/hdf5",
        "output/ini/txt/single",
        "output/ini/txt/by",
        "output/ini/sts/ALL",
        "output/ini/sts/BH",
        "output/ini/sts/MS",
        "output/ini/sts/NS",
        "output/ini/sts/WD",
        "output/ini/sts/BHB",
        "output/ini/sts/BHNS",
        "output/ini/sts/NSB",
        "output/ecev/bin/single",
        "output/ecev/dms",
        "output/indvd/emri/sg/bh",
        "output/indvd/emri/sg/ms",
        "output/indvd/plunge/bh",
        "output/indvd/td",
        "output/pro/MS",
        "output/pro/BH",
        "output/pro/BD",
        "output/pro/NS/norm",
        "output/pro/NS/merge",
        "output/pro/NS/other",
        "output/pro/WD",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Output directories created.")


def read_ntasks_from_model(model_file: str = "model.in") -> int:
    """Read number of tasks from model.in file."""
    try:
        with open(model_file, 'r') as f:
            # Skip comments
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return int(line)
    except Exception as e:
        print(f"Warning: Could not read ntasks from {model_file}: {e}")
    return 1


def check_cfuns_exists(cfs_dir: str) -> bool:
    """Check if C-functions file exists."""
    # Normalize path separators
    cfs_dir = cfs_dir.replace("\\", "/")
    
    # Handle relative path from examples directory
    if os.path.exists(cfs_dir + ".bin"):
        return True
    # Try from script directory (GNC/main)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gnc_dir = os.path.dirname(script_dir)  # GNC directory
    
    # Strip leading ../ and join with GNC directory
    rel_path = cfs_dir.lstrip("./").lstrip("../")
    abs_path = os.path.normpath(os.path.join(gnc_dir, rel_path))
    
    if os.path.exists(abs_path + ".bin"):
        return True
    
    # Also check from current working directory
    cwd_path = os.path.normpath(os.path.join(os.getcwd(), cfs_dir))
    return os.path.exists(cwd_path + ".bin")


def get_cfs_dir_from_model(model_file: str = "model.in") -> str:
    """Read cfs directory from model.in."""
    try:
        with open(model_file, 'r') as f:
            for line in f:
                if "cfs dir" in line.lower() or "cfuns" in line.lower():
                    parts = line.split('=')
                    if len(parts) > 1:
                        return parts[-1].strip()
    except Exception:
        pass
    return "../../common_data/cfuns_34"


def run_cfuns(ntasks: int, output_dir: str = None):
    """Run cfuns to generate C-functions file."""
    if output_dir is None:
        output_dir = get_cfs_dir_from_model()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfuns_script = os.path.join(script_dir, "cfuns.py")
    
    # Convert relative path to absolute path from GNC directory
    if output_dir.startswith("../") or output_dir.startswith("..\\"):
        # Path is relative to examples/1comp, convert to absolute
        gnc_dir = os.path.dirname(script_dir)  # GNC directory
        output_dir_abs = os.path.normpath(os.path.join(gnc_dir, output_dir.lstrip("../")))
    else:
        output_dir_abs = os.path.abspath(output_dir)
    
    # Ensure the parent directory exists
    parent_dir = os.path.dirname(output_dir_abs)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    # Parameters for cfuns - use fewer bins for faster generation
    num_bins = 256  # Smaller for faster generation (can be increased to 1024 for production)
    log_j_min = -3.5
    log_j_max = 0
    
    print(f"\nGenerating C-functions file: {output_dir_abs}.bin")
    print(f"  num_bins={num_bins}, log_j_min={log_j_min}, log_j_max={log_j_max}")
    print(f"  This may take a few minutes...")
    
    # Adjust ntasks to be divisible by num_bins
    effective_ntasks = ntasks
    while num_bins % effective_ntasks != 0 and effective_ntasks > 1:
        effective_ntasks -= 1
    
    if effective_ntasks != ntasks:
        print(f"  Adjusted ntasks from {ntasks} to {effective_ntasks} (must divide {num_bins})")
    
    cmd = [sys.executable, cfuns_script, str(num_bins), str(log_j_min), str(log_j_max), output_dir_abs]
    
    if effective_ntasks > 1:
        # Try to use mpiexec
        try:
            mpi_cmd = ["mpiexec", "-n", str(effective_ntasks)] + cmd
            subprocess.run(mpi_cmd, check=True)
            return
        except FileNotFoundError:
            print("Warning: mpiexec not found, running in single-process mode")
        except subprocess.CalledProcessError as e:
            print(f"MPI execution failed, trying single-process mode: {e}")
    
    # Single process mode
    subprocess.run(cmd, check=True)


def run_ini(ntasks: int):
    """Run initialization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ini_script = os.path.join(script_dir, "ini.py")
    
    print("\n" + "="*60)
    print("Running initialization (ini.py)...")
    print("="*60)
    
    cmd = [sys.executable, ini_script]
    
    if ntasks > 1:
        try:
            mpi_cmd = ["mpiexec", "-n", str(ntasks)] + cmd
            subprocess.run(mpi_cmd, check=True)
            return
        except FileNotFoundError:
            print("Warning: mpiexec not found, running in single-process mode")
    
    subprocess.run(cmd, check=True)


def run_main(ntasks: int):
    """Run main evolution loop."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "main.py")
    
    print("\n" + "="*60)
    print("Running main evolution (main.py)...")
    print("="*60)
    
    cmd = [sys.executable, main_script]
    
    if ntasks > 1:
        try:
            mpi_cmd = ["mpiexec", "-n", str(ntasks)] + cmd
            subprocess.run(mpi_cmd, check=True)
            return
        except FileNotFoundError:
            print("Warning: mpiexec not found, running in single-process mode")
    
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run GNC Monte Carlo simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Run full simulation
    python run.py --cfuns            # Generate cfuns if needed, then run
    python run.py --cfuns-only       # Only generate cfuns file
    python run.py --ini-only         # Only run initialization
    python run.py --main-only        # Only run main evolution
        """
    )
    parser.add_argument("--cfuns", action="store_true", 
                        help="Generate C-functions file if it doesn't exist")
    parser.add_argument("--cfuns-only", action="store_true",
                        help="Only generate C-functions file and exit")
    parser.add_argument("--ini-only", action="store_true",
                        help="Only run initialization")
    parser.add_argument("--main-only", action="store_true",
                        help="Only run main evolution")
    parser.add_argument("-n", "--ntasks", type=int, default=None,
                        help="Number of MPI tasks (default: read from model.in)")
    
    args = parser.parse_args()
    
    # Read ntasks from model.in if not specified
    ntasks = args.ntasks if args.ntasks else read_ntasks_from_model()
    print(f"Number of tasks: {ntasks}")
    
    # Ensure output directories exist
    ensure_directories()
    
    # Get cfuns path
    cfs_dir = get_cfs_dir_from_model()
    
    # Handle --cfuns-only
    if args.cfuns_only:
        run_cfuns(ntasks, cfs_dir)
        print("\nC-functions generation complete.")
        return
    
    # Always check if cfuns exists and generate if needed
    if not check_cfuns_exists(cfs_dir):
        print(f"\nC-functions file not found: {cfs_dir}.bin")
        print("Generating C-functions file (this may take a few minutes)...")
        run_cfuns(ntasks, cfs_dir)
    else:
        print(f"C-functions file found: {cfs_dir}.bin")
    
    # Handle --ini-only
    if args.ini_only:
        run_ini(ntasks)
        print("\nInitialization complete.")
        return
    
    # Handle --main-only
    if args.main_only:
        run_main(ntasks)
        print("\nMain evolution complete.")
        return
    
    # Run full simulation
    run_ini(ntasks)
    run_main(ntasks)
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)
    print("\nOutput files:")
    print("  - HDF5 snapshots: output/ecev/dms/dms_*.hdf5")
    print("  - Binary snapshots: output/ecev/bin/single/")
    print("\nTo plot results, run:")
    print("  python plot/ge.py")


if __name__ == "__main__":
    main()


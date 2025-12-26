#!/usr/bin/env python
"""
cfuns.py - Generate auxiliary C-functions file for GNC simulation.

Usage (with MPI):
    mpirun -np <num_cores> python cfuns.py <num_bins> <log_j_min> <log_j_max> <output_file>

Example:
    mpirun -np 8 python cfuns.py 1024 -3.5 0 ../common_data/cfuns_34

Note: num_bins must be divisible by num_cores
"""
from __future__ import annotations

import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from com_main_gw import init_mpi, stop_mpi, rid, ctl
from md_cfuns import cfs


def cfuns(argv: list[str]) -> None:
    """Main function to generate C-functions auxiliary file."""
    
    # Parse arguments
    if len(argv) < 5:
        print("Usage: python cfuns.py <num_bins> <log_j_min> <log_j_max> <output_file>")
        print("Example: python cfuns.py 1024 -3.5 0 ../common_data/cfuns_34")
        sys.exit(1)
    
    str_num_bin = argv[1]
    num_bin = int(str_num_bin)

    str_jmin = argv[2]
    logjmin = float(str_jmin)

    str_jmax = argv[3]
    logjmax = float(str_jmax)

    str_output_file = argv[4] if len(argv) > 4 else "../common_data/cfuns"
    
    # Initialize MPI
    init_mpi()
    
    # Re-import rid after MPI init (it may have changed)
    from com_main_gw import rid as current_rid
    
    # Initialize cfs
    cfs.init(num_bin, num_bin, logjmin, logjmax)
    print(f"[Rank {current_rid}] start cfuns ..., nx, ny= {cfs.nj}, {cfs.ns}")

    if (num_bin % ctl.ntasks) != 0:
        print(f"error! mpi_threads should be integer times of num_bin! num_bin={num_bin}, ntasks={ctl.ntasks}")
        stop_mpi()
        sys.exit(1)

    # Calculate C-functions using MPI
    print(f"[Rank {current_rid}] Computing C-functions (this may take a few minutes)...")
    cfs.get_cfs_s_mpi(current_rid, ctl.ntasks)

    # Output from root process only
    if current_rid == 0:
        # Ensure directory exists
        output_dir = os.path.dirname(str_output_file.strip())
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cfs.output_bin(str_output_file.strip())
        print(f"C-functions written to {str_output_file.strip()}.bin")

    stop_mpi()


if __name__ == "__main__":
    cfuns(sys.argv)

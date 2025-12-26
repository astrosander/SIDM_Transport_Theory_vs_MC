#!/usr/bin/env python
"""
ini.py - Initialize the GNC Monte Carlo simulation.

This script:
1. Reads model parameters from model.in and mfrac.in
2. Initializes particle samples
3. Computes initial distribution function g(x)
4. Outputs initial state to binary/HDF5 files

Usage (with MPI):
    mpirun -np <num_cores> python ini.py

Example:
    cd examples/1comp
    mpirun -np 12 python ../../main/ini.py
"""
from __future__ import annotations

import sys
import os
import time

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from com_main_gw import (
    init_mpi, stop_mpi, mpi_barrier, rid, mpi_master_id, ctl,
    bksams_arr_ini, bksams, particle_samples_arr_type,
    MPI_COMM_WORLD
)
from md_cfuns import cfs
from read_ini_single import readin_model_par, print_model_par, print_current_code_version
from gen_ge import (
    set_dm_init, get_ge_by_root, update_weights, print_num_boundary, print_num_all,
    dms, set_seed, get_dms, collection_data_single_bympi
)
from ini_single import get_init_samples, set_chain_samples
from io_hdf5 import output_dms_hdf5_pdf


def output_all_barge_txt(dm, fl: str) -> None:
    """Output distribution function to HDF5."""
    try:
        output_dms_hdf5_pdf(dm, fl)
    except Exception as e:
        print(f"Warning: Could not write HDF5 output: {e}")


def init_model() -> None:
    """Initialize model parameters after reading config."""
    from com_main_gw import MBH, rh, ctl, emax_factor
    from read_ini_single import set_simu_time
    import math
    
    # Calculate influence radius
    # rh = MBH / (sigma^2) where sigma is velocity dispersion
    # Using approximate relation from M-sigma relation
    import com_main_gw as cmg
    
    cmg.rh = 1.0  # Normalized
    cmg.rhmax = cmg.rh * emax_factor * 2
    cmg.rhmin = cmg.rh / emax_factor / 2
    
    # Set simulation time parameters
    set_simu_time()
    
    # Set MPI block parameters
    ctl.nblock_size = ctl.grid_bins // ctl.ntasks
    ctl.nblock_mpi_bg = rid * ctl.nblock_size + 1
    ctl.nblock_mpi_ed = (rid + 1) * ctl.nblock_size


def ini() -> None:
    """Main initialization routine."""
    # Read model parameters
    readin_model_par("model.in")
    
    # Initialize MPI
    init_mpi()
    
    # Initialize model
    init_model()

    if rid == mpi_master_id:
        print(f"mpi_master_id= {mpi_master_id}")

    # Read C-functions auxiliary file
    # Resolve path - it may be relative to different locations
    cfs_path = ctl.cfs_file_dir.strip().replace("\\", "/")
    if not os.path.exists(cfs_path + ".bin"):
        # Try relative to script directory (GNC/main)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gnc_dir = os.path.dirname(script_dir)
        rel_path = cfs_path.lstrip("./").lstrip("../")
        alt_path = os.path.normpath(os.path.join(gnc_dir, rel_path))
        if os.path.exists(alt_path + ".bin"):
            cfs_path = alt_path
        else:
            # Try from current working directory
            cwd_path = os.path.normpath(os.path.join(os.getcwd(), cfs_path))
            if os.path.exists(cwd_path + ".bin"):
                cfs_path = cwd_path
            else:
                raise FileNotFoundError(
                    f"C-functions file not found!\n"
                    f"  Tried: {cfs_path}.bin\n"
                    f"  Tried: {alt_path}.bin\n"
                    f"  Tried: {cwd_path}.bin\n"
                    f"Run with --cfuns flag to generate it first."
                )
    
    cfs.input_bin(cfs_path)
    print(f"readin cfs finished: nj={cfs.nj}, ns={cfs.ns}, rid={rid}")

    t1 = None
    if rid == 0:
        print_model_par()
        print_current_code_version()
        t1 = time.process_time()

    # Create sample arrays for MPI collection
    smsa = [particle_samples_arr_type() for _ in range(ctl.ntasks)]

    print(f"proc rid start {rid}")
    set_seed(ctl.same_rseed_ini, ctl.seed_value + rid)

    tmpi = f"{rid + ctl.ntask_bg + 1:4d}"

    # Initialize diffuse mass spectrum
    print("start dms init")
    set_dm_init(dms)

    # Generate initial particle samples
    get_init_samples(bksams_arr_ini)

    # Check for errors in initialization
    from com_main_gw import exit_normal
    for i in range(bksams_arr_ini.n):
        if bksams_arr_ini.sp[i].exit_flag != exit_normal:
            print(f"i= {i + 1}, exit_flag={bksams_arr_ini.sp[i].exit_flag}")
            stop_mpi()
            sys.exit(1)

    # Set up chain structure
    set_chain_samples(bksams, bksams_arr_ini)

    dms.weight_asym = 1.0
    
    # Collect data via MPI
    collection_data_single_bympi(smsa, ctl.ntasks)
    mpi_barrier(MPI_COMM_WORLD)

    # Compute distribution function
    get_ge_by_root(smsa, ctl.ntasks, True)
    update_weights()
    get_dms(dms)

    if rid == 0:
        print_num_boundary(dms)
        print_num_all(dms)

    mpi_barrier(MPI_COMM_WORLD)

    # Output binary snapshot
    bksams.output_bin(f"output/ini/bin/single/samchn{tmpi.strip()}")

    if rid == 0:
        output_all_barge_txt(dms, "output/ini/hdf5/dms_0_0")
        dms.output_bin("output/ini/bin/dms.bin")
        print("output control...")
        print("init finished")
        t2 = time.process_time()
        print(f"total_time= {t2 - t1:.2f}s")

    stop_mpi()


if __name__ == "__main__":
    ini()

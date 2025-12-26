#!/usr/bin/env python
"""
main.py - Main evolution loop for GNC Monte Carlo simulation.

This script evolves the particle distribution over time using
Fokker-Planck methods with Monte Carlo sampling.

Usage (with MPI):
    mpirun -np <num_cores> python main.py

Example:
    cd examples/1comp
    mpirun -np 12 python ../../main/main.py
"""
from __future__ import annotations

import sys
import os
import time

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from com_main_gw import (
    init_mpi, stop_mpi, mpi_barrier, rid, mpi_master_id, ctl,
    bksams, bksams_arr, particle_samples_arr_type, proc_id,
    MPI_COMM_WORLD
)
from md_cfuns import cfs
from read_ini_single import readin_model_par, print_model_par, print_current_code_version
from gen_ge import (
    set_dm_init, dms, set_seed, set_clone_weight, set_real_weight,
    update_arrays_single, all_chain_to_arr_single
)
from io_hdf5 import output_dms_hdf5_pdf
from io_txt import output_sams_sg_track_txt


# Chattery output unit (stdout wrapper)
class ChatteryUnit:
    def write(self, s: str) -> None:
        print(s, end='')
    def flush(self) -> None:
        pass

chattery_out_unit = ChatteryUnit()


def output_all_barge_txt(dm, fl: str) -> None:
    """Output distribution function to HDF5."""
    try:
        output_dms_hdf5_pdf(dm, fl)
    except Exception as e:
        print(f"Warning: Could not write HDF5 output: {e}")


def init_model() -> None:
    """Initialize model parameters after reading config."""
    from com_main_gw import emax_factor
    from read_ini_single import set_simu_time
    import com_main_gw as cmg
    
    cmg.rh = 1.0
    cmg.rhmax = cmg.rh * emax_factor * 2
    cmg.rhmin = cmg.rh / emax_factor / 2
    
    set_simu_time()
    
    ctl.nblock_size = ctl.grid_bins // ctl.ntasks
    ctl.nblock_mpi_bg = rid * ctl.nblock_size + 1
    ctl.nblock_mpi_ed = (rid + 1) * ctl.nblock_size
    
    # Initialize dms with proper data structures
    set_dm_init(dms)


def prepare_ini_data(tmprid: str) -> None:
    """Load initial data from ini phase."""
    # Read dms from binary
    if os.path.exists("output/ini/bin/dms.bin"):
        dms.input_bin("output/ini/bin/dms.bin")
    
    # Read particle chain
    chain_path = f"output/ini/bin/single/samchn{tmprid.strip()}"
    if os.path.exists(chain_path + ".pkl"):
        bksams.input_bin(chain_path)
    else:
        print(f"Warning: Could not find initial chain file {chain_path}")


def run_one_snap(cur_time_i: float, cur_time_f: float, smsa, ntasks: int, update_dms: bool) -> None:
    """Run one snapshot evolution step."""
    from gen_ge import collection_data_single_bympi, get_ge_by_root, update_weights, get_dms
    from ac_ec_evl_single import evolve_particles_single
    
    # Evolve particles (placeholder - actual implementation needed)
    try:
        evolve_particles_single(bksams, cur_time_i, cur_time_f)
    except Exception:
        # Placeholder if evolution not implemented
        pass
    
    # Update arrays and collect via MPI
    update_arrays_single()
    collection_data_single_bympi(smsa, ntasks)
    mpi_barrier(MPI_COMM_WORLD)
    
    # Compute new distribution function
    if update_dms:
        get_ge_by_root(smsa, ntasks, False)
        update_weights()
        get_dms(dms)


def convert_sams(chain) -> None:
    """Convert samples for next evolution step."""
    # Placeholder - implement sample conversion logic
    pass


def run_snapshots(tmprid: str, tmpssnapid: str, pid_root_str: str, pid_cld_str: str, i: int) -> None:
    """Run all update steps within a snapshot."""
    ctl.burn_in_phase = False
    str_ = tmprid.strip() + "_" + tmpssnapid.strip()

    smsa = [particle_samples_arr_type() for _ in range(ctl.ntasks)]

    for j in range(1, ctl.num_update_per_snap + 1):
        tmpj = f"{j:4d}"
        cur_time_i = ctl.ts_spshot * float(i - 1) + ctl.update_dt * float(j - 1)
        cur_time_f = ctl.update_dt + cur_time_i

        chattery_out_unit.write(
            f"start.. snap, i, j, cur_time_i, f= {i} {j} {cur_time_i:.4e} {cur_time_f:.4e}\n"
        )

        update_dms = False if ctl.trace_all_sample >= 1 else True

        run_one_snap(cur_time_i, cur_time_f, smsa, ctl.ntasks, update_dms)

        chattery_out_unit.write(f"finished snapshot i, rid, update_j= {i} {rid} {j}\n")

        if rid == mpi_master_id:
            print(f"start output rid={rid}")
            dms.output_bin(f"output/ecev/dms/dms_{tmpssnapid.strip()}_{tmpj.strip()}")
            output_all_barge_txt(dms, f"output/ecev/dms/dms_{tmpssnapid.strip()}_{tmpj.strip()}")
            print(f"output finished rid={rid}")

    print("start output bins")
    bksams.output_bin(f"output/ecev/bin/single/samchn{str_.strip()}")

    if ctl.trace_all_sample >= 1:
        all_chain_to_arr_single(bksams, bksams_arr)
        output_sams_sg_track_txt(bksams_arr, "output/indvd/")

    if i != ctl.n_spshot_total:
        print("convert_sams")
        convert_sams(bksams)


def deallocate_chains_arrs() -> None:
    """Deallocate chain and array memory."""
    bksams.destory()


def main() -> None:
    """Main evolution routine."""
    readin_model_par("model.in")

    init_mpi()
    init_model()

    if rid == 0:
        print_model_par()
        print_current_code_version()

    if rid == 0:
        print(f"root process id = {proc_id}")
    if rid == 1:
        print(f"cld process id = {proc_id}")

    # Resolve path - it may be relative to different locations
    cfs_path = ctl.cfs_file_dir.strip()
    if not os.path.exists(cfs_path + ".bin"):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gnc_dir = os.path.dirname(script_dir)
        alt_path = os.path.normpath(os.path.join(gnc_dir, cfs_path.lstrip("../")))
        if os.path.exists(alt_path + ".bin"):
            cfs_path = alt_path
    
    cfs.input_bin(cfs_path)
    print(f"{rid} readin cfs finished: nj={cfs.nj}, ns={cfs.ns}")

    global mpi_master_id
    mpi_master_id = 0

    tmprid = f"{rid + 1 + ctl.ntask_bg:4d}"

    prepare_ini_data(tmprid)

    t1 = None
    if rid == 0:
        t1 = time.process_time()

    set_seed(ctl.same_rseed_evl, ctl.seed_value + rid)
    update_arrays_single()
    set_clone_weight(bksams)
    set_real_weight(bksams)

    pid_root_str = ""
    pid_cld_str = ""

    for i in range(1, ctl.n_spshot_total + 1):
        tmpssnapid = f"{i:4d}"
        run_snapshots(tmprid, tmpssnapid, pid_root_str, pid_cld_str, i)

    if rid == 0:
        t2 = time.process_time()
        print(f"total time: {t2 - t1:.2f}s")

    print(f"finished main rid={rid}")
    stop_mpi()
    deallocate_chains_arrs()


if __name__ == "__main__":
    main()

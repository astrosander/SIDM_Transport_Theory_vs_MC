"""
read_ini_single.py - Read initialization parameters from model.in and mfrac.in files.
"""
from __future__ import annotations

import math
import os

from com_main_gw import (
    ctl, MBH, rh, rhmin, rhmax, mbh_radius,
    jmin_value, jmax_value, emin_factor, emax_factor,
    log10emin_factor, log10emax_factor, clone_e0_factor, clone_emax,
    log10clone_emax, my_unit_vel_c, model_intej_fast,
    method_int_nearst, method_int_linear, boundary_fj_iso, boundary_fj_ls,
    Jbin_type_lin, Jbin_type_log, jbin_type_sqr,
    star_type_MS, star_type_BH, star_type_NS, star_type_WD, star_type_BD,
    readpar_int_sp, readpar_dbl_sp, readpar_str_sp, READPAR_STR_AUTO_sp,
    skip_comments, get_tnr_timescale_at_rh, get_ccidx_from_type, rid,
    CoreCompType
)
import com_main_gw as cmg


# Module-level variable for tidal radius minimum
rtmin: float = 1.0


def print_current_code_version() -> None:
    """Print the current code version."""
    print("="*60)
    print("GNC Python Version 1.0 (Python port, 2024)")
    print("Original Fortran code version: 1.0 (updated 2023-8-13)")
    print("="*60)


def readin_model_par(fl: str) -> None:
    """Read model parameters from model.in file."""
    global rtmin
    
    if not os.path.exists(fl):
        raise FileNotFoundError(f"Model file not found: {fl}")
    
    funit = 999
    fp = open(fl, "r")

    # Number of threads
    ctl.ntasks = readpar_int_sp(funit, fp, "#", "=")

    # Model of DE and DJ
    model_intej = readpar_str_sp(funit, fp, "#", "=")
    ctl.str_model_intej = model_intej.strip()
    if ctl.str_model_intej == "FAST":
        ctl.model_intej = model_intej_fast
    else:
        raise RuntimeError(f"ERROR, define dejmodel: {ctl.str_model_intej}")

    # Interpolation method
    str_method_int = readpar_str_sp(funit, fp, "#", "=")
    ctl.str_method_interpolate = str_method_int.strip()
    if str_method_int.strip() == "near":
        ctl.method_interpolate = method_int_nearst
    elif str_method_int.strip() == "li2d":
        ctl.method_interpolate = method_int_linear

    # jmin and jmax
    strall, strs, nnum = READPAR_STR_AUTO_sp(funit, fp, "#", ",", " ", 2)
    cmg.jmin_value = float(strs[0])
    cmg.jmax_value = float(strs[1])

    # CFS file directory
    ctl.cfs_file_dir = readpar_str_sp(funit, fp, "#", "=")

    # MBH
    cmg.MBH = readpar_dbl_sp(funit, fp, "#", "=")
    cmg.mbh = cmg.MBH
    cmg.mbh_radius = cmg.MBH / (my_unit_vel_c ** 2)

    # Energy factors
    cmg.emin_factor = readpar_dbl_sp(funit, fp, "#", "=")
    cmg.log10emin_factor = math.log10(cmg.emin_factor)

    cmg.emax_factor = readpar_dbl_sp(funit, fp, "#", "=")
    cmg.log10emax_factor = math.log10(cmg.emax_factor)

    # Boundary
    ctl.x_boundary = readpar_dbl_sp(funit, fp, "#", "=")

    # J-bin type
    str_jbin_bd = readpar_str_sp(funit, fp, "#", "=").strip()
    ctl.str_jbin_bd = str_jbin_bd
    if str_jbin_bd == "LIN":
        ctl.jbin_type = Jbin_type_lin
    elif str_jbin_bd == "LOG":
        ctl.jbin_type = Jbin_type_log
    elif str_jbin_bd == "SQR":
        ctl.jbin_type = jbin_type_sqr
    else:
        raise RuntimeError(f"error! define jbin type {ctl.str_jbin_bd}")

    # Boundary fj type
    str_fj_bd = readpar_str_sp(funit, fp, "#", "=").strip()
    ctl.str_fj_bd = str_fj_bd
    if str_fj_bd == "ISO":
        ctl.boundary_fj = boundary_fj_iso
    elif str_fj_bd == "LS":
        ctl.boundary_fj = boundary_fj_ls
    else:
        raise RuntimeError(f"error! define fj type {ctl.str_fj_bd}")

    # Random seed settings
    ctl.seed_value = readpar_int_sp(funit, fp, "#", "=")
    ctl.same_rseed_ini = readpar_int_sp(funit, fp, "#", "=")

    # Grid bins
    strall, strs, nnum = READPAR_STR_AUTO_sp(funit, fp, "#", ",", " ", 2)
    ctl.gx_bins = int(float(strs[0]))
    ctl.grid_bins = int(float(strs[1]))

    if ctl.grid_bins % ctl.ntasks != 0:
        raise RuntimeError(
            f"read_ini_single.py: grid_bins, ntasks= {ctl.grid_bins} {ctl.ntasks}\n"
            "error! grid_bin number must be integer times of ntasks"
        )

    # Evolution seed
    ctl.same_rseed_evl = readpar_int_sp(funit, fp, "#", "=")

    # Task and snapshot modes
    readin_task_mode(fp)
    readin_snap_mode(fp)

    # Initial alpha
    ctl.alpha_ini = readpar_dbl_sp(funit, fp, "#", "=")
    
    # Loss cone
    ctl.include_loss_cone = readpar_int_sp(funit, fp, "#", "=")

    rtmin = 1.0

    # Clone scheme
    ctl.clone_scheme = readpar_int_sp(funit, fp, "#", "=")
    if ctl.clone_scheme >= 1:
        cmg.clone_e0_factor = readpar_dbl_sp(funit, fp, "#", "=")

    ctl.del_cross_clone = ctl.clone_scheme

    # Chattery and trace settings
    ctl.chattery = readpar_int_sp(funit, fp, "#", "=")
    ctl.trace_all_sample = readpar_int_sp(funit, fp, "#", "=")
    if ctl.trace_all_sample >= 1:
        ctl.output_track_td = readpar_int_sp(funit, fp, "#", "=")
        ctl.output_track_plunge = readpar_int_sp(funit, fp, "#", "=")
        ctl.output_track_emri = readpar_int_sp(funit, fp, "#", "=")

    fp.close()
    print("readin model.in finished")
    
    # Read mass bins
    readin_mass_bins("mfrac.in")
    check_readin()


def readin_mass_bins(fl: str) -> None:
    """Read mass bin configuration from mfrac.in file."""
    import numpy as np
    
    if not os.path.exists(fl):
        raise FileNotFoundError(f"Mass fraction file not found: {fl}")
    
    have_comp = [False] * 7
    num_comp = 0
    funit = 1999

    fp = open(fl, "r")
    strall, strs, nnum = READPAR_STR_AUTO_sp(funit, fp, "#", ",", " ", 12)

    ctl.m_bins = int(float(strs[0]))
    str_massbin_mode = strs[1].strip()

    if str_massbin_mode == "GIVEN":
        for i in range(ctl.m_bins):
            skip_comments(fp, "#")
            parts1 = fp.readline().strip().split()
            ctl.bin_mass_m1[i] = float(parts1[0])
            ctl.bin_mass[i] = float(parts1[1])
            ctl.bin_mass_m2[i] = float(parts1[2])
            ctl.asymptot[0, i] = float(parts1[3])
            ctl.ini_weight_n[i] = float(parts1[4])
            ctl.clone_factor[i] = int(float(parts1[5]))
            
            # Set particle number based on weight
            ctl.bin_mass_particle_number[i] = int(ctl.ini_weight_n[i])
            ctl.weight_n[i] = ctl.ini_weight_n[i]
            
            parts2 = fp.readline().strip().split()
            for k in range(min(5, len(parts2))):
                ctl.asymptot[k + 1, i] = float(parts2[k])

        for i in range(1, ctl.m_bins):
            if ctl.bin_mass_m1[i] == ctl.bin_mass_m2[i - 1]:
                raise RuntimeError(
                    f"error! bin_mass_m1=bin_mass_m2 {i+1}\n"
                    f"m1={ctl.bin_mass_m1[i]} {ctl.bin_mass_m2[i-1]}"
                )

        for j in range(7):
            for i in range(ctl.m_bins):
                if ctl.asymptot[j + 1, i] != 0:
                    have_comp[j] = True
                    num_comp += 1
                    break

        ctl.num_bk_comp = num_comp
        ctl.cc = [CoreCompType() for _ in range(max(1, num_comp))]

        jj = 0
        for i in range(7):
            if have_comp[i]:
                if i == 0:
                    ctl.cc[jj].bktypemodel = star_type_MS
                    ctl.idxstar = jj + 1
                elif i == 1:
                    ctl.cc[jj].bktypemodel = star_type_BH
                    ctl.idxsbh = jj + 1
                elif i == 2:
                    ctl.cc[jj].bktypemodel = star_type_NS
                    ctl.idxns = jj + 1
                elif i == 3:
                    ctl.cc[jj].bktypemodel = star_type_WD
                    ctl.idxwd = jj + 1
                elif i == 4:
                    ctl.cc[jj].bktypemodel = star_type_BD
                    ctl.idxbd = jj + 1
                jj += 1
    else:
        raise RuntimeError("readin_mass_bins: only GIVEN mode is supported")

    ctl.idx_ref = 1
    ctl.mass_ref = ctl.bin_mass[ctl.idx_ref - 1]
    fp.close()
    print(f"readin mfrac.in finished: m_bins={ctl.m_bins}, num_bk_comp={ctl.num_bk_comp}")


def find_close_number(nin: float) -> float:
    """Find a close round number."""
    if nin <= 0:
        raise RuntimeError("nin should be larger than zero! stopped!")
    dg = int(math.log10(nin)) + 1
    if nin > 5 * 10 ** (dg - 1):
        nout = 5 * 10 ** (dg - 1)
    else:
        nout = 10 ** (dg - 1)
    print(f"nin, nout={nin} {nout}")
    return float(nout)


def readin_snap_mode(fp) -> None:
    """Read snapshot mode parameters."""
    funit = 999
    ctl.num_update_per_snap = readpar_int_sp(funit, fp, "#", "=")
    ctl.ts_snap_input = readpar_dbl_sp(funit, fp, "#", "=")
    time_unit = readpar_str_sp(funit, fp, "#", "=")
    ctl.n_spshot = readpar_int_sp(funit, fp, "#", "=")
    ctl.time_unit = time_unit.strip()


def set_simu_time() -> None:
    """Set simulation time parameters based on relaxation time."""
    tnr = get_tnr_timescale_at_rh()
    ctl.tnr = tnr
    ctl.ts_spshot = tnr * ctl.ts_snap_input
    ctl.update_dt = ctl.ts_spshot / float(ctl.num_update_per_snap)
    ctl.n_spshot_total = ctl.n_spshot
    ctl.total_time = ctl.ts_spshot * ctl.n_spshot_total
    
    # Set velocity and density scales
    ctl.v0 = math.sqrt(cmg.MBH / cmg.rh) if cmg.rh > 0 else 1.0
    
    # Estimate n0 from asymptotic normalization
    n_tot = 0
    for i in range(ctl.m_bins):
        n_tot += ctl.asymptot[0, i]
    ctl.n0 = n_tot / (4.0 / 3.0 * math.pi * cmg.rh ** 3) if cmg.rh > 0 else 1.0
    ctl.n_basic = n_tot
    
    # Set energy scales
    ctl.energy0 = ctl.v0 ** 2
    ctl.energy_boundary = ctl.energy0 * ctl.x_boundary
    ctl.energy_min = ctl.energy0 * cmg.emin_factor
    ctl.energy_max = ctl.energy0 * cmg.emax_factor
    
    # Clone energy threshold
    ctl.clone_e0 = ctl.energy0 * cmg.clone_e0_factor
    
    print(f"Simulation time parameters:")
    print(f"  TNR = {tnr:.4e} Myr")
    print(f"  ts_spshot = {ctl.ts_spshot:.4e}")
    print(f"  n_spshot = {ctl.n_spshot}")
    print(f"  total_time = {ctl.total_time:.4e}")


def readin_task_mode(fp) -> None:
    """Read task mode parameters."""
    ctl.ntask_bg = 0
    ctl.ntask_total = ctl.ntask_bg + ctl.ntasks


def print_model_par() -> None:
    """Print model parameters."""
    print("\n" + "="*60)
    print("MODEL PARAMETERS")
    print("="*60)
    print(f"{'num of components=':25s}{ctl.num_bk_comp:10d}")
    print(f"{'MBH=':25s}{cmg.MBH:10.3e}")
    print(f"{'rh=':25s}{cmg.rh:10.3e}")
    print(f"{'Jmin, Jmax=':25s}{cmg.jmin_value:10.3e}{cmg.jmax_value:10.3e}")
    print(f"{'ExMIN, ExMAX=':25s}{cmg.emin_factor:10.3e}{cmg.emax_factor:10.3e}")
    print(f"{'x_boundary=':25s}{ctl.x_boundary:10.3e}")
    print(f"{'Integration MODEL=':25s}{ctl.str_model_intej.strip():25s}")
    print(f"{'Interpolate Method=':25s}{ctl.str_method_interpolate.strip():25s}")

    if ctl.include_loss_cone >= 1:
        print(f"{'include_loss_cone=':25s}{ctl.include_loss_cone:10d}")

    print(f"\n{'Mass bin mode:':25s}GIVEN")
    for i in range(ctl.m_bins):
        print(f"  bin {i+1}: m={ctl.bin_mass[i]:.3f}, "
              f"m1={ctl.bin_mass_m1[i]:.3f}, m2={ctl.bin_mass_m2[i]:.3f}, "
              f"N={ctl.bin_mass_particle_number[i]}, clone_factor={ctl.clone_factor[i]}")

    print(f"\n{'N_basic=':25s}{ctl.n_basic:10.3f}")
    print(f"{'NTASK, NTASK_TOT=':25s}{ctl.ntasks:10d}{ctl.ntask_total:10d}")
    print(f"{'NSNAP, NSNAP_TOT=':25s}{ctl.n_spshot:10d}{ctl.n_spshot_total:10d}")
    print(f"{'TIMEUNIT=':25s}{ctl.time_unit:10s}")
    print(f"{'TNR=':25s}{ctl.tnr:10.4e}")
    print(f"{'TOTAL_TIME=':25s}{ctl.total_time:10.4e}")
    print(f"{'CLONE_SCHEME=':25s}{ctl.clone_scheme:10d}")
    print(f"{'SAME_EVL_SEED=':25s}{ctl.same_rseed_evl:10d}")
    print("="*60 + "\n")


def check_readin() -> None:
    """Check that all required parameters were read correctly."""
    idxstar = get_ccidx_from_type(0, star_type_MS)
    idxsbh = get_ccidx_from_type(0, star_type_BH)
    
    if ctl.m_bins <= 0:
        raise RuntimeError("Error: m_bins must be > 0")
    
    if ctl.grid_bins <= 0:
        raise RuntimeError("Error: grid_bins must be > 0")
    
    print("Parameter check passed.")

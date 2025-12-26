import math
from com_main_gw import (
    ctl,
    MBH,
    rh,
    rhmin,
    rhmax,
    mbh_radius,
    jmin_value,
    jmax_value,
    emin_factor,
    emax_factor,
    log10emin_factor,
    log10emax_factor,
    clone_e0_factor,
    clone_emax,
    log10clone_emax,
    my_unit_vel_c,
    model_intej_fast,
    method_int_nearst,
    method_int_linear,
    boundary_fj_iso,
    boundary_fj_ls,
    Jbin_type_lin,
    Jbin_type_log,
    jbin_type_sqr,
    star_type_MS,
    star_type_BH,
    star_type_NS,
    star_type_WD,
    star_type_BD,
)
from com_main_gw import (
    readpar_int_sp,
    readpar_dbl_sp,
    readpar_str_sp,
    READPAR_STR_AUTO_sp,
    skip_comments,
    get_tnr_timescale_at_rh,
    get_ccidx_from_type,
)


def print_current_code_version() -> None:
    print("current code version: 1.0 (updated in 2023-8-13)")


def readin_model_par(fl: str) -> None:
    global MBH, mbh_radius, emin_factor, emax_factor, log10emin_factor, log10emax_factor
    global jmin_value, jmax_value, clone_e0_factor, clone_emax, log10clone_emax
    global rtmin

    with open(fl, "r") as f:
        funit = 999
        ier = 0

    funit = 999
    fp = open(fl, "r")

    ctl.ntasks = readpar_int_sp(funit, fp, "#", "=")

    model_intej = readpar_str_sp(funit, fp, "#", "=")
    ctl.str_model_intej = model_intej.strip()
    if ctl.str_model_intej == "FAST":
        ctl.model_intej = model_intej_fast
    else:
        raise RuntimeError(f"ERROR, define dejmodel: {ctl.str_model_intej}")

    str_method_int = readpar_str_sp(funit, fp, "#", "=")
    ctl.str_method_interpolate = str_method_int.strip()
    if str_method_int.strip() == "near":
        ctl.method_interpolate = method_int_nearst
    elif str_method_int.strip() == "li2d":
        ctl.method_interpolate = method_int_linear

    strall, strs, nnum = READPAR_STR_AUTO_sp(funit, fp, "#", ",", " ", 2)
    jmin_value = float(strs[0])
    jmax_value = float(strs[1])

    ctl.cfs_file_dir = readpar_str_sp(funit, fp, "#", "=")

    MBH = readpar_dbl_sp(funit, fp, "#", "=")
    mbh_radius = MBH / (my_unit_vel_c**2)

    emin_factor = readpar_dbl_sp(funit, fp, "#", "=")
    log10emin_factor = math.log10(emin_factor)

    emax_factor = readpar_dbl_sp(funit, fp, "#", "=")
    log10emax_factor = math.log10(emax_factor)

    ctl.x_boundary = readpar_dbl_sp(funit, fp, "#", "=")

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

    str_fj_bd = readpar_str_sp(funit, fp, "#", "=").strip()
    ctl.str_fj_bd = str_fj_bd
    if str_fj_bd == "ISO":
        ctl.boundary_fj = boundary_fj_iso
    elif str_fj_bd == "LS":
        ctl.boundary_fj = boundary_fj_ls
    else:
        raise RuntimeError(f"error! define fj type {ctl.str_fj_bd}")

    ctl.seed_value = readpar_int_sp(funit, fp, "#", "=")
    ctl.same_rseed_ini = readpar_int_sp(funit, fp, "#", "=")

    strall, strs, nnum = READPAR_STR_AUTO_sp(funit, fp, "#", ",", " ", 2)
    ctl.gx_bins = int(float(strs[0]))
    ctl.grid_bins = int(float(strs[1]))

    if ctl.grid_bins % ctl.ntasks != 0:
        raise RuntimeError(
            f"read_ini_single.f90: 101: grid_bins, ntasks= {ctl.grid_bins} {ctl.ntasks}\n"
            "read_ini_single.f90: 101: error! grid_bin number must be integer times of ntasks"
        )

    ctl.same_rseed_evl = readpar_int_sp(funit, fp, "#", "=")

    readin_task_mode(fp)
    readin_snap_mode(fp)

    ctl.alpha_ini = readpar_dbl_sp(funit, fp, "#", "=")
    ctl.include_loss_cone = readpar_int_sp(funit, fp, "#", "=")

    rtmin = 1.0

    ctl.clone_scheme = readpar_int_sp(funit, fp, "#", "=")
    if ctl.clone_scheme >= 1:
        clone_e0_factor = readpar_dbl_sp(funit, fp, "#", "=")

    ctl.del_cross_clone = ctl.clone_scheme

    ctl.chattery = readpar_int_sp(funit, fp, "#", "=")
    ctl.trace_all_sample = readpar_int_sp(funit, fp, "#", "=")
    if ctl.trace_all_sample >= 1:
        ctl.output_track_td = readpar_int_sp(funit, fp, "#", "=")
        ctl.output_track_plunge = readpar_int_sp(funit, fp, "#", "=")
        ctl.output_track_emri = readpar_int_sp(funit, fp, "#", "=")

    fp.close()
    print("single finished")
    readin_mass_bins("mfrac.in")
    check_readin()


def readin_mass_bins(fl: str) -> None:
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
            parts2 = fp.readline().strip().split()
            for k in range(5):
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
        ctl.cc = [type(ctl).cc.__args__[0]() for _ in range(num_comp)]  # placeholder allocation

        jj = 0
        for i in range(7):
            if have_comp[i]:
                jj += 1
                if i == 0:
                    ctl.cc[jj - 1].bktypemodel = star_type_MS
                elif i == 1:
                    ctl.cc[jj - 1].bktypemodel = star_type_BH
                elif i == 2:
                    ctl.cc[jj - 1].bktypemodel = star_type_NS
                elif i == 3:
                    ctl.cc[jj - 1].bktypemodel = star_type_WD
                elif i == 4:
                    ctl.cc[jj - 1].bktypemodel = star_type_BD
    else:
        raise RuntimeError("readin_mass_bins:error!")

    ctl.idx_ref = 1
    ctl.mass_ref = ctl.bin_mass[ctl.idx_ref - 1]
    fp.close()


def find_close_number(nin: float) -> float:
    if nin <= 0:
        raise RuntimeError("nin should be larger than zero! stoped!")
    dg = int(math.log10(nin)) + 1
    if nin > 5 * 10 ** (dg - 1):
        nout = 5 * 10 ** (dg - 1)
    else:
        nout = 10 ** (dg - 1)
    print(f"nin, nout={nin} {nout}")
    return float(nout)


def readin_snap_mode(fp) -> None:
    funit = 999
    ctl.num_update_per_snap = readpar_int_sp(funit, fp, "#", "=")
    ctl.ts_snap_input = readpar_dbl_sp(funit, fp, "#", "=")
    time_unit = readpar_str_sp(funit, fp, "#", "=")
    ctl.n_spshot = readpar_int_sp(funit, fp, "#", "=")
    ctl.time_unit = time_unit.strip()


def set_simu_time() -> None:
    get_tnr_timescale_at_rh(ctl.tnr)
    ctl.ts_spshot = ctl.tnr * ctl.ts_snap_input
    ctl.update_dt = ctl.ts_spshot / float(ctl.num_update_per_snap)
    ctl.n_spshot_total = ctl.n_spshot
    ctl.total_time = ctl.ts_spshot * ctl.n_spshot_total


def readin_task_mode(fp) -> None:
    ctl.ntask_bg = 0
    ctl.ntask_total = ctl.ntask_bg + ctl.ntasks


def print_model_par() -> None:
    print(f"{'num of components=':25s}{ctl.num_bk_comp:10d}")
    print(f"{'MBH,rh,rhmin,rhmax=':25s}{MBH:10.3e}{rh:10.3e}{rhmin:10.3e}{rhmax:10.3e}")
    print(f"{'Jmin, Jmax=':25s}{jmin_value:10.3e}{jmax_value:10.3e}")
    print(f"{'ExMIN,ExMAX, EB=':25s}{emin_factor:10.3e}{emax_factor:10.3e}{ctl.x_boundary:10.3e}")
    print(f"{'Integration MODEL=':25s}{ctl.str_model_intej.strip():25s}")
    print(f"{'Intepolate Method=':25s}{ctl.str_method_interpolate.strip():25s}")

    if ctl.include_loss_cone >= 1:
        print(f"include_loss_cone={ctl.include_loss_cone}")

    print(f"{'mass bin mode: given':25s}")
    mass_bins = " ".join(f"{ctl.bin_mass[k]:10.3e}" for k in range(ctl.m_bins))
    print(f"{'mass_bin=':25s}{mass_bins}")

    for i in range(ctl.m_bins):
        print(
            f"{'I,IWN,WN,PN,CL=':25s}"
            f"{i+1:8d}{ctl.ini_weight_n[i]:10.3f}{ctl.weight_n[i]:10.3f}"
            f"{ctl.bin_mass_particle_number[i]:10d}{ctl.clone_factor[i]:10d}"
        )

    print(f"{'ctl%N_basic=':25s}{ctl.n_basic:10.3f}")
    print(f"{'NTASK, NTASK_TOT=':25s}{ctl.ntasks:10d}{ctl.ntask_total:10d}")
    print(f"{'NSNAP, NSNAP_TOT=':25s}{ctl.n_spshot:10d}{ctl.n_spshot_total:10d}")
    print(f"{'TIMEUNIT,TNR=':25s}{ctl.time_unit:4s}{ctl.tnr:10.2f}")
    print(f"{'TOTAL_TIME=':25s}{ctl.total_time:10.2f}")
    print(f"{'CLONE=':25s}{bool(ctl.clone_scheme)}")
    print(f"{'SEVSEED=':25s}{bool(ctl.same_rseed_evl)}")


def check_readin() -> None:
    idxstar = 0
    idxsbh = 0
    get_ccidx_from_type(idxstar, star_type_MS)
    get_ccidx_from_type(idxsbh, star_type_BH)

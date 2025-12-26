import os
import subprocess
import math
import numpy as np

from com_main_gw import (
    ctl,
    mbh,
    jbin_type_lin,
    jbin_type_log,
    jbin_type_sqr,
    exit_boundary_max,
    exit_boundary_min,
    exit_invtransit,
    exit_normal,
    star_type_BH,
    star_type_MS,
    star_type_NS,
    star_type_WD,
    star_type_BD,
    rid,
    chattery_out_unit,
    boundary_sts_emin_cros,
    boundary_sts_emin_dir,
    boundary_sts_create,
    boundary_sts_replace,
    boundary_cross_up,
    boundary_cross_down,
    boundary_sts_emax_cros,
    boundary_cross_net,
    dms,
    bksams,
    bksams_arr,
    bksams_arr_norm,
    bksams_pointer_arr,
    cfs,
    n_tot_comp,
    df,
)


def sams_get_weight_clone_single(sps):
    for i in range(sps.n):
        call_get_mass_idx = get_mass_idx
        mass_idx = -1
        call_get_mass_idx(sps.sp[i].m, mass_idx_holder := {"v": mass_idx})
        mass_idx = mass_idx_holder["v"]
        amplifier = ctl.clone_factor[mass_idx - 1] if mass_idx != -1 else 0
        if sps.sp[i].exit_flag == exit_boundary_max:
            en = -mbh / 2.0 / sps.sp[i].byot_bf.a_bin
        else:
            en = sps.sp[i].en
        particle_sample_get_weight_clone(en, ctl.clone_scheme, amplifier, ctl.clone_e0, sps.sp[i].weight_clone)
        if math.isnan(sps.sp[i].weight_clone):
            raise RuntimeError(
                f"sams_get_weight_clone_single:NaN {sps.sp[i].weight_clone} "
                f"{sps.sp[i].obtype} {sps.sp[i].obidx} {sps.sp[i].en} {sps.sp[i].id}"
            )


def set_clone_weight(sms):
    pt = sms.head
    while pt is not None and getattr(pt, "ob", None) is not None:
        ob = pt.ob
        mass_idx_holder = {"v": -1}
        get_mass_idx(ob.m, mass_idx_holder)
        mass_idx = mass_idx_holder["v"]
        amplifier = ctl.clone_factor[mass_idx - 1] if mass_idx != -1 else 0
        en = ob.en
        particle_sample_get_weight_clone(en, ctl.clone_scheme, amplifier, ctl.clone_e0, ob.weight_clone)
        pt = pt.next


def sams_select_merge_single(sps, sps_out):
    def selection(pt):
        return getattr(pt, "append_left", None) is not None
    chain_select_by_condition(sps, sps_out, selection)


def chain_select_type_single(sps, sps_out, obtype):
    def selection(pt):
        return pt.ob.obtype == obtype
    chain_select_by_condition(sps, sps_out, selection)


def output_sams_merge_hierarchy(pt, fl):
    path = f"{fl.strip()}_hierarch.txt"
    with open(path, "w") as f:
        f.write(f"{'':15s}{'nhiar':6s}{'m':20s}{'weight_real':20s}{'weight_clone':20s}"
                f"{'weight_asym':20s}{'weight_N':20s}{'ctime':20s}{'abin':20s}\n")
        ps = pt
        while ps is not None:
            write_txt(f, "-", ps)
            ps = ps.next


def write_txt(f, marker, pt_now):
    f.write(
        f"{marker.strip():15s}"
        f"{int(getattr(pt_now.ob, 'm', 0.0)):6d}"
        f"{pt_now.ob.weight_real:20.6f}"
        f"{pt_now.ob.weight_clone:20.6f}"
        f"{pt_now.ob.weight_N:20.6f}"
        f"{pt_now.ob.create_time:20.6f}"
        f"{pt_now.ob.byot.a_bin:20.6f}\n"
    )
    marker_here = f"{marker.strip()}-"
    if getattr(pt_now, "append_left", None) is not None:
        write_txt(f, marker_here, pt_now.append_left)
    if getattr(pt_now, "append_right", None) is not None:
        write_txt(f, marker_here, pt_now.append_right)


def sams_arr_select_type_single(sps, sps_out, obtype):
    nsel = 0
    for i in range(sps.n):
        if sps.sp[i].obtype == obtype:
            nsel += 1
    sps_out.init(nsel)
    nsel2 = 0
    for i in range(sps.n):
        if sps.sp[i].obtype == obtype:
            sps_out.sp[nsel2] = sps.sp[i]
            nsel2 += 1


def output_all_sts(fout):
    flag = 2 if ctl.include_loss_cone >= 1 else 1
    output_eveset_txt(pteve_star, f"{fout}/pro/MS/", flag)
    if ctl.idxsbh != -1:
        output_eveset_txt(pteve_sbh, f"{fout}/pro/BH/", flag)
    if ctl.idxbd != -1:
        output_eveset_txt(pteve_bd, f"{fout}/pro/BD/", flag)
    if ctl.idxns != -1:
        output_eveset_txt(pteve_ns, f"{fout}/pro/NS/", flag)
    if ctl.idxwd != -1:
        output_eveset_txt(pteve_wd, f"{fout}/pro/WD/", flag)


def get_memo_usage(pid):
    pid_str = f"{pid:d}"
    cmd = f"ps axo pid,rss | grep {pid_str}"
    print("pid, memory occupy(kB)")
    subprocess.run(cmd, shell=True, check=False)


def get_ccidx_from_type(idx_holder, bktype):
    idx_holder["v"] = -1
    for i in range(ctl.num_bk_comp):
        if ctl.cc[i].bktypemodel == bktype:
            idx_holder["v"] = i + 1
            return


def get_mass_idx(m, idx_holder):
    idx_holder["v"] = -1
    for i in range(ctl.m_bins):
        if m >= ctl.bin_mass_m1[i] and m <= ctl.bin_mass_m2[i]:
            idx_holder["v"] = i + 1
            return
    print(f"get_mass_idx idx=-1: m={m}")


def get_javg_coef(dm):
    for i in range(dm.n):
        s1_dee = dm.mb[i].dc.s1_dee
        s1_de = dm.mb[i].dc.s1_de
        s2_de_0 = dm.mb[i].dc.s2_de_0

        s1_dee.init(dm.emin_factor, dm.emax_factor, s2_de_0.nx, "linear")
        s1_dee.xb = s2_de_0.xcenter

        for j in range(s2_de_0.nx):
            ss = 0.0
            for k in range(s2_de_0.ny):
                for l in range(dm.n):
                    s2d = dm.mb[l].dc.s2_dee
                    s2d.ystep = s2d.ycenter[1] - s2d.ycenter[0]
                    if dm.jbin_type == jbin_type_lin:
                        ss += s2d.fxy[j, k] * s2d.ycenter[k] * s2d.ystep
                    elif dm.jbin_type == jbin_type_log:
                        ss += s2d.fxy[j, k] * (10.0 ** s2d.ycenter[k]) ** 2 * s2d.ystep * math.log(10.0)
                    elif dm.jbin_type == jbin_type_sqr:
                        ss += s2d.fxy[j, k] / 2.0 * s2d.ystep
            s1_dee.fx[j] = ss * 2.0

        s1_de.init(dm.emin_factor, dm.emax_factor, s2_de_0.nx, "linear")
        s1_de.xb = s2_de_0.xcenter

        for j in range(s2_de_0.nx):
            ss = 0.0
            for k in range(s2_de_0.ny):
                for l in range(dm.n):
                    s2dl1 = dm.mb[l].dc.s2_de_0
                    s2dl2 = dm.mb[l].dc.s2_de_110
                    s2dl1.ystep = s2dl1.ycenter[1] - s2dl1.ycenter[0]
                    term = s2dl1.fxy[j, k] + dm.mb[i].mc / dm.mb[l].mc * s2dl2.fxy[j, k]
                    if dm.jbin_type == jbin_type_lin:
                        ss += term * s2dl1.ycenter[k] * s2dl1.ystep
                    elif dm.jbin_type == jbin_type_log:
                        ss += term * (10.0 ** s2dl1.ycenter[k]) ** 2 * s2dl1.ystep * math.log(10.0)
                    elif dm.jbin_type == jbin_type_sqr:
                        ss += term / 2.0 * s2dl1.ystep
            s1_de.fx[j] = ss * 2.0


def print_javg_coef_theory_from_pow(xb, n, mc, m, b0, xmin, xmax, alpha, m1, res, res2):
    sigma32 = (2.0 * math.pi * ctl.v0 ** 2) ** (-1.5)
    n0 = ctl.n0
    const = 16.0 * math.pi ** 2 * math.log(mbh) * sigma32 * n0

    for i in range(n):
        ss = 0.0
        for j in range(m):
            aj = b0[j] / (xmin ** alpha[j])
            ss += (
                (m1 * mc[j] * aj / (alpha[j] - 1.5) * (xmax ** (alpha[j] - 1.5) - xb[i] ** (alpha[j] - 1.5))
                 - mc[j] ** 2 * (aj / (alpha[j] + 1.0) / (xb[i] ** 2.5) * (xb[i] ** (alpha[j] + 1.0) - xmin ** (alpha[j] + 1.0))
                               + b0[j] / (xb[i] ** 2.5)))
            )
        res[i] = const * ss * xb[i] ** 1.5

        ss = 0.0
        for j in range(m):
            aj = b0[j] / (xmin ** alpha[j])
            if alpha[j] != 0.5:
                ss += mc[j] ** 2 * (
                    aj / (alpha[j] - 0.5) * (xmax ** (alpha[j] - 0.5) - xb[i] ** (alpha[j] - 0.5))
                    + (aj / (alpha[j] + 1.0) / (xb[i] ** 1.5) * (xb[i] ** (alpha[j] + 1.0) - xmin ** (alpha[j] + 1.0))
                       + b0[j] / (xb[i] ** 1.5))
                )
            else:
                ss += mc[j] ** 2 * (
                    aj * math.log(xmax / xb[i])
                    + (aj / (alpha[j] + 1.0) / (xb[i] ** 1.5) * (xb[i] ** (alpha[j] + 1.0) - xmin ** (alpha[j] + 1.0))
                       + b0[j] / (xb[i] ** 1.5))
                )
        res2[i] = 4.0 / 3.0 * const * ss * xb[i] ** 0.5
        print(xb[i], res[i], res2[i])


def get_boundary_flux(en0, en1, sample):
    if en0 > ctl.energy_boundary and en1 < ctl.energy_boundary:
        idxmass_holder = {"v": -1}
        get_mass_idx(sample.m, idxmass_holder)
        idxmass = idxmass_holder["v"]
        idxob_holder = {"v": -1}
        get_type_idx(sample, idxob_holder)
        idxob = idxob_holder["v"]
        if idxob != -1 and idxmass != -1:
            ctl.bin_mass_flux_in[idxmass - 1][idxob - 1] += sample.weight_n
        return

    if en1 > ctl.energy_boundary and en0 < ctl.energy_boundary and en1 < ctl.energy_min:
        idxmass_holder = {"v": -1}
        get_mass_idx(sample.m, idxmass_holder)
        idxmass = idxmass_holder["v"]
        idxob_holder = {"v": -1}
        get_type_idx(sample, idxob_holder)
        idxob = idxob_holder["v"]
        if idxob != -1 and idxmass != -1:
            ctl.bin_mass_flux_out[idxmass - 1][idxob - 1] += sample.weight_n
        return

    if en1 < ctl.energy_max:
        idxmass_holder = {"v": -1}
        get_mass_idx(sample.m, idxmass_holder)
        idxmass = idxmass_holder["v"]
        idxob_holder = {"v": -1}
        get_type_idx(sample, idxob_holder)
        idxob = idxob_holder["v"]
        if idxob != -1 and idxmass != -1:
            ctl.bin_mass_emax_out[idxmass - 1][idxob - 1] += sample.weight_n


def get_num_at_boundary(bksam, bysam):
    ctl.bin_mass_N[:] = 0
    pt = bksam.head
    while pt is not None:
        if pt.ob.en > ctl.energy_boundary and pt.ob.en < ctl.energy_min and pt.ob.exit_flag == exit_normal:
            obmidx_holder = {"v": -1}
            get_mass_idx(pt.ob.m, obmidx_holder)
            obmidx = obmidx_holder["v"]
            obtidx_holder = {"v": -1}
            get_type_idx(pt.ob, obtidx_holder)
            obtidx = obtidx_holder["v"]
            if obtidx != -1 and obmidx != -1:
                ctl.bin_mass_N[obmidx - 1][obtidx - 1] += pt.ob.weight_n
        pt = pt.next

    pt = bysam.head
    while pt is not None:
        if getattr(pt, "ob", None) is not None and pt.ob.en > ctl.energy_boundary and pt.ob.en < ctl.energy_min and pt.ob.exit_flag == exit_normal:
            obmidx_holder = {"v": -1}
            get_mass_idx(pt.ob.m, obmidx_holder)
            obmidx = obmidx_holder["v"]
            obtidx_holder = {"v": -1}
            get_type_idx(pt.ob, obtidx_holder)
            obtidx = obtidx_holder["v"]
            if obtidx != -1 and obmidx != -1:
                ctl.bin_mass_N[obmidx - 1][obtidx - 1] += pt.ob.weight_n
        pt = pt.next


def print_bin_mass_N():
    print("BIN_MASS_N:", file=os.fdopen(os.dup(chattery_out_unit), "w"))
    out = os.fdopen(os.dup(chattery_out_unit), "w")
    out.write("BIN_MASS_N:\n")
    out.write(f"{'id':12s}{'mI':12s}{'star':12s}{'sbh':12s}{'bbh':12s}\n")
    for i in range(ctl.m_bins):
        for j in range(ctl.num_bk_comp):
            if ctl.bin_mass_N[i][j] != 0:
                out.write(f"{'BM':8s}{rid:12d}{(i+1):12d}{int(ctl.bin_mass_N[i][0]):12d}"
                          f"{int(ctl.bin_mass_N[i][1]):12d}{int(ctl.bin_mass_N[i][2]):12d}\n")
    out.flush()


def print_boundary_sts():
    global boundary_cross_net
    boundary_cross_net = boundary_cross_up - boundary_cross_down
    out = os.fdopen(os.dup(chattery_out_unit), "w")
    out.write(f"{'emax_cros':10s}{'emin_dir':10s}{'emin_cros':10s}{'replace':10s}{'create':10s}"
              f"{'up':10s}{'down':10s}{'net':10s}\n")
    out.write(f"{boundary_sts_emax_cros:10d}{boundary_sts_emin_dir:10d}{boundary_sts_emin_cros:10d}"
              f"{boundary_sts_replace:10d}{boundary_sts_create:10d}{boundary_cross_up:10d}"
              f"{boundary_cross_down:10d}{boundary_cross_net:10d}\n")
    out.flush()

    globals()["boundary_sts_emin_cros"] = 0
    globals()["boundary_sts_replace"] = 0
    globals()["boundary_sts_create"] = 0
    globals()["boundary_sts_emin_dir"] = 0
    globals()["boundary_cross_up"] = 0
    globals()["boundary_cross_down"] = 0
    globals()["boundary_sts_emax_cros"] = 0


def check(str_pos):
    pt = bksams.head
    while pt is not None:
        if pt.ob.en < ctl.energy_min and pt.ob.exit_flag == exit_normal:
            raise RuntimeError(f"{str_pos}\nerror! {pt.ob.en / ctl.clone_e0} {ctl.clone_e0} {pt.ob.id}")
        pt = pt.next


def get_star_type(type_int, str_holder):
    if type_int == star_type_BH:
        str_holder["v"] = "BH"
    elif type_int == star_type_MS:
        str_holder["v"] = "MS"
    elif type_int == star_type_NS:
        str_holder["v"] = "NS"
    elif type_int == star_type_WD:
        str_holder["v"] = "WD"
    elif type_int == star_type_BD:
        str_holder["v"] = "BD"
    else:
        str_holder["v"] = "UNKNOWN"


def deallocate_chains_arrs():
    bksams.destory()
    if hasattr(bksams_arr, "sp") and bksams_arr.sp is not None:
        bksams_arr.sp = None
    if hasattr(bksams_arr_norm, "sp") and bksams_arr_norm.sp is not None:
        bksams_arr_norm.sp = None
    if hasattr(bksams_pointer_arr, "pt") and bksams_pointer_arr.pt is not None:
        bksams_pointer_arr.pt = None

    for i in range(dms.n):
        mb = dms.mb[i]
        for attr in ("bstar", "bbh", "sbh", "all", "bd", "wd", "ns"):
            comp = getattr(mb, attr, None)
            if comp is not None:
                if hasattr(comp, "nejw") and comp.nejw is not None:
                    comp.nejw = None
                if hasattr(comp, "deallocation"):
                    comp.deallocation()
        for j in range(n_tot_comp):
            mb.dsp[j].p = None

    dms.mb = None

    if getattr(cfs, "cfs_110", None) is not None:
        cfs.cfs_110 = None
        cfs.cfs_111 = None
        cfs.cfs_131 = None
        cfs.cfs_13_1 = None
        cfs.cfs_130 = None
        cfs.cfs_330 = None
        cfs.cfs_310 = None
        cfs.jum = None
        cfs.s = None

    if df is not None:
        globals()["df"] = None

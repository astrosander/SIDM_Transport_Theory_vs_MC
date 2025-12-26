from __future__ import annotations

import math


def get_tnr_timescale_at_rh():
    nm2_tot = 0.0
    for i in range(1, ctl.m_bins + 1):
        nm2_tot += ctl.n0 * ctl.asymptot[1, i] * (ctl.bin_mass[i] ** 2)

    lglambda = math.log(mbh)
    tnr = 0.34 * (ctl.v0 ** 3) / nm2_tot / lglambda / 2.0 / math.pi / 1.0e6

    if rid == 0:
        print("TNR(rh)=", tnr, " Myr")
    return tnr


def output_sg_sample_track_txt(sp, fl):
    tmpid = f"{sp.id:15d}"
    path = f"{str(fl).strip()}_{tmpid.strip()}_simple.txt"
    with open(path, "w") as f:
        f.write(f"{'time':29s}{'ac':20s}{'ec':20s}{'inc':20s}{'om':20s}{'flag':10s}\n")
        for i in range(sp.length):
            tr = sp.track[i]
            f.write(
                f"{tr.time:29.15E}"
                f"{tr.ac:20.10E}"
                f"{tr.ec:20.10E}"
                f"{tr.incout:20.10E}"
                f"{tr.omout:20.10E}"
                f"{int(tr.state_flag):10d}\n"
            )


def output_sams_sg_track_txt(sps, fl):
    nummax = 500
    print("track n=", sps.n)

    if not hasattr(sps, "sp") or sps.sp is None:
        print("sps_arr not allocated")
        return

    num = 0
    num2 = 0

    for i in range(sps.n):
        sp = sps.sp[i]
        if sp.write_down_track >= record_track_nes or (ctl.trace_all_sample >= record_track_detail):
            itmp = f"{i + 1 + rid * nummax:10d}".strip()

            if sp.exit_flag == exit_boundary_max:
                num2 += 1
                if num2 < nummax and ctl.output_track_emri >= 1:
                    if sp.obtype == star_type_ms:
                        output_sample_track_txt(sp, f"{str(fl).strip()}/emri/sg/ms/evl_sg_{itmp}")
                    elif sp.obtype == star_type_bh:
                        output_sample_track_txt(sp, f"{str(fl).strip()}/emri/sg/bh/evl_sg_{itmp}")

            elif sp.exit_flag == exit_plunge_single:
                if sp.obtype == star_type_bh and ctl.output_track_plunge >= 1:
                    output_sample_track_txt(sp, f"{str(fl).strip()}/plunge/bh/evl_sg_{itmp}")

            elif sp.exit_flag in (exit_tidal_empty, exit_tidal_full):
                num += 1
                if num < nummax and ctl.output_track_td >= 1:
                    output_sample_track_txt(sp, f"{str(fl).strip()}/td/evl_sg_{itmp}")

            elif sp.exit_flag == exit_normal:
                pass

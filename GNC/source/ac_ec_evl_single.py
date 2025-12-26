import math
from typing import Any, Optional

pi = math.pi

_run_one_sample_num_out = 0
_run_one_sample_particle_num_out = 0


def run_one_sample(pt: Any, run_time: float) -> None:
    global _run_one_sample_num_out

    sample = pt.ob

    time = sample.simu_bgtime * 1e6 * 2.0 * pi
    total_time = run_time * 1e6 * 2.0 * pi

    if sample.weight_real == 0.0 or sample.weight_n == 0.0:
        print("ac_ec_evl_single:error, sample%weight_real or n=0")
        sample.print("ac_ec_evl single")
        raise RuntimeError("invalid weights")

    reset_sample_init(sample, total_time, time)

    out_flag_boundary = 0

    while True:
        if sample.en > ctl.energy_boundary:
            out_flag_boundary = 0
            run_boundary_state(sample, total_time, time, out_flag_boundary)
            if out_flag_boundary == 100:
                time_create = time
                sample.create_time = time_create / 2e6 / pi

                if ctl.chattery >= 3:
                    print("cross at", time / 2e6 / pi)
                    print("en0, jm0=", sample.en0 / ctl.energy0, sample.jm0)
                    print("en, jm, time_create=", sample.en / ctl.energy0, sample.jm)

                if isinstance(sample, particle_sample_type):
                    run_one_sample_particle_inside_cluster(pt, time, total_time)

                if sample.exit_flag == exit_boundary_min:
                    sample.jm = sample.jm0
                    sample.en = sample.en0
                    sample.byot.a_bin = -mbh / 2.0 / sample.en
                    sample.byot.e_bin = (1.0 - sample.jm**2) ** 0.5
                    time = time_create
                    set_star_radius(sample.byot.ms)
                    get_sample_r_td(sample)
                    if isinstance(sample, particle_sample_type):
                        init_particle_sample_common(sample)
                    continue

                update_samples(sample, pt, time_create / 1e6 / 2.0 / pi, flag_ini_or)

                if ctl.chattery >= 3:
                    print(
                        "create at:",
                        time_create / 1e6 / 2.0 / pi,
                        bksams.head.ed.ob.en / ctl.energy0,
                        sample.en / ctl.energy0,
                    )
            else:
                sample.exit_time = time / 2e6 / pi
        else:
            if isinstance(sample, particle_sample_type):
                run_one_sample_particle_inside_cluster(pt, time, total_time)
            if pt.ob.exit_flag == exit_boundary_min:
                ctl.num_boundary_elim = ctl.num_boundary_elim + 1

        if ctl.chattery >= 1:
            if ctl.chattery == 1:
                if sample.exit_flag != exit_boundary_min:
                    if isinstance(sample, particle_sample_type):
                        print_results_single(pt, pt.idx, pt.ed.idx, sample)
            else:
                if isinstance(sample, particle_sample_type):
                    print_results_single(pt, pt.idx, pt.ed.idx, sample)

            if ctl.chattery >= 4 or ctl.debug >= 1:
                input()

        break


def run_one_sample_particle_inside_cluster(pt: Any, time: float, total_time: float) -> None:
    global _run_one_sample_particle_num_out

    sample = pt.ob

    j = 0
    time_next = time

    sample_mass_idx = get_mass_idx(sample.m)

    while time < total_time:
        update_sample_ej(sample)
        coeNr = get_coeff(sample)
        get_sample_r_td(sample)
        if_sample_within_lc(sample)
        steps = get_step(sample, coeNr, total_time, time)

        en0 = sample.en
        jm0 = sample.jm

        if steps > 1e99:
            print("af get_steps steps=", steps, ieee_is_finite(steps))
            sample.print("sample")
            raise RuntimeError("steps too large")

        period = P(sample.byot.a_bin)
        time_dt = steps * period

        get_de_dj(sample, coeNr, time, time_dt, steps, period)

        sample_enf, sample_jmf, sample_mef, sample_af = get_move_result(
            sample, sample.den, sample.djp, steps
        )

        if ctl.include_loss_cone >= 1 and sample.en < ctl.energy_boundary:
            if sample.within_jt == 1:
                flag_pass_rp = if_sample_pass_rp(sample, steps)
                if flag_pass_rp >= 1:
                    if sample.obtype == star_type_ms:
                        if abs(sample.djp0) < math.sqrt(sample.r_td * mbh * 2.0):
                            sample.exit_flag = exit_tidal_empty
                        else:
                            sample.exit_flag = exit_tidal_full
                        if (
                            ctl.trace_all_sample >= record_track_nes
                            or sample.write_down_track >= record_track_detail
                        ):
                            add_track(time / 1e6 / (2.0 * pi), sample, state_td)
                        break
                    elif sample.obtype in (star_type_bh, star_type_ns, star_type_wd, star_type_bd):
                        sample.exit_flag = exit_plunge_single
                        if (
                            ctl.trace_all_sample >= record_track_nes
                            or sample.write_down_track >= record_track_detail
                        ):
                            add_track(time / 2e6 / pi, sample, state_plunge)
                        break
                    else:
                        print("define star type:", sample.obtype)
                        raise RuntimeError("unknown star type")

        update_track(sample, j)

        if (
            sample.write_down_track >= record_track_detail
            or ctl.trace_all_sample >= record_track_detail
        ):
            add_track(time / 1e6 / (2.0 * pi), sample, state_ae_evl)

        j += 1

        if j > MAX_RUN_LENGTH:
            sample.exit_flag = exit_max_reach
            print("j=", j)
            break

        if j == MAX_RUN_LENGTH // 20 or j == MAX_RUN_LENGTH // 2:
            print("single:warning, j, rid=", j, rid)
            print(
                "step,ao,eo, rp=",
                steps,
                sample.byot.a_bin,
                sample.byot.e_bin,
                sample.byot.a_bin * (1.0 - sample.byot.e_bin),
                sample.r_td,
            )
            print("within_jt=", sample.within_jt)
            print("sample%obtype=", sample.obtype)
            print(time, period)

        if steps > 1e99 or (isinstance(steps, float) and (math.isnan(steps))) or steps < 0.0:
            print("bf get_dedj steps=", steps, ieee_is_finite(steps))
            sample.print("ac_ec_evl_single")
            print("time, rid=", time, rid)
            raise RuntimeError("invalid steps")

        move_de_dj_one(sample, sample_enf, sample_jmf, sample_mef, sample_af)

        en1 = sample.en
        time_next = time + time_dt

        if en1 < ctl.energy_max:
            sample.exit_flag = exit_boundary_max
            boundary_sts_emax_cros = boundary_sts_emax_cros + 1
            if ctl.trace_all_sample >= record_track_nes:
                add_track(time_next / 2e6 / pi, sample, state_emax)
            break

        if ctl.clone_scheme >= 1:
            out_flag_clone = clone_scheme(
                pt,
                en0,
                en1,
                ctl.clone_factor[sample_mass_idx],
                time_next / 1e6 / 2.0 / pi,
            )
            if out_flag_clone == 100:
                sample.exit_flag = exit_invtransit
                break

        if en1 > ctl.energy_boundary:
            sample.exit_flag = exit_boundary_min
            break

        time = time_next

    if ctl.chattery >= 4:
        print("exit time=", time / 1e6 / 2.0 / pi, sample.exit_flag)

    sample.exit_time = time_next / 1e6 / 2.0 / pi

    if ctl.chattery >= 5:
        print("finished, rid, flag, ac=", rid, sample.byot.a_bin)
        print("-------time exit:", sample.exit_time)
        print("-------exit flag:", sample.exit_flag)
        input()


def print_results_single(pt: Any, id: int, eid: int, sample: Any) -> None:
    spid = sample.id
    str_type = get_star_type(sample.obtype)

    def w(s: str) -> None:
        chattery_out_unit.write(s + "\n")

    if sample.exit_flag == exit_normal:
        w(f"{'-------time out':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_tidal_empty:
        w(f"{'-------td empty':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_tidal_full:
        w(f"{'-------td full':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_max_reach:
        w(f"{'-------MORE STEPS NEEDED':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
        w(f"{'--ac,ec,ain,ein=':25s} {sample.byot.a_bin:10.3e} {sample.byot.e_bin:10.3e}")
    elif sample.exit_flag == exit_plunge_single:
        w(f"{'----plunge':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_boundary_min:
        w(f"{'-------exit_to_emin':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_boundary_max:
        w(f"{'-------exit_to_emax':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
        if (isinstance(sample.byot.a_bin, float) and math.isnan(sample.byot.a_bin)) or sample.byot.a_bin == 0.0:
            print("aout,eout=", sample.byot.a_bin, sample.byot.e_bin, sample.id)
            sample.print("print_results_single")
            print("????", rid)
            input()
    elif sample.exit_flag == exit_ejection:
        w(f"{'-------escape':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_other:
        w(f"{'-------other':25s} {str_type:5s} {id:10d} {rid:10d} {eid:10d} {spid:10d} {pt.ed.ob.id:12d}")
    elif sample.exit_flag == exit_invtransit:
        return
    else:
        chattery_out_unit.write(f"define state: {sample.exit_flag}\n")

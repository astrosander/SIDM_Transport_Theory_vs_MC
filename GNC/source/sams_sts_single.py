from com_main_gw import (
    ctl,
    exit_normal,
    exit_plunge_single,
    exit_boundary_max,
    exit_tidal_full,
    exit_tidal_empty,
)
from com_main_gw import sams_arr_select_condition_single, sams_get_eve_num, get_sams_event_rate


def get_sams_events_sbh(
    sa,
    sams_sel_norm,
    sams_sel_norm_inbd,
    sams_emri_by,
    sams_emri_sg,
    sams_emri,
    sams_plunge,
    eveset,
    isnap,
):
    sams_sel = type(sa)()
    get_sams_sts_single(sa, sams_sel, -1, eveset.etot, isnap)
    get_sams_sts_single(sa, sams_sel_norm, exit_normal, eveset.enorm, isnap)

    get_sams_sts_condition_single(
        sams_sel_norm, sams_sel_norm_inbd, eveset.enorm_withinbd, isnap, selection_within_bd
    )

    get_sams_event_rate(eveset.etot, eveset.enorm, eveset.etot, isnap)
    get_sams_event_rate(eveset.enorm, eveset.enorm, eveset.etot, isnap)
    get_sams_event_rate(eveset.egw, eveset.enorm, eveset.etot, isnap)

    get_sams_sts_single(sa, sams_plunge, exit_plunge_single, eveset.egw_plunge, isnap)
    get_sams_event_rate(eveset.egw_plunge, eveset.enorm, eveset.etot, isnap)

    get_sams_sts_single(sa, sams_sel, exit_boundary_max, eveset.eemax, isnap)
    get_sams_event_rate(eveset.eemax, eveset.enorm, eveset.etot, isnap)


def get_sams_events_star(sa, sams_sel_norm, sams_sel_norm_inbd, sams_sel_td, eveset, isnap):
    sams_sel = type(sa)()
    get_sams_sts_single(sa, sams_sel, -1, eveset.etot, isnap)
    get_sams_sts_single(sa, sams_sel_norm, exit_normal, eveset.enorm, isnap)

    get_sams_sts_condition_single(
        sams_sel_norm, sams_sel_norm_inbd, eveset.enorm_withinbd, isnap, selection_within_bd
    )

    get_sams_event_rate(eveset.etot, eveset.enorm, eveset.etot, isnap)
    get_sams_event_rate(eveset.enorm, eveset.enorm, eveset.etot, isnap)

    if ctl.include_loss_cone >= 1:
        get_sams_sts_condition_single(sa, sams_sel_td, eveset.etd, isnap, selection_td)
        get_sams_sts_single(sa, sams_sel, exit_tidal_full, eveset.etdfull, isnap)
        get_sams_sts_single(sa, sams_sel, exit_tidal_empty, eveset.etdempty, isnap)
        get_sams_event_rate(eveset.etd, eveset.enorm, eveset.etot, isnap)
        get_sams_event_rate(eveset.etdfull, eveset.enorm, eveset.etot, isnap)
        get_sams_event_rate(eveset.etdempty, eveset.enorm, eveset.etot, isnap)

    get_sams_sts_single(sa, sams_sel, exit_boundary_max, eveset.eemax, isnap)
    get_sams_event_rate(eveset.eemax, eveset.enorm, eveset.etot, isnap)


def selection_within_bd(sp):
    return sp.en < ctl.energy_boundary


def selection_emri_sgsource(sp):
    return sp.exit_flag == exit_boundary_max


def get_sams_sts_single(sa, sams_sel, flag, eve, isnap):
    sa.select(sams_sel, flag, -1.0, -1.0)
    weights = [sp.weight_real for sp in sams_sel.sp[: sams_sel.n]]
    sams_get_eve_num(weights, sams_sel.n, eve, isnap)


def get_sams_sts_condition_single(sa, sams_sel, eve, isnap, sub_condition):
    sams_arr_select_condition_single(sa, sams_sel, sub_condition)
    weights = [sp.weight_real for sp in sams_sel.sp[: sams_sel.n]]
    sams_get_eve_num(weights, sams_sel.n, eve, isnap)


def get_avgmass_single(sa):
    if sa.n == 0:
        return 0.0
    return sum(sp.m for sp in sa.sp[: sa.n]) / float(sa.n)


def selection_td(sp):
    flag = sp.exit_flag
    return (flag == exit_tidal_full) or (flag == exit_tidal_empty and sp.m > 0.1)

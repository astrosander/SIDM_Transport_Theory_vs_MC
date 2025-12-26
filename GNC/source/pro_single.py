import math
from com_main_gw import (
    ctl,
    rh,
    emin_factor,
    emax_factor,
    aomax,
    aomin,
    bksams_arr,
    pteve_star,
    pteve_sbh,
    pteve_ns,
    pteve_wd,
    pteve_bd,
    star_type_BH,
    star_type_NS,
    star_type_MS,
    star_type_WD,
    star_type_BD,
)
from com_main_gw import (
    sams_get_weight_clone_single,
    set_real_weight_arr_single,
    sams_arr_select_type_single,
    get_sams_events_sbh,
    get_sams_events_star,
    print_events,
)


def pro_single(isnap: int) -> None:
    print("pro star")
    pro_single_star(isnap)

    if ctl.idxsbh != -1:
        print("pro sbh")
        pro_single_compact(isnap, "BH", star_type_BH, pteve_sbh)

    if ctl.idxns != -1:
        print("pro ns")
        pro_single_compact(isnap, "NS", star_type_NS, pteve_ns)

    if ctl.idxwd != -1:
        print("pro wd")
        pro_single_compact(isnap, "WD", star_type_WD, pteve_wd)

    if ctl.idxbd != -1:
        print("pro bd")
        pro_single_compact(isnap, "BD", star_type_BD, pteve_bd)


def pro_single_compact(isnap: int, str_comp: str, comp_type: int, eve_sets) -> None:
    bksma_arr_sel = type(bksams_arr)()
    bksma_arr_sel_norm = type(bksams_arr)()
    bksma_arr_sel_norm_bd = type(bksams_arr)()
    bksma_by_source_emri = type(bksams_arr)()
    bksma_sg_source_emri = type(bksams_arr)()
    bksma_emri = type(bksams_arr)()
    bksma_plunge = type(bksams_arr)()

    sams_get_weight_clone_single(bksams_arr)
    set_real_weight_arr_single(bksams_arr)

    sams_arr_select_type_single(bksams_arr, bksma_arr_sel, comp_type)
    print("arr selection finished")
    if bksma_arr_sel.n == 0:
        return

    get_sams_events_sbh(
        bksma_arr_sel,
        bksma_arr_sel_norm,
        bksma_arr_sel_norm_bd,
        bksma_by_source_emri,
        bksma_sg_source_emri,
        bksma_emri,
        bksma_plunge,
        eve_sets,
        isnap,
    )
    print("get_sams_events_single finshed!")
    print_events(eve_sets.etot, eve_sets.enorm, isnap)
    print("print finished")

    print(f"{'tot':>10}{'tot_sbh':>10}{'norm':>10}{'norm_inbd':>10}")
    print(f"{bksams_arr.n:10d}{bksma_arr_sel.n:10d}{bksma_arr_sel_norm.n:10d}{bksma_arr_sel_norm_bd.n:10d}")


def pro_single_star(isnap: int) -> None:
    bksma_arr_sel = type(bksams_arr)()
    bksma_arr_sel_norm = type(bksams_arr)()
    bksma_arr_sel_norm_bd = type(bksams_arr)()
    bksma_arr_sel_td = type(bksams_arr)()

    sams_get_weight_clone_single(bksams_arr)
    set_real_weight_arr_single(bksams_arr)

    sams_arr_select_type_single(bksams_arr, bksma_arr_sel, star_type_MS)
    print("arr selection finished")

    get_sams_events_star(
        bksma_arr_sel,
        bksma_arr_sel_norm,
        bksma_arr_sel_norm_bd,
        bksma_arr_sel_td,
        pteve_star,
        isnap,
    )
    print("get_sams_events_single finshed!")
    print_events(pteve_star.etot, pteve_star.enorm, isnap)
    print("print finished")

    print(f"{'tot':>10}{'tot_star':>10}{'norm':>10}{'norm_inbd':>10}")
    print(f"{bksams_arr.n:10d}{bksma_arr_sel.n:10d}{bksma_arr_sel_norm.n:10d}{bksma_arr_sel_norm_bd.n:10d}")


def get_sts_one_species_single(sma, sma_arr, pteve, isnap: int) -> None:
    return


def init_pro() -> None:
    global aomax, aomin
    aomax = rh / 2.0 / emin_factor
    aomin = rh / 2.0 / emax_factor

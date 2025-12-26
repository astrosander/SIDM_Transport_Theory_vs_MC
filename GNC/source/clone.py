import math
from typing import Tuple


class Ctl:
    def __init__(self) -> None:
        self.clone_e0: float = 1.0
        self.chattery: int = 0
        self.ntasks: int = 1
        self.del_cross_clone: int = 0


ctl = Ctl()
log10clone_emax: int = 0
clone_emax: float = 0.0
chattery_out_unit = None


def istranst(en0: float, en1: float) -> Tuple[bool, int]:
    lvl = -1
    en0p = math.log10(en0 / ctl.clone_e0)
    en1p = math.log10(en1 / ctl.clone_e0)
    for jk in range(1, log10clone_emax + 1):
        if en0p < jk and en1p > jk:
            return True, jk
    return False, lvl


def invtransit(en0: float, en1: float) -> Tuple[bool, int]:
    lvl = -1
    en0p = math.log10(en0 / ctl.clone_e0)
    en1p = math.log10(en1 / ctl.clone_e0)
    if ctl.chattery >= 4:
        print("en0p, en1p, clone_e0, clone_emax=", en0p, en1p, ctl.clone_e0, clone_emax)
    for jk in range(1, log10clone_emax + 1):
        if en0p > jk and en1p < jk:
            return True, jk
    return False, lvl


def create_init_clone_particle(pt, spen0: float, time: float) -> None:
    nlvl = int(math.log10(spen0 / ctl.clone_e0))
    if nlvl >= 1:
        ca = getattr(pt, "ob", None)
        if ca is None:
            return
        get_mass_idx(ca.m, sample_mass_idx)
        create_clone_particle(pt, nlvl, int((ctl.clone_factor(sample_mass_idx) ** nlvl)), time)


def clone_scheme(pt, en0: float, en1: float, amplifier: int, time: float) -> int:
    out_flag_clone = 0
    ok, lvl = istranst(en0, en1)
    if ok:
        if ctl.chattery >= 4:
            t = time / 1e6 / (2.0 * math.pi)
            if ctl.ntasks > 1:
                print("time,en0,en1=", t, en0, en1)
                print("crossing ", lvl, ", create clone particle", getattr(pt, "idx", -1))
            else:
                print("time,en0,en1=", t, en0, en1)
                print("crossing ", lvl, ", create clone particle", getattr(pt, "idx", -1))
        create_clone_particle(pt, lvl, amplifier, time)
        out_flag_clone = 1
        if ctl.chattery >= 4:
            ed_idx = getattr(getattr(pt, "ed", None), "idx", -1)
            print("-------clone particle--", lvl, getattr(pt, "idx", -1), ed_idx)

    if ctl.del_cross_clone >= 1:
        ok2, lvl2 = invtransit(en0, en1)
        if ok2:
            tmp = rnd(0.0, float(amplifier))
            if tmp > 1.0:
                if ctl.chattery >= 4:
                    if ctl.ntasks > 1:
                        print("en0,en1=", en0, en1)
                        print("Invcrossing, deleting clone particle", getattr(pt, "idx", -1))
                    else:
                        print("en0,en1=", en0, en1)
                        print("Invcrossing, deleting clone particle", getattr(pt, "idx", -1))
                return 100
    return out_flag_clone

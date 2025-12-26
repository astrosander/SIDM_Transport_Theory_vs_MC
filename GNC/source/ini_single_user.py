import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Control:
    alpha_ini: float = 0.0
    x_boundary: float = 1.0


ctl = Control()

rid = 0

pi = math.pi
rh = 1.0
rhmax = 1.0
mbh = 1.0
MBH = mbh
mbh_radius = 1.0
AU_SI = 1.0
my_unit_vel_c = 1.0

exit_normal = 0
state_ae_evl = 0

star_type_ms = 1
star_type_bh = 2
star_type_ns = 3
star_type_wd = 4
star_type_bd = 5

an_in_mode_mean = 0


def fpowerlaw(alpha: float, xmin: float, xmax: float) -> float:
    if xmin <= 0.0 or xmax <= xmin:
        return xmin
    u = np.random.random()
    a = float(alpha)
    if abs(a + 1.0) < 1e-14:
        return xmin * (xmax / xmin) ** u
    p = a + 1.0
    return (u * (xmax ** p - xmin ** p) + xmin ** p) ** (1.0 / p)


def rnd(a: float, b: float) -> float:
    return float(a + (b - a) * np.random.random())


@dataclass
class Particle:
    m: float = 0.0
    obtype: int = 0
    radius: float = 0.0
    id: int = 0


@dataclass
class BinaryOrbit:
    a_bin: float = 0.0
    e_bin: float = 0.0
    mtot: float = 0.0
    Inc: float = 0.0
    Om: float = 0.0
    pe: float = 0.0
    me: float = 0.0
    bname: str = ""
    an_in_mode: int = 0
    ms: Particle = field(default_factory=Particle)
    mm: Particle = field(default_factory=Particle)


@dataclass
class ParticleSampleType:
    m: float = 0.0
    obtype: int = 0
    en: float = 0.0
    en0: float = 0.0
    jm: float = 0.0
    jm0: float = 0.0
    djp: float = 0.0
    elp: float = 0.0
    id: int = 0
    state_flag_last: int = 0
    exit_flag: int = 0
    rid: int = 0
    r_td: float = 0.0
    byot: BinaryOrbit = field(default_factory=BinaryOrbit)
    byot_ini: BinaryOrbit = field(default_factory=BinaryOrbit)
    byot_bf: BinaryOrbit = field(default_factory=BinaryOrbit)


def by_em2st(by: BinaryOrbit) -> None:
    return


def by_split_from_rd(by: BinaryOrbit) -> None:
    return


def track_init(bkps: ParticleSampleType, _: int) -> None:
    return


def star_Radius(m: float) -> float:
    return 0.0


def white_dwarf_radius(m: float) -> float:
    return 0.0


def set_ini_byot_abin() -> float:
    return float(fpowerlaw(ctl.alpha_ini, 0.01 * rh / (ctl.x_boundary * 2.0), rhmax))


def init_particle_sample_common(bkps: ParticleSampleType) -> None:
    bkps.en = -MBH / (2.0 * bkps.byot.a_bin)
    bkps.en0 = bkps.en
    bkps.jm = math.sqrt(1.0 - bkps.byot.e_bin ** 2)
    bkps.jm0 = bkps.jm
    bkps.djp = 0.0
    bkps.elp = 0.0
    bkps.id = int(rnd(0.0, 100000000.0))
    bkps.byot.ms.id = bkps.id
    bkps.state_flag_last = state_ae_evl
    bkps.exit_flag = exit_normal
    bkps.rid = rid
    set_particle_sample_other(bkps)
    bkps.byot_ini = bkps.byot
    track_init(bkps, 0)


def set_particle_sample_other(bkps: ParticleSampleType) -> None:
    bkps.byot.ms.m = bkps.m
    bkps.byot.mm.m = mbh
    bkps.byot.mtot = bkps.m + mbh
    bkps.byot.Inc = rnd(0.0, pi)
    bkps.byot.Om = rnd(0.0, 2.0 * pi)
    bkps.byot.pe = rnd(0.0, 2.0 * pi)
    bkps.byot.bname = "byot"
    bkps.byot_bf.bname = "byot_bf"
    bkps.byot.ms.id = bkps.id
    bkps.byot.mm.obtype = star_type_bh
    bkps.byot.mm.radius = mbh / 1e8
    bkps.byot.me = rnd(0.0, 2.0 * pi)
    bkps.byot.an_in_mode = an_in_mode_mean
    by_em2st(bkps.byot)
    by_split_from_rd(bkps.byot)


def get_sample_r_td_single(sp: ParticleSampleType) -> None:
    if sp.obtype == star_type_ms:
        if sp.byot.ms.radius == 0.0:
            raise RuntimeError("ms radius=0?? check")
        sp.r_td = (3.0 * mbh / sp.m) ** (1.0 / 3.0) * sp.byot.ms.radius
    elif sp.obtype in (star_type_bh, star_type_ns, star_type_wd, star_type_bd):
        sp.r_td = 16.0 * mbh_radius / (1.0 + sp.byot.e_bin)
    else:
        raise RuntimeError(f"error! star type not defined {sp.obtype}")


def set_star_radius(pr: Particle) -> None:
    if pr.obtype == star_type_ms:
        pr.radius = float(star_Radius(pr.m))
    elif pr.obtype == star_type_bd:
        pr.radius = pr.m / (my_unit_vel_c ** 2)
    elif pr.obtype == star_type_bh:
        pr.radius = pr.m / (my_unit_vel_c ** 2)
    elif pr.obtype == star_type_wd:
        pr.radius = float(white_dwarf_radius(pr.m))
    elif pr.obtype == star_type_ns:
        pr.radius = 1.0e4 / AU_SI
    else:
        pr.radius = 0.0
        raise RuntimeError(f"star type={pr.obtype}")

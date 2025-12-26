import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


pi = math.pi

star_type_ms = 1
star_type_bh = 2
star_type_ns = 3
star_type_wd = 4
star_type_bd = 5

flag_ini_or = 1
flag_ini_ini = 2

exit_normal = 0
state_ae_evl = 0

rid = 0
mpi_master_id = 0

mbh = 1.0
MBH = mbh
mbh_radius = 1.0

rh = 1.0
rhmax = 1.0

jmin_value = 0.0
emax_factor = 1.0
log10emin_factor = 0.0

AU_SI = 1.0
my_unit_vel_c = 1.0
rd_sun = 1.0


def rnd(a: float, b: float) -> float:
    return float(a + (b - a) * np.random.random())


def fpowerlaw(alpha: float, xmin: float, xmax: float) -> float:
    if xmin <= 0.0 or xmax <= xmin:
        return xmin
    u = np.random.random()
    a = float(alpha)
    if abs(a + 1.0) < 1e-14:
        return xmin * (xmax / xmin) ** u
    p = a + 1.0
    return (u * (xmax ** p - xmin ** p) + xmin ** p) ** (1.0 / p)


@dataclass
class Control:
    m_bins: int = 0
    idxstar: int = 0
    idxsbh: int = -1
    idxns: int = -1
    idxwd: int = -1
    idxbd: int = -1
    alpha_ini: float = 0.0
    x_boundary: float = 1.0
    clone_scheme: int = 0
    asymptot: Optional[np.ndarray] = None
    bin_mass_particle_number: Optional[np.ndarray] = None
    bin_mass: Optional[np.ndarray] = None
    weight_n: Optional[np.ndarray] = None


ctl = Control()


def get_mass_idx(m: float) -> int:
    if ctl.bin_mass is None or len(ctl.bin_mass) == 0:
        return 0
    arr = np.asarray(ctl.bin_mass, dtype=float)
    return int(np.argmin(np.abs(arr - m)))


def create_init_clone_particle(ptbk, en0: float, time: float) -> None:
    return


@dataclass
class Particle:
    m: float = 0.0
    obtype: int = 0
    obidx: int = 0
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
    obidx: int = 0
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
    rp: float = 0.0
    create_time: float = 0.0
    simu_bgtime: float = 0.0
    weight_real: float = 1.0
    weight_n: float = 0.0
    weight_clone: float = 1.0
    byot: BinaryOrbit = field(default_factory=BinaryOrbit)
    byot_ini: BinaryOrbit = field(default_factory=BinaryOrbit)
    byot_bf: BinaryOrbit = field(default_factory=BinaryOrbit)


@dataclass
class ParticleSamplesArrType:
    n: int = 0
    sp: List[ParticleSampleType] = field(default_factory=list)

    def init(self, n: int) -> None:
        self.n = int(n)
        self.sp = [ParticleSampleType() for _ in range(self.n)]


@dataclass
class ChainPointer:
    ob: Optional[ParticleSampleType] = None
    next: Optional["ChainPointer"] = None


@dataclass
class ChainType:
    n: int = 0
    head: Optional[ChainPointer] = None

    def init(self, n: int) -> None:
        self.n = int(n)
        if self.n <= 0:
            self.head = None
            return
        self.head = ChainPointer(ob=ParticleSampleType())
        p = self.head
        for _ in range(1, self.n):
            p.next = ChainPointer(ob=ParticleSampleType())
            p = p.next


def by_em2st(by: BinaryOrbit) -> None:
    return


def by_split_from_rd(by: BinaryOrbit) -> None:
    return


def track_init(bkps: ParticleSampleType, _: int) -> None:
    return


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
    by_em2st(bkps.byot)
    by_split_from_rd(bkps.byot)


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


def set_ini_byot_abin() -> float:
    return float(fpowerlaw(ctl.alpha_ini, 0.01 * rh / (ctl.x_boundary * 2.0), rhmax))


def star_Radius(mass: float) -> float:
    if mass <= 0.06:
        return 0.1 * rd_sun
    if 0.06 < mass <= 1.0:
        return (mass ** 0.8) * rd_sun
    return (mass ** 0.56) * rd_sun


def white_dwarf_radius(mass: float) -> float:
    if mass < 1.44:
        return 0.01 * rd_sun * (mass ** (-1.0 / 3.0))
    raise RuntimeError("error, white dwarf mass should be smaller than 1.44 solar mass")


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


def get_sample_r_td_single(sp: ParticleSampleType) -> None:
    if sp.obtype == star_type_ms:
        if sp.byot.ms.radius == 0.0:
            raise RuntimeError("ms radius=0?? check")
        sp.r_td = (3.0 * mbh / sp.m) ** (1.0 / 3.0) * sp.byot.ms.radius
    elif sp.obtype in (star_type_bh, star_type_ns, star_type_wd, star_type_bd):
        sp.r_td = 16.0 * mbh_radius / (1.0 + sp.byot.e_bin)
    else:
        raise RuntimeError(f"error! star type not defined {sp.obtype}")


def get_sample_r_td(bkps: ParticleSampleType) -> None:
    get_sample_r_td_single(bkps)


def set_jm_init(bkps: ParticleSampleType) -> None:
    if bkps.byot.e_bin < 0.0:
        bkps.byot.e_bin = 0.0
    if bkps.byot.e_bin >= 1.0:
        bkps.byot.e_bin = 1.0 - 1e-12
    bkps.jm = math.sqrt(1.0 - bkps.byot.e_bin ** 2)


def init_particle_sample_one_model_rnd(bkps: ParticleSampleType, flag: int) -> None:
    if bkps.obtype == 0 or bkps.obidx == 0:
        raise RuntimeError(f"error:particle type not assigned {bkps.obtype} {bkps.obidx}")
    if bkps.m <= 0.0:
        raise RuntimeError(f"error:particle mass should be assigned {bkps.m}")

    if flag == flag_ini_or:
        if bkps.byot.a_bin == 0.0:
            raise RuntimeError("error! flag_ini_or should assume abin first")
        bkps.byot.ms.m = bkps.m
        bkps.byot.ms.obtype = bkps.obtype
        bkps.byot.ms.obidx = bkps.obidx
        set_star_radius(bkps.byot.ms)
        if bkps.byot.ms.radius == 0.0:
            raise RuntimeError(f"ini {bkps.byot.ms.radius}")
        get_sample_r_td(bkps)
        bkps.rp = bkps.byot.a_bin * (1.0 - bkps.byot.e_bin)
        init_particle_sample_common(bkps)
        if bkps.weight_real == 0.0:
            raise RuntimeError(
                f"error in ini run particle: bkps%weight_real=0,id={bkps.id} "
                f"{bkps.weight_real} {bkps.weight_n} {bkps.weight_clone}"
            )

    elif flag == flag_ini_ini:
        bkps.byot.a_bin = set_ini_byot_abin()
        bkps.byot.ms.m = bkps.m
        bkps.byot.ms.obtype = bkps.obtype
        bkps.byot.ms.obidx = bkps.obidx
        set_star_radius(bkps.byot.ms)
        if bkps.byot.ms.radius == 0.0:
            raise RuntimeError(f"ini {bkps.byot.ms.radius}")
        get_sample_r_td(bkps)
        set_jm_init(bkps)
        bkps.rp = bkps.byot.a_bin * (1.0 - bkps.byot.e_bin)
        init_particle_sample_common(bkps)
        midx = get_mass_idx(bkps.m)
        if ctl.weight_n is not None and len(ctl.weight_n) > 0:
            bkps.weight_n = float(ctl.weight_n[midx])
        else:
            bkps.weight_n = 0.0
    else:
        raise RuntimeError("undefined flag in init_particle_sample_one_model_rnd")


def init_particle_sample_one(sample: ParticleSampleType, m: float, flag: int) -> None:
    sample.m = float(m)
    init_particle_sample_one_model_rnd(sample, flag)


def get_numbers_each_bin(n: int):
    nstar = np.zeros(n, dtype=int)
    nsbh = np.zeros(n, dtype=int)
    nns = np.zeros(n, dtype=int)
    nwd = np.zeros(n, dtype=int)
    nbd = np.zeros(n, dtype=int)

    for i in range(ctl.m_bins):
        nstar[i] = int(ctl.asymptot[1, i] * ctl.bin_mass_particle_number[i])
        nsbh[i] = int(ctl.asymptot[2, i] * ctl.bin_mass_particle_number[i]) if ctl.idxsbh != -1 else 0
        nns[i] = int(ctl.asymptot[3, i] * ctl.bin_mass_particle_number[i]) if ctl.idxns != -1 else 0
        nwd[i] = int(ctl.asymptot[4, i] * ctl.bin_mass_particle_number[i]) if ctl.idxwd != -1 else 0
        nbd[i] = int(ctl.asymptot[5, i] * ctl.bin_mass_particle_number[i]) if ctl.idxbd != -1 else 0

    nstar_tot = int(nstar.sum())
    nsbh_tot = int(nsbh.sum()) if ctl.idxsbh != -1 else 0
    nwd_tot = int(nwd.sum()) if ctl.idxwd != -1 else 0
    nns_tot = int(nns.sum()) if ctl.idxns != -1 else 0
    nbd_tot = int(nbd.sum()) if ctl.idxbd != -1 else 0

    return nstar_tot, nsbh_tot, nns_tot, nwd_tot, nbd_tot, nstar, nsbh, nns, nwd, nbd


def get_init_samples_given(bksps_arr_ini: ParticleSamplesArrType) -> None:
    (nstar_tot, nsbh_tot, nns_tot, nwd_tot, nbd_tot,
     nstar, nsbh, nns, nwd, nbd) = get_numbers_each_bin(ctl.m_bins)

    total = nstar_tot + nsbh_tot + nwd_tot + nns_tot + nbd_tot
    bksps_arr_ini.init(total)

    nsg0 = 0
    for i in range(ctl.m_bins):
        for j in range(int(nstar[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_ms
            sp.obidx = int(ctl.idxstar)
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nstar[i])

        for j in range(int(nsbh[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_bh
            sp.obidx = int(ctl.idxsbh)
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nsbh[i])

        for j in range(int(nwd[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_wd
            sp.obidx = int(ctl.idxwd)
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nwd[i])

        for j in range(int(nns[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_ns
            sp.obidx = int(ctl.idxns)
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nns[i])

        for j in range(int(nbd[i])):
            sp = bksps_arr_ini.sp[j + nsg0]
            sp.obtype = star_type_bd
            sp.obidx = int(ctl.idxbd)
            sp.m = float(ctl.bin_mass[i])
            init_particle_sample_one_model_rnd(sp, flag_ini_ini)
        nsg0 += int(nbd[i])


def get_init_samples(bksps_arr_ini: ParticleSamplesArrType) -> None:
    get_init_samples_given(bksps_arr_ini)


def set_chain_samples(cbk: ChainType, bksps_arr: ParticleSamplesArrType) -> None:
    set_chain_samples_single(cbk, bksps_arr)


def set_chain_samples_single(cbk: ChainType, bksps_arr: ParticleSamplesArrType) -> None:
    cbk.init(bksps_arr.n)
    ptbk = cbk.head
    for i in range(bksps_arr.n):
        if ptbk is None:
            break
        ca = bksps_arr.sp[i]
        ca.create_time = 0.0
        ca.simu_bgtime = 0.0
        ca.en0 = -mbh / (2.0 * ca.byot.a_bin)
        ca.jm0 = math.sqrt(1.0 - ca.byot.e_bin ** 2)
        if ca.jm0 < jmin_value:
            raise RuntimeError(f"jm0={ca.jm0}")
        if ctl.clone_scheme >= 1:
            create_init_clone_particle(ptbk, ca.en0, 0.0)
        ptbk.ob = ca
        ptbk = ptbk.next

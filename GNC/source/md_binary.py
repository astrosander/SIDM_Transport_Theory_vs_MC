from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any


@dataclass
class Particle:
    x: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    vx: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    M: float = 0.0
    radius: float = 0.0
    id: int = 0
    obtype: int = 0
    obidx: int = 0


def get_distance(p1: Particle, p2: Particle) -> float:
    return math.sqrt(
        (p1.x[0] - p2.x[0]) ** 2
        + (p1.x[1] - p2.x[1]) ** 2
        + (p1.x[2] - p2.x[2]) ** 2
    )


def get_velmag(p: Particle) -> float:
    return math.sqrt(p.vx[0] ** 2 + p.vx[1] ** 2 + p.vx[2] ** 2)


def get_rmag(p: Particle) -> float:
    return math.sqrt(p.x[0] ** 2 + p.x[1] ** 2 + p.x[2] ** 2)


an_in_mode_f0 = 1
an_in_mode_t0 = 2
an_in_mode_mean = 3

nint_by = 10
nreal_by = 17 + 24
nstr_by = 100


@dataclass
class Binary:
    Ms: Particle = field(default_factory=Particle)
    Mm: Particle = field(default_factory=Particle)
    rd: Particle = field(default_factory=Particle)
    E: float = 0.0
    l: float = 0.0
    k: float = 0.0
    miu: float = 0.0
    Mtot: float = 0.0
    Jc: float = 0.0
    a_bin: float = 0.0
    e_bin: float = 0.0
    lum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    f0: float = 0.0
    Inc: float = 0.0
    Om: float = 0.0
    pe: float = 0.0
    t0: float = 0.0
    me: float = 0.0
    bname: str = ""
    an_in_mode: int = 0


def by_st2em(by: Binary) -> int:
    f_return = 0

    gm = by.Ms.M + by.Mm.M
    x, y, z = by.rd.x
    vx, vy, vz = by.rd.vx

    q, e, inc, p, n, l = mco_x2el(gm, x, y, z, vx, vy, vz)

    if e > 1.0e-6:
        tau = get_true_anomaly(gm, x, y, z, vx, vy, vz)
    else:
        tau = l

    om = n
    pe = p - n
    lm = l

    by.a_bin = q / (1.0 - e)
    by.e_bin = e
    by.Inc = inc
    by.Om = om
    by.pe = pe
    by.f0 = tau
    by.me = lm

    if math.isnan(tau):
        print("st2em:nan!")
        print_binary(by)
        f_return = 1

    return f_return


def get_center_particle_p(m1: Particle, m2: Particle, mc: Particle) -> None:
    mtot = m1.M + m2.M
    mc.x = [(m1.x[i] * m1.M + m2.x[i] * m2.M) / mtot for i in range(3)]
    mc.vx = [(m1.vx[i] * m1.M + m2.vx[i] * m2.M) / mtot for i in range(3)]
    mc.M = mtot


def get_center_particle(by: Binary) -> Particle:
    gcp = Particle()
    by.Mtot = by.Ms.M + by.Mm.M
    gcp.x = [(by.Ms.x[i] * by.Ms.M + by.Mm.x[i] * by.Mm.M) / by.Mtot for i in range(3)]
    gcp.vx = [(by.Ms.vx[i] * by.Ms.M + by.Mm.vx[i] * by.Mm.M) / by.Mtot for i in range(3)]
    gcp.M = by.Mtot
    return gcp


def by_move_to_mass_center(by: Binary) -> None:
    c = get_center_particle(by)
    by.Ms.x = [by.Ms.x[i] - c.x[i] for i in range(3)]
    by.Ms.vx = [by.Ms.vx[i] - c.vx[i] for i in range(3)]
    by.Mm.x = [by.Mm.x[i] - c.x[i] for i in range(3)]
    by.Mm.vx = [by.Mm.vx[i] - c.vx[i] for i in range(3)]


def by_get_el(by: Binary, kind_in: Optional[int] = None) -> None:
    kind = 0 if kind_in is None else kind_in
    if kind == 0:
        by.k = by.Ms.M * by.Mm.M
        by.Mtot = by.Ms.M + by.Mm.M
        by.miu = by.k / by.Mtot
        by.E = -by.k / 2.0 / by.a_bin
        by.l = math.sqrt(by.miu * by.k * by.a_bin * (1.0 - by.e_bin ** 2))
    elif kind == 1:
        by.Mtot = by.Ms.M + by.Mm.M
        by.E = -by.Mtot / 2.0 / by.a_bin
        by.l = math.sqrt(by.Mtot * by.a_bin * (1.0 - by.e_bin ** 2))


def by_get_lm(by: Binary) -> None:
    get_l_m = [0.0, 0.0, 0.0]
    get_l_m[0] = math.sin(by.Inc) * math.sin(by.Om)
    get_l_m[1] = -math.sin(by.Inc) * math.cos(by.Om)
    get_l_m[2] = math.cos(by.Inc)
    by.lum = get_l_m


def by_get_energy_lum(by: Binary) -> None:
    ms = by.Ms
    mm = by.Mm
    bin_em = Particle(
        x=[ms.x[i] - mm.x[i] for i in range(3)],
        vx=[ms.vx[i] - mm.vx[i] for i in range(3)],
    )
    by.E = -by.k / get_rmag(bin_em) + by.miu * (get_velmag(bin_em) ** 2) / 2.0
    by_get_lm(by)
    l_m = by.lum
    by.l = by.miu * math.sqrt(l_m[0] ** 2 + l_m[1] ** 2 + l_m[2] ** 2)


def by_get_Jc(by: Binary) -> None:
    by.Jc = math.sqrt((by.Ms.M + by.Mm.M) * by.a_bin * (1.0 - by.e_bin ** 2))


def by_get_period(by: Binary) -> float:
    return 2.0 * PI * by.a_bin * math.sqrt(by.a_bin / by.Mtot)


def by_em2st(by: Binary) -> None:
    gm = by.Ms.M + by.Mm.M
    e = by.e_bin
    q = by.a_bin * (1.0 - e)
    inc = by.Inc
    n = by.Om
    p = by.pe + n

    if by.an_in_mode == an_in_mode_t0:
        l = -math.sqrt(by.Mtot / (by.a_bin ** 3)) * by.t0
    elif by.an_in_mode == an_in_mode_mean:
        l = by.me
        by.t0 = -l / math.sqrt(by.Mtot / (by.a_bin ** 3))
    elif by.an_in_mode == an_in_mode_f0:
        if by.e_bin < 1.0:
            ecc_ano = math.atan(math.tan(by.f0 / 2.0) * math.sqrt(1.0 - by.e_bin) / math.sqrt(1.0 + by.e_bin)) * 2.0
            if by.f0 > PI - 30.0 / 180.0 * PI and by.f0 <= PI:
                ecc_ano = math.acos((math.cos(by.f0) + by.e_bin) / (1.0 + by.e_bin * math.cos(by.f0)))
            elif by.f0 < PI + 30.0 / 180.0 * PI and by.f0 > PI:
                ecc_ano = -math.acos((math.cos(by.f0) + by.e_bin) / (1.0 + by.e_bin * math.cos(by.f0)))
            if by.f0 == PI:
                ecc_ano = PI
            l = ecc_ano - by.e_bin * math.sin(ecc_ano)
        elif by.e_bin > 1.0:
            ecc_ano = math.atanh(math.tan(by.f0 / 2.0) * math.sqrt(by.e_bin - 1.0) / math.sqrt(1.0 + by.e_bin)) * 2.0
            if by.f0 > PI - 30.0 / 180.0 * PI and by.f0 < PI:
                ecc_ano = math.acosh((math.cos(by.f0) + by.e_bin) / (1.0 + by.e_bin * math.cos(by.f0)))
            elif by.f0 < PI + 30.0 / 180.0 * PI and by.f0 > PI:
                ecc_ano = -math.acosh((math.cos(by.f0) + by.e_bin) / (1.0 + by.e_bin * math.cos(by.f0)))
            if by.f0 == PI:
                ecc_ano = PI
            l = by.e_bin * math.sinh(ecc_ano) - ecc_ano
        else:
            ecc_ano = math.tan(by.f0 / 2.0)
            l = ecc_ano + (ecc_ano ** 3) / 3.0
        by.t0 = -l / math.sqrt(by.Mtot / (by.a_bin ** 3))
    else:
        raise RuntimeError("ERROR: an_in_mode not defined!")

    by.me = l

    x, y, z, u, v, w = mco_el2x(gm, q, e, inc, p, n, l)

    if by.an_in_mode == an_in_mode_t0:
        if by.e_bin > 1.0e-6:
            by.f0 = get_true_anomaly(gm, x, y, z, u, v, w)
        else:
            by.f0 = l

    by.rd.x = [x, y, z]
    by.rd.vx = [u, v, w]
    by.rd.M = by.Ms.M * by.Mm.M / gm

    if math.isnan(by.f0) and by.an_in_mode == an_in_mode_f0:
        print("em2st:nan!")
        print_binary(by)


def by_split_from_rd(by: Binary) -> None:
    mtot = by.Ms.M + by.Mm.M
    by.Ms.x = [by.rd.x[i] * by.Mm.M / mtot for i in range(3)]
    by.Mm.x = [-by.rd.x[i] * by.Ms.M / mtot for i in range(3)]
    by.Ms.vx = [by.rd.vx[i] * by.Mm.M / mtot for i in range(3)]
    by.Mm.vx = [-by.rd.vx[i] * by.Ms.M / mtot for i in range(3)]


def by_get_rd(by: Binary) -> None:
    by.rd.x = [by.Ms.x[i] - by.Mm.x[i] for i in range(3)]
    by.rd.vx = [by.Ms.vx[i] - by.Mm.vx[i] for i in range(3)]


def vector_mag(v: list[float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def by_get_rdmag(by: Binary) -> float:
    return vector_mag(by.rd.x)


def print_binary(by: Binary, funit: Optional[Any] = None) -> None:
    out = funit if funit is not None else None

    def w(s: str) -> None:
        if out is None:
            print(s)
        else:
            out.write(s + "\n")

    str_ms_obtype = get_star_type(by.Ms.obtype)
    str_mm_obtype = get_star_type(by.Mm.obtype)

    w(f"{'NAME=':20s}{by.bname.strip():20s}")
    w(f"{'ANMODE=':20s}{by.an_in_mode:4d}")
    w(f"{'MS,MM=':20s}{by.Ms.M:20.8E}{by.Mm.M:20.8E}")
    w(f"{'TYPE_MS,TYPE_MM=':20s}{str_ms_obtype:20s}{str_mm_obtype:20s}")
    w(f"{'Sma,Ecc, Inc=':20s}{by.a_bin:20.8E}{by.e_bin:20.8E}{by.Inc:20.8E}")
    w(f"{'E,L=':20s}{by.E:20.8E}{by.l:20.8E}")
    if by.a_bin < 0.0:
        w(f"{'Vinf(kms)=':20s}{math.sqrt(-(by.Ms.M + by.Mm.M) / by.a_bin) * 29.79:20.8E}")
    w(f"{'Ome,Pe,Me=':20s}{by.Om:20.10f}{by.pe:20.10f}{by.me:20.10f}")
    w(f"{'f=':20s}{by.f0:20.10f}")
    w(f"{'RD X=':20s}{by.rd.x[0]:20.10f}{by.rd.x[1]:20.10f}{by.rd.x[2]:20.10f}")
    w(f"{'RD MAG(X)=':20s}{vector_mag(by.rd.x):20.10f}")
    w(f"{'RD VX=':20s}{by.rd.vx[0]:20.10f}{by.rd.vx[1]:20.10f}{by.rd.vx[2]:20.10f}")
    w(f"{'MS X=':20s}{by.Ms.x[0]:20.10f}{by.Ms.x[1]:20.10f}{by.Ms.x[2]:20.10f}")
    w(f"{'MS VX=':20s}{by.Ms.vx[0]:20.10f}{by.Ms.vx[1]:20.10f}{by.Ms.vx[2]:20.10f}")
    w(f"{'MM X=':20s}{by.Mm.x[0]:20.10f}{by.Mm.x[1]:20.10f}{by.Mm.x[2]:20.10f}")
    w(f"{'MM VX=':20s}{by.Mm.vx[0]:20.10f}{by.Mm.vx[1]:20.10f}{by.Mm.vx[2]:20.10f}")


@dataclass
class Triple:
    pass


def split_by3p(byin: Binary, byot: Binary, p1: Particle, p2: Particle, pc: Particle) -> None:
    p1.x = [byin.Ms.x[i] + byot.Ms.x[i] for i in range(3)]
    p2.x = [byin.Mm.x[i] + byot.Ms.x[i] for i in range(3)]
    pc.x = [byot.Mm.x[i] for i in range(3)]

    p1.vx = [byin.Ms.vx[i] + byot.Ms.vx[i] for i in range(3)]
    p2.vx = [byin.Mm.vx[i] + byot.Ms.vx[i] for i in range(3)]
    pc.vx = [byot.Mm.vx[i] for i in range(3)]

    p1.M = byin.Ms.M
    p2.M = byin.Mm.M
    pc.M = byot.Mm.M


def get_triple_energy(p1: Particle, p2: Particle, p3: Particle) -> float:
    r12 = get_distance(p1, p2)
    r23 = get_distance(p2, p3)
    r31 = get_distance(p3, p1)
    v1 = get_velmag(p1)
    v2 = get_velmag(p2)
    v3 = get_velmag(p3)
    return (
        0.5 * p1.M * (v1 ** 2)
        + 0.5 * p2.M * (v2 ** 2)
        + 0.5 * p3.M * (v3 ** 2)
        - p1.M * p2.M / r12
        - p1.M * p3.M / r31
        - p2.M * p3.M / r23
    )

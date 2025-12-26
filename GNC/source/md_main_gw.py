from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import math

PI = math.pi

evaluate_dej_method_rk = 1
evaluate_dej_method_grid = 2
evaluate_dej_method_cfs = 3

boundary_fj_iso = 1
boundary_fj_ls = 2

flag_ini_or = 1
flag_ini_ini = 2

exit_normal = 0

record_track_nes = 1
record_track_all = 2

sts_type_grid = 1

star_type_ms = 1
star_type_bh = 2
star_type_ns = 3
star_type_wd = 4
star_type_bd = 5

chattery_out_unit_0 = 1000

pc = 1.0
AU_SI = 1.0

jmin_value = 1e-6
jmax_value = 0.9999999999

emin_factor = 1.0
emax_factor = 1.0
clone_e0_factor = 1.0
log10emax_factor = 0.0
log10emin_factor = 0.0

emin_value = 0.0
emax_value = 0.0

mbh = 0.0
mbh_radius = 0.0
rh = 0.0
rhmin = 0.0
rhmax = 0.0
log10rh = 0.0

rid = 0
chattery_out_unit = 6

clone_emax = 0.0
log10clone_emax = 0.0

sample_jc = 0.0
sample_jlc = 0.0
sample_jm = 0.0
sample_jlc_dimless = 0.0
sample_enf = 0.0
sample_jmf = 0.0
sample_mef = 0.0
sample_af = 0.0
sample_mass_idx = 0

DEJ_GRID_FILE_DIR = ""

ge_profile: list[Any] = []


def isnan(x: float) -> bool:
    return math.isnan(x)


def ieee_is_finite(x: float) -> bool:
    return math.isfinite(x)


def P(a: float) -> float:
    return 2.0 * PI


def fpowerlaw(alpha: float, xmin: float, xmax: float) -> float:
    return xmin


def rnd(a: float, b: float) -> float:
    return a + (b - a) * 0.5


def gen_gaussian(sigma: float) -> float:
    return 0.0


def gen_gaussian_correlate(r1_out: Any, r2_out: Any, rho: float) -> tuple[float, float]:
    return (0.0, 0.0)


def particle_sample_get_weight_clone(en: float, clone_scheme: int, amplifier: int, clone_e0: float) -> float:
    return 1.0


def set_simu_time() -> None:
    return


def init_pro() -> None:
    return


def get_ccidx_from_type(out_idx: Any, obtype: int) -> None:
    return


def get_coenr(even: float, evjum: float, mass: float, en: float, jc: float, coenr: "CoeffType") -> tuple[int, int]:
    return (0, 0)


@dataclass
class CoeffType:
    ee: float = 0.0
    e: float = 0.0
    jj: float = 0.0
    j: float = 0.0
    ej: float = 0.0


@dataclass
class TrackType:
    time: float = 0.0
    ac: float = 0.0
    ec: float = 0.0
    incout: float = 0.0
    omout: float = 0.0
    state_flag: int = 0


@dataclass
class Binary:
    a_bin: float = 0.0
    e_bin: float = 0.0
    inc: float = 0.0
    om: float = 0.0
    pe: float = 0.0
    me: float = 0.0


@dataclass
class ParticleSampleType:
    id: int = 0
    obtype: int = 0
    obidx: int = 0
    m: float = 0.0
    en: float = 0.0
    jm: float = 0.0
    en0: float = 0.0
    jm0: float = 0.0
    rp: float = 0.0
    r_td: float = 0.0
    within_jt: int = 0
    den: float = 0.0
    djp: float = 0.0
    djp0: float = 0.0
    weight_real: float = 0.0
    weight_n: float = 0.0
    weight_N: float = 0.0
    weight_clone: float = 1.0
    exit_flag: int = exit_normal
    exit_time: float = 0.0
    write_down_track: int = 0
    track_step: int = 1
    length: int = 0
    length_to_expand: int = 0
    track: list[TrackType] = field(default_factory=list)
    byot: Binary = field(default_factory=Binary)
    byot_bf: Binary = field(default_factory=Binary)

    def init(self) -> None:
        return


@dataclass
class ChainPointerType:
    idx: int = 0
    next: Optional["ChainPointerType"] = None
    ed: Optional["ChainPointerType"] = None
    ob: Optional[ParticleSampleType] = None

    def create_chain(self, n: int) -> None:
        cur = self
        while cur.next is not None:
            cur = cur.next
        for _ in range(n):
            node = ChainPointerType()
            cur.next = node
            cur = node

    def copy(self, dst: "ChainPointerType") -> None:
        if self.ob is None:
            return
        dst.ob = ParticleSampleType(**self.ob.__dict__)


@dataclass
class ChainType:
    head: ChainPointerType = field(default_factory=ChainPointerType)

    def init(self, n: int) -> None:
        self.head = ChainPointerType()
        self.head.ed = self.head
        self.head.create_chain(n)

    def destory(self) -> None:
        self.head = ChainPointerType()

    def get_length(self, type: int = 0) -> int:
        n = 0
        p = self.head
        while p is not None:
            if type == 0:
                n += 1
            elif type == 1 and isinstance(p.ob, ParticleSampleType):
                n += 1
            p = p.next
        return n


@dataclass
class CoreCompType:
    alpha_ini: float = 0.0
    n_in_rh: float = 0.0
    alpha: float = 0.0
    blkmass: float = 0.0
    n0: float = 0.0
    mc: float = 0.0


@dataclass
class CtlType:
    grid_type: int = 0
    sigma: float = 0.0
    v0: float = 0.0
    n0: float = 0.0
    energy0: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 0.0
    energy_boundary: float = 0.0
    x_boundary: float = 0.0
    rbd: float = 0.0
    clone_scheme: int = 0
    clone_e0: float = 0.0
    bd_thickness: float = 0.0
    num_bk_comp: int = 0
    m_bins: int = 0
    asymptot: Any = None
    ini_weight_n: Any = None
    bin_mass_particle_number: Any = None
    weight_n: Any = None
    n_basic: float = 1.0
    include_loss_cone: int = 0
    boundary_fj: int = boundary_fj_iso
    chattery: int = 0
    trace_all_sample: int = 0
    ntasks: int = 1
    grid_bins: int = 0
    nblock_size: int = 0
    nblock_mpi_bg: int = 0
    nblock_mpi_ed: int = 0
    num_boundary_created: int = 0
    num_boundary_elim: int = 0
    num_clone_created: int = 0
    idxstar: int = -1
    idxsbh: int = -1
    idxns: int = -1
    idxwd: int = -1
    idxbd: int = -1
    cc: list[CoreCompType] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cc is None:
            self.cc = []


ctl = CtlType()
bksams = ChainType()


@dataclass
class DmsType:
    weight_asym: float = 1.0


dms = DmsType()


def init_particle_sample_one(sample: ParticleSampleType, m: float, flag: int) -> None:
    return


def output_sg_sample_track_txt(sp: ParticleSampleType, fl: str) -> None:
    return


def update_sample_ej(sample: ParticleSampleType) -> None:
    global sample_jc, sample_jm, sample_jlc_dimless, sample_jlc
    if (abs(sample.en + mbh / (2.0 * sample.byot.a_bin)) > 1e-5 or
        abs(sample.jm - math.sqrt(max(0.0, 1.0 - sample.byot.e_bin**2))) > 1e-5):
        raise RuntimeError("warning: sample en/jm inconsistent with orbital elements")
    sample.en = -mbh / (2.0 * sample.byot.a_bin)
    sample.jm = math.sqrt(max(0.0, 1.0 - sample.byot.e_bin**2))
    sample_jc = math.sqrt(mbh * sample.byot.a_bin)
    sample_jm = sample.jm * sample_jc
    if ctl.include_loss_cone >= 1:
        sample.rp = sample.byot.a_bin * (1.0 - sample.byot.e_bin)
        ratio = min(2.0, sample.r_td / sample.byot.a_bin)
        sample_jlc_dimless = math.sqrt(ratio) * math.sqrt(2.0 - ratio)
        sample_jlc = sample_jlc_dimless * sample_jc


def get_coeff(sample: ParticleSampleType, coenr: CoeffType) -> None:
    evjum = sample.jm
    if sample.jm < jmin_value:
        sample.jm = 2.0 * jmin_value - sample.jm
        sample.byot.e_bin = math.sqrt(max(0.0, 1.0 - sample.jm**2))
        evjum = jmin_value
    if sample.jm > jmax_value:
        sample.jm = jmax_value
        sample.byot.e_bin = math.sqrt(max(0.0, 1.0 - sample.jm**2))
        evjum = jmax_value
    even = math.log10(sample.en / ctl.energy0)
    if even > log10emax_factor:
        even = log10emax_factor
    if even < log10emin_factor:
        even = log10emin_factor
    idx, idy = get_coenr(even, evjum, sample.m, sample.en, sample_jc, coenr)
    _ = (idx, idy)


def create_clone_particle(pt: ChainPointerType, lvl: int, amplifier: int, time: float) -> None:
    pe = pt.ed if pt.ed is not None else pt
    pe.create_chain(amplifier - 1)
    ctl.num_clone_created += 1
    ps = pe.next
    for _ in range(amplifier - 1):
        pt.copy(ps)
        if ps.ob is None:
            ps = ps.next
            continue
        ps.ob.create_time = time
        ps.ob.simu_bgtime = time
        ps.ob.weight_clone = particle_sample_get_weight_clone(ps.ob.en, ctl.clone_scheme, amplifier, ctl.clone_e0)
        ps = ps.next


def reset_sample_init(sample: ParticleSampleType, total_time: float, time: float) -> None:
    sample.exit_flag = exit_normal


def create_one_to_chain(
    pt_chain: ChainPointerType,
    sample: ParticleSampleType,
    time: float,
    obj_type: int,
    obtype: int,
    typeidx: int,
    flag: int,
) -> None:
    pt = pt_chain.ed if pt_chain.ed is not None else pt_chain
    pt.create_chain(1)
    ctl.num_boundary_created += 1
    pt = pt.next
    if obj_type == 1:
        pt.ob = ParticleSampleType()
    pt.ob.init()
    pt.ob.obtype = obtype
    pt.ob.obidx = typeidx
    pt.ob.weight_real = sample.weight_real
    if flag == flag_ini_or:
        pt.ob.en = sample.en0
        pt.ob.jm = sample.jm0
        pt.ob.byot.e_bin = math.sqrt(max(0.0, 1.0 - pt.ob.jm**2))
        pt.ob.byot.a_bin = -mbh / (2.0 * pt.ob.en)
    if obj_type == 1:
        init_particle_sample_one(pt.ob, sample.m, flag)
    pt.ob.create_time = time
    pt.ob.simu_bgtime = time
    pt.ob.weight_N = sample.weight_N
    if ctl.clone_scheme >= 1:
        create_init_clone_particle(pt, pt.ob.en0, time)


def create_init_clone_particle(pt: ChainPointerType, en0: float, time: float) -> None:
    return


def if_sample_within_lc(sample: ParticleSampleType) -> None:
    sample.rp = sample.byot.a_bin * (1.0 - sample.byot.e_bin)
    sample.within_jt = 1 if sample.rp < sample.r_td else 0


def if_sample_pass_rp(sample: ParticleSampleType, steps: float) -> int:
    npi = sample.byot.me
    npf = npi + steps * PI * 2.0
    ipdi = npi / (2.0 * PI) - int(npi / (2.0 * PI))
    ipdf = npf / (2.0 * PI) - int(npf / (2.0 * PI))
    return 1 if ((ipdi < 0.5 and ipdf > 0.5) or steps >= 1.0) else 0


def get_steps_nr_EJ(en: float, jm: float, coenr: CoeffType, jc: float) -> float:
    if coenr.ee != 0.0:
        time_dt_e = min((en * 0.15) ** 2 / coenr.ee, abs(en * 0.15) / abs(coenr.e) if coenr.e != 0.0 else 1e6)
    else:
        time_dt_e = 1e6
    if coenr.jj != 0.0:
        time_dt_j = min(
            (jc * 0.1) ** 2 / coenr.jj,
            (0.4 * (1.0075 - jm) * jc) ** 2 / coenr.jj,
        )
    else:
        time_dt_j = 1e6
    return min(time_dt_e, time_dt_j)


def get_steps_nr_xj(en: float, jm: float, coenr: CoeffType) -> float:
    if coenr.ee != 0.0:
        enev = en / ctl.energy0
        time_dt_e = min((enev * 0.15) ** 2 / coenr.ee, abs(enev * 0.15) / abs(coenr.e) if coenr.e != 0.0 else 1e6)
    else:
        time_dt_e = 1e6
    if coenr.jj != 0.0:
        time_dt_j = min(
            0.1**2 / coenr.jj,
            (0.4 * (1.0075 - jm)) ** 2 / coenr.jj,
            (0.25 * abs(jm)) ** 2 / coenr.jj,
        )
    else:
        time_dt_j = 1e6
    return min(time_dt_e, time_dt_j)


def get_sample_r_td(sp: ParticleSampleType) -> None:
    return


def set_jm_init(bkps: ParticleSampleType) -> None:
    if ctl.boundary_fj == boundary_fj_iso:
        bkps.jm = fpowerlaw(1.0, 0.0044, 0.99999)
        bkps.byot.e_bin = math.sqrt(max(0.0, 1.0 - bkps.jm**2))
    elif ctl.boundary_fj == boundary_fj_ls:
        while True:
            bkps.jm = fpowerlaw(1.0, 0.01, 0.99999)
            if bkps.obtype == star_type_ms:
                rtd = (3.0 * mbh / bkps.m) ** (1.0 / 3.0) * 0.0
                jlc = math.sqrt(max(0.0, 1.0 - (1.0 - rtd / ctl.rbd) ** 2))
            elif bkps.obtype in (star_type_bh, star_type_ns, star_type_wd, star_type_bd):
                jlc = 4.0 * math.sqrt(mbh_radius / bkps.byot.a_bin) if bkps.byot.a_bin != 0.0 else jmin_value
            else:
                raise RuntimeError("star type not defined")
            tmp = rnd(0.0, 1.0)
            if bkps.jm < jlc:
                continue
            if tmp > math.log(bkps.jm / jlc) / math.log(1.0 / jlc):
                continue
            bkps.byot.e_bin = math.sqrt(max(0.0, 1.0 - bkps.jm**2))
            break
    else:
        raise RuntimeError("define flag INI")


def update_samples(sample: ParticleSampleType, pt: ChainPointerType, time: float, flag_bd: int) -> None:
    obtype = sample.obtype
    typeidx = sample.obidx
    create_one_to_chain(bksams.head.ed if bksams.head.ed else bksams.head, sample, time, 1, obtype, typeidx, flag_bd)


def get_type_idx(sample: ParticleSampleType) -> int:
    if sample.obtype == star_type_ms:
        return 1
    if sample.obtype == star_type_bh:
        return 2
    if sample.obtype == star_type_ns:
        return 3
    if sample.obtype == star_type_wd:
        return 4
    if sample.obtype == star_type_bd:
        return 5
    raise RuntimeError("error in get_type_idx")


MAX_LENGTH = 10_000_000
track_length_expand_block = 1000


def update_track(sp: ParticleSampleType, j: int) -> None:
    if sp.write_down_track >= record_track_nes or ctl.trace_all_sample >= record_track_all:
        if sp.track_step != 0 and (j % sp.track_step) == 0:
            if sp.length_to_expand > MAX_LENGTH:
                sp.track_step *= 10
                track_compress(sp, 10)


def track_compress(sp: ParticleSampleType, ns: int) -> None:
    tk = sp.track[:sp.length]
    out = []
    for i in range(0, sp.length, ns):
        out.append(tk[i])
    sp.track[:len(out)] = out
    sp.length = len(out)


def get_de_dj_nr(coenr: CoeffType, dt: float, steps: float) -> tuple[float, float, float]:
    rho = coenr.ej / math.sqrt(abs(coenr.ee * coenr.jj)) if (coenr.ee * coenr.jj) != 0.0 else 0.0
    y1, y2 = gen_gaussian_correlate(None, None, rho)
    y1 = max(min(y1, 6.0), -6.0)
    y2 = max(min(y2, 6.0), -6.0)
    den = coenr.e * dt + y1 * math.sqrt(max(0.0, coenr.ee * dt))
    n2j = math.sqrt(max(0.0, coenr.jj * dt))
    djp = coenr.j * dt + y2 * n2j
    djp0 = (coenr.j * dt / steps + y2 * math.sqrt(max(0.0, coenr.jj * dt / steps))) if steps != 0.0 else 0.0
    return den, djp, djp0


def get_de_dj(sample: ParticleSampleType, coenr: CoeffType, time: float, dt: float, steps: float, period: float) -> None:
    if isnan(dt) or (not ieee_is_finite(dt)) or (not ieee_is_finite(steps)):
        raise RuntimeError("dt/steps invalid")
    if isnan(sample.jm) or isnan(sample.en):
        raise RuntimeError("sample jm/en is NaN")
    den, djp, djp0 = get_de_dj_nr(coenr, dt, steps)
    sample.den = den
    sample.djp = djp
    sample.djp0 = djp0


def get_sample_weight_real(sp: ParticleSampleType) -> None:
    sp.weight_real = sp.weight_clone * dms.weight_asym * sp.weight_n * ctl.n_basic


def output_sample_track_txt(sp: ParticleSampleType, fl: str) -> None:
    output_sg_sample_track_txt(sp, fl)


def add_track(t: float, sp: ParticleSampleType, state_flag: int) -> None:
    i = sp.length
    if i == sp.length_to_expand:
        sp.length_to_expand += track_length_expand_block
        tk = sp.track[:i]
        sp.track = tk + [TrackType() for _ in range(sp.length_to_expand - i)]
    i += 1
    sp.length = i
    if len(sp.track) < i:
        sp.track.append(TrackType())
    sp.track[i - 1].time = t
    sp.track[i - 1].ac = sp.byot.a_bin
    sp.track[i - 1].ec = sp.byot.e_bin
    sp.track[i - 1].incout = sp.byot.inc
    sp.track[i - 1].omout = sp.byot.om
    sp.track[i - 1].state_flag = state_flag


def get_move_result(sample: ParticleSampleType, den: float, djp: float, steps: float) -> tuple[float, float, float, float]:
    sample.byot.a_bin = -mbh / (2.0 * sample.en)
    ai = sample.byot.a_bin
    eni = sample.en
    ji = sample.jm * math.sqrt(mbh * ai)
    enf = eni + den
    af = -mbh / (2.0 * enf)
    jf = ji + djp
    jmf = jf / math.sqrt(mbh * af)
    mef = sample.byot.me + (steps - int(steps)) * 2.0 * PI
    while True:
        if jmf < jmin_value:
            jmf = 2.0 * jmin_value - jmf
        if jmf > jmax_value:
            jmf = jmax_value
        if jmf >= jmin_value:
            break
    return enf, jmf, mef, af


def move_de_dj_one(sample: ParticleSampleType, enf: float, jmf: float, mef: float, af: float) -> None:
    sample.byot.a_bin = -mbh / (2.0 * sample.en)
    ai = sample.byot.a_bin
    sample.byot_bf.a_bin = sample.byot.a_bin
    sample.byot_bf.e_bin = sample.byot.e_bin
    sample.byot.a_bin = af
    sample.en = enf
    sample.jm = jmf
    sample.byot.e_bin = math.sqrt(max(0.0, 1.0 - sample.jm**2))
    sample.byot.me = mef - 2.0 * PI if mef > 2.0 * PI else mef
    if sample.byot.e_bin == 1.0:
        sample.byot.e_bin = 0.9999999999
    if isnan(sample.jm) and sample.en < 0.0:
        raise RuntimeError("sample jm is NaN")


def run_boundary_state(sample: ParticleSampleType, total_time: float, time_now: float) -> tuple[float, int]:
    if time_now >= total_time:
        return time_now, 0
    evjum = max(min(sample.jm, jmax_value), jmin_value)
    even = math.log10(sample.en / ctl.energy0)
    en = sample.en
    jc = mbh / math.sqrt(-2.0 * en) if en < 0.0 else 0.0
    coenr = CoeffType()
    idx, idy = get_coenr(even, evjum, sample.m, en, jc, coenr)
    _ = (idx, idy)
    period = P(sample.byot.a_bin)
    while time_now < total_time:
        time_dt_t = total_time - time_now
        time_dt_nr = get_steps_nr_EJ(en, sample.jm, coenr, jc)
        time_dt = min(time_dt_nr, time_dt_t)
        time_now += time_dt
        steps = time_dt / period if period != 0.0 else 0.0
        den, djp, djp0 = get_de_dj_nr(coenr, time_dt, steps)
        if sample.en + den < ctl.energy_boundary:
            sample.den = den
            sample.djp = djp
            enf, jmf, mef, af = get_move_result(sample, den, djp, steps)
            move_de_dj_one(sample, enf, jmf, mef, af)
            return time_now, 100
    return time_now, 0


def get_step(sample: ParticleSampleType, coenr: CoeffType, ttot: float, tnow: float) -> float:
    period = P(sample.byot.a_bin)
    if not isnan(sample.byot.a_bin):
        sample.en = -mbh / (2.0 * sample.byot.a_bin)
    else:
        raise RuntimeError("get_step: a_bin is NaN")
    if sample.byot.e_bin < 1.0:
        sample.jm = math.sqrt(max(0.0, 1.0 - sample.byot.e_bin**2))
    else:
        raise RuntimeError("get_step: e_bin > 1")
    jm_dim = sample.jm * sample_jc
    time_dt_nr = get_steps_nr_EJ(sample.en, sample.jm, coenr, sample_jc)
    steps = time_dt_nr / period if period != 0.0 else 0.0
    ntmax = (ttot - tnow) / period if period != 0.0 else 0.0
    steps = min(steps, ntmax)
    if ctl.include_loss_cone >= 1 and sample.en < ctl.energy_boundary:
        jnr2 = coenr.jj * period
        nmjl = (max(0.1 * sample_jlc, 0.25 * abs(jm_dim - sample_jlc))) ** 2 / jnr2 if jnr2 != 0.0 else steps
        steps = min(steps, nmjl)
    if steps <= 0.0 or isnan(steps):
        raise RuntimeError("get_step produced invalid steps")
    return steps


def get_rh_vh_nh(mbh_in: float) -> tuple[float, float, float]:
    rgc = 8.32e3
    r0 = 3.1
    n0 = 2e4
    rh_out = r0 * (mbh_in / 4e6) ** 0.55 * pc
    vh_out = math.sqrt(mbh_in / rh_out) if rh_out != 0.0 else 0.0
    nh_out = n0 / (pc**3) * (mbh_in / 4e6) ** (-0.65)
    return rh_out, nh_out, vh_out


def init_model_ctl() -> None:
    global emin_value, emax_value, rh, rhmin, rhmax, log10rh, clone_emax, log10clone_emax
    ctl.grid_type = sts_type_grid
    emin_value = math.sqrt(max(0.0, 1.0 - jmax_value**2))
    emax_value = math.sqrt(max(0.0, 1.0 - jmin_value**2))
    rh_out, n0_out, v0_out = get_rh_vh_nh(mbh)
    rh = rh_out
    ctl.n0 = n0_out
    ctl.v0 = v0_out
    log10rh = math.log10(rh) if rh > 0.0 else 0.0
    rhmin = rh / emax_factor / 2.0
    rhmax = rh / emin_factor / 2.0
    ctl.sigma = math.sqrt(mbh / rh) if rh != 0.0 else 0.0
    ctl.energy0 = -mbh / rh if rh != 0.0 else 0.0
    ctl.energy_min = ctl.energy0 * emin_factor
    ctl.energy_max = ctl.energy0 * emax_factor
    set_simu_time()
    ctl.rbd = rh * 0.5 / ctl.x_boundary if ctl.x_boundary != 0.0 else 0.0
    if ctl.clone_scheme >= 1:
        ctl.clone_e0 = clone_e0_factor / 10.0 * ctl.energy0
        clone_emax = ctl.energy_max / ctl.clone_e0 if ctl.clone_e0 != 0.0 else 0.0
        log10clone_emax = math.log10(clone_emax) if clone_emax > 0.0 else 0.0
    for i in range(ctl.num_bk_comp):
        ci = ctl.cc[i]
        denom = (4.0 / (3.0 - ci.alpha) * PI * rh**3) if (3.0 - ci.alpha) != 0.0 else 1.0
        ci.n0 = ci.n_in_rh / denom
    ctl.energy_boundary = ctl.x_boundary * ctl.energy0
    for i in range(ctl.num_bk_comp):
        ctl.cc[i].alpha_ini = 7.0 / 4.0
    ctl.num_boundary_created = 0
    ctl.num_boundary_elim = 0
    ctl.num_clone_created = 0
    get_ccidx_from_type(ctl.idxstar, star_type_ms)
    get_ccidx_from_type(ctl.idxsbh, star_type_bh)
    get_ccidx_from_type(ctl.idxns, star_type_ns)
    get_ccidx_from_type(ctl.idxwd, star_type_wd)
    get_ccidx_from_type(ctl.idxbd, star_type_bd)
    ctl.nblock_size = int(ctl.grid_bins / ctl.ntasks) if ctl.ntasks != 0 else 0
    ctl.nblock_mpi_bg = ctl.nblock_size * rid + 1
    ctl.nblock_mpi_ed = ctl.nblock_size * (rid + 1)


def init_model() -> None:
    init_model_ctl()
    init_chattery()
    init_pro()


def init_chattery() -> None:
    global chattery_out_unit
    if ctl.chattery >= 1:
        chattery_out_unit = chattery_out_unit_0 + rid
    chattery_out_unit = 6

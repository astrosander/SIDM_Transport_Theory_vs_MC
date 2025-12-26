from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any
import math
import random

PI = math.pi

MBH: float = 0.0
rh: float = 0.0
log10rh: float = 0.0
mbh_radius: float = 0.0
rhmin: float = 0.0
rhmax: float = 0.0

fgx_g0: float = 0.0
NJbin_tot = 8

ncd_maxsize = 200000
chattery_out_unit_0 = 1383829393
chattery_out_unit: int = 0

my_unit_vel_c = 1.0
my_unit_vel_c5 = my_unit_vel_c ** 5
my_unit_vel_c3 = my_unit_vel_c ** 3

clone_e0_factor: float = 0.0
clone_emax: float = 0.0

jmin_value: float = 0.0
jmax_value: float = 0.0
emin_value: float = 0.0
emax_value: float = 0.0

method_int_nearst = 1
method_int_linear = 2

boundary_fj_iso = 1
boundary_fj_ls = 2
model_intej_fast = 1
task_mode_new = 1
task_mode_append = 2
snap_mode_new = 1
snap_mode_append = 2

outcome_ejection = 15
ls_type_compact = 1
ls_type_binary = 2
MAX_LENGTH = 100000
MAX_RUN_LENGTH = int(1e7)
n_orb_track_block = 100000
n_orb_track_block_max = n_orb_track_block * 16

ini_bhb_mbh_r0: float = 0.0
ini_by_3body_r0: float = 0.0

collection_data_through_mpi = 0

nsize_chain_bk: int = 0
nsize_chain_by: int = 0
nsize_arr_bk: int = 0
nsize_arr_by: int = 0
nsize_arr_bk_norm: int = 0
nsize_arr_by_norm: int = 0
nsize_arr_bk_pointer: int = 0
nsize_arr_by_pointer: int = 0
nsize_tot_bk: int = 0
nsize_tot_by: int = 0
nsize_tot: int = 0

boundary_sts_emin_cros: int = 0
boundary_sts_emin_dir: int = 0
boundary_sts_create: int = 0
boundary_sts_replace: int = 0
boundary_cross_up: int = 0
boundary_cross_down: int = 0
boundary_cross_net: int = 0
boundary_sts_emax_cros: int = 0

exit_normal = 1
exit_tidal = 2
exit_max_reach = 3
exit_boundary_min = 4
exit_boundary_max = 5
exit_invtransit = 7
exit_tidal_empty = 8
exit_tidal_full = 9
exit_ejection = 11
exit_gw_iso = 99


@dataclass
class CoreCompType:
    bk_evl_mode: int = 0
    blkmass: float = 0.0
    gamma: float = 0.0
    n0: float = 0.0
    alpha: float = 0.0
    N_in_rh: float = 0.0
    Mass_in_rh: float = 0.0
    N_pre_in: float = 0.0
    N_pre: int = 0
    nbd_tot: int = 0
    rb_min: float = 0.0
    rb_max: float = 0.0
    frac: float = 0.0
    rb: float = 0.0
    rb_ini: float = 0.0
    alpha_ini: float = 0.0
    rb_min_ini: float = 0.0
    rb_max_ini: float = 0.0
    clone_amplifier: int = 0
    str_bktypemodel: str = ""
    str_bkmassmodel: str = ""
    str_bkacmodel: str = ""
    str_bkacinimodel: str = ""
    bktypemodel: int = 0
    bkmassmodel: int = 0
    bkacmodel: int = 0
    bkacinimodel: int = 0


@dataclass
class ControlType:
    cc: List[CoreCompType] = field(default_factory=list)

    mass_ref: float = 0.0
    bin_mass_min: float = 0.0
    bin_mass_max: float = 0.0
    n_basic: float = 0.0

    bin_mass: List[float] = field(default_factory=lambda: [0.0] * 20)
    bin_mass_m1: List[float] = field(default_factory=lambda: [0.0] * 20)
    bin_mass_m2: List[float] = field(default_factory=lambda: [0.0] * 20)
    asymptot: List[List[float]] = field(default_factory=lambda: [[0.0] * 20 for _ in range(8)])
    ini_weight_n: List[float] = field(default_factory=lambda: [0.0] * 20)

    Weight_n: List[float] = field(default_factory=lambda: [0.0] * 20)
    alpha_ini: float = 0.0

    bin_mass_N: List[List[float]] = field(default_factory=lambda: [[0.0] * 4 for _ in range(20)])
    total_time: float = 0.0

    sigma: float = 0.0
    energy0: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 0.0

    clone_e0: float = 0.0
    v0: float = 0.0
    n0: float = 0.0

    rbd: float = 0.0
    x_boundary: float = 0.0
    energy_boundary: float = 0.0

    ts_spshot: float = 0.0
    tnr: float = 0.0
    ts_snap_input: float = 0.0

    update_dt: float = 0.0

    str_jbin_bd: str = ""
    str_fj_bd: str = ""
    time_unit: str = ""
    str_model_intej: str = ""
    str_method_interpolate: str = ""
    cfs_file_dir: str = ""
    burn_in_dir: str = ""

    bin_mass_particle_number: List[int] = field(default_factory=lambda: [0] * 20)
    clone_factor: List[int] = field(default_factory=lambda: [0] * 20)

    EJ_mode: int = 0
    num_bk_comp: int = 0
    num_clone_created: int = 0
    num_boundary_created: int = 0
    num_boundary_elim: int = 0
    boundary_method: int = 0
    boundary_fj: int = 0

    num_update_per_snap: int = 0
    n_spshot: int = 0
    n_spshot_total: int = 0
    include_loss_cone: int = 0
    model_intej: int = 0

    same_rseed_evl: int = 0
    same_rseed_ini: int = 0
    clone_scheme: int = 0
    trace_all_sample: int = 0
    del_cross_clone: int = 0
    burn_in_snap: int = 0
    jbin_type: int = 0

    idxstar: int = 0
    idxsbh: int = 0
    idxns: int = 0
    idxwd: int = 0
    idxbd: int = 0

    method_interpolate: int = 0
    m_bins: int = 0
    grid_bins: int = 0
    gx_bins: int = 0
    grid_type: int = 0
    idx_ref: int = 0

    debug: int = 0

    bin_mass_flux_in: List[List[int]] = field(default_factory=lambda: [[0] * 4 for _ in range(20)])
    bin_mass_flux_out: List[List[int]] = field(default_factory=lambda: [[0] * 4 for _ in range(20)])
    bin_mass_emax_out: List[List[int]] = field(default_factory=lambda: [[0] * 4 for _ in range(20)])

    chattery: int = 0
    ntasks: int = 0
    ntask_total: int = 0
    seed_value: int = 0
    ntask_bg: int = 0

    nblock_mpi_bg: int = 0
    nblock_mpi_ed: int = 0
    nblock_size: int = 0

    output_track_td: int = 0
    output_track_emri: int = 0
    output_track_plunge: int = 0

    burn_in_phase: bool = False


ctl = ControlType()


class ChainType:
    def __init__(self) -> None:
        self.head: Optional[Any] = None

    def get_length(self, type: int = 0) -> int:
        raise NotImplementedError


class ChainPointerType:
    def chain_to_arr_single(self, out_list: List[Any], n: int) -> None:
        raise NotImplementedError


class ParticleSamplesArrType:
    def __init__(self) -> None:
        self.n: int = 0
        self.sp: List[Any] = []

    def init(self, n: int) -> None:
        self.n = n
        self.sp = [None] * n


class SamplesTypePointerArr:
    pass


class DiffuseMSpec:
    pass


Allsams = ChainType()

bksams_arr_ini = ParticleSamplesArrType()
bkstars = ChainType()
bkbystars = ChainType()
bkbbhs = ChainType()
bksbhs = ChainType()
bksams = ChainType()
bksams_norm = ChainType()
bksams_merge = ChainType()

bkstars_arr = ParticleSamplesArrType()
bkbystars_arr = ParticleSamplesArrType()
bkbbhs_arr = ParticleSamplesArrType()
bksbhs_arr = ParticleSamplesArrType()
bksbhs_arr_norm = ParticleSamplesArrType()
bksams_arr_norm = ParticleSamplesArrType()
bksams_arr_norm_sbh = ParticleSamplesArrType()
bksams_arr = ParticleSamplesArrType()

bksams_pointer_arr = SamplesTypePointerArr()
dms = DiffuseMSpec()


def p(a: float) -> float:
    return 2.0 * PI * math.sqrt(a ** 3 / MBH)


def all_chain_to_arr_single(sps: ChainType, sps_arr: ParticleSamplesArrType) -> None:
    nr = sps.get_length(type=1)
    if nr == 0:
        return
    sps_arr.init(nr)
    sp = sps.head
    if sp is None:
        return
    sp.chain_to_arr_single(sps_arr.sp[0:nr], nr)


def get_exit_flag_str(exit_flag: int) -> str:
    if exit_flag == exit_normal:
        return "NORMAL"
    if exit_flag == exit_tidal:
        return "TIDAL"
    if exit_flag == exit_max_reach:
        return "NMAX"
    if exit_flag == exit_boundary_min:
        return "BOUNDAY_MIN"
    if exit_flag == exit_boundary_max:
        return "BOUNDARY MAX"
    if exit_flag == exit_invtransit:
        return "INVERSE TRANSIT"
    if exit_flag == exit_tidal_empty:
        return "TD EMPTY"
    if exit_flag == exit_tidal_full:
        return "TD FULL"
    if exit_flag == exit_ejection:
        return "EJECT"
    if exit_flag == exit_gw_iso:
        return "GW_ISO"
    return "Null"


def get_lvl(en: float) -> int:
    return int(math.log10(en / ctl.clone_e0))


def same_random_seed(seed_value: int) -> None:
    random.seed(seed_value)


def set_seed(same_seed: int, seed_value: int) -> None:
    if same_seed > 0:
        same_random_seed(seed_value)
    else:
        random.seed()

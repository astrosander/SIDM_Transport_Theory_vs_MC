from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, BinaryIO, Optional
import math

log10clone_emax: float = 0.0

track_length_expand_block = 100
nint_particle = 10
nreal_particle = 19

flag_ini_ini = 1
flag_ini_or = 5

record_track_nes = 1
record_track_detail = 2
record_track_all = 3

star_type_BH = 1
star_type_NS = 2
star_type_MS = 3
star_type_WD = 4
star_type_BD = 5
star_type_other = 6

exit_normal = 1
exit_other = 100
exit_ejection = 11
exit_by_gw_embhb = 17
exit_emri_single = 180
exit_plunge_single = 181
exit_tidal = 2
exit_merge_eject = 200
exit_max_reach = 3
exit_boundary_min = 4
exit_boundary_max = 5
exit_invtransit = 7
exit_tidal_empty = 8
exit_tidal_full = 9
exit_gw_iso = 99

state_ae_evl = 1
state_emax = 19
state_plunge = 199
state_td = 71


def isnan(x: float) -> bool:
    return math.isnan(x)


@dataclass
class Binary:
    a_bin: float = 0.0
    e_bin: float = 0.0
    Inc: float = 0.0
    Om: float = 0.0


@dataclass
class TrackType:
    time: float = 0.0
    ac: float = 0.0
    ec: float = 0.0
    ain: float = 0.0
    ein: float = 0.0
    Incin: float = 0.0
    incout: float = 0.0
    omin: float = 0.0
    omout: float = 0.0
    state_flag: int = 0
    ms_star_type: int = 0
    mm_star_type: int = 0


def star_type(type_int: int) -> str:
    if type_int == star_type_BH:
        return "BH"
    if type_int == star_type_MS:
        return "MS"
    if type_int == star_type_NS:
        return "NS"
    if type_int == star_type_WD:
        return "WD"
    if type_int == star_type_BD:
        return "BD"
    return "UNKNOWN"


@dataclass
class ParticleSampleType:
    id: int = 0
    rid: int = 0
    idx: int = 0
    obtype: int = 0
    obidx: int = 0
    state_flag_last: int = 0
    exit_flag: int = 0
    length: int = 0
    length_to_expand: int = 0
    track_step: int = 1
    write_down_track: int = 0
    within_jt: int = 0

    r_td: float = 0.0
    m: float = 0.0
    en0: float = 0.0
    jm0: float = 0.0
    pd: float = 0.0
    rp: float = 0.0
    tgw: float = 0.0
    simu_bgtime: float = 0.0

    En: float = 0.0
    Jm: float = 0.0
    create_time: float = 0.0
    exit_time: float = 0.0

    djp: float = 0.0
    elp: float = 0.0
    den: float = 0.0
    djp0: float = 0.0

    weight_clone: float = -1e99
    weight_N: float = -1e99
    weight_real: float = -1e99

    byot: Binary = field(default_factory=Binary)
    byot_ini: Binary = field(default_factory=Binary)
    byot_bf: Binary = field(default_factory=Binary)

    track: List[TrackType] = field(default_factory=list)

    def print(self, s: str) -> None:
        print(s)
        print(f"{'rid,idx,length=':20s}{self.rid:15d}{self.idx:15d}{self.length:15d}")
        print(f"{'id, exit_flag=':20s}{self.id:15d}{self.exit_flag:15d}")
        print(f"{'state_last=':20s}{self.state_flag_last:15d}")
        print(
            f"{'otby:a,e, I, Om=':20s}"
            f"{self.byot.a_bin:15.6f}{self.byot.e_bin:15.6f}{self.byot.Inc:15.6f}{self.byot.Om:15.6f}"
        )
        print(
            f"{'ctime, w(real,  clone, n)=':30s}"
            f"{self.create_time:15.6e}{self.weight_real:15.6e}{self.weight_clone:15.6e}{self.weight_N:15.6e}"
        )

    def track_init(self, n: int) -> None:
        if n > 0:
            self.track = [TrackType() for _ in range(n)]
        else:
            self.track = []
        self.length = 0
        self.length_to_expand = n

    def init(self) -> None:
        self.write_down_track = 0
        self.within_jt = 0
        self.byot.a_bin = 0.0
        self.byot.e_bin = 0.0
        self.track_step = 1
        self.pd = 0.0
        self.exit_flag = 0
        self.m = 0.0
        self.weight_real = -1e99
        self.weight_clone = -1e99
        self.weight_N = -1e99
        self.exit_time = 0.0
        self.track_init(0)

    def read_info(self, f: BinaryIO) -> None:
        raise NotImplementedError

    def write_info(self, f: BinaryIO) -> None:
        raise NotImplementedError

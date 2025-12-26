from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List
import math
import struct
import random


@dataclass
class SamplesTypePointer:
    sp: Any = None
    rid: int = 0
    index: int = 0


@dataclass
class SamplesTypePointerArr:
    n: int = 0
    pt: List[SamplesTypePointer] = field(default_factory=list)

    def init(self, n: int) -> None:
        self.pt = [SamplesTypePointer() for _ in range(n)]
        self.n = n


@dataclass
class ParticleSamplesArrType:
    n: int = 0
    sp: List[Any] = field(default_factory=list)

    def init(self, n: int) -> None:
        self.sp = [type(self._proto())() if callable(self._proto) else self._proto() for _ in range(n)]
        self.n = n
        for i in range(n):
            if hasattr(self.sp[i], "init") and callable(self.sp[i].init):
                self.sp[i].init()

    def _proto(self):
        if self.sp:
            return self.sp[0].__class__
        return ParticleSampleType

    def output_bin(self, fl: str) -> None:
        with open(f"{fl}.bin", "wb") as f:
            f.write(struct.pack("<i", int(self.n)))
            for i in range(self.n):
                self.sp[i].write_info(f)

    def input_bin(self, fl: str) -> None:
        with open(f"{fl}.bin", "rb") as f:
            (n,) = struct.unpack("<i", f.read(4))
            self.init(int(n))
            for i in range(self.n):
                self.sp[i].read_info(f)

    def select(self, sps_out: "ParticleSamplesArrType", exitflag: int, timebg: float, timeed: float) -> None:
        selected_idx = [i for i in range(self.n) if sams_selection_function(self, i, exitflag, timebg, timeed)]
        sps_out.init(len(selected_idx))
        for k, i in enumerate(selected_idx):
            sps_out.sp[k] = self.sp[i]

    def output_txt(self, fl: str) -> None:
        with open(f"{fl}.txt", "w") as f:
            f.write(
                "m aout eout inc om pe rp en jm weight x y z exit_flag obtype\n"
            )
            for i in range(self.n):
                sp = self.sp[i]
                if hasattr(sp, "byot"):
                    sp.byot.an_in_mode = an_in_mode_mean
                    sp.byot.me = rnd(0.0, 2.0 * PI)
                    by_em2st(sp.byot)
                    by_split_from_rd(sp.byot)
                    x, y, z = sp.byot.rd.x
                else:
                    x = y = z = 0.0
                f.write(
                    f"{sp.m:.10E} {sp.byot.a_bin:.10E} {sp.byot.e_bin:.10E} "
                    f"{sp.byot.Inc:.10E} {sp.byot.Om:.10E} {sp.byot.pe:.10E} "
                    f"{sp.rp:.10E} {sp.en:.10E} {sp.jm:.10E} {sp.weight_real:.10E} "
                    f"{x:.10E} {y:.10E} {z:.10E} {int(sp.exit_flag)} {int(sp.obtype)}\n"
                )


def init_pointer_arr(this: SamplesTypePointerArr, n: int) -> None:
    this.init(n)


def convert_sams_pointer_arr(sps: Any, sps_pointer: SamplesTypePointerArr, type: Optional[int] = None) -> None:
    typeI = 0 if type is None else type
    nsel = 0
    nsel = sps.get_length(type=typeI)
    sps_pointer.init(nsel)
    nsel = 0
    ps = sps.head
    while ps is not None:
        if typeI == 0:
            nsel += 1
            sps_pointer.pt[nsel - 1].sp = ps
        elif typeI == 1:
            ob = getattr(ps, "ob", None)
            if isinstance(ob, ParticleSampleType):
                nsel += 1
                sps_pointer.pt[nsel - 1].sp = ps
        ps = ps.next


def sams_arr_select_condition_single(
    sps: ParticleSamplesArrType,
    sps_out: ParticleSamplesArrType,
    selection_func: Callable[..., bool],
    ipar: Optional[list[int]] = None,
    rpar: Optional[list[float]] = None,
) -> None:
    nsel = 0
    for i in range(sps.n):
        if ipar is not None:
            selected = selection_func(sps.sp[i], ipar, rpar)
        else:
            selected = selection_func(sps.sp[i])
        if selected:
            nsel += 1
    sps_out.init(nsel)
    nsel = 0
    for i in range(sps.n):
        if ipar is not None:
            selected = selection_func(sps.sp[i], ipar, rpar)
        else:
            selected = selection_func(sps.sp[i])
        if selected:
            nsel += 1
            sps_out.sp[nsel - 1] = sps.sp[i]


def copy_particle_sample_arr(scopy: ParticleSamplesArrType, sp: ParticleSamplesArrType) -> None:
    sp.init(scopy.n)
    for i in range(scopy.n):
        sp.sp[i] = scopy.sp[i]


def smmerge_arr_single(sma: list[ParticleSamplesArrType], n: int, smam: ParticleSamplesArrType) -> None:
    nsam = 0
    for i in range(n):
        nsam += sma[i].n
    smam.init(nsam)
    k = 0
    for i in range(n):
        for j in range(sma[i].n):
            smam.sp[k] = sma[i].sp[j]
            k += 1


def set_sample_arr_indexs_rid_particle(sams_arr: ParticleSamplesArrType, rid: int) -> None:
    for i in range(sams_arr.n):
        sams_arr.sp[i].idx = i + 1
        sams_arr.sp[i].rid = rid


def sams_arr_select_single(
    sps: ParticleSamplesArrType,
    sps_out: ParticleSamplesArrType,
    exitflag: int,
    timebg: float,
    timeed: float,
) -> None:
    selected_idx = [i for i in range(sps.n) if sams_selection_function(sps, i, exitflag, timebg, timeed)]
    sps_out.init(len(selected_idx))
    for k, i in enumerate(selected_idx):
        sps_out.sp[k] = sps.sp[i]


def sams_selection_function(
    sps: ParticleSamplesArrType,
    i: int,
    exitflag: int,
    timebg: float,
    timeed: float,
) -> bool:
    sp = sps.sp[i]
    if math.isnan(sp.weight_clone) or math.isnan(sp.weight_N):
        raise RuntimeError(f"selection error: {sp.weight_clone} {sp.weight_N} {sp.id}")
    ok_exit = (sp.exit_flag == exitflag) or (exitflag == -1) or (exitflag == -2 and sp.exit_flag != exit_invtransit)
    if not ok_exit:
        return False
    ok_bg = (sp.exit_time > timebg) or (timebg < 0.0)
    if not ok_bg:
        return False
    ok_ed = (sp.exit_time < timeed) or (timeed < 0.0)
    return ok_ed


def output_particle_sams_txt(sps: ParticleSamplesArrType, fl: str) -> None:
    sps.output_txt(fl)


PI = math.pi


def rnd(a: float, b: float) -> float:
    return a + (b - a) * random.random()


an_in_mode_mean = 3


class ParticleSampleType:
    def __init__(self):
        self.idx = 0
        self.rid = 0
        self.id = 0
        self.m = 0.0
        self.en = 0.0
        self.jm = 0.0
        self.rp = 0.0
        self.exit_flag = 0
        self.obtype = 0
        self.exit_time = 0.0
        self.weight_clone = 0.0
        self.weight_N = 0.0
        self.weight_real = 0.0
        self.byot = Binary()

    def init(self) -> None:
        pass

    def write_info(self, f) -> None:
        raise NotImplementedError

    def read_info(self, f) -> None:
        raise NotImplementedError


class Binary:
    def __init__(self):
        self.a_bin = 0.0
        self.e_bin = 0.0
        self.Inc = 0.0
        self.Om = 0.0
        self.pe = 0.0
        self.me = 0.0
        self.an_in_mode = 0
        self.rd = type("RD", (), {})()
        self.rd.x = [0.0, 0.0, 0.0]


def by_em2st(by: Binary) -> None:
    raise NotImplementedError


def by_split_from_rd(by: Binary) -> None:
    raise NotImplementedError


exit_invtransit = 0

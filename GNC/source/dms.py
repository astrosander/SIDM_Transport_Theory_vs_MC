import math
import pickle
from dataclasses import dataclass, field
from typing import BinaryIO, List, Optional, Sequence

import numpy as np

n_tot_comp = 7


@dataclass
class DMSStellarObject:
    n: int = 0
    n_real: float = 0.0
    barge: Optional["S1DType"] = None
    fden: Optional["S1DType"] = None
    fden_simu: Optional["S1DType"] = None
    fNa: Optional["S1DType"] = None
    fMa: Optional["S1DType"] = None
    asymp: float = 0.0
    nxj: Optional["S2DHstBasicType"] = None
    gxj: Optional["S2DType"] = None

    def init(self, nbin_grid: int, nbin_gx: int, emin: float, emax: float, jmin: float, jmax: float, rh: float, jb_type: int) -> None:
        self.n = 0
        self.n_real = 0.0

    def write_info(self, funit: BinaryIO) -> None:
        pickle.dump(self, funit, protocol=pickle.HIGHEST_PROTOCOL)

    def read_info(self, funit: BinaryIO) -> None:
        obj = pickle.load(funit)
        self.__dict__.update(obj.__dict__)


@dataclass
class DSOPointer:
    p: Optional[DMSStellarObject] = None


@dataclass
class MassBins:
    bstar: DMSStellarObject = field(default_factory=DMSStellarObject)
    star: DMSStellarObject = field(default_factory=DMSStellarObject)
    sbh: DMSStellarObject = field(default_factory=DMSStellarObject)
    bbh: DMSStellarObject = field(default_factory=DMSStellarObject)
    ns: DMSStellarObject = field(default_factory=DMSStellarObject)
    all: DMSStellarObject = field(default_factory=DMSStellarObject)
    wd: DMSStellarObject = field(default_factory=DMSStellarObject)
    bd: DMSStellarObject = field(default_factory=DMSStellarObject)
    dsp: List[DSOPointer] = field(default_factory=lambda: [DSOPointer() for _ in range(n_tot_comp)])

    mc: float = 0.0
    m1: float = 0.0
    m2: float = 0.0
    nbin_grid: int = 0
    nbin_gx: int = 0
    frac: float = 0.0
    emin: float = 0.0
    emax: float = 0.0
    jmin: float = 0.0
    jmax: float = 0.0
    mbh: float = 0.0
    v0: float = 0.0
    n0: float = 0.0
    barmin: float = 0.0
    rh: float = 0.0
    dc: "DiffuseCoefficientType" = field(default_factory=lambda: DiffuseCoefficientType())

    def init(
        self,
        nbin_grid: int,
        nbin_gx: int,
        emin: float,
        emax: float,
        jmin: float,
        jmax: float,
        mbh: float,
        v0: float,
        n0: float,
        rh: float,
        jb_type: int,
    ) -> None:
        self.nbin_grid = int(nbin_grid)
        self.nbin_gx = int(nbin_gx)
        self.emin = float(emin)
        self.emax = float(emax)
        self.jmin = float(jmin)
        self.jmax = float(jmax)
        self.mbh = float(mbh)
        self.v0 = float(v0)
        self.n0 = float(n0)
        self.rh = float(rh)

        _ = math.log10(0.5 * rh / (10.0 ** emax))
        _ = math.log10(0.5 * rh / (10.0 ** emin))

        self.all.init(nbin_grid, nbin_gx, emin, emax, jmin, jmax, rh, jb_type)

        self.dsp[0].p = self.star
        self.dsp[1].p = self.sbh
        self.dsp[2].p = self.ns
        self.dsp[3].p = self.wd
        self.dsp[4].p = self.bd
        self.dsp[5].p = self.bstar
        self.dsp[6].p = self.bbh

        for i in range(n_tot_comp):
            self.dsp[i].p.init(nbin_grid, nbin_gx, emin, emax, jmin, jmax, rh, jb_type)

    def write_mb(self, funit: BinaryIO) -> None:
        self.dc.write_grid(funit)
        pickle.dump((self.mc, self.m1, self.m2, self.nbin_grid, self.nbin_gx), funit, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((self.emin, self.emax, self.jmin, self.jmax, self.mbh, self.v0, self.n0, self.rh), funit, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((self.all.n, self.star.n, self.sbh.n, self.wd.n, self.ns.n, self.bd.n, self.bstar.n, self.bbh.n), funit, protocol=pickle.HIGHEST_PROTOCOL)

        def dump_obj_fields(obj: DMSStellarObject) -> None:
            pickle.dump((obj.barge, obj.fden, obj.fden_simu, obj.asymp), funit, protocol=pickle.HIGHEST_PROTOCOL)

        dump_obj_fields(self.all)

        if self.star.n > 0:
            dump_obj_fields(self.star)
        if self.sbh.n > 0:
            dump_obj_fields(self.sbh)
        if self.ns.n > 0:
            dump_obj_fields(self.ns)
        if self.wd.n > 0:
            dump_obj_fields(self.wd)
        if self.bd.n > 0:
            dump_obj_fields(self.bd)
        if self.bstar.n > 0:
            dump_obj_fields(self.bstar)
        if self.bbh.n > 0:
            dump_obj_fields(self.bbh)

    def read_mb(self, funit: BinaryIO) -> None:
        self.dc.read_grid(funit)
        self.mc, self.m1, self.m2, self.nbin_grid, self.nbin_gx = pickle.load(funit)
        self.emin, self.emax, self.jmin, self.jmax, self.mbh, self.v0, self.n0, self.rh = pickle.load(funit)
        (self.all.n, self.star.n, self.sbh.n, self.wd.n, self.ns.n, self.bd.n, self.bstar.n, self.bbh.n) = pickle.load(funit)

        def load_obj_fields(obj: DMSStellarObject) -> None:
            obj.barge, obj.fden, obj.fden_simu, obj.asymp = pickle.load(funit)

        load_obj_fields(self.all)

        if self.star.n > 0:
            load_obj_fields(self.star)
        if self.sbh.n > 0:
            load_obj_fields(self.sbh)
        if self.ns.n > 0:
            load_obj_fields(self.ns)
        if self.wd.n > 0:
            load_obj_fields(self.wd)
        if self.bd.n > 0:
            load_obj_fields(self.bd)
        if self.bstar.n > 0:
            load_obj_fields(self.bstar)
        if self.bbh.n > 0:
            load_obj_fields(self.bbh)


@dataclass
class NEJWType:
    e: float = 0.0
    j: float = 0.0
    w: float = 0.0
    idx: int = 0


@dataclass
class DiffuseMSpec:
    n: int = 0
    mb: List[MassBins] = field(default_factory=list)
    all: MassBins = field(default_factory=MassBins)
    weight_asym: float = 0.0
    idx_ref: int = 0
    dc0: "DiffuseCoefficientType" = field(default_factory=lambda: DiffuseCoefficientType())
    nbin_grid: int = 0
    nbin_gx: int = 0
    jbin_type: int = 0
    grid_type: int = 0
    emin: float = 0.0
    emax: float = 0.0
    jmin: float = 0.0
    jmax: float = 0.0
    mbh: float = 0.0
    v0: float = 0.0
    n0: float = 0.0
    rh: float = 0.0
    acmin: float = 0.0
    acmax: float = 0.0
    x_boundary: float = 0.0

    def set_diffuse_mspec(
        self,
        nbin_grid: int,
        nbin_gx: int,
        emin: float,
        emax: float,
        jmin: float,
        jmax: float,
        mbh: float,
        v0: float,
        n0: float,
        rh: float,
        xb: float,
        idx_ref: int,
        jb_type: int,
        grid_type: int,
    ) -> None:
        self.idx_ref = int(idx_ref)
        self.nbin_grid = int(nbin_grid)
        self.nbin_gx = int(nbin_gx)
        self.emin = float(emin)
        self.emax = float(emax)
        self.mbh = float(mbh)
        self.v0 = float(v0)
        self.n0 = float(n0)
        self.rh = float(rh)
        self.acmin = math.log10(0.5 * rh / (10.0 ** emax))
        self.acmax = math.log10(0.5 * rh / (10.0 ** emin))
        self.x_boundary = float(xb)
        self.grid_type = int(grid_type)

        self.jmin = float(conv_j_space(jmin, jb_type))
        self.jmax = float(conv_j_space(jmax, jb_type))
        self.jbin_type = int(jb_type)
        self.grid_type = int(grid_type)

    def init(self, n: int) -> None:
        self.mb = [MassBins() for _ in range(int(n))]
        self.n = int(n)
        self.dc0.init(self.nbin_grid, self.emin, self.emax, self.jmin, self.jmax, self.grid_type)
        for i in range(self.n):
            self.mb[i].dc.init(self.nbin_grid, self.emin, self.emax, self.jmin, self.jmax, self.grid_type)
            self.mb[i].init(self.nbin_grid, self.nbin_gx, self.emin, self.emax, self.jmin, self.jmax, self.mbh, self.v0, self.n0, self.rh, self.jbin_type)
        self.all.init(self.nbin_grid, self.nbin_gx, self.emin, self.emax, self.jmin, self.jmax, self.mbh, self.v0, self.n0, self.rh, self.jbin_type)

    def output_bin(self, fl: str) -> None:
        with open(fl, "wb") as f:
            pickle.dump(
                (
                    self.n,
                    self.idx_ref,
                    self.nbin_grid,
                    self.nbin_gx,
                    self.emin,
                    self.emax,
                    self.jmin,
                    self.jmax,
                    self.mbh,
                    self.v0,
                    self.n0,
                    self.rh,
                    self.acmin,
                    self.acmax,
                    self.x_boundary,
                    self.jbin_type,
                    self.grid_type,
                ),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            for i in range(self.n):
                self.mb[i].write_mb(f)
            self.dc0.write_grid(f)
            pickle.dump(self.weight_asym, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.all.all.fNa, f, protocol=pickle.HIGHEST_PROTOCOL)

    def input_bin(self, fl: str) -> None:
        with open(fl, "rb") as f:
            (
                n,
                self.idx_ref,
                self.nbin_grid,
                self.nbin_gx,
                self.emin,
                self.emax,
                self.jmin,
                self.jmax,
                self.mbh,
                self.v0,
                self.n0,
                self.rh,
                self.acmin,
                self.acmax,
                self.x_boundary,
                self.jbin_type,
                self.grid_type,
            ) = pickle.load(f)
            self.init(n)
            for i in range(self.n):
                self.mb[i].read_mb(f)
            self.dc0.read_grid(f)
            self.weight_asym = pickle.load(f)
            self.all.all.fNa = pickle.load(f)


def set_mass_bin_mass_given(dm: DiffuseMSpec, masses: Sequence[float], m1: Sequence[float], m2: Sequence[float], asym: np.ndarray, n: int) -> None:
    for i in range(int(n)):
        dm.mb[i].mc = float(masses[i])
        dm.mb[i].m1 = float(m1[i])
        dm.mb[i].m2 = float(m2[i])
        dm.mb[i].all.asymp = float(asym[0, i])
        for j in range(n_tot_comp):
            dm.mb[i].dsp[j].p.asymp = float(asym[j + 1, i] * asym[0, i])


def get_n_from_particlex(mstar: np.ndarray, xstar: np.ndarray, m1: float, m2: float, x_b: float) -> np.ndarray:
    if mstar.size == 0:
        return np.zeros(0, dtype=np.int64)
    mask = (mstar >= m1) & (mstar <= m2) & (xstar >= x_b)
    return np.nonzero(mask)[0].astype(np.int64)


def get_n_from_particle(mstar: np.ndarray, m1: float, m2: float) -> np.ndarray:
    if mstar.size == 0:
        return np.zeros(0, dtype=np.int64)
    mask = (mstar >= m1) & (mstar <= m2)
    return np.nonzero(mask)[0].astype(np.int64)


def get_ejw_from_particle(
    estar: np.ndarray,
    jstar: np.ndarray,
    wstar: np.ndarray,
    mstar: np.ndarray,
    m1: float,
    m2: float,
    mbh: float,
    v0: float,
    xb: float,
) -> List[NEJWType]:
    xstar = np.abs(estar) / (v0 * v0)
    idx = get_n_from_particle(mstar, m1, m2)
    out: List[NEJWType] = []
    for ii in idx.tolist():
        e = math.log10(float(xstar[ii]))
        out.append(NEJWType(e=e, j=float(jstar[ii]), w=float(wstar[ii]), idx=int(ii)))
    return out

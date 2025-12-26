import math
import pickle
from dataclasses import dataclass, field
from typing import BinaryIO, List

import numpy as np

STS_TYPE_GRID = 1
STS_TYPE_DSTR = 0
STS_TYPE_DSTR_LOG = 2

Invns = 101
inf = -50.1
tiny_ = 1e-9

emin_factor = 0.0
emax_factor = 0.0
log10emin_factor = 0.0
log10emax_factor = 0.0

Jbin_type_lin = 1
Jbin_type_log = 2
Jbin_type_sqr = 3

coeff_chattery = 0

df: List["DiffuseCoefficientType"] = []


def set_range_1d(xmin: float, xmax: float, n: int, flag: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if flag == 0:
        xstep = (xmax - xmin) / float(n)
        i = np.arange(1, n + 1, dtype=np.float64)
        return xmin + xstep * (i - 0.5)
    if flag == 1:
        if n == 1:
            return np.array([xmin], dtype=np.float64)
        xstep = (xmax - xmin) / float(n - 1)
        i = np.arange(1, n + 1, dtype=np.float64)
        return xmin + xstep * (i - 1.0)
    raise ValueError(f"define flag {flag}")


@dataclass
class S1DType:
    xmin: float = 0.0
    xmax: float = 0.0
    xstep: float = 0.0
    bin_type: int = STS_TYPE_GRID
    nbin: int = 0
    xb: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    fx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    pn: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def init(self, xmin: float, xmax: float, n: int, bin_type: int) -> None:
        self.nbin = int(n)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.bin_type = int(bin_type)
        self.xb = np.zeros(self.nbin, dtype=np.float64)
        self.fx = np.zeros(self.nbin, dtype=np.float64)
        self.pn = np.zeros(self.nbin, dtype=np.float64)
        self.xstep = 0.0

    def set_range(self) -> None:
        self.xb = set_range_1d(self.xmin, self.xmax, self.nbin, self.bin_type)
        if self.nbin >= 2:
            self.xstep = float(self.xb[1] - self.xb[0])
        else:
            self.xstep = 0.0


@dataclass
class S2DType:
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    xstep: float = 0.0
    ystep: float = 0.0
    bin_type: int = STS_TYPE_GRID
    nx: int = 0
    ny: int = 0
    xcenter: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    ycenter: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    fxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))

    def init(self, nx: int, ny: int, xmin: float, xmax: float, ymin: float, ymax: float, bin_type: int) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.bin_type = int(bin_type)
        self.xcenter = np.zeros(self.nx, dtype=np.float64)
        self.ycenter = np.zeros(self.ny, dtype=np.float64)
        self.fxy = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.xstep = 0.0
        self.ystep = 0.0

    def set_range(self) -> None:
        self.xcenter = set_range_1d(self.xmin, self.xmax, self.nx, self.bin_type)
        self.ycenter = set_range_1d(self.ymin, self.ymax, self.ny, self.bin_type)
        if self.nx >= 2:
            self.xstep = float(self.xcenter[1] - self.xcenter[0])
        else:
            self.xstep = 0.0
        if self.ny >= 2:
            self.ystep = float(self.ycenter[1] - self.ycenter[0])
        else:
            self.ystep = 0.0


@dataclass
class CoeffType:
    e: float = 0.0
    j: float = 0.0
    ee: float = 0.0
    jj: float = 0.0
    ej: float = 0.0
    e_110: float = 0.0
    e_0: float = 0.0
    j_111: float = 0.0
    j_rest: float = 0.0
    m_avg: float = 0.0


@dataclass
class InvType:
    n: int = 0
    xprime: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    I_nvs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    y2: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def init(self, n: int) -> None:
        self.n = int(n)
        self.xprime = np.zeros(self.n, dtype=np.float64)
        self.I_nvs = np.zeros(self.n, dtype=np.float64)
        self.y2 = np.zeros(self.n, dtype=np.float64)


Invs2 = InvType()
Invs3 = InvType()
Invs5 = InvType()
Invs6 = InvType()
Invs7 = InvType()
Invs8 = InvType()
Invs9 = InvType()
Invs10 = InvType()
Invs11 = InvType()
GBW76 = InvType()
eps1 = InvType()
eps2 = InvType()


def conv_j_space(jin: float, jbin_type: int) -> float:
    if jbin_type == Jbin_type_log:
        return math.log10(jin)
    if jbin_type == Jbin_type_lin:
        return jin
    if jbin_type == Jbin_type_sqr:
        return jin * jin
    raise ValueError(f"define jbin type {jbin_type}")


@dataclass
class DiffuseCoefficientType:
    s2_de: S2DType = field(default_factory=S2DType)
    s2_dee: S2DType = field(default_factory=S2DType)
    s2_dj: S2DType = field(default_factory=S2DType)
    s2_djj: S2DType = field(default_factory=S2DType)
    s2_dej: S2DType = field(default_factory=S2DType)
    s2_dRRJJ: S2DType = field(default_factory=S2DType)
    s2_dRRJ: S2DType = field(default_factory=S2DType)
    s2_de_110: S2DType = field(default_factory=S2DType)
    s2_de_0: S2DType = field(default_factory=S2DType)
    s2_dj_111: S2DType = field(default_factory=S2DType)
    s2_dj_rest: S2DType = field(default_factory=S2DType)
    s1_de: S1DType = field(default_factory=S1DType)
    s1_dee: S1DType = field(default_factory=S1DType)
    emin: float = 0.0
    emax: float = 0.0
    jmin: float = 0.0
    jmax: float = 0.0
    nbin: int = 0

    def init(self, nbin: int, emin: float, emax: float, jmin: float, jmax: float, sts_type_dc: int) -> None:
        self.nbin = int(nbin)
        self.emin = float(emin)
        self.emax = float(emax)
        self.jmin = float(jmin)
        self.jmax = float(jmax)

        self.s2_de_110.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_de_110.set_range()

        self.s2_de_0.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_de_0.set_range()

        self.s2_dee.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_dee.set_range()

        self.s2_dj_111.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_dj_111.set_range()

        self.s2_dj_rest.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_dj_rest.set_range()

        self.s2_djj.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_djj.set_range()

        self.s2_dej.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_dej.set_range()

        self.s2_dRRJJ.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_dRRJJ.set_range()

        self.s2_dRRJ.init(nbin, nbin, emin, emax, jmin, jmax, sts_type_dc)
        self.s2_dRRJ.set_range()

        self.s1_de.init(emin, emax, nbin, sts_type_dc)
        self.s1_de.xb = self.s2_de_0.xcenter.copy()

        self.s1_dee.init(emin, emax, nbin, sts_type_dc)
        self.s1_dee.xb = self.s2_dee.xcenter.copy()

    def write_grid(self, fp: BinaryIO) -> None:
        pickle.dump(self.nbin, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_de_110, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_de_0, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_dee, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_dj_111, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_dj_rest, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_djj, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_dej, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_dRRJJ, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.s2_dRRJ, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def read_grid(self, fp: BinaryIO) -> None:
        self.nbin = pickle.load(fp)
        self.s2_de_110 = pickle.load(fp)
        self.s2_de_0 = pickle.load(fp)
        self.s2_dee = pickle.load(fp)
        self.s2_dj_111 = pickle.load(fp)
        self.s2_dj_rest = pickle.load(fp)
        self.s2_djj = pickle.load(fp)
        self.s2_dej = pickle.load(fp)
        self.s2_dRRJJ = pickle.load(fp)
        self.s2_dRRJ = pickle.load(fp)


df_tot = DiffuseCoefficientType()

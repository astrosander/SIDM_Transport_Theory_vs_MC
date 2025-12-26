from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import struct


PI = math.pi
DSMIN_VALUE = -8.0


def _alloc_2d(n1: int, n2: int, fill: float = 0.0) -> list[list[float]]:
    return [[fill for _ in range(n2)] for _ in range(n1)]


def _alloc_1d(n: int, fill: float = 0.0) -> list[float]:
    return [fill for _ in range(n)]


def my_integral_none(a: float, b: float, fcn, nsteps: int = 4096) -> float:
    if a == b:
        return 0.0
    nsteps = max(2, int(nsteps))
    if nsteps % 2 == 1:
        nsteps += 1
    h = (b - a) / nsteps
    s = fcn(a) + fcn(b)
    for i in range(1, nsteps):
        x = a + h * i
        s += (4.0 if i % 2 == 1 else 2.0) * fcn(x)
    return s * h / 3.0


def collect_data_mpi(arr_2d: list[list[float]], xbin: int, nbg: int, ned: int, nblock: int, ntasks: int) -> None:
    return


def mpi_barrier() -> None:
    return


def get_cfunction(s: float, e: float, ld: int, md: int, nd: int) -> float:
    ymin = -PI / 2.0
    arg = (2.0 / s - 1.0) / e if e != 0.0 else 1.0
    arg = max(min(1.0, arg), -1.0)
    ymax = math.asin(arg)

    def fcn(x: float) -> float:
        return (
            (1.0 + e * math.sin(x)) ** (ld + nd)
            / (1.0 - e * math.sin(x)) ** (nd + md / 2.0)
            * (abs(2.0 - s - s * e * math.sin(x))) ** (md / 2.0)
        )

    y_out = my_integral_none(ymin, ymax, fcn) if ymin != ymax else 0.0
    return (2.0 ** (1.0 - ld)) * y_out


def set_grid_cfs(jum: list[float], s: list[list[float]], nj: int, ns: int, jmin: float, jmax: float) -> None:
    dsmin = -7.0
    for i in range(nj):
        jum_i = (jmax - jmin) * float(i) / float(nj - 1) + jmin
        jum[i] = jum_i
        e = math.sqrt(max(0.0, 1.0 - (10.0 ** jum_i) ** 2))
        dsmax = math.log10(2.0 / (1.0 - e) - 1.0) if (1.0 - e) != 0.0 else dsmin
        for j in range(ns):
            ds = (dsmax - dsmin) * float(j) / float(ns - 1) + dsmin
            s[i][j] = math.log10(10.0 ** ds + 1.0)


def get_cfunction_grid_s(ld: int, md: int, nd: int, cfs: list[list[float]], xbin: int, ybin: int, jmin: float, jmax: float) -> None:
    dsmin = -7.0
    for i in range(xbin):
        jum = (jmax - jmin) * float(i) / float(xbin - 1) + jmin
        e = math.sqrt(max(0.0, 1.0 - (10.0 ** jum) ** 2))
        dsmax = math.log10(2.0 / (1.0 - e) - 1.0) if (1.0 - e) != 0.0 else dsmin
        for j in range(ybin):
            ds = (dsmax - dsmin) * float(j) / float(ybin - 1) + dsmin
            s = math.log10(10.0 ** ds + 1.0)
            cfs[i][j] = get_cfunction(10.0 ** s, e, ld, md, nd)


def get_cfunction_grid_s_mpi(ld: int, md: int, nd: int, cfs: list[list[float]], xbin: int, ybin: int, jmin: float, jmax: float,
                            rid: int, ntasks: int) -> None:
    dsmin = -7.0
    nblock_size = int(xbin / ntasks) if ntasks > 0 else xbin
    nbg = nblock_size * rid + 1
    ned = nblock_size * (rid + 1)
    nbg0 = max(1, nbg)
    ned0 = min(xbin, ned)

    for i1 in range(nbg0, ned0 + 1):
        i = i1 - 1
        jum = (jmax - jmin) * float(i) / float(xbin - 1) + jmin
        e = math.sqrt(max(0.0, 1.0 - (10.0 ** jum) ** 2))
        dsmax = math.log10(2.0 / (1.0 - e) - 1.0) if (1.0 - e) != 0.0 else dsmin
        for j in range(ybin):
            ds = (dsmax - dsmin) * float(j) / float(ybin - 1) + dsmin
            s = math.log10(10.0 ** ds + 1.0)
            cfs[i][j] = get_cfunction(10.0 ** s, e, ld, md, nd)

    mpi_barrier()
    collect_data_mpi(cfs, xbin, nbg0, ned0, nblock_size, ntasks)


@dataclass
class CfunsType:
    nj: int = 0
    ns: int = 0
    jmin: float = 0.0
    jmax: float = 0.0
    cfs_110: list[list[float]] = field(default_factory=list)
    cfs_111: list[list[float]] = field(default_factory=list)
    cfs_131: list[list[float]] = field(default_factory=list)
    cfs_13_1: list[list[float]] = field(default_factory=list)
    cfs_130: list[list[float]] = field(default_factory=list)
    cfs_330: list[list[float]] = field(default_factory=list)
    cfs_310: list[list[float]] = field(default_factory=list)
    jum: list[float] = field(default_factory=list)
    s: list[list[float]] = field(default_factory=list)

    def init(self, nj: int, ns: int, jmin: float, jmax: float) -> None:
        self.nj = nj
        self.ns = ns
        self.jmin = jmin
        self.jmax = jmax
        self.cfs_110 = _alloc_2d(nj, ns)
        self.cfs_111 = _alloc_2d(nj, ns)
        self.cfs_131 = _alloc_2d(nj, ns)
        self.cfs_13_1 = _alloc_2d(nj, ns)
        self.cfs_130 = _alloc_2d(nj, ns)
        self.cfs_330 = _alloc_2d(nj, ns)
        self.cfs_310 = _alloc_2d(nj, ns)
        self.s = _alloc_2d(nj, ns)
        self.jum = _alloc_1d(nj)

    def get_size(self) -> int:
        n = 0
        n += self.nj * self.ns * 8
        n *= 7
        n += self.nj * self.ns * 8
        n += self.nj * 8
        return (n // 1024) * 8 + (self.nj * 8 // 1024)

    def input_bin(self, fl: str) -> None:
        with open(f"{fl}.bin", "rb") as f:
            nj, ns = struct.unpack("<ii", f.read(8))
            jmin, jmax = struct.unpack("<dd", f.read(16))
            self.init(nj, ns, jmin, jmax)

            def read_2d() -> list[list[float]]:
                data = struct.unpack(f"<{nj*ns}d", f.read(8 * nj * ns))
                out = _alloc_2d(nj, ns)
                k = 0
                for i in range(nj):
                    for j in range(ns):
                        out[i][j] = float(data[k])
                        k += 1
                return out

            def read_1d(n: int) -> list[float]:
                data = struct.unpack(f"<{n}d", f.read(8 * n))
                return [float(x) for x in data]

            self.cfs_110 = read_2d()
            self.cfs_111 = read_2d()
            self.cfs_130 = read_2d()
            self.cfs_13_1 = read_2d()
            self.cfs_131 = read_2d()
            self.cfs_310 = read_2d()
            self.cfs_330 = read_2d()
            self.s = read_2d()
            self.jum = read_1d(nj)

    def output_bin(self, fl: str) -> None:
        nj, ns = self.nj, self.ns
        with open(f"{fl}.bin", "wb") as f:
            f.write(struct.pack("<ii", int(nj), int(ns)))
            f.write(struct.pack("<dd", float(self.jmin), float(self.jmax)))

            def write_2d(a: list[list[float]]) -> None:
                flat = []
                for i in range(nj):
                    for j in range(ns):
                        flat.append(float(a[i][j]))
                f.write(struct.pack(f"<{nj*ns}d", *flat))

            def write_1d(a: list[float]) -> None:
                f.write(struct.pack(f"<{len(a)}d", *[float(x) for x in a]))

            write_2d(self.cfs_110)
            write_2d(self.cfs_111)
            write_2d(self.cfs_130)
            write_2d(self.cfs_13_1)
            write_2d(self.cfs_131)
            write_2d(self.cfs_310)
            write_2d(self.cfs_330)
            write_2d(self.s)
            write_1d(self.jum)

    def get_cfs_s(self) -> None:
        set_grid_cfs(self.jum, self.s, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(1, 1, 1, self.cfs_111, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(1, 1, 0, self.cfs_110, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(1, 3, 0, self.cfs_130, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(1, 3, -1, self.cfs_13_1, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(1, 3, 1, self.cfs_131, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(3, 1, 0, self.cfs_310, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s(3, 3, 0, self.cfs_330, self.nj, self.ns, self.jmin, self.jmax)

    def get_cfs_s_mpi(self, rid: int, ntasks: int) -> None:
        set_grid_cfs(self.jum, self.s, self.nj, self.ns, self.jmin, self.jmax)
        get_cfunction_grid_s_mpi(1, 1, 1, self.cfs_111, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)
        get_cfunction_grid_s_mpi(1, 1, 0, self.cfs_110, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)
        get_cfunction_grid_s_mpi(1, 3, 0, self.cfs_130, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)
        get_cfunction_grid_s_mpi(1, 3, -1, self.cfs_13_1, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)
        get_cfunction_grid_s_mpi(1, 3, 1, self.cfs_131, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)
        get_cfunction_grid_s_mpi(3, 1, 0, self.cfs_310, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)
        get_cfunction_grid_s_mpi(3, 3, 0, self.cfs_330, self.nj, self.ns, self.jmin, self.jmax, rid, ntasks)


cfs = CfunsType()

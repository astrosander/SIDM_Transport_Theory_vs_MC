from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from md_s2d_basic_type import S2DBasicType

STS_TYPE_DSTR = 0
STS_TYPE_GRID = 1


def set_range(n: int, xmin: float, xmax: float, flag: int) -> np.ndarray:
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
    raise ValueError("flag must be 0 or 1")


def return_idxy(
    x: float,
    y: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int,
    flag: int,
) -> Tuple[int, int]:
    if nx <= 0 or ny <= 0:
        return -9999, -9999

    if flag == 1:
        if nx == 1:
            xstep = 0.0
        else:
            xstep = (xmax - xmin) / float(nx - 1)
        if ny == 1:
            ystep = 0.0
        else:
            ystep = (ymax - ymin) / float(ny - 1)

        if x >= xmin and x < xmax and xstep != 0.0:
            idx = int(np.rint((x - xmin) / xstep)) + 1
        elif x == xmax:
            idx = nx
        else:
            idx = -9999

        if y >= ymin and y < ymax and ystep != 0.0:
            idy = int(np.rint((y - ymin) / ystep)) + 1
        elif y == ymax:
            idy = ny
        else:
            idy = -9999

        return idx, idy

    if flag == 0:
        xstep = (xmax - xmin) / float(nx)
        ystep = (ymax - ymin) / float(ny)

        if x > xmin and x < xmax and xstep != 0.0:
            idx = int((x - xmin) / xstep + 1.0)
        elif x == xmax:
            idx = nx
        elif x == xmin:
            idx = 1
        else:
            idx = -9999

        if y > ymin and y < ymax and ystep != 0.0:
            idy = int((y - ymin) / ystep + 1.0)
        elif y == ymax:
            idy = ny
        elif y == ymin:
            idy = 1
        else:
            idy = -9999

        return idx, idy

    raise ValueError("flag must be 0 or 1")


def bin2_weight(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    xmin: float,
    xmax: float,
    nx: int,
    ymin: float,
    ymax: float,
    ny: int,
    flag: int,
) -> np.ndarray:
    abin = np.zeros((nx, ny), dtype=np.float64)
    n = int(x.shape[0])
    for i in range(n):
        idx, idy = return_idxy(float(x[i]), float(y[i]), xmin, xmax, ymin, ymax, nx, ny, flag)
        if 1 <= idx <= nx and 1 <= idy <= ny:
            abin[idx - 1, idy - 1] += float(w[i])
    return abin


def cal_bin2_arr_weight(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    xmin: float,
    xmax: float,
    rxn: int,
    ymin: float,
    ymax: float,
    ryn: int,
    bflag: int,
    mflag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = set_range(rxn, xmin, xmax, bflag)
    yy = set_range(ryn, ymin, ymax, bflag)
    abin2d = bin2_weight(x, y, weight, xmin, xmax, rxn, ymin, ymax, ryn, bflag)
    sumi = float(np.sum(abin2d))

    if rxn >= 2:
        xstep0 = float(xx[1] - xx[0])
    else:
        xstep0 = float(xmax - xmin)

    if ryn >= 2:
        ystep0 = float(yy[1] - yy[0])
    else:
        ystep0 = float(ymax - ymin)

    if mflag == 0:
        fabin2d = abin2d.copy()
        return xx, yy, fabin2d

    if mflag == 1:
        fabin2d = np.zeros_like(abin2d)
        if sumi == 0.0:
            return xx, yy, fabin2d
        if bflag == 0:
            fabin2d = abin2d / (sumi * xstep0 * ystep0)
        else:
            for i in range(rxn):
                for j in range(ryn):
                    xstep = xstep0 * 0.5 if (i == 0 or i == rxn - 1) else xstep0
                    ystep = ystep0 * 0.5 if (j == 0 or j == ryn - 1) else ystep0
                    fabin2d[i, j] = abin2d[i, j] / (sumi * xstep * ystep)
        return xx, yy, fabin2d

    if mflag == 2:
        fabin2d = np.zeros_like(abin2d)
        if bflag == 0:
            fabin2d = abin2d / (xstep0 * ystep0)
        else:
            for i in range(rxn):
                for j in range(ryn):
                    xstep = xstep0 * 0.5 if (i == 0 or i == rxn - 1) else xstep0
                    ystep = ystep0 * 0.5 if (j == 0 or j == ryn - 1) else ystep0
                    fabin2d[i, j] = abin2d[i, j] / (xstep * ystep)
        return xx, yy, fabin2d

    raise ValueError("mflag must be 0, 1, or 2")


@dataclass
class S2DHstBasicType(S2DBasicType):
    nxyw: Optional[np.ndarray] = field(default=None, repr=False)
    nxy: Optional[np.ndarray] = field(default=None, repr=False)
    fxyw: Optional[np.ndarray] = field(default=None, repr=False)
    xmean: Optional[np.ndarray] = field(default=None, repr=False)
    ymean: Optional[np.ndarray] = field(default=None, repr=False)
    xsct: Optional[np.ndarray] = field(default=None, repr=False)
    ysct: Optional[np.ndarray] = field(default=None, repr=False)
    use_weight: bool = False

    def init_hst(self, nx: int, ny: int, xmin: float, xmax: float, ymin: float, ymax: float, use_weight: bool = False) -> None:
        super().init(nx, ny, xmin, xmax, ymin, ymax, STS_TYPE_DSTR)
        self.use_weight = bool(use_weight)

        self.nxy = np.zeros((self.nx, self.ny), dtype=np.int64)
        self.ymean = np.zeros(self.nx, dtype=np.float64)
        self.xmean = np.zeros(self.ny, dtype=np.float64)
        self.xsct = np.zeros(self.ny, dtype=np.float64)
        self.ysct = np.zeros(self.nx, dtype=np.float64)

        if self.use_weight:
            self.nxyw = np.zeros((self.nx, self.ny), dtype=np.float64)
            self.fxyw = np.zeros((self.nx, self.ny), dtype=np.float64)
        else:
            self.nxyw = None
            self.fxyw = None

    def get_stats_weight(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        xx, yy, fxyw = cal_bin2_arr_weight(
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            weight=np.asarray(w, dtype=np.float64),
            xmin=float(self.xmin),
            xmax=float(self.xmax),
            rxn=int(self.nx),
            ymin=float(self.ymin),
            ymax=float(self.ymax),
            ryn=int(self.ny),
            bflag=STS_TYPE_DSTR,
            mflag=2,
        )
        self.xcenter = xx
        self.ycenter = yy
        if self.fxyw is None or self.fxyw.shape != fxyw.shape:
            self.fxyw = np.zeros_like(fxyw)
        self.fxyw[:, :] = fxyw

        _, _, nxyw = cal_bin2_arr_weight(
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            weight=np.asarray(w, dtype=np.float64),
            xmin=float(self.xmin),
            xmax=float(self.xmax),
            rxn=int(self.nx),
            ymin=float(self.ymin),
            ymax=float(self.ymax),
            ryn=int(self.ny),
            bflag=STS_TYPE_DSTR,
            mflag=0,
        )
        if self.nxyw is None or self.nxyw.shape != nxyw.shape:
            self.nxyw = np.zeros_like(nxyw)
        self.nxyw[:, :] = nxyw

    def output_txt(self, fn: str) -> None:
        if self.xcenter is None or self.ycenter is None or self.fxy is None:
            return
        np.savetxt(f"{fn}_fxy.txt", np.asarray(self.fxy, dtype=np.float64), fmt="%.10e")
        np.savetxt(f"{fn}_fxy_x.txt", np.asarray(self.xcenter, dtype=np.float64), fmt="%.10e")
        np.savetxt(f"{fn}_fxy_y.txt", np.asarray(self.ycenter, dtype=np.float64), fmt="%.10e")


@dataclass
class S2DType(S2DBasicType):
    def init_s2d(self, nx: int, ny: int, xmin: float, xmax: float, ymin: float, ymax: float, bin_type: int) -> None:
        super().init(nx, ny, xmin, xmax, ymin, ymax, int(bin_type))

    def write_unformatted(self) -> Tuple[Tuple[int, int, float, float, float, float, int], Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        header = (int(self.nx), int(self.ny), float(self.xmin), float(self.xmax), float(self.ymin), float(self.ymax), int(self.bin_type))
        payload = (self.xcenter.copy(), self.ycenter.copy(), self.fxy.copy(), float(self.xstep), float(self.ystep))
        return header, payload

    def read_unformatted(
        self,
        header: Tuple[int, int, float, float, float, float, int],
        payload: Tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    ) -> None:
        nx, ny, xmin, xmax, ymin, ymax, bin_type = header
        self.init_s2d(int(nx), int(ny), float(xmin), float(xmax), float(ymin), float(ymax), int(bin_type))
        xcenter, ycenter, fxy, xstep, ystep = payload
        self.xcenter[:] = np.asarray(xcenter, dtype=np.float64)[: self.nx]
        self.ycenter[:] = np.asarray(ycenter, dtype=np.float64)[: self.ny]
        self.fxy[:, :] = np.asarray(fxy, dtype=np.float64)[: self.nx, : self.ny]
        self.xstep = float(xstep)
        self.ystep = float(ystep)


@dataclass
class S2DHstType(S2DHstBasicType):
    def init(self, nx: int, ny: int, xmin: float, xmax: float, ymin: float, ymax: float, use_weight: bool = False) -> None:
        self.init_hst(nx, ny, xmin, xmax, ymin, ymax, use_weight=use_weight)

    def write_unformatted(
        self,
    ) -> Tuple[
        Tuple[int, int, float, float, float, float, bool],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float],
    ]:
        header = (int(self.nx), int(self.ny), float(self.xmin), float(self.xmax), float(self.ymin), float(self.ymax), bool(self.use_weight))
        payload = (
            self.xcenter.copy(),
            self.ycenter.copy(),
            self.fxy.copy(),
            self.nxy.copy(),
            self.xmean.copy(),
            self.ymean.copy(),
            self.xsct.copy(),
            self.ysct.copy(),
            float(self.xstep),
            float(self.ystep),
        )
        return header, payload

    def read_unformatted(
        self,
        header: Tuple[int, int, float, float, float, float, bool],
        payload: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float],
    ) -> None:
        nx, ny, xmin, xmax, ymin, ymax, use_weight = header
        self.init(int(nx), int(ny), float(xmin), float(xmax), float(ymin), float(ymax), use_weight=bool(use_weight))

        xcenter, ycenter, fxy, nxy, xmean, ymean, xsct, ysct, xstep, ystep = payload
        self.xcenter[:] = np.asarray(xcenter, dtype=np.float64)[: self.nx]
        self.ycenter[:] = np.asarray(ycenter, dtype=np.float64)[: self.ny]
        self.fxy[:, :] = np.asarray(fxy, dtype=np.float64)[: self.nx, : self.ny]

        self.nxy[:, :] = np.asarray(nxy, dtype=np.int64)[: self.nx, : self.ny]
        self.xmean[:] = np.asarray(xmean, dtype=np.float64)[: self.ny]
        self.ymean[:] = np.asarray(ymean, dtype=np.float64)[: self.nx]
        self.xsct[:] = np.asarray(xsct, dtype=np.float64)[: self.ny]
        self.ysct[:] = np.asarray(ysct, dtype=np.float64)[: self.nx]

        self.xstep = float(xstep)
        self.ystep = float(ystep)

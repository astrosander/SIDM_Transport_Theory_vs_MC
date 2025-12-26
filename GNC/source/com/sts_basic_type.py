from __future__ import annotations

import numpy as np


class bin_constants:
    sts_type_grid = 1
    sts_type_dstr = 0
    sts_type_dstr_log = 2
    sts_type_default = sts_type_grid

    method_intp_s1d = 1
    method_direct_s1d = 2
    method_linear_s1d = 3
    method_linear_log_s1d = 4


def set_range(xb: np.ndarray, n: int, xmin: float, xmax: float, bin_type: int):
    if n <= 0:
        return
    if n == 1:
        xb[0] = 0.5 * (xmin + xmax)
        return
    if bin_type == bin_constants.sts_type_grid:
        step = (xmax - xmin) / float(n - 1)
        xb[:] = xmin + step * np.arange(n, dtype=np.float64)
    elif bin_type == bin_constants.sts_type_dstr:
        step = (xmax - xmin) / float(n)
        xb[:] = xmin + (np.arange(n, dtype=np.float64) + 0.5) * step
    elif bin_type == bin_constants.sts_type_dstr_log:
        xbg = np.log10(xmin)
        xed = np.log10(xmax)
        step = (xed - xbg) / float(n)
        xb[:] = 10.0 ** (xbg + (np.arange(n, dtype=np.float64) + 0.5) * step)
    else:
        step = (xmax - xmin) / float(n)
        xb[:] = xmin + (np.arange(n, dtype=np.float64) + 0.5) * step


def return_idx(x: float, xmin: float, xmax: float, nbin: int, bin_type: int):
    if nbin <= 0:
        return 0
    if x < xmin or x > xmax:
        return 0
    if bin_type == bin_constants.sts_type_grid:
        if nbin == 1:
            return 1
        step = (xmax - xmin) / float(nbin - 1)
        idx0 = int(np.rint((x - xmin) / step))
        return idx0 + 1
    if bin_type == bin_constants.sts_type_dstr:
        step = (xmax - xmin) / float(nbin)
        idx0 = int(np.floor((x - xmin) / step))
        if idx0 == nbin:
            idx0 = nbin - 1
        return idx0 + 1
    if bin_type == bin_constants.sts_type_dstr_log:
        if x <= 0 or xmin <= 0 or xmax <= 0:
            return 0
        xbg = np.log10(xmin)
        xed = np.log10(xmax)
        step = (xed - xbg) / float(nbin)
        xv = np.log10(x)
        idx0 = int(np.floor((xv - xbg) / step))
        if idx0 == nbin:
            idx0 = nbin - 1
        return idx0 + 1
    step = (xmax - xmin) / float(nbin)
    idx0 = int(np.floor((x - xmin) / step))
    if idx0 == nbin:
        idx0 = nbin - 1
    return idx0 + 1


def return_idxy(vx: float, vy: float, xmin: float, xmax: float, ymin: float, ymax: float,
                nx: int, ny: int, bin_type: int):
    idx = return_idx(vx, xmin, xmax, nx, bin_type)
    idy = return_idx(vy, ymin, ymax, ny, bin_type)
    return idx, idy


def linear_int(x, y, n: int, xev: float):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    yv = np.asarray(y, dtype=np.float64).reshape((n,))

    if n <= 1:
        return float(yv[0]) if n == 1 else 0.0

    if xev < xv[0]:
        return float(yv[0] - (yv[1] - yv[0]) / (xv[1] - xv[0]) * (xv[0] - xev))
    if xev > xv[-1]:
        return float(yv[-1] + (yv[-1] - yv[-2]) / (xv[-1] - xv[-2]) * (xev - xv[-1]))
    if xev == xv[-1]:
        return float(yv[-1])

    for i in range(n - 1):
        if xv[i] >= xv[i + 1]:
            raise ValueError(f"bad input in linear_int: x(i)>x(i+1) at i={i+1}")
        if xv[i] < xev < xv[i + 1]:
            return float(yv[i + 1] - (yv[i + 1] - yv[i]) / (xv[i + 1] - xv[i]) * (xv[i + 1] - xev))
        if xv[i] == xev:
            return float(yv[i])

    raise ValueError("linear_int: unreachable state")


def linear_int_2d(xmin: float, ymin: float, nx: int, ny: int, xstep: float, ystep: float,
                  z, vx: float, vy: float):
    z = np.asarray(z, dtype=np.float64).reshape((nx, ny))

    rdx = (vx - xmin) / xstep
    rdy = (vy - ymin) / ystep
    idx = int(rdx) + 1
    idy = int(rdy) + 1

    if idx < 0 or idx > nx or idy < 0 or idy > ny:
        return 0.0

    if idx == nx:
        idxn, idxm = idx - 1, idx
    elif idx == 1:
        idxn, idxm = 1, 2
    else:
        idxn, idxm = idx, idx + 1

    if idy == ny:
        idyn, idym = idy - 1, idy
    elif idy == 1:
        idyn, idym = 1, 2
    else:
        idyn, idym = idy, idy + 1

    y1 = z[idxn - 1, idyn - 1]
    y2 = z[idxm - 1, idyn - 1]
    y3 = z[idxm - 1, idym - 1]
    y4 = z[idxn - 1, idym - 1]

    t = rdx - idxn + 1.0
    u = rdy - idyn + 1.0

    return float((1.0 - t) * (1.0 - u) * y1 + t * (1.0 - u) * y2 + t * u * y3 + (1.0 - t) * u * y4)


class BinFunctionType:
    def __init__(self):
        self.xb: np.ndarray | None = None
        self.fx: np.ndarray | None = None
        self.y2: np.ndarray | None = None
        self.nbin: int = 0
        self.is_spline_prepared: bool = False

    def init_bin_function(self, n: int):
        self.deallocate()
        self.nbin = int(n)
        self.xb = np.zeros((self.nbin,), dtype=np.float64)
        self.fx = np.zeros((self.nbin,), dtype=np.float64)
        self.is_spline_prepared = False

    def deallocate(self):
        self.xb = None
        self.fx = None
        self.y2 = None
        self.nbin = 0
        self.is_spline_prepared = False

    def print(self, name: str | None = None):
        if name is not None:
            print("Name=", str(name).strip())
        print(f"{'X':>20}{'Y':>20}")
        if self.xb is None or self.fx is None:
            return
        for i in range(self.nbin):
            print(f"{self.xb[i]:20.10f}{self.fx[i]:20.10f}")

    def form_d2(self, x, y, n: int, reverse: int | None = None):
        rev = 1 if reverse is not None else 0
        self.init_bin_function(n)
        xv = np.asarray(x, dtype=np.float64).reshape((n,))
        yv = np.asarray(y, dtype=np.float64).reshape((n,))
        if rev == 0:
            self.xb[:] = xv
            self.fx[:] = yv
        else:
            self.xb[:] = xv[::-1]
            self.fx[:] = yv[::-1]

    def get_value_l(self, x: float, expolate: bool | None = None):
        expolate_in = bool(expolate) if expolate is not None else False
        if self.xb is None or self.fx is None or self.nbin <= 0:
            return 0.0
        if not expolate_in:
            if self.xb[0] > x:
                return float(self.fx[0])
            if self.xb[self.nbin - 1] < x:
                return float(self.fx[self.nbin - 1])
        return float(linear_int(self.xb, self.fx, self.nbin, x))


class S1DBasicType(BinFunctionType):
    def __init__(self):
        super().__init__()
        self.xmin: float = 0.0
        self.xmax: float = 0.0
        self.xstep: float = 0.0
        self.bin_type: int = bin_constants.sts_type_dstr

    def init_s1d_basic(self, xmin: float, xmax: float, n: int, bin_type: int):
        if n == 0:
            pass
        self.init_bin_function(n)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.bin_type = int(bin_type)
        self.xstep = 0.0

    def set_range(self):
        if self.xb is None:
            return
        set_range(self.xb, self.nbin, self.xmin, self.xmax, self.bin_type)
        self.xstep = float(self.xb[1] - self.xb[0]) if self.nbin >= 2 else 0.0

    def print(self, name: str | None = None):
        if name is not None:
            print("for s1d=", str(name).strip())
        if self.bin_type == bin_constants.sts_type_grid:
            str_bin_type = "GRID"
        elif self.bin_type == bin_constants.sts_type_dstr:
            str_bin_type = "DSTR"
        else:
            str_bin_type = "OTHER"
        print(f"{'bin_type':>15}{'nbin':>15}{'xmin':>15}{'xmax':>15}{'xstep':>15}")
        print(f"{str_bin_type:>15}{self.nbin:15d}{self.xmin:15.5e}{self.xmax:15.5e}{self.xstep:15.5e}")
        print(f"{'X':>20}{'FX':>20}")
        if self.xb is None or self.fx is None:
            return
        for i in range(self.nbin):
            print(f"{self.xb[i]:20.10e}{self.fx[i]:20.10e}")

    def get_value_d(self, x: float):
        idx = return_idx(float(x), self.xmin, self.xmax, self.nbin, self.bin_type)
        if idx < 1 or idx > self.nbin or self.fx is None:
            return 0.0
        return float(self.fx[idx - 1])


class S2DBasicType:
    def __init__(self):
        self.xmin: float = 0.0
        self.xmax: float = 0.0
        self.ymin: float = 0.0
        self.ymax: float = 0.0
        self.xstep: float = 0.0
        self.ystep: float = 0.0
        self.bin_type: int = bin_constants.sts_type_dstr
        self._is_spline_prepared: bool = False
        self.xcenter: np.ndarray | None = None
        self.ycenter: np.ndarray | None = None
        self.fxy: np.ndarray | None = None
        self.nx: int = 0
        self.ny: int = 0
        self._y2: np.ndarray | None = None

    def init_s2d_basic(self, nx: int, ny: int, xmin: float, xmax: float, ymin: float, ymax: float, bin_type: int):
        if nx == 0 or ny == 0:
            pass
        self.nx = int(nx)
        self.ny = int(ny)
        self.xcenter = np.zeros((self.nx,), dtype=np.float64)
        self.ycenter = np.zeros((self.ny,), dtype=np.float64)
        self.fxy = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        if self.xmax < self.xmin:
            raise ValueError("error! s2d.xmax < s2d.xmin")
        self.bin_type = int(bin_type)
        self._is_spline_prepared = False

    def set_range(self):
        if self.xcenter is None or self.ycenter is None:
            return
        set_range(self.xcenter, self.nx, self.xmin, self.xmax, self.bin_type)
        set_range(self.ycenter, self.ny, self.ymin, self.ymax, self.bin_type)
        self.xstep = float(self.xcenter[1] - self.xcenter[0]) if self.nx >= 2 else 0.0
        self.ystep = float(self.ycenter[1] - self.ycenter[0]) if self.ny >= 2 else 0.0

    def print(self, name: str | None = None):
        if name is not None:
            print("s2d=", str(name).strip())
        if self.bin_type == bin_constants.sts_type_grid:
            str_bin_type = "GRID"
        elif self.bin_type == bin_constants.sts_type_dstr:
            str_bin_type = "DSTR"
        else:
            str_bin_type = "OTHER"
        print(f"{'bin_type':>15}{'nx':>15}{'xmin':>15}{'xmax':>15}{'xstep':>15}"
              f"{'ny':>15}{'ymin':>15}{'ymax':>15}{'ystep':>15}")
        print(f"{str_bin_type:>15}{self.nx:15d}{self.xmin:15.5e}{self.xmax:15.5e}{self.xstep:15.5e}"
              f"{self.ny:15d}{self.ymin:15.5e}{self.ymax:15.5e}{self.ystep:15.5e}")
        if self.fxy is None:
            return
        for i in range(self.nx):
            row = self.fxy[i, :]
            print(" ".join(f"{v:12.3e}" for v in row))

    def get_value_d(self, vx: float, vy: float):
        if vx < self.xmin or vx > self.xmax or vy < self.ymin or vy > self.ymax:
            return 0.0
        if self.fxy is None:
            return 0.0
        idx, idy = return_idxy(vx, vy, self.xmin, self.xmax, self.ymin, self.ymax, self.nx, self.ny, self.bin_type)
        if 1 <= idx <= self.nx and 1 <= idy <= self.ny:
            return float(self.fxy[idx - 1, idy - 1])
        return 0.0

    def get_value_l(self, vx: float, vy: float):
        if vx < self.xmin or vx > self.xmax or vy < self.ymin or vy > self.ymax:
            return 0.0
        if self.fxy is None:
            return 0.0
        return float(linear_int_2d(self.xmin, self.ymin, self.nx, self.ny, self.xstep, self.ystep, self.fxy, vx, vy))

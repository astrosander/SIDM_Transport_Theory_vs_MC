import math
import pickle
from dataclasses import dataclass
from typing import BinaryIO, Callable, List, Optional, Tuple

import numpy as np


Jbin_type_lin = 1
Jbin_type_log = 2
Jbin_type_sqr = 3

method_int_nearst = 1
method_int_linear = 2

sts_type_grid = 1


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
class Ctl:
    x_boundary: float
    m_bins: int
    jbin_type: int
    method_interpolate: int
    grid_type: int
    chattery: int = 0


@dataclass
class CCTShare:
    alpha: float


@dataclass
class FCShare:
    xb: np.ndarray
    fx: np.ndarray
    xmin: float
    xmax: float

    @property
    def nbin(self) -> int:
        return int(self.xb.size)

    def get_value_l(self, xlog: float) -> float:
        return float(linear_int(self.xb, self.fx, xlog))


@dataclass
class S2D:
    fxy: np.ndarray


@dataclass
class DiffuseCoeffGrid:
    s2_de_110: S2D
    s2_de_0: S2D
    s2_dee: S2D
    s2_dj_111: S2D
    s2_dj_rest: S2D
    s2_djj: S2D
    s2_dej: S2D


@dataclass
class MassBin:
    mc: float
    dc: DiffuseCoeffGrid


@dataclass
class DMS:
    n: int
    nbin_grid: int
    emin: float
    emax: float
    jmin: float
    jmax: float
    mb: List[MassBin]
    dc0: DiffuseCoeffGrid


def _nint(x: float) -> int:
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def linear_int(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return float("nan")
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])
    k = int(np.searchsorted(x, xq, side="right") - 1)
    x0 = float(x[k])
    x1 = float(x[k + 1])
    y0 = float(y[k])
    y1 = float(y[k + 1])
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def linear_int_2d(xmin: float, ymin: float, nx: int, ny: int, xstep: float, ystep: float, arr: np.ndarray, xq: float, yq: float) -> float:
    arr = np.asarray(arr, dtype=float)
    if nx <= 1 or ny <= 1:
        return float("nan")
    rx = (xq - xmin) / xstep
    ry = (yq - ymin) / ystep
    ix = int(math.floor(rx))
    iy = int(math.floor(ry))
    ix = max(0, min(nx - 2, ix))
    iy = max(0, min(ny - 2, iy))
    tx = rx - ix
    ty = ry - iy
    v00 = float(arr[ix, iy])
    v10 = float(arr[ix + 1, iy])
    v01 = float(arr[ix, iy + 1])
    v11 = float(arr[ix + 1, iy + 1])
    v0 = v00 + tx * (v10 - v00)
    v1 = v01 + tx * (v11 - v01)
    return v0 + ty * (v1 - v0)


def return_idx(xq: float, xmin: float, xmax: float, nbin: int, mode: int = 1) -> int:
    if nbin <= 1:
        return 0
    step = (xmax - xmin) / float(nbin)
    t = (xq - xmin) / step
    if mode == 1:
        idx = _nint(t)
    else:
        idx = int(math.floor(t))
    idx = max(0, min(nbin - 1, idx))
    return idx


def fx_g(x: float, ctl: Ctl, cct_share: CCTShare, fgx_g0: float) -> float:
    alpha = cct_share.alpha
    if x <= 0.0:
        return fgx_g0 * math.exp(x)
    return fgx_g0 * (x / ctl.x_boundary) ** (alpha - 1.5)


def fgx_mb_star(x: float, fc_share: FCShare, fgx_g0: float) -> float:
    x1 = 10.0 ** float(fc_share.xb[0])
    x2 = 10.0 ** float(fc_share.xb[fc_share.nbin - 1])
    if x >= x1 and x <= x2:
        return fc_share.get_value_l(math.log10(x))
    if x <= 0.0:
        return math.exp(x) * fgx_g0
    if x > x2 or (x >= 0.0 and x < x1):
        return 0.0
    return 0.0


def fgx_mb(x: float, ctl: Ctl, fc_share: FCShare, fgx_g0: float, emax_factor: float) -> float:
    if x > ctl.x_boundary and x <= emax_factor:
        return linear_int(fc_share.xb[: fc_share.nbin], fc_share.fx[: fc_share.nbin], math.log10(x))
    if x <= 0.0:
        return math.exp(x) * fgx_g0
    if x > emax_factor or (x <= ctl.x_boundary and x > 0.0):
        return 0.0
    return 0.0


def fgx_mb_direct(x: float, ctl: Ctl, fc_share: FCShare, fgx_g0: float, emax_factor: float) -> float:
    if x > ctl.x_boundary and x <= emax_factor:
        idx = return_idx(math.log10(x), fc_share.xmin, fc_share.xmax, fc_share.nbin, 1)
        return float(fc_share.fx[idx])
    if x <= 0.0:
        return math.exp(x) * fgx_g0
    if x > emax_factor or (x <= ctl.x_boundary and x > 0.0):
        return 0.0
    return 0.0


def output_coef_ba16(fn: str, ctl: Ctl, df: List, df_tot) -> None:
    path = f"{fn.rstrip('/')}/coef_ba16.bin"
    with open(path, "wb") as f:
        for i in range(int(ctl.m_bins)):
            df[i].write_grid(f)
        df_tot.write_grid(f)


def input_coef_ba16(fn: str, ctl: Ctl, df_factory, df_tot) -> List:
    path = f"{fn.rstrip('/')}/coef_ba16.bin"
    df: List = [df_factory() for _ in range(int(ctl.m_bins))]
    with open(path, "rb") as f:
        for i in range(int(ctl.m_bins)):
            df[i].read_grid(f)
        df_tot.read_grid(f)
    return df


def get_coeff_sigma_funcs_cfs_grid(
    energy: float,
    jum: float,
    barg: Callable,
    cfs,
    get_sigma_funcs_cfs_grid_fn: Callable,
) -> Tuple[float, float, float, float, float, float, float]:
    sigma110 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_110, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma111 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_111, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma131 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_131, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma130 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_130, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma13_1 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_13_1, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma330 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_330, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma310 = get_sigma_funcs_cfs_grid_fn(energy, jum, barg, cfs.cfs_310, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    return sigma110, sigma111, sigma131, sigma130, sigma13_1, sigma330, sigma310


def get_coeff_xr(
    rj: float,
    sigma0: float,
    sigma110: float,
    sigma111: float,
    sigma131: float,
    sigma130: float,
    sigma13_1: float,
    sigma330: float,
    sigma310: float,
) -> CoeffType:
    coe = CoeffType()
    coe.e_110 = sigma110
    coe.e_0 = -sigma0
    coe.ee = 4.0 / 3.0 * (sigma0 + sigma13_1)
    coe.j_111 = rj * (sigma110 - sigma111)
    coe.j_rest = (
        (5.0 - 10.0 * rj) / 3.0 * sigma0
        + 4.0 * sigma310
        - 4.0 / 3.0 * sigma330
        + rj / 2.0 * sigma131
        - 3.0 / 2.0 * rj * sigma111
        - 4.0 / 3.0 * rj * sigma130
    )
    coe.jj = (
        10.0 / 3.0 * (rj - rj * rj) * sigma0
        + 2.0 * rj * rj * sigma131
        - 2.0 * rj * rj * sigma111
        + 8.0 * rj * sigma310
        - 8.0 / 3.0 * rj * sigma330
        + rj * rj * 4.0 / 3.0 * sigma13_1
        - 8.0 / 3.0 * rj * rj * sigma130
    )
    coe.ej = 4.0 / 3.0 * rj * (sigma13_1 - sigma130)
    return coe


def get_coeff_xj(
    j: float,
    sigma0: float,
    sigma110: float,
    sigma111: float,
    sigma131: float,
    sigma130: float,
    sigma13_1: float,
    sigma330: float,
    sigma310: float,
) -> CoeffType:
    rj = j * j
    coe = CoeffType()
    coe.e_110 = sigma110
    coe.e_0 = -sigma0
    coe.ee = 4.0 / 3.0 * (sigma0 + sigma13_1)
    coe.j_111 = j * sigma111
    coe.j_rest = (
        (5.0 * (1.0 - 3.0 * rj) / 12.0) * sigma0
        + sigma310
        - (1.0 / 3.0) * sigma330
        + rj / 2.0 * sigma111
        - (1.0 / 3.0) * rj * sigma130
        - (1.0 / 6.0) * rj * sigma13_1
    ) / j
    coe.jj = (
        5.0 / 6.0 * (1.0 - rj) * sigma0
        + rj / 2.0 * sigma131
        - rj / 2.0 * sigma111
        + 2.0 * sigma310
        - 2.0 / 3.0 * sigma330
        + rj / 3.0 * sigma13_1
        - 2.0 / 3.0 * rj * sigma130
    )
    coe.ej = 2.0 / 3.0 * j * (sigma13_1 - sigma130)
    return coe


def get_coeff_ej(
    jum: float,
    sigma0: float,
    sigma110: float,
    sigma111: float,
    sigma131: float,
    sigma130: float,
    sigma13_1: float,
    sigma330: float,
    sigma310: float,
) -> CoeffType:
    coe = CoeffType()
    coe.e_110 = sigma110
    coe.e_0 = -sigma0
    coe.ee = 4.0 / 3.0 * (sigma0 + sigma13_1)
    coe.j_111 = -jum * sigma111
    coe.j_rest = ((5.0 - 3.0 * jum * jum) / 12.0 * sigma0 + sigma310 - sigma330 / 3.0) / jum
    coe.jj = (
        (5.0 - 3.0 * jum * jum) / 6.0 * sigma0
        + (jum * jum) / 2.0 * sigma131
        - (jum * jum) / 2.0 * sigma111
        + 2.0 * sigma310
        - 2.0 / 3.0 * sigma330
    )
    coe.ej = -2.0 / 3.0 * jum * (sigma0 + sigma130)
    return coe


def get_coenr(
    even: float,
    evjum: float,
    m: float,
    en: float,
    jc: float,
    ctl: Ctl,
    dms: DMS,
    dc_grid_xstep: float,
    dc_grid_ystep: float,
    idx_in: Optional[int] = None,
    idy_in: Optional[int] = None,
) -> Tuple[CoeffType, int, int]:
    evj = float(evjum)
    if ctl.jbin_type == Jbin_type_log:
        evj = math.log10(evj)
    elif ctl.jbin_type == Jbin_type_sqr:
        evj = evj * evj

    idx = int(idx_in) if idx_in is not None else 0
    idy = int(idy_in) if idy_in is not None else 0

    de_110 = np.zeros(dms.n, dtype=float)
    dj_111 = np.zeros(dms.n, dtype=float)

    if ctl.method_interpolate == method_int_nearst:
        if ctl.grid_type == sts_type_grid:
            rdx = (even - dms.emin) / dc_grid_xstep
            rdy = (evj - dms.jmin) / dc_grid_ystep
            idx = _nint(rdx) + 1
            idy = _nint(rdy) + 1

        if idx < -1:
            idx = 1
        if idx > dms.nbin_grid:
            idx = dms.nbin_grid
        if idy < -1:
            idy = 1
        if idy > dms.nbin_grid:
            idy = dms.nbin_grid

        ix = idx - 1
        iy = idy - 1

        for i in range(dms.n):
            de_110[i] = float(dms.mb[i].dc.s2_de_110.fxy[ix, iy])
            dj_111[i] = float(dms.mb[i].dc.s2_dj_111.fxy[ix, iy])

        de_0 = float(dms.dc0.s2_de_0.fxy[ix, iy])
        dee = float(dms.dc0.s2_dee.fxy[ix, iy])
        dj_rest = float(dms.dc0.s2_dj_rest.fxy[ix, iy])
        djj = float(dms.dc0.s2_djj.fxy[ix, iy])
        dej = float(dms.dc0.s2_dej.fxy[ix, iy])

    else:
        for i in range(dms.n):
            de_110[i] = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.mb[i].dc.s2_de_110.fxy, even, evj)
            dj_111[i] = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.mb[i].dc.s2_dj_111.fxy, even, evj)

        de_0 = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.dc0.s2_de_0.fxy, even, evj)
        dee = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.dc0.s2_dee.fxy, even, evj)
        dj_rest = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.dc0.s2_dj_rest.fxy, even, evj)
        djj = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.dc0.s2_djj.fxy, even, evj)
        dej = linear_int_2d(dms.emin, dms.jmin, dms.nbin_grid, dms.nbin_grid, dc_grid_xstep, dc_grid_ystep, dms.dc0.s2_dej.fxy, even, evj)

    coe = CoeffType()
    coe.jj = djj * (jc * jc)
    coe.e = de_0
    coe.j = dj_rest

    for i in range(dms.n):
        mc = float(dms.mb[i].mc)
        coe.e += m / mc * float(de_110[i])
        coe.j += float(dj_111[i]) * (m + mc) / mc / 2.0

    coe.ee = dee * (en * en)
    coe.e *= en
    coe.j *= jc
    coe.ej = dej * en * jc

    return coe, idx, idy


def get_coeff_sigma_funcs_cfs_rk(
    energy: float,
    jum: float,
    barg: Callable,
    cfs,
    get_sigma_funcs_cfs_rk_fn: Callable,
) -> Tuple[float, float, float, float, float, float, float]:
    sigma110 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_110, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma111 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_111, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma131 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_131, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma130 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_130, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma13_1 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_13_1, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma330 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_330, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    sigma310 = get_sigma_funcs_cfs_rk_fn(energy, jum, barg, cfs.cfs_310, cfs.nj, cfs.ns, cfs.jmin, cfs.jmax)
    return sigma110, sigma111, sigma131, sigma130, sigma13_1, sigma330, sigma310

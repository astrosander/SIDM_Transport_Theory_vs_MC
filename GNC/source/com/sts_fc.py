from __future__ import annotations

import numpy as np


def set_range(n: int, xmin: float, xmax: float, flag: int) -> np.ndarray:
    x = np.empty(n, dtype=np.float64)
    if flag == 0:
        xstep = (xmax - xmin) / float(n)
        for i in range(1, n + 1):
            x[i - 1] = xmin + xstep * (i - 0.5)
    elif flag == 1:
        xstep = (xmax - xmin) / float(n - 1)
        for i in range(1, n + 1):
            x[i - 1] = xmin + xstep * (i - 1)
    else:
        raise ValueError("flag must be 0 or 1")
    return x


def get_dstr_num_in_each_bin(x: np.ndarray, n: int, xbg: float, xstep: float, nbin: int):
    fx = np.zeros(nbin, dtype=np.int64)
    n_num = 0
    for i in range(1, n + 1):
        indx = int((x[i - 1] - xbg) / xstep + 1)
        if 0 < indx <= nbin:
            fx[indx - 1] += 1
            n_num += 1
    return fx, n_num


def get_dstr_num_in_each_bin_weight(x: np.ndarray, w: np.ndarray, n: int, xbg: float, xstep: float, nbin: int):
    fxw = np.zeros(nbin, dtype=np.float64)
    n_numw = 0.0
    for i in range(1, n + 1):
        indx = int((x[i - 1] - xbg) / xstep + 1)
        if 0 < indx <= nbin:
            fxw[indx - 1] += float(w[i - 1])
            n_numw += float(w[i - 1])
    return fxw, n_numw


def return_idx(x: float, xmin: float, xmax: float, nx: int, flag: int) -> int:
    if flag == 1:
        xstep = (xmax - xmin) / float(nx - 1)
        if x >= xmin and x < xmax:
            return int(np.rint((x - xmin) / xstep)) + 1
        if x == xmax:
            return nx
        return -99999
    if flag == 0:
        xstep = (xmax - xmin) / float(nx)
        if x > xmin and x < xmax:
            return int((x - xmin) / xstep + 1)
        if x == xmin:
            return 1
        if x == xmax:
            return nx
        return -99999
    raise ValueError("flag must be 0 or 1")


def return_idxy(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float, nx: int, ny: int, flag: int):
    if flag == 1:
        xstep = (xmax - xmin) / float(nx - 1)
        ystep = (ymax - ymin) / float(ny - 1)

        if x >= xmin and x < xmax:
            idx = int(np.rint((x - xmin) / xstep)) + 1
        elif x == xmax:
            idx = nx
        else:
            idx = -9999

        if y >= ymin and y < ymax:
            idy = int(np.rint((y - ymin) / ystep)) + 1
        elif y == ymax:
            idy = ny
        else:
            idy = -9999

        return idx, idy

    if flag == 0:
        xstep = (xmax - xmin) / float(nx)
        ystep = (ymax - ymin) / float(ny)

        if x > xmin and x < xmax:
            idx = int((x - xmin) / xstep + 1)
        elif x == xmax:
            idx = nx
        elif x == xmin:
            idx = 1
        else:
            idx = -9999

        if y > ymin and y < ymax:
            idy = int((y - ymin) / ystep + 1)
        elif y == ymax:
            idy = ny
        elif y == ymin:
            idy = 1
        else:
            idy = -9999

        return idx, idy

    raise ValueError("flag must be 0 or 1")


def bin2(x: np.ndarray, y: np.ndarray, n: int, xmin: float, xmax: float, abinx: int, ymin: float, ymax: float, abiny: int, flag: int):
    abin2d = np.zeros((abinx, abiny), dtype=np.int64)
    for i in range(1, n + 1):
        idx, idy = return_idxy(float(x[i - 1]), float(y[i - 1]), xmin, xmax, ymin, ymax, abinx, abiny, flag)
        if 1 <= idx <= abinx and 1 <= idy <= abiny:
            abin2d[idx - 1, idy - 1] += 1
    return abin2d


def bin2_weight(x: np.ndarray, y: np.ndarray, w: np.ndarray, n: int, xmin: float, xmax: float, abinx: int, ymin: float, ymax: float, abiny: int, flag: int):
    abin2d = np.zeros((abinx, abiny), dtype=np.float64)
    for i in range(1, n + 1):
        idx, idy = return_idxy(float(x[i - 1]), float(y[i - 1]), xmin, xmax, ymin, ymax, abinx, abiny, flag)
        if 0 <= idx <= abinx and 0 <= idy <= abiny:
            if idx >= 1 and idy >= 1:
                abin2d[idx - 1, idy - 1] += float(w[i - 1])
    return abin2d


def cal_bin2_arr(x: np.ndarray, y: np.ndarray, n: int, xmin: float, xmax: float, rxn: int, ymin: float, ymax: float, ryn: int, bflag: int, mflag: int):
    xx = set_range(rxn, xmin, xmax, bflag)
    yy = set_range(ryn, ymin, ymax, bflag)

    abin2d = bin2(x, y, n, xmin, xmax, rxn, ymin, ymax, ryn, bflag)

    sumi = float(np.sum(abin2d))
    xstep = float(xx[1] - xx[0])
    ystep = float(yy[1] - yy[0])

    if mflag == 0:
        fabin2d = abin2d.astype(np.float64)
    elif mflag == 1:
        if bflag == 0:
            fabin2d = abin2d.astype(np.float64) / sumi / xstep / ystep
        else:
            fabin2d = np.empty((rxn, ryn), dtype=np.float64)
            for i in range(1, rxn + 1):
                for j in range(1, ryn + 1):
                    xs = (xx[1] - xx[0]) / 2.0 if (i == 1 or i == rxn) else (xx[1] - xx[0])
                    ys = (yy[1] - yy[0]) / 2.0 if (j == 1 or j == ryn) else (yy[1] - yy[0])
                    fabin2d[i - 1, j - 1] = float(abin2d[i - 1, j - 1]) / sumi / float(xs) / float(ys)
    elif mflag == 2:
        if bflag == 0:
            fabin2d = abin2d.astype(np.float64) / xstep / ystep
        else:
            fabin2d = np.empty((rxn, ryn), dtype=np.float64)
            for i in range(1, rxn + 1):
                for j in range(1, ryn + 1):
                    xs = (xx[1] - xx[0]) / 2.0 if (i == 1 or i == rxn) else (xx[1] - xx[0])
                    ys = (yy[1] - yy[0]) / 2.0 if (j == 1 or j == ryn) else (yy[1] - yy[0])
                    fabin2d[i - 1, j - 1] = float(abin2d[i - 1, j - 1]) / float(xs) / float(ys)
    else:
        raise ValueError("mflag must be 0, 1, or 2")

    return xx, yy, fabin2d


def cal_bin2_arr_weight(x: np.ndarray, y: np.ndarray, weight: np.ndarray, n: int, xmin: float, xmax: float, rxn: int, ymin: float, ymax: float, ryn: int, bflag: int, mflag: int):
    xx = set_range(rxn, xmin, xmax, bflag)
    yy = set_range(ryn, ymin, ymax, bflag)

    abin2d = bin2_weight(x, y, weight, n, xmin, xmax, rxn, ymin, ymax, ryn, bflag)

    sumi = float(np.sum(abin2d))
    xstep = float(xx[1] - xx[0])
    ystep = float(yy[1] - yy[0])

    if mflag == 0:
        fabin2d = abin2d.astype(np.float64)
    elif mflag == 1:
        if bflag == 0:
            fabin2d = abin2d.astype(np.float64) / sumi / xstep / ystep
        else:
            fabin2d = np.empty((rxn, ryn), dtype=np.float64)
            for i in range(1, rxn + 1):
                for j in range(1, ryn + 1):
                    xs = (xx[1] - xx[0]) / 2.0 if (i == 1 or i == rxn) else (xx[1] - xx[0])
                    ys = (yy[1] - yy[0]) / 2.0 if (j == 1 or j == ryn) else (yy[1] - yy[0])
                    fabin2d[i - 1, j - 1] = float(abin2d[i - 1, j - 1]) / sumi / float(xs) / float(ys)
    elif mflag == 2:
        if bflag == 0:
            fabin2d = abin2d.astype(np.float64) / xstep / ystep
        else:
            fabin2d = np.empty((rxn, ryn), dtype=np.float64)
            for i in range(1, rxn + 1):
                for j in range(1, ryn + 1):
                    xs = (xx[1] - xx[0]) / 2.0 if (i == 1 or i == rxn) else (xx[1] - xx[0])
                    ys = (yy[1] - yy[0]) / 2.0 if (j == 1 or j == ryn) else (yy[1] - yy[0])
                    fabin2d[i - 1, j - 1] = float(abin2d[i - 1, j - 1]) / float(xs) / float(ys)
    else:
        raise ValueError("mflag must be 0, 1, or 2")

    return xx, yy, fabin2d

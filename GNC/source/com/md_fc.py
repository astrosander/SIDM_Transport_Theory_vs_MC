import math
import pickle
import numpy as np


fc_spacing_log = 1
fc_spacing_linear = 0


def set_range(arr, n: int, xbg: float, xed: float, mode: int = 0):
    step = (xed - xbg) / float(n)
    for i in range(n):
        arr[i] = xbg + (i + 0.5) * step


def get_dstr_num_in_each_bin(x, n: int, xbg: float, xstep: float, nbin: int):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    idx = np.floor((xv - xbg) / xstep).astype(np.int64) + 1
    mask = (idx >= 1) & (idx <= nbin)
    nb = np.zeros((nbin,), dtype=np.int64)
    if np.any(mask):
        ii = idx[mask] - 1
        np.add.at(nb, ii, 1)
    ns = int(np.sum(mask))
    return nb, ns


def get_dstr_num_in_each_bin_weight(x, w, n: int, xbg: float, xstep: float, nbin: int):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    wv = np.asarray(w, dtype=np.float64).reshape((n,))
    idx = np.floor((xv - xbg) / xstep).astype(np.int64) + 1
    mask = (idx >= 1) & (idx <= nbin)
    nbw = np.zeros((nbin,), dtype=np.float64)
    if np.any(mask):
        ii = idx[mask] - 1
        np.add.at(nbw, ii, wv[mask])
    nsw = float(np.sum(wv[mask])) if np.any(mask) else 0.0
    return nbw, nsw


def linear_int_arb(x, y, n: int, xev: float):
    if n <= 1:
        raise ValueError(f"linear_int_arb, n<=1: {n}")
    x = np.asarray(x, dtype=np.float64).reshape((n,))
    y = np.asarray(y, dtype=np.float64).reshape((n,))
    xev = float(xev)

    if x[0] > x[n - 1]:
        x1, xn = x[n - 1], x[0]
        yn = y[0]
        increase = False
    else:
        x1, xn = x[0], x[n - 1]
        yn = y[n - 1]
        increase = True

    if xev < x1 or xev > xn:
        raise ValueError(f"xev out of range: {xev} not in [{x1}, {xn}]")

    if xev == xn:
        return float(yn)

    if increase:
        for i in range(n - 1):
            if x[i] < xev and x[i + 1] > xev:
                return float(y[i + 1] - (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x[i + 1] - xev))
            if x[i] == xev:
                return float(y[i])
    else:
        for i in range(n - 1, 0, -1):
            if x[i] < xev and x[i - 1] > xev:
                return float(y[i - 1] - (y[i - 1] - y[i]) / (x[i - 1] - x[i]) * (x[i - 1] - xev))
            if x[i] == xev:
                return float(y[i])

    raise RuntimeError("linear_int_arb: no interval found")


def search_for_position(y, x, n: int, per: float):
    y = np.asarray(y, dtype=np.float64).reshape((n,))
    x = np.asarray(x, dtype=np.float64).reshape((n,))
    per = float(per)
    for i in range(n - 1):
        if per == y[i]:
            return float(x[i])
        if per > y[i] and per < y[i + 1]:
            return float((x[i + 1] - x[i]) / (y[i + 1] - y[i]) * (per - y[i]) + x[i])
        return float(x[i])
    return float(x[n - 1])


class S1DHstBasicType:
    def __init__(self):
        self.nbin = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.xstep = 0.0
        self.ns = 0
        self.nsw = 0.0
        self.use_weight = False
        self.is_spline_prepared = False
        self.xb = None
        self.fx = None
        self.fxw = None
        self.nbw = None
        self.y2 = None

    def deallocate(self):
        self.xb = None
        self.fx = None
        self.fxw = None
        self.nbw = None
        self.y2 = None
        self.nbin = 0
        self.ns = 0
        self.nsw = 0.0
        self.is_spline_prepared = False


def init_s1_hst_basic(obj: S1DHstBasicType, xmin: float, xmax: float, n: int, use_weight: bool):
    obj.xmin = float(xmin)
    obj.xmax = float(xmax)
    obj.nbin = int(n)
    obj.use_weight = bool(use_weight)
    obj.xb = np.zeros((obj.nbin,), dtype=np.float64)
    obj.fx = np.zeros((obj.nbin,), dtype=np.float64)
    obj.ns = 0
    obj.nsw = 0.0
    obj.is_spline_prepared = False
    obj.y2 = None
    if obj.use_weight:
        obj.fxw = np.zeros((obj.nbin,), dtype=np.float64)
        obj.nbw = np.zeros((obj.nbin,), dtype=np.float64)
    else:
        obj.fxw = np.zeros((obj.nbin,), dtype=np.float64)
        obj.nbw = np.zeros((obj.nbin,), dtype=np.float64)


class StsFCType(S1DHstBasicType):
    def __init__(self):
        super().__init__()
        self.cnb = None
        self.cnbw = None
        self.cfx = None
        self.cfxw = None
        self.pn = None
        self.pnw = None
        self.y2_y = None
        self.nb = None
        self.flaglog = fc_spacing_linear
        self.type_int_size = 0
        self.type_real_size = 0
        self.type_log_size = 2

    def init(self, xmin: float, xmax: float, n: int, flag_log: int, use_weight: bool = False):
        if int(n) < 1:
            raise ValueError(f"n<1 in init_fc: {n}")
        init_s1_hst_basic(self, xmin, xmax, n, bool(use_weight))
        self.cfx = np.zeros((self.nbin,), dtype=np.float64)
        self.cnb = np.zeros((self.nbin,), dtype=np.int64)
        self.pn = np.zeros((self.nbin,), dtype=np.float64)
        self.y2_y = np.zeros((self.nbin,), dtype=np.float64)
        self.nb = np.zeros((self.nbin,), dtype=np.int64)
        if self.use_weight:
            self.cfxw = np.zeros((self.nbin,), dtype=np.float64)
            self.cnbw = np.zeros((self.nbin,), dtype=np.float64)
            self.pnw = np.zeros((self.nbin,), dtype=np.float64)
        else:
            self.cfxw = np.zeros((self.nbin,), dtype=np.float64)
            self.cnbw = np.zeros((self.nbin,), dtype=np.float64)
            self.pnw = np.zeros((self.nbin,), dtype=np.float64)
        self.flaglog = int(flag_log)
        self.type_int_size = self.nbin * 2 + 3
        self.type_real_size = self.nbin * 10 + 4
        return self

    def deallocate(self):
        super().deallocate()
        self.cnb = None
        self.cnbw = None
        self.cfx = None
        self.cfxw = None
        self.pn = None
        self.pnw = None
        self.y2_y = None
        self.nb = None

    def set_range(self):
        if self.flaglog == fc_spacing_log:
            xbg = math.log10(self.xmin)
            xed = math.log10(self.xmax)
        else:
            xbg = self.xmin
            xed = self.xmax
        set_range(self.xb, self.nbin, xbg, xed, 0)
        if self.flaglog == fc_spacing_log:
            self.xb = 10.0 ** self.xb
        self.xstep = (xed - xbg) / float(self.nbin)

    def set_fc_xb(self):
        self.set_range()

    def normalize(self):
        wtot = float(np.sum(self.xstep * self.fx))
        if wtot != 0.0:
            self.fx = self.fx / wtot

    def print(self, str_: str | None = None):
        if str_ is not None:
            print("for fc=", str(str_).strip())
        spacing = "LOG" if self.flaglog == fc_spacing_log else "LINEAR"
        print(f"{'spacing':>15}{'nbin':>15}{'xmin':>15}{'xmax':>15}{'xstep':>15}{'nsw':>15}{'ns':>15}")
        print(f"{spacing:>15}{self.nbin:15d}{self.xmin:15.5e}{self.xmax:15.5e}{self.xstep:15.5e}{self.nsw:15.5e}{self.ns:15d}")
        print(f"{'X':>20}{'FX':>20}{'FXW':>20}{'cfx':>20}{'cfxw':>20}")
        for i in range(self.nbin):
            print(f"{self.xb[i]:20.10e}{self.fx[i]:20.10e}{self.fxw[i]:20.10e}{self.cfx[i]:20.10e}{self.cfxw[i]:20.10e}")

    def output_fc(self, fn: str):
        with open(f"{fn.strip()}_fc.txt", "w") as f:
            f.write(f"{'xb':>27}{'fx':>27}{'pn':>27}{'fxw':>27}{'pnw':>27}{'cfx':>27}{'cfxw':>27}\n")
            for i in range(self.nbin):
                f.write(f"{self.xb[i]:27.12e} {self.fx[i]:27.12e} {self.pn[i]:27.12e} {self.fxw[i]:27.12e} {self.pnw[i]:27.12e} {self.cfx[i]:27.12e} {self.cfxw[i]:27.12e}\n")

    def output_nfc(self, fn: str):
        with open(f"{fn.strip()}_nfc.txt", "w") as f:
            f.write(f"{'xb':>27}{'nbw':>27}{'cnbw':>27}{'nb':>10}{'cnb':>10}\n")
            for i in range(self.nbin):
                f.write(f"{self.xb[i]:27.12e} {self.nbw[i]:27.12e} {self.cnbw[i]:27.12e} {int(self.nb[i]):10d} {int(self.cnb[i]):10d}\n")

    def output_fc_bin(self, fn: str):
        payload = self.to_dict()
        with open(f"{fn.strip()}_fc.bin", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def input_fc_bin(self, fn: str):
        with open(f"{fn.strip()}_fc.bin", "rb") as f:
            payload = pickle.load(f)
        self.from_dict(payload)

    def input_fc(self, fn: str, xmin: float, xmax: float, rn: int, flaglog: int):
        self.init(xmin, xmax, rn, flaglog, use_weight=True)
        with open(f"{fn.strip()}_fc.txt", "r") as f:
            _ = f.readline()
            for i in range(self.nbin):
                parts = f.readline().split()
                self.xb[i] = float(parts[0])
                self.fx[i] = float(parts[1])
                self.pn[i] = float(parts[2])
                self.fxw[i] = float(parts[3])
                self.pnw[i] = float(parts[4])
                self.cfx[i] = float(parts[5])
                self.cfxw[i] = float(parts[6])

    def get_value_l_y(self, va: float):
        return linear_int_arb(self.fx, self.xb, self.nbin, float(va))

    def conv_fc_int_real_arrays(self):
        logarr = np.array([bool(self.use_weight), bool(self.is_spline_prepared)], dtype=bool)
        intarr = np.zeros((self.type_int_size,), dtype=np.int64)
        realarr = np.zeros((self.type_real_size,), dtype=np.float64)

        intarr[0:3] = np.array([self.flaglog, self.nbin, int(self.ns)], dtype=np.int64)
        intarr[3:3 + self.nbin] = self.nb[: self.nbin]
        intarr[3 + self.nbin:3 + 2 * self.nbin] = self.cnb[: self.nbin]

        realarr[0:self.nbin] = self.xb[: self.nbin]
        realarr[self.nbin:2 * self.nbin] = self.fx[: self.nbin]
        realarr[5 * self.nbin:6 * self.nbin] = self.cfx[: self.nbin]
        realarr[7 * self.nbin:8 * self.nbin] = self.pn[: self.nbin]
        if self.is_spline_prepared and self.y2 is not None:
            realarr[9 * self.nbin:10 * self.nbin] = self.y2[: self.nbin]
        if self.use_weight:
            realarr[2 * self.nbin:3 * self.nbin] = self.fxw[: self.nbin]
            realarr[3 * self.nbin:4 * self.nbin] = self.nbw[: self.nbin]
            realarr[4 * self.nbin:5 * self.nbin] = self.cnbw[: self.nbin]
            realarr[8 * self.nbin:9 * self.nbin] = self.pnw[: self.nbin]
            realarr[6 * self.nbin:7 * self.nbin] = self.cfxw[: self.nbin]

        realarr[10 * self.nbin:10 * self.nbin + 4] = np.array([self.xmin, self.xmax, float(self.nsw), self.xstep], dtype=np.float64)
        return intarr, realarr, logarr

    def conv_int_real_arrays_fc(self, intarr, realarr, logarr):
        self.use_weight = bool(logarr[0])
        self.is_spline_prepared = bool(logarr[1])
        self.flaglog = int(intarr[0])
        self.nbin = int(intarr[1])
        self.ns = int(intarr[2])

        if self.xb is None or len(self.xb) != self.nbin:
            init_s1_hst_basic(self, 0.0, 1.0, self.nbin, self.use_weight)
            self.cfx = np.zeros((self.nbin,), dtype=np.float64)
            self.cnb = np.zeros((self.nbin,), dtype=np.int64)
            self.pn = np.zeros((self.nbin,), dtype=np.float64)
            self.y2_y = np.zeros((self.nbin,), dtype=np.float64)
            self.nb = np.zeros((self.nbin,), dtype=np.int64)
            if self.use_weight:
                self.cfxw = np.zeros((self.nbin,), dtype=np.float64)
                self.cnbw = np.zeros((self.nbin,), dtype=np.float64)
                self.pnw = np.zeros((self.nbin,), dtype=np.float64)
            else:
                self.cfxw = np.zeros((self.nbin,), dtype=np.float64)
                self.cnbw = np.zeros((self.nbin,), dtype=np.float64)
                self.pnw = np.zeros((self.nbin,), dtype=np.float64)

        self.nb[: self.nbin] = np.asarray(intarr[3:3 + self.nbin], dtype=np.int64)
        self.cnb[: self.nbin] = np.asarray(intarr[3 + self.nbin:3 + 2 * self.nbin], dtype=np.int64)

        self.xb[: self.nbin] = np.asarray(realarr[0:self.nbin], dtype=np.float64)
        self.fx[: self.nbin] = np.asarray(realarr[self.nbin:2 * self.nbin], dtype=np.float64)
        self.cfx[: self.nbin] = np.asarray(realarr[5 * self.nbin:6 * self.nbin], dtype=np.float64)
        self.pn[: self.nbin] = np.asarray(realarr[7 * self.nbin:8 * self.nbin], dtype=np.float64)

        if self.is_spline_prepared:
            if self.y2 is None or len(self.y2) != self.nbin:
                self.y2 = np.zeros((self.nbin,), dtype=np.float64)
            self.y2[: self.nbin] = np.asarray(realarr[9 * self.nbin:10 * self.nbin], dtype=np.float64)

        if self.use_weight:
            self.fxw[: self.nbin] = np.asarray(realarr[2 * self.nbin:3 * self.nbin], dtype=np.float64)
            self.nbw[: self.nbin] = np.asarray(realarr[3 * self.nbin:4 * self.nbin], dtype=np.float64)
            self.cnbw[: self.nbin] = np.asarray(realarr[4 * self.nbin:5 * self.nbin], dtype=np.float64)
            self.pnw[: self.nbin] = np.asarray(realarr[8 * self.nbin:9 * self.nbin], dtype=np.float64)
            self.cfxw[: self.nbin] = np.asarray(realarr[6 * self.nbin:7 * self.nbin], dtype=np.float64)

        self.xmin = float(realarr[10 * self.nbin + 0])
        self.xmax = float(realarr[10 * self.nbin + 1])
        self.nsw = float(realarr[10 * self.nbin + 2])
        self.xstep = float(realarr[10 * self.nbin + 3])

    def to_dict(self):
        return {
            "nbin": self.nbin,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "xstep": self.xstep,
            "ns": self.ns,
            "nsw": float(self.nsw),
            "flaglog": self.flaglog,
            "use_weight": bool(self.use_weight),
            "is_spline_prepared": bool(self.is_spline_prepared),
            "xb": None if self.xb is None else self.xb.copy(),
            "fx": None if self.fx is None else self.fx.copy(),
            "fxw": None if self.fxw is None else self.fxw.copy(),
            "nbw": None if self.nbw is None else self.nbw.copy(),
            "cnbw": None if self.cnbw is None else self.cnbw.copy(),
            "cfx": None if self.cfx is None else self.cfx.copy(),
            "cfxw": None if self.cfxw is None else self.cfxw.copy(),
            "pn": None if self.pn is None else self.pn.copy(),
            "pnw": None if self.pnw is None else self.pnw.copy(),
            "nb": None if self.nb is None else self.nb.copy(),
            "cnb": None if self.cnb is None else self.cnb.copy(),
            "y2": None if self.y2 is None else self.y2.copy(),
            "y2_y": None if self.y2_y is None else self.y2_y.copy(),
        }

    def from_dict(self, d):
        self.init(d["xmin"], d["xmax"], int(d["nbin"]), int(d["flaglog"]), bool(d["use_weight"]))
        self.xstep = float(d["xstep"])
        self.ns = int(d["ns"])
        self.nsw = float(d["nsw"])
        self.is_spline_prepared = bool(d["is_spline_prepared"])
        if d.get("xb") is not None:
            self.xb[:] = d["xb"]
        if d.get("fx") is not None:
            self.fx[:] = d["fx"]
        if d.get("fxw") is not None:
            self.fxw[:] = d["fxw"]
        if d.get("nbw") is not None:
            self.nbw[:] = d["nbw"]
        if d.get("cnbw") is not None:
            self.cnbw[:] = d["cnbw"]
        if d.get("cfx") is not None:
            self.cfx[:] = d["cfx"]
        if d.get("cfxw") is not None:
            self.cfxw[:] = d["cfxw"]
        if d.get("pn") is not None:
            self.pn[:] = d["pn"]
        if d.get("pnw") is not None:
            self.pnw[:] = d["pnw"]
        if d.get("nb") is not None:
            self.nb[:] = d["nb"]
        if d.get("cnb") is not None:
            self.cnb[:] = d["cnb"]
        if self.is_spline_prepared and d.get("y2") is not None:
            self.y2 = np.asarray(d["y2"], dtype=np.float64).copy()
        if d.get("y2_y") is not None:
            self.y2_y[:] = d["y2_y"]


def get_fc(x, n: int, fc: StsFCType):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    if fc.flaglog == fc_spacing_log:
        xbg = math.log10(fc.xmin)
        xed = math.log10(fc.xmax)
        xtp = np.log10(xv)
    else:
        xbg = fc.xmin
        xed = fc.xmax
        xtp = xv.copy()

    set_range(fc.xb, fc.nbin, xbg, xed, 0)
    if fc.flaglog == fc_spacing_log:
        fc.xb = 10.0 ** fc.xb
    fc.xstep = (xed - xbg) / float(fc.nbin)

    nb, ns = get_dstr_num_in_each_bin(xtp, n, xbg, fc.xstep, fc.nbin)
    fc.nb[:] = nb
    fc.ns = ns

    if fc.ns > 0 and fc.xstep != 0.0:
        fc.fx[:] = fc.nb.astype(np.float64) / float(fc.ns) / float(fc.xstep)
    else:
        fc.fx[:] = 0.0

    for i in range(fc.nbin):
        if fc.fx[i] != 0.0 and fc.nb[i] > 0:
            fc.pn[i] = fc.fx[i] / math.sqrt(float(fc.nb[i]))
        else:
            fc.pn[i] = 0.0

    if fc.nbin > 0:
        fc.cnb[0] = fc.nb[0]
        fc.cfx[0] = fc.fx[0] * fc.xstep
        for i in range(fc.nbin - 1):
            fc.cnb[i + 1] = fc.cnb[i] + fc.nb[i + 1]
            fc.cfx[i + 1] = fc.cfx[i] + fc.fx[i + 1] * fc.xstep


def get_fc_weight(x, w, n: int, fc: StsFCType):
    if not fc.use_weight:
        raise ValueError("fc.use_weight is False")
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    wv = np.asarray(w, dtype=np.float64).reshape((n,))
    if fc.flaglog == fc_spacing_log:
        xbg = math.log10(fc.xmin)
        xed = math.log10(fc.xmax)
        xtp = np.log10(xv)
    else:
        xbg = fc.xmin
        xed = fc.xmax
        xtp = xv.copy()

    set_range(fc.xb, fc.nbin, xbg, xed, 0)
    if fc.flaglog == fc_spacing_log:
        fc.xb = 10.0 ** fc.xb
    fc.xstep = (xed - xbg) / float(fc.nbin)

    nb, ns = get_dstr_num_in_each_bin(xtp, n, xbg, fc.xstep, fc.nbin)
    fc.nb[:] = nb
    fc.ns = ns

    nbw, nsw = get_dstr_num_in_each_bin_weight(xtp, wv, n, xbg, fc.xstep, fc.nbin)
    fc.nbw[:] = nbw
    fc.nsw = nsw

    if fc.ns > 0 and fc.xstep != 0.0:
        fc.fx[:] = fc.nb.astype(np.float64) / float(fc.ns) / float(fc.xstep)
    else:
        fc.fx[:] = 0.0

    if fc.nsw > 0.0 and fc.xstep != 0.0:
        fc.fxw[:] = fc.nbw / float(fc.nsw) / float(fc.xstep)
    else:
        fc.fxw[:] = 0.0

    for i in range(fc.nbin):
        if fc.fx[i] != 0.0 and fc.nb[i] > 0:
            fc.pn[i] = fc.fx[i] / math.sqrt(float(fc.nb[i]))
        else:
            fc.pn[i] = 0.0
        if fc.fxw[i] != 0.0 and fc.nb[i] > 0:
            fc.pnw[i] = fc.fxw[i] / math.sqrt(float(fc.nb[i]))
        else:
            fc.pnw[i] = 0.0

    if fc.nbin > 0:
        fc.cnb[0] = fc.nb[0]
        fc.cnbw[0] = fc.nbw[0]
        fc.cfx[0] = fc.fx[0] * fc.xstep
        fc.cfxw[0] = fc.fxw[0] * fc.xstep
        for i in range(fc.nbin - 1):
            fc.cnb[i + 1] = fc.cnb[i] + fc.nb[i + 1]
            fc.cnbw[i + 1] = fc.cnbw[i] + fc.nbw[i + 1]
            fc.cfx[i + 1] = fc.cfx[i] + fc.fx[i + 1] * fc.xstep
            fc.cfxw[i + 1] = fc.cfxw[i] + fc.fxw[i + 1] * fc.xstep


def output_sts_data_weight(x, w, n: int, xmin: float, xmax: float, rn: int, flaglog: int, fn: str):
    fc = StsFCType().init(xmin, xmax, rn, flaglog, use_weight=True)
    get_fc_weight(np.asarray(x)[:n], np.asarray(w)[:n], n, fc)
    fc.output_fc(fn.strip())
    fc.output_nfc(fn.strip())


def output_sts_data_weight_auto(x, w, n: int, rn: int, flaglog: int, fn: str):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    xmin = float(np.min(xv))
    xmax = float(np.max(xv))
    output_sts_data_weight(xv, np.asarray(w)[:n], n, xmin, xmax, rn, flaglog, fn)


def output_sts_data(x, n: int, xmin: float, xmax: float, rn: int, flaglog: int, fn: str):
    fc = StsFCType().init(xmin, xmax, rn, flaglog, use_weight=False)
    get_fc(np.asarray(x)[:n], n, fc)
    fc.output_fc(fn.strip())
    fc.output_nfc(fn.strip())


def output_sts_data_auto(x, n: int, rn: int, flaglog: int, fn: str):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    xmin = float(np.min(xv))
    xmax = float(np.max(xv))
    output_sts_data(xv, n, xmin, xmax, rn, flaglog, fn)


def get_sts_data_weight(x, w, n: int, xmin: float, xmax: float, rn: int, flaglog: int):
    fc = StsFCType().init(xmin, xmax, rn, flaglog, use_weight=True)
    get_fc_weight(np.asarray(x)[:n], np.asarray(w)[:n], n, fc)
    return fc


def get_sts_data_weight_auto(x, w, n: int, rn: int, flaglog: int):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    xmin = float(np.min(xv))
    xmax = float(np.max(xv))
    return get_sts_data_weight(xv, np.asarray(w)[:n], n, xmin, xmax, rn, flaglog)


def get_sts_data(x, n: int, xmin: float, xmax: float, rn: int, flaglog: int):
    fc = StsFCType().init(xmin, xmax, rn, flaglog, use_weight=False)
    get_fc(np.asarray(x)[:n], n, fc)
    return fc


def get_sts_data_auto(x, n: int, rn: int, flaglog: int):
    xv = np.asarray(x, dtype=np.float64).reshape((n,))
    xmin = float(np.min(xv))
    xmax = float(np.max(xv))
    return get_sts_data(xv, n, xmin, xmax, rn, flaglog)


def get_percentage_position(fc: StsFCType, percent, n: int, method: int = 1):
    per = np.asarray(percent, dtype=np.float64).reshape((n,))
    pos = np.zeros((4, n), dtype=np.float64)
    for i in range(n):
        pos[0, i] = search_for_position(fc.cfx, fc.xb, fc.nbin, per[i])
        pos[1, i] = search_for_position(fc.cfxw, fc.xb, fc.nbin, per[i])
    return pos

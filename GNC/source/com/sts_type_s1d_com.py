from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from md_s1d_basic_type import S1DBasicType


@dataclass
class S1DType(S1DBasicType):
    type_int_size: int = 0
    type_real_size: int = 0
    type_log_size: int = 0
    pn: Optional[np.ndarray] = field(default=None, repr=False)

    def init(self, xmin: float, xmax: float, n: int, bin_type: int) -> None:
        super().init(xmin, xmax, n, bin_type)
        self.pn = np.zeros(self.nbin, dtype=np.float64)
        self.type_int_size = 2
        self.type_real_size = self.nbin * 3 + 3
        self.type_log_size = 1

    def write_unformatted(self) -> Tuple[Tuple[int, float, float, int], Tuple[np.ndarray, np.ndarray]]:
        n = self.nbin
        header = (n, float(self.xmin), float(self.xmax), int(self.bin_type))
        payload = (self.xb[:n].copy(), self.fx[:n].copy())
        return header, payload

    def read_unformatted(
        self,
        header: Tuple[int, float, float, int],
        payload: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        n, xmin, xmax, bin_type = header
        self.init(float(xmin), float(xmax), int(n), int(bin_type))
        xb, fx = payload
        self.xb[: self.nbin] = np.asarray(xb, dtype=np.float64)[: self.nbin]
        self.fx[: self.nbin] = np.asarray(fx, dtype=np.float64)[: self.nbin]

    def conv_to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        intarr = np.zeros(self.type_int_size, dtype=np.int64)
        logarr = np.zeros(self.type_log_size, dtype=np.bool_)
        realarr = np.zeros(self.type_real_size, dtype=np.float64)

        intarr[0] = int(self.nbin)
        intarr[1] = int(self.bin_type)

        logarr[0] = bool(self.is_spline_prepared)

        n = self.nbin
        realarr[0:n] = self.xb[:n]
        realarr[n:2 * n] = self.fx[:n]

        if self.is_spline_prepared and self.y2 is not None:
            realarr[2 * n:3 * n] = self.y2[:n]

        realarr[3 * n:3 * n + 3] = np.array([self.xmin, self.xmax, self.xstep], dtype=np.float64)
        return intarr, realarr, logarr

    def conv_from_arrays(self, intarr: np.ndarray, realarr: np.ndarray, logarr: np.ndarray) -> None:
        self.nbin = int(intarr[0])
        self.bin_type = int(intarr[1])
        self.is_spline_prepared = bool(logarr[0])

        if self.xb is None or len(self.xb) != self.nbin:
            self.xb = np.zeros(self.nbin, dtype=np.float64)
        if self.fx is None or len(self.fx) != self.nbin:
            self.fx = np.zeros(self.nbin, dtype=np.float64)

        n = self.nbin
        self.xb[:n] = np.asarray(realarr[0:n], dtype=np.float64)
        self.fx[:n] = np.asarray(realarr[n:2 * n], dtype=np.float64)

        if self.is_spline_prepared:
            if self.y2 is None or len(self.y2) != n:
                self.y2 = np.zeros(n, dtype=np.float64)
            self.y2[:n] = np.asarray(realarr[2 * n:3 * n], dtype=np.float64)

        self.xmin = float(realarr[3 * n])
        self.xmax = float(realarr[3 * n + 1])
        self.xstep = float(realarr[3 * n + 2])

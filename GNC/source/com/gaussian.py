import math
import random


_gauss_flag = 0
_gauss_v1 = 0.0
_gauss_v2 = 0.0
_gauss_s = 0.0


def gen_gaussian(sigma: float) -> float:
    global _gauss_flag, _gauss_v1, _gauss_v2, _gauss_s
    if _gauss_flag == 0:
        _gauss_s = 0.0
        while _gauss_s >= 1.0 or _gauss_s == 0.0:
            u1 = random.random()
            u2 = random.random()
            _gauss_v1 = 2.0 * u1 - 1.0
            _gauss_v2 = 2.0 * u2 - 1.0
            _gauss_s = _gauss_v1 * _gauss_v1 + _gauss_v2 * _gauss_v2
        x = _gauss_v1 * math.sqrt(-2.0 * math.log(_gauss_s) / _gauss_s)
        _gauss_flag = 1
    else:
        x = _gauss_v2 * math.sqrt(-2.0 * math.log(_gauss_s) / _gauss_s)
        _gauss_flag = 0
    return x * float(sigma)


def gen_gaussian_correlate(coeff: float):
    y1p = gen_gaussian(1.0)
    y2p = gen_gaussian(1.0)
    y1 = y1p
    c = float(coeff)
    if abs(c) <= 1.0:
        y2 = y1p * c + y2p * math.sqrt(1.0 - c * c)
    else:
        y2 = (c / abs(c)) * y1
    return y1, y2

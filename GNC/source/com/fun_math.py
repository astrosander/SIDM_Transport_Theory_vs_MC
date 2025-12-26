import numpy as np


def farravg(arr, n: int):
    a = np.asarray(arr, dtype=np.float64)
    return float(np.sum(a) / float(n))


def farrsct(arr, n: int):
    a = np.asarray(arr, dtype=np.float64)
    mean = float(np.sum(a) / float(n))
    return float(np.sqrt(np.sum((a - mean) ** 2) / float(n - 1)))


def farrsct2(arr, mean, n: int):
    a = np.asarray(arr, dtype=np.float64)
    m = float(mean)
    return float(np.sqrt(np.sum((a - m) ** 2) / float(n - 1)))

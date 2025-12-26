import math
import random
import time


def rnd(min_val: float, max_val: float) -> float:
    temp = random.random()
    if min_val < max_val:
        return temp * (max_val - min_val) + min_val
    if min_val > max_val:
        return temp * (min_val - max_val) + max_val
    return float(min_val)


def set_system_random_seed() -> None:
    random.seed(time.time_ns())


def same_random_seed(seed_value: int) -> None:
    random.seed(int(seed_value))


def rndI(min_val: int, max_val: int) -> int:
    temp = random.random()
    if min_val < max_val:
        return int(temp * (max_val - min_val + 1) + min_val)
    if min_val > max_val:
        return int(temp * (min_val - max_val + 1) + max_val)
    return int(min_val)


def fPowerLaw(yta: float, xmin: float, xmax: float) -> float:
    if yta == -1.0:
        return math.exp(rnd(math.log(xmin), math.log(xmax)))
    return rnd(xmin ** (1.0 + yta), xmax ** (1.0 + yta)) ** (1.0 / (1.0 + yta))

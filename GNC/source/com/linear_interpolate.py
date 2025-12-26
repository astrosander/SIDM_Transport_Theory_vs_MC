import numpy as np


def linear_int(x, y, n: int, xev: float):
    if n <= 1:
        raise ValueError(f"linear_int, n<=1: {n}")
    x = np.asarray(x, dtype=np.float64).reshape((n,))
    y = np.asarray(y, dtype=np.float64).reshape((n,))
    xev = float(xev)

    if xev < x[0]:
        return float(y[0] - (y[1] - y[0]) / (x[1] - x[0]) * (x[0] - xev))
    if xev > x[n - 1]:
        return float(y[n - 1] + (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) * (xev - x[n - 1]))
    if xev == x[n - 1]:
        return float(y[n - 1])

    for i in range(n - 1):
        if x[i] >= x[i + 1]:
            raise ValueError(f"bad input in linear_int: x[{i}]>=x[{i+1}] ({x[i]}, {x[i+1]})")
        if x[i] < xev and x[i + 1] > xev:
            return float(y[i + 1] - (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x[i + 1] - xev))
        if x[i] == xev:
            return float(y[i])

    raise RuntimeError("linear_int: no interval found")


def linear_int_2d(xmin: float, ymin: float, nx: int, ny: int, xstep: float, ystep: float, z, vx: float, vy: float):
    z = np.asarray(z, dtype=np.float64).reshape((nx, ny))
    xmin = float(xmin)
    ymin = float(ymin)
    xstep = float(xstep)
    ystep = float(ystep)
    vx = float(vx)
    vy = float(vy)

    rdx = (vx - xmin) / xstep
    rdy = (vy - ymin) / ystep
    idx = int(rdx) + 1
    idy = int(rdy) + 1

    if idx < 0 or idx > nx or idy < 0 or idy > ny:
        return 0.0

    if idx == nx:
        idxn = idx - 1
        idxm = idx
    elif idx == 1:
        idxn = 1
        idxm = 2
    else:
        idxn = idx
        idxm = idx + 1

    if idy == ny:
        idyn = idy - 1
        idym = idy
    elif idy == 1:
        idyn = 1
        idym = 2
    else:
        idyn = idy
        idym = idy + 1

    y1 = z[idxn - 1, idyn - 1]
    y2 = z[idxm - 1, idyn - 1]
    y3 = z[idxm - 1, idym - 1]
    y4 = z[idxn - 1, idym - 1]

    t = rdx - idxn + 1.0
    u = rdy - idyn + 1.0

    return float((1.0 - t) * (1.0 - u) * y1 + t * (1.0 - u) * y2 + t * u * y3 + (1.0 - t) * u * y4)


def linear_int_arb(x, y, n: int, xev: float):
    if n <= 1:
        raise ValueError(f"linear_int_arb, n<=1: {n}")
    x = np.asarray(x, dtype=np.float64).reshape((n,))
    y = np.asarray(y, dtype=np.float64).reshape((n,))
    xev = float(xev)

    if x[0] > x[n - 1]:
        x1, xn = x[n - 1], x[0]
        y1, yn = y[n - 1], y[0]
        increase = False
    else:
        x1, xn = x[0], x[n - 1]
        y1, yn = y[0], y[n - 1]
        increase = True

    if xev < x1 or xev > xn:
        raise ValueError(f"xev out of range: {xev} not in [{x1}, {xn}]")

    if increase:
        if xev == xn:
            return float(yn)
        for i in range(n - 1):
            if x[i] < xev and x[i + 1] > xev:
                return float(y[i + 1] - (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x[i + 1] - xev))
            if x[i] == xev:
                return float(y[i])
    else:
        if xev == xn:
            return float(yn)
        for i in range(n - 1, 0, -1):
            if x[i] < xev and x[i - 1] > xev:
                return float(y[i - 1] - (y[i - 1] - y[i]) / (x[i - 1] - x[i]) * (x[i - 1] - xev))
            if x[i] == xev:
                return float(y[i])

    raise RuntimeError("linear_int_arb: no interval found")

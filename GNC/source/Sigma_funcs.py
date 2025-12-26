import math

def get_sigma0(energyx, funcs, emin_factor, fgx_g0):
    int_out = 0.0
    idid = 0

    if energyx / emin_factor > 1.0 + 1e-7:
        def fcn(x):
            fx = funcs(x * energyx)
            if math.isnan(fx) or math.isnan(x):
                raise RuntimeError(f"{emin_factor} {x * energyx} {fx}")
            return fx

        int_out, idid = my_integral_none(emin_factor / energyx, 1.0, fcn)

        if idid < 0:
            raise RuntimeError(f"error! idid={idid} energyx={energyx} emin_factor={emin_factor}")

    sigma = int_out + fgx_g0 / energyx
    if idid == -2:
        print("sigma0=", sigma)
    return sigma

def get_sigma_funcs_cfs_grid(energyx, jum, funcs, cfs_grid, nx, ny, jmin, jmax):
    if jum > 1.0:
        raise RuntimeError(f"error, jum>1: {jum}")

    ecc = math.sqrt(1.0 - jum * jum)
    dsmin = -7.0
    dsmax = math.log10(2.0 / (1.0 - ecc) - 1.0)

    nstep = nx * 6
    sstep = (dsmax - dsmin) / float(nstep - 1)

    yout = 0.0
    for i in range(1, nstep + 1):
        ds = (dsmax - dsmin) * float(i - 1) / float(nstep - 1) + dsmin
        ss = math.log10(10.0 ** ds + 1.0)
        cfs = get_cfs(jum, ds, cfs_grid, nx, ny, jmin, jmax, dsmin, dsmax)
        fx = funcs((10.0 ** ss) * energyx)
        yout += fx * cfs * (10.0 ** ss - 1.0) * sstep

    sigma = (1.0 / math.pi) * yout * math.log(10.0)
    return sigma

def _return_idxy_nearest(x, y, xmin, xmax, ymin, ymax, nx, ny):
    if xmax == xmin:
        idx = 0
    else:
        idx = int(round((x - xmin) / (xmax - xmin) * (nx - 1)))
    if ymax == ymin:
        idy = 0
    else:
        idy = int(round((y - ymin) / (ymax - ymin) * (ny - 1)))
    idx = max(0, min(nx - 1, idx))
    idy = max(0, min(ny - 1, idy))
    return idx, idy

def get_cfs(jum, y, cfs, nx, ny, jmin, jmax, dsmin, dsmax):
    x = math.log10(jum)
    xmin, xmax = jmin, jmax
    ymin, ymax = dsmin, dsmax

    if x < xmin:
        x = xmin
    if x > xmax:
        print("warnning, jmax, j=", jum, 10.0 ** x)
        x = xmax

    if y < ymin:
        if abs(y - ymin) > 1e-6:
            print("warnning, smin,s=", ymin, y)
        y = ymin
    if y > ymax:
        if abs(y - ymax) > 1e-6:
            print("warnning, smax,s=", ymax, y)
        y = ymax

    idx, idy = _return_idxy_nearest(x, y, xmin, xmax, ymin, ymax, nx, ny)
    return cfs[idx][idy]

def get_cfs_blinear(jum, y, cfs, nx, ny, jmin, jmax, dsmin, dsmax):
    x = math.log10(jum)

    if x > jmax:
        print("warnning, jmax, j=", jum, 10.0 ** x)

    if y > dsmax:
        if abs(y - dsmax) > 1e-6:
            print("warnning, dsmax,s=", dsmax, y)
        y = dsmax

    x0, y0 = jmin, dsmin
    xstep = (jmax - jmin) / float(nx - 1)
    ystep = (dsmax - dsmin) / float(ny - 1)

    tx = (x - x0) / xstep if xstep != 0.0 else 0.0
    ty = (y - y0) / ystep if ystep != 0.0 else 0.0

    ix = int(math.floor(tx))
    iy = int(math.floor(ty))

    ix = max(0, min(nx - 2, ix))
    iy = max(0, min(ny - 2, iy))

    fx = tx - ix
    fy = ty - iy

    v00 = cfs[ix][iy]
    v10 = cfs[ix + 1][iy]
    v01 = cfs[ix][iy + 1]
    v11 = cfs[ix + 1][iy + 1]

    return (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11

def get_sigma_funcs_cfs_rk(energyx, jum, funcs, cfs_grid, nx, ny, jmin, jmax, dsmin_value):
    dsmin = dsmin_value
    ecc = math.sqrt(1.0 - jum * jum)
    dsmax = math.log10(2.0 / (1.0 - ecc) - 1.0)

    a = math.log10(1.0 + 10.0 ** dsmin)
    b = math.log10(2.0 / (1.0 - ecc))

    def fcn(x):
        cfs = get_cfs_blinear(jum, math.log10(10.0 ** x - 1.0), cfs_grid, nx, ny, jmin, jmax, dsmin, dsmax)
        fx = funcs((10.0 ** x) * energyx)
        return fx * cfs * (10.0 ** x)

    yout, idid = my_integral_acc(a, b, fcn, 1e-13, 1e-12)
    sigma = (1.0 / math.pi) * yout * math.log(10.0)
    return sigma

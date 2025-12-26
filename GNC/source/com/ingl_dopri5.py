import math
import numpy as np


def cdopri():
    c2 = 0.2
    c3 = 0.3
    c4 = 0.8
    c5 = 8.0 / 9.0

    a21 = 0.2
    a31 = 3.0 / 40.0
    a32 = 9.0 / 40.0
    a41 = 44.0 / 45.0
    a42 = -56.0 / 15.0
    a43 = 32.0 / 9.0
    a51 = 19372.0 / 6561.0
    a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0
    a54 = -212.0 / 729.0
    a61 = 9017.0 / 3168.0
    a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0
    a64 = 49.0 / 176.0
    a65 = -5103.0 / 18656.0
    a71 = 35.0 / 384.0
    a73 = 500.0 / 1113.0
    a74 = 125.0 / 192.0
    a75 = -2187.0 / 6784.0
    a76 = 11.0 / 84.0

    e1 = 71.0 / 57600.0
    e3 = -71.0 / 16695.0
    e4 = 71.0 / 1920.0
    e5 = -17253.0 / 339200.0
    e6 = 22.0 / 525.0
    e7 = -1.0 / 40.0

    d1 = -12715105075.0 / 11282082432.0
    d3 = 87487479700.0 / 32700410799.0
    d4 = -10690763975.0 / 1880347072.0
    d5 = 701980252875.0 / 199316789632.0
    d6 = -1453857185.0 / 822651844.0
    d7 = 69997945.0 / 29380423.0

    return (
        c2, c3, c4, c5,
        e1, e3, e4, e5, e6, e7,
        a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
        a61, a62, a63, a64, a65, a71, a73, a74, a75, a76,
        d1, d3, d4, d5, d6, d7
    )


def _as_array(x, n):
    a = np.asarray(x, dtype=np.float64)
    if a.shape == (n,):
        return a.copy()
    return a.reshape((n,)).astype(np.float64, copy=False).copy()


def _get_scalar_or_vec(v, n, itol):
    if itol == 0:
        if np.isscalar(v):
            return float(v), None
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        return float(a[0]), None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return None, np.full((n,), float(a[0]), dtype=np.float64)
    if a.size != n:
        raise ValueError("tolerance vector has wrong length")
    return None, a.astype(np.float64, copy=False)


def _fcn_eval(fcn, n, x, y, rpar, ipar):
    try:
        out = fcn(n, x, y, rpar, ipar)
    except TypeError:
        try:
            out = fcn(x, y, rpar, ipar)
        except TypeError:
            try:
                out = fcn(x, y)
            except TypeError:
                out = fcn(n, x, y)
    return _as_array(out, n)


def hinit(n, fcn, x, y, xend, posneg, f0, iord, hmax, atol, rtol, itol, rpar, ipar):
    atols, atolv = _get_scalar_or_vec(atol, n, itol)
    rtols, rtolv = _get_scalar_or_vec(rtol, n, itol)

    dnf = 0.0
    dny = 0.0

    if itol == 0:
        for i in range(n):
            sk = atols + rtols * abs(y[i])
            dnf += (f0[i] / sk) ** 2
            dny += (y[i] / sk) ** 2
    else:
        for i in range(n):
            sk = atolv[i] + rtolv[i] * abs(y[i])
            dnf += (f0[i] / sk) ** 2
            dny += (y[i] / sk) ** 2

    if dnf <= 1e-10 or dny <= 1e-10:
        h = 1e-6
    else:
        h = math.sqrt(dny / dnf) * 0.01

    h = min(h, hmax)
    h = math.copysign(h, posneg)

    y1 = y + h * f0
    f1 = _fcn_eval(fcn, n, x + h, y1, rpar, ipar)

    der2 = 0.0
    if itol == 0:
        for i in range(n):
            sk = atols + rtols * abs(y[i])
            der2 += ((f1[i] - f0[i]) / sk) ** 2
    else:
        for i in range(n):
            sk = atolv[i] + rtolv[i] * abs(y[i])
            der2 += ((f1[i] - f0[i]) / sk) ** 2

    der2 = math.sqrt(der2) / abs(h) if h != 0.0 else float("inf")
    der12 = max(abs(der2), math.sqrt(dnf))

    if der12 <= 1e-15:
        h1 = max(1e-6, abs(h) * 1e-3)
    else:
        h1 = (0.01 / der12) ** (1.0 / float(iord))

    h = min(100.0 * abs(h), h1, hmax)
    return math.copysign(h, posneg), 1


def contd5(ii, x, con, icomp, nd, xold, h):
    jpos = -1
    for j, c in enumerate(icomp):
        if c == ii:
            jpos = j
            break
    if jpos < 0:
        raise ValueError("no dense output available for component")
    theta = (x - xold) / h
    theta1 = 1.0 - theta
    i = jpos
    return con[i] + theta * (
        con[nd + i] + theta1 * (
            con[2 * nd + i] + theta * (con[3 * nd + i] + theta1 * con[4 * nd + i])
        )
    )


def _call_solout(solout, nr, xold, x, y, n, con, icomp, nd, rpar, ipar, irtrn):
    if solout is None:
        return irtrn
    args_list = [
        (nr, xold, x, y, n, con, icomp, nd, rpar, ipar, irtrn),
        (nr, xold, x, y, n, con, icomp, nd, rpar, ipar),
        (nr, xold, x, y, con, icomp),
        (nr, xold, x, y),
    ]
    for args in args_list:
        try:
            res = solout(*args)
            if res is None:
                return irtrn
            if isinstance(res, (int, np.integer)):
                return int(res)
            try:
                return int(res)
            except Exception:
                return irtrn
        except TypeError:
            continue
    return irtrn


def dopcor(
    n, fcn, x, y, xend, hmax, h, rtol, atol, itol, iprint,
    solout, iout, nmax, uround, meth, nstiff, safe, beta, fac1, fac2,
    rpar, ipar, nrd, icomp
):
    (
        c2, c3, c4, c5,
        e1, e3, e4, e5, e6, e7,
        a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
        a61, a62, a63, a64, a65, a71, a73, a74, a75, a76,
        d1, d3, d4, d5, d6, d7
    ) = cdopri()

    atols, atolv = _get_scalar_or_vec(atol, n, itol)
    rtols, rtolv = _get_scalar_or_vec(rtol, n, itol)

    nonsti = 0
    facold = 1e-4
    expo1 = 0.2 - beta * 0.75
    facc1 = 1.0 / fac1
    facc2 = 1.0 / fac2
    posneg = 1.0 if (xend - x) >= 0.0 else -1.0
    last = False
    hlamb = 0.0
    iasti = 0

    k1 = _fcn_eval(fcn, n, x, y, rpar, ipar)
    nfcn = 1
    nstep = 0
    naccpt = 0
    nrejct = 0
    reject = False

    hmax = abs(hmax)
    iord = 5
    if h == 0.0:
        h, extra = hinit(n, fcn, x, y, xend, posneg, k1, iord, hmax, atol, rtol, itol, rpar, ipar)
        nfcn += extra

    xold = x
    irtrn = 0
    con = np.zeros((5 * nrd,), dtype=np.float64) if (iout >= 2 and nrd > 0) else np.zeros((0,), dtype=np.float64)

    if iout != 0:
        irtrn = 1
        irtrn = _call_solout(solout, naccpt + 1, xold, x, y.copy(), n, con, icomp, nrd, rpar, ipar, irtrn)
        if irtrn < 0:
            return x, y, h, 2, nfcn, nstep, naccpt, nrejct, con, icomp, nrd, xold

    while True:
        if nstep > nmax:
            return x, y, h, -2, nfcn, nstep, naccpt, nrejct, con, icomp, nrd, xold
        if 0.1 * abs(h) <= abs(x) * uround:
            return x, y, h, -3, nfcn, nstep, naccpt, nrejct, con, icomp, nrd, xold
        if (x + 1.01 * h - xend) * posneg > 0.0:
            h = xend - x
            last = True

        nstep += 1

        if irtrn >= 2:
            k1 = _fcn_eval(fcn, n, x, y, rpar, ipar)
            nfcn += 1

        y1 = y + h * a21 * k1
        k2 = _fcn_eval(fcn, n, x + c2 * h, y1, rpar, ipar)
        y1 = y + h * (a31 * k1 + a32 * k2)
        k3 = _fcn_eval(fcn, n, x + c3 * h, y1, rpar, ipar)
        y1 = y + h * (a41 * k1 + a42 * k2 + a43 * k3)
        k4 = _fcn_eval(fcn, n, x + c4 * h, y1, rpar, ipar)
        y1 = y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k5 = _fcn_eval(fcn, n, x + c5 * h, y1, rpar, ipar)
        ysti = y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        xph = x + h
        k6 = _fcn_eval(fcn, n, xph, ysti, rpar, ipar)
        y1 = y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        k7 = _fcn_eval(fcn, n, xph, y1, rpar, ipar)
        nfcn += 6

        if iout >= 2 and nrd > 0:
            for j in range(nrd):
                i = icomp[j] - 1
                con[4 * nrd + j] = h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i] + d7 * k7[i])

        errv = (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7) * h

        err = 0.0
        if itol == 0:
            for i in range(n):
                sk = atols + rtols * max(abs(y[i]), abs(y1[i]))
                err += (errv[i] / sk) ** 2
        else:
            for i in range(n):
                sk = atolv[i] + rtolv[i] * max(abs(y[i]), abs(y1[i]))
                err += (errv[i] / sk) ** 2
        err = math.sqrt(err / float(n))

        fac11 = err ** expo1 if err > 0.0 else 0.0
        fac = fac11 / (facold ** beta)
        fac = max(facc2, min(facc1, fac / safe))
        hnew = h / fac

        if err <= 1.0:
            facold = max(err, 1e-4)
            naccpt += 1

            if (naccpt % nstiff == 0) or (iasti > 0):
                stnum = float(np.sum((k7 - k6) ** 2))
                stden = float(np.sum((y1 - ysti) ** 2))
                if stden > 0.0:
                    hlamb = h * math.sqrt(stnum / stden)
                if hlamb > 3.25:
                    nonsti = 0
                    iasti += 1
                    if iasti == 15:
                        return x, y, h, -4, nfcn, nstep, naccpt, nrejct, con, icomp, nrd, xold
                else:
                    nonsti += 1
                    if nonsti == 6:
                        iasti = 0

            if iout >= 2 and nrd > 0:
                for j in range(nrd):
                    i = icomp[j] - 1
                    yd0 = y[i]
                    ydiff = y1[i] - yd0
                    bspl = h * k1[i] - ydiff
                    con[j] = yd0
                    con[nrd + j] = ydiff
                    con[2 * nrd + j] = bspl
                    con[3 * nrd + j] = -h * k7[i] + ydiff - bspl

            k1 = k7
            y = y1
            xold = x
            x = xph

            if iout != 0:
                irtrn = _call_solout(solout, naccpt + 1, xold, x, y.copy(), n, con, icomp, nrd, rpar, ipar, irtrn)
                if irtrn < 0:
                    return x, y, hnew, 2, nfcn, nstep, naccpt, nrejct, con, icomp, nrd, xold
            else:
                irtrn = 0

            if last:
                return x, y, hnew, 1, nfcn, nstep, naccpt, nrejct, con, icomp, nrd, xold

            if abs(hnew) > hmax:
                hnew = posneg * hmax
            if reject:
                hnew = posneg * min(abs(hnew), abs(h))
            reject = False
        else:
            hnew = h / min(facc1, fac11 / safe if fac11 != 0.0 else facc1)
            reject = True
            if naccpt >= 1:
                nrejct += 1
            last = False

        h = hnew


def dopri5(
    n, fcn, x, y, xend,
    rtol, atol, itol=0,
    solout=None, iout=0,
    work=None, iwork=None,
    rpar=None, ipar=None
):
    y = _as_array(y, n)
    work = list(work) if work is not None else [0.0] * 21
    iwork = list(iwork) if iwork is not None else [0] * 21

    iprint = iwork[2] if len(iwork) > 2 and iwork[2] != 0 else 6

    nmax = iwork[0] if len(iwork) > 0 and iwork[0] != 0 else 100000
    meth = iwork[1] if len(iwork) > 1 and iwork[1] != 0 else 1

    nstiff = iwork[3] if len(iwork) > 3 else 0
    if nstiff == 0:
        nstiff = 1000
    if nstiff < 0:
        nstiff = nmax + 10

    nrd = iwork[4] if len(iwork) > 4 else 0
    if nrd < 0 or nrd > n:
        return x, y, -1, work, iwork, None

    if nrd == n:
        icomp = [i + 1 for i in range(n)]
    elif nrd > 0:
        need = 20 + nrd
        if len(iwork) < need + 1:
            iwork.extend([0] * (need + 1 - len(iwork)))
        icomp = [int(iwork[20 + j]) for j in range(nrd)]
        if any(c <= 0 or c > n for c in icomp):
            return x, y, -1, work, iwork, None
    else:
        icomp = []

    uround = work[0] if len(work) > 0 and work[0] != 0.0 else 2.3e-16
    safe = work[1] if len(work) > 1 and work[1] != 0.0 else 0.9
    fac1 = work[2] if len(work) > 2 and work[2] != 0.0 else 0.2
    fac2 = work[3] if len(work) > 3 and work[3] != 0.0 else 10.0
    if len(work) > 4 and work[4] != 0.0:
        beta = 0.0 if work[4] < 0.0 else float(work[4])
    else:
        beta = 0.04
    hmax = work[5] if len(work) > 5 and work[5] != 0.0 else (xend - x)
    h = work[6] if len(work) > 6 else 0.0

    rpar = [] if rpar is None else rpar
    ipar = [] if ipar is None else ipar

    x2, y2, h2, idid, nfcn, nstep, naccpt, nrejct, con, icomp2, nrd2, xold = dopcor(
        n, fcn, float(x), y, float(xend), float(hmax), float(h), rtol, atol, int(itol), int(iprint),
        solout, int(iout), int(nmax), float(uround), int(meth), int(nstiff), float(safe), float(beta), float(fac1), float(fac2),
        rpar, ipar, int(nrd), icomp
    )

    if len(work) < 7:
        work.extend([0.0] * (7 - len(work)))
    work[6] = float(h2)

    if len(iwork) < 20:
        iwork.extend([0] * (20 - len(iwork)))
    if len(iwork) < 21:
        iwork.append(0)

    need_stats = 20
    if len(iwork) < need_stats + 1:
        iwork.extend([0] * (need_stats + 1 - len(iwork)))
    iwork[16] = int(nfcn)
    iwork[17] = int(nstep)
    iwork[18] = int(naccpt)
    iwork[19] = int(nrejct)

    dense = None
    if iout >= 2 and nrd2 > 0:
        def _dense(ii, s):
            return contd5(int(ii), float(s), con, icomp2, nrd2, float(xold), float(x2 - xold))
        dense = _dense

    return x2, y2, idid, work, iwork, dense

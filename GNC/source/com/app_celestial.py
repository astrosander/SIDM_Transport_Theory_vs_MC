import math

PI = 3.141592653589793
TWOPI = 2.0 * PI
TINY = 1.0e-12


def _fmod(a, p):
    return math.fmod(a, p)


def _sign(a, b):
    return math.copysign(abs(a), b)


def _cbrt(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def mco_sine(x):
    if x > 0.0:
        x = _fmod(x, TWOPI)
    else:
        x = _fmod(x, TWOPI) + TWOPI
    cx = math.cos(x)
    if x > PI:
        sx = -math.sqrt(max(0.0, 1.0 - cx * cx))
    else:
        sx = math.sqrt(max(0.0, 1.0 - cx * cx))
    return sx, cx


def mco_sinh(x):
    sx = math.sinh(x)
    cx = math.sqrt(1.0 + sx * sx)
    return sx, cx


def orbel_zget(q):
    iflag = 0
    if q < 0.0:
        iflag = 1
        q = -q
    if q < 1.0e-3:
        z = q * (1.0 - (q * q / 3.0) * (1.0 - q * q))
    else:
        x = 0.5 * (3.0 * q + math.sqrt(9.0 * (q * q) + 4.0))
        tmp = _cbrt(x)
        z = tmp - 1.0 / tmp
    if iflag == 1:
        z = -z
        q = -q
    return z


def orbel_fget(e, capn, tiny=TINY):
    if capn < 0.0:
        tmp = -2.0 * capn / e + 1.8
        x = -math.log(tmp)
    else:
        tmp = 2.0 * capn / e + 1.8
        x = math.log(tmp)

    for _ in range(10):
        shx, chx = mco_sinh(x)
        esh = e * shx
        ech = e * chx
        f = esh - x - capn
        fp = ech - 1.0
        fpp = esh
        fppp = ech
        dx = -f / fp
        dx = -f / (fp + dx * fpp / 2.0)
        dx = -f / (fp + dx * fpp / 2.0 + dx * dx * fppp / 6.0)
        x_new = x + dx
        if abs(dx) <= tiny:
            return x_new
        x = x_new
    return x


def orbel_flon(e, capn, tiny=TINY):
    iflag = 0
    if capn < 0.0:
        iflag = 1
        capn = -capn

    a11 = 156.0
    a9 = 17160.0
    a7 = 1235520.0
    a5 = 51891840.0
    a3 = 1037836800.0
    b11 = 11.0 * a11
    b9 = 9.0 * a9
    b7 = 7.0 * a7
    b5 = 5.0 * a5
    b3 = 3.0 * a3

    a1 = 6227020800.0 * (1.0 - 1.0 / e)
    a0 = -6227020800.0 * capn / e
    b1 = a1

    a = 6.0 * (e - 1.0) / e
    b = -6.0 * capn / e
    sq = math.sqrt(0.25 * b * b + a * a * a / 27.0)
    biga = _cbrt(-0.5 * b + sq)
    bigb = -_cbrt(0.5 * b + sq)
    x = biga + bigb

    if capn < tiny:
        if iflag == 1:
            return -x
        return x

    for _ in range(10):
        x2 = x * x
        f = a0 + x * (a1 + x2 * (a3 + x2 * (a5 + x2 * (a7 + x2 * (a9 + x2 * (a11 + x2))))))
        fp = b1 + x2 * (b3 + x2 * (b5 + x2 * (b7 + x2 * (b9 + x2 * (b11 + 13.0 * x2)))))
        dx = -f / fp
        x_new = x + dx
        if abs(dx) <= tiny:
            x = x_new
            break
        x = x_new

    if iflag == 1:
        capn = -capn
        return -x
    return x


def orbel_fhybrid(e, n, tiny=TINY):
    abn = -n if n < 0.0 else n
    if abn < 0.636 * e - 0.6:
        return orbel_flon(e, n, tiny=tiny)
    return orbel_fget(e, n, tiny=tiny)


def mco_kep(e, oldl):
    if oldl >= 0.0:
        l = _fmod(oldl, TWOPI)
    else:
        l = _fmod(oldl, TWOPI) + TWOPI

    sign = 1.0
    if l > PI:
        l = TWOPI - l
        sign = -1.0

    piby2 = 0.5 * PI
    ome = 1.0 - e

    if l >= 0.45 or e < 0.55:
        if l < ome:
            u1 = ome
        else:
            if l > (PI - 1.0 - e):
                u1 = (l + e * PI) / (1.0 + e)
            else:
                u1 = l + e

        flag = u1 > piby2
        x = PI - u1 if flag else u1
        x2 = x * x
        sn = x * (1.0 + x2 * (-0.16605 + x2 * 0.00761))
        dsn = 1.0 + x2 * (-0.49815 + x2 * 0.03805)
        if flag:
            dsn = -dsn
        f2 = e * sn
        f0 = u1 - f2 - l
        f1 = 1.0 - e * dsn
        u2 = u1 - f0 / (f1 - 0.5 * f0 * f2 / f1)
    else:
        z1 = 4.0 * e + 0.5
        p = ome / z1
        q = 0.5 * l / z1
        p2 = p * p
        z2 = math.exp(math.log(math.sqrt(p2 * p + q * q) + q) / 1.5)
        u1 = 2.0 * q / (z2 + p + p2 / z2)
        z2 = u1 * u1
        z3 = z2 * z2
        u2 = u1 - 0.075 * u1 * z3 / (ome + z1 * z2 + 0.375 * z3)
        u2 = l + e * u2 * (3.0 - 4.0 * u2 * u2)

    bigg = u2 > piby2
    z3 = PI - u2 if bigg else u2

    big = z3 > (0.5 * piby2)
    x = piby2 - z3 if big else z3
    x2 = x * x

    ss = (x * x2 / 6.0) * (
        1.0
        - x2 / 20.0
        * (
            1.0
            - x2 / 42.0
            * (
                1.0
                - x2 / 72.0
                * (
                    1.0
                    - x2 / 110.0
                    * (
                        1.0
                        - x2 / 156.0
                        * (
                            1.0
                            - x2 / 210.0
                            * (1.0 - x2 / 272.0)
                        )
                    )
                )
            )
        )
    )

    cc = (x2 / 2.0) * (
        1.0
        - x2 / 12.0
        * (
            1.0
            - x2 / 30.0
            * (
                1.0
                - x2 / 56.0
                * (
                    1.0
                    - x2 / 90.0
                    * (
                        1.0
                        - x2 / 132.0
                        * (
                            1.0
                            - x2 / 182.0
                            * (
                                1.0
                                - x2 / 240.0
                                * (1.0 - x2 / 306.0)
                            )
                        )
                    )
                )
            )
        )
    )

    if big:
        z1 = cc + z3 - 1.0
        z2 = ss + z3 + 1.0 - piby2
    else:
        z1 = ss
        z2 = cc

    if bigg:
        z1 = 2.0 * u2 + z1 - PI
        z2 = 2.0 - z2

    f0 = l - u2 * ome - e * z1
    f1 = ome + e * z2
    f2 = 0.5 * e * (u2 - z1)
    f3 = (e / 6.0) * (1.0 - z2)
    z1v = f0 / f1
    z2v = f0 / (f2 * z1v + f1)
    return sign * (u2 + f0 / ((f3 * z1v + f2) * z2v + f1))


def mco_el2x(gm, q, e, i, p, n, l, tiny=TINY):
    g = p - n

    si, ci = mco_sine(i)
    sg, cg = mco_sine(g)
    sn, cn = mco_sine(n)

    z1 = cg * cn
    z2 = cg * sn
    z3 = sg * cn
    z4 = sg * sn

    d11 = z1 - z4 * ci
    d12 = z2 + z3 * ci
    d13 = sg * si
    d21 = -z3 - z2 * ci
    d22 = -z4 + z1 * ci
    d23 = cg * si

    a = q / (1.0 - e)

    if e < 1.0:
        romes = math.sqrt(1.0 - e * e)
        temp = mco_kep(e, l)
        se, ce = mco_sine(temp)
        z1 = a * (ce - e)
        z2 = a * romes * se
        temp2 = math.sqrt(gm / a) / (1.0 - e * ce)
        z3 = -se * temp2
        z4 = romes * ce * temp2
    else:
        if e == 1.0:
            ce = orbel_zget(l)
            z1 = q * (1.0 - ce * ce)
            z2 = 2.0 * q * ce
            z4 = math.sqrt(2.0 * gm / q) / (1.0 + ce * ce)
            z3 = -ce * z4
        else:
            romes = math.sqrt(e * e - 1.0)
            temp = orbel_fhybrid(e, l, tiny=tiny)
            se, ce = mco_sinh(temp)
            z1 = a * (ce - e)
            z2 = -a * romes * se
            temp2 = math.sqrt(gm / abs(a)) / (e * ce - 1.0)
            z3 = -se * temp2
            z4 = romes * ce * temp2

    x = d11 * z1 + d21 * z2
    y = d12 * z1 + d22 * z2
    z = d13 * z1 + d23 * z2
    u = d11 * z3 + d21 * z4
    v = d12 * z3 + d22 * z4
    w = d13 * z3 + d23 * z4
    return x, y, z, u, v, w


def get_acc_of_orb(mg, r, x):
    return [-mg / (r ** 3) * x[0], -mg / (r ** 3) * x[1], -mg / (r ** 3) * x[2]]


def get_t_given_r_hypo(a, e, M, t_0, r):
    if e <= 1.0:
        raise ValueError("get_t_given_r_hypo: e<=1")
    cosf = (a * (1.0 - e * e) / r - 1.0) / e
    coshE = (e + cosf) / (e * cosf + 1.0)
    ecc_ano = math.acosh(coshE)
    mean_ano = e * math.sinh(ecc_ano) - ecc_ano
    mean_motion = math.sqrt(M / ((-a) ** 3))
    return t_0 + mean_ano / mean_motion


def get_ecc_ano_given_t(a, e, M, t, t_0):
    if e >= 1.0:
        raise ValueError(f"get_ecc_ano_given_t: e>1 {e}")
    n = (M ** 0.5) / (a ** 1.5)
    mean = n * (t - t_0)
    return mco_kep(e, mean)


def get_r_given_t(a, e, M, t):
    if e >= 1.0:
        raise ValueError(f"get_r_given_t: e>1 {e}")
    n = (M ** 0.5) / (a ** 1.5)
    mean = n * t
    ecc_ano = mco_kep(e, mean)
    return a * (1.0 - e * math.cos(ecc_ano))


def get_r_given_mean(a, e, mean):
    if e >= 1.0:
        raise ValueError(f"get_r_given_mean: e>=1 {e}")
    ecc_ano = mco_kep(e, mean)
    return a * (1.0 - e * math.cos(ecc_ano))


def vector_dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def vector_x(v1, v2):
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def vector_mag(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vector_unit(v):
    m = vector_mag(v)
    return [v[0] / m, v[1] / m, v[2] / m]


def get_true_anomaly(mass, x, y, z, vx, vy, vz):
    vv = [vx, vy, vz]
    vr = [x, y, z]
    vh = vector_x(vr, vv)
    vtmp = vector_x(vv, vh)
    vur = vector_unit(vr)
    ve = [vtmp[0] - vur[0] * mass, vtmp[1] - vur[1] * mass, vtmp[2] - vur[2] * mass]
    vue = vector_unit(ve)
    vdot = vector_dot(vue, vur)
    rvdot = vector_dot(vr, vv)
    if vdot > 1.0:
        vdot = 1.0
    if vdot < -1.0:
        vdot = -1.0
    if rvdot > 0.0:
        ta = math.acos(vdot)
    else:
        ta = 2.0 * PI - math.acos(vdot)
    return ta


def galactic_equatorial(Dir, Equinox, l, b, al, de):
    e1 = 62.6 / 180.0 * PI
    fa1 = 282.25 / 180.0 * PI
    fl1 = 33.0 / 180.0 * PI

    e2 = 62.8717 / 180.0 * PI
    fa2 = 282.8596 / 180.0 * PI
    fl2 = 32.93192 / 180.0 * PI

    if Equinox == 1950.0:
        e = e1
        fa = fa1
        fl = fl1
    elif Equinox == 2000.0:
        e = e2
        fa = fa2
        fl = fl2
    else:
        raise ValueError("In galactic_equatorial only Equinox=1950.0 or 2000.0 are supported!")

    if Dir:
        w = math.atan2(
            math.sin(l - PI / 2.0 - fl),
            (math.cos(l - PI / 2.0 - fl) * math.cos(e) - math.tan(b) * math.sin(e)),
        )
        al = (w + fa - 1.5 * PI)
        de = math.asin(math.sin(b) * math.cos(e) + math.cos(b) * math.sin(e) * math.cos(l - PI / 2.0 - fl))
        al = al - 2.0 * PI * int(al / (2.0 * PI))
        if al < 0.0:
            al = al + 2.0 * PI
    else:
        l = math.atan2(
            math.cos(de) * math.sin(al - fa) * math.cos(e) + math.sin(de) * math.sin(e),
            math.cos(de) * math.cos(al - fa),
        ) + fl
        l = l - 2.0 * PI * int(l / (2.0 * PI))
        if l < 0.0:
            l = l + 2.0 * PI
        b = math.asin(math.sin(de) * math.cos(e) - math.cos(de) * math.sin(al - fa) * math.sin(e))

    return l, b, al, de


def Delta(Numbers, nn):
    if nn > 3 or nn < 1:
        raise ValueError("Wrong nn in Delta!")
    if nn == 1:
        return Numbers[0]
    if nn == 2:
        return math.copysign(1.0, Numbers[0]) * (abs(Numbers[0]) + Numbers[1] / 60.0)
    return math.copysign(1.0, Numbers[0]) * (abs(Numbers[0]) + Numbers[1] / 60.0 + Numbers[2] / 3600.0)


def strlen_trail(string):
    return len(string.rstrip(" "))


def mco_x2el(gm, x, y, z, u, v, w):
    hx = y * w - z * v
    hy = z * u - x * w
    hz = x * v - y * u
    h2 = hx * hx + hy * hy + hz * hz
    v2 = u * u + v * v + w * w
    rv = x * u + y * v + z * w
    r = math.sqrt(x * x + y * y + z * z)
    h = math.sqrt(h2)
    s = h2 / gm

    ci = hz / h
    if abs(ci) < 1.0:
        inc = math.acos(ci)
        node = math.atan2(hx, -hy)
        if node < 0.0:
            node = node + TWOPI
    else:
        inc = 0.0 if ci > 0.0 else PI
        node = 0.0

    temp = 1.0 + s * (v2 / gm - 2.0 / r)
    if temp <= 0.0:
        e = 0.0
    else:
        e = math.sqrt(temp)
    q = s / (1.0 + e)

    if hy != 0.0:
        to = -hx / hy
        temp2 = (1.0 - ci) * to
        tmp2 = to * to
        true = math.atan2((y * (1.0 + tmp2 * ci) - x * temp2), (x * (tmp2 + ci) - y * temp2))
    else:
        true = math.atan2(y * ci, x)
    if ci < 0.0:
        true = true + PI

    if e < 3.0e-8:
        p = 0.0
        l = true
    else:
        ce = (v2 * r - gm) / (e * gm)

        if e < 1.0:
            if abs(ce) > 1.0:
                ce = _sign(1.0, ce)
            bige = math.acos(ce)
            if rv < 0.0:
                bige = TWOPI - bige
            l = bige - e * math.sin(bige)
        else:
            if ce < 1.0:
                ce = 1.0
            bige = math.log(ce + math.sqrt(ce * ce - 1.0))
            if rv < 0.0:
                bige = -bige
            l = e * math.sinh(bige) - bige

        cf = (s - r) / (e * r)
        if abs(cf) > 1.0:
            cf = _sign(1.0, cf)
        f = math.acos(cf)
        if rv < 0.0:
            f = TWOPI - f
        p = true - f
        p = (p + TWOPI + TWOPI) % TWOPI

    if l < 0.0:
        l = l + TWOPI
    if l > TWOPI:
        l = l % TWOPI

    return q, e, inc, p, node, l

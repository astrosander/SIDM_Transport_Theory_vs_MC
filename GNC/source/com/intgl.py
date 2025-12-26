import numpy as np


def my_solout_empty(nr, xold, x, y, n, con, icomp, nd, rpar, ipar, irtrn):
    return irtrn


class MyIntgl:
    def __init__(self):
        self.ipar = [0] * 100
        self.rpar = [0.0] * 100

    def my_integral_none(self, xs: float, xe: float, y: float, fcn):
        rtol = 1e-12
        atol = 1e-12
        itol = 0
        lwork = 400
        liwork = 400
        work = [0.0] * lwork
        iwork = [0] * liwork
        iwork[0] = 10000000
        work[6] = 0.0
        iout = 0
        x_start = float(xs)
        x_end = float(xe)
        if x_start == x_end:
            return 0.0, 0
        yout = np.array([float(y)], dtype=np.float64)
        x2, y2, idid, work, iwork, dense = dopri5(
            1, fcn, x_start, yout, x_end,
            rtol, atol, itol,
            solout=my_solout_empty, iout=iout,
            work=work, iwork=iwork,
            rpar=self.rpar, ipar=self.ipar
        )
        return float(y2[0]), int(idid)

    def my_integral_acc(self, xs: float, xe: float, y: float, atol_in: float, rtol_in: float, fcn):
        itol = 0
        lwork = 400
        liwork = 400
        work = [0.0] * lwork
        iwork = [0] * liwork
        iwork[0] = 10000000
        work[6] = 0.0
        iout = 0
        x_start = float(xs)
        x_end = float(xe)
        if x_start == x_end:
            return float(y), 1
        yout = np.array([float(y)], dtype=np.float64)
        x2, y2, idid, work, iwork, dense = dopri5(
            1, fcn, x_start, yout, x_end,
            float(rtol_in), float(atol_in), int(itol),
            solout=my_solout_empty, iout=iout,
            work=work, iwork=iwork,
            rpar=self.rpar, ipar=self.ipar
        )
        return float(y2[0]), int(idid)

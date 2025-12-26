class Astron_constant:
    pc = 206264.98
    rd_sun = 4.65e-3
    AU_GS = 1.4959787e13
    AU_SI = 1.4959787e11
    m_sun_GS = 1.98855e33
    m_sun_SI = 1.98855e30
    one_year = 3.1556e7
    pi4degree2 = 41252.96125


class unitless_value:
    PI = 3.141592653589793
    TWO_PI = 3.141592653589793 * 2.0
    e_nature = 2.718281828459045


class my_unit:
    my_unit_vel_c = 3e5 / 29.784

    @staticmethod
    def myu_conv_t2second(t: float) -> float:
        return t * 365.2425 / 2.0 / unitless_value.PI * 86400.0


class constant(unitless_value, Astron_constant, my_unit):
    pass

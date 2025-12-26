import math


def _next_non_comment_line(f, comstr: str) -> str:
    while True:
        line = f.readline()
        if line == "":
            raise EOFError("unexpected end of file")
        s = line.rstrip("\n")
        stripped = s.lstrip()
        if stripped == "":
            continue
        if stripped[0] == comstr:
            continue
        return s


def _value_part(line: str) -> str:
    p1 = line.rfind("=")
    p2 = line.rfind(":")
    p = max(p1, p2)
    return (line[p + 1 :] if p >= 0 else line).strip()


def _value_part_sp(line: str, sp: str) -> str:
    p = line.rfind(sp)
    return (line[p + 1 :] if p >= 0 else line).strip()


def _parse_bool(tok: str) -> bool:
    t = tok.strip().lower()
    if t in ("t", "true", ".true.", "1", "y", "yes", "on"):
        return True
    if t in ("f", "false", ".false.", "0", "n", "no", "off"):
        return False
    raise ValueError(f"cannot parse logical: {tok!r}")


def READPAR_INT_AUTO(N: int, file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    vals = _value_part(line).replace(",", " ").split()
    out = [int(x) for x in vals[:N]]
    if len(out) != N:
        raise ValueError("not enough integers in line")
    return out


def READPAR_DBL_AUTO(N: int, file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    vals = _value_part(line).replace(",", " ").split()
    out = [float(x) for x in vals[:N]]
    if len(out) != N:
        raise ValueError("not enough doubles in line")
    return out


def READPAR_DBL_ONE(file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part(line).split()[0]
    return float(s)


def READPAR_STR(file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    vp = _value_part(line)
    if vp != line.strip():
        return vp
    return line.strip().split()[-1]


def READPAR_INT(file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part(line).split()[0]
    return int(s)


def READPAR_LOG(file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part(line).split()[0]
    return _parse_bool(s)


def READPAR_DBL(file_unit, comstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part(line).split()[0]
    return float(s)


def READPAR_INT_AUTO_SP(N: int, file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    vals = _value_part_sp(line, sp).replace(",", " ").split()
    out = [int(x) for x in vals[:N]]
    if len(out) != N:
        raise ValueError("not enough integers in line")
    return out


def READPAR_DBL_AUTO_SP(N: int, file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    vals = _value_part_sp(line, sp).replace(",", " ").split()
    out = [float(x) for x in vals[:N]]
    if len(out) != N:
        raise ValueError("not enough doubles in line")
    return out


def READPAR_DBL_ONE_SP(file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part_sp(line, sp).split()[0]
    return float(s)


def READPAR_STR_SP(file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    return _value_part_sp(line, sp)


def READPAR_STR_AUTO_SP(N: int, file_unit, comstr: str, sp: str, sepstr: str):
    line = _next_non_comment_line(file_unit, comstr)
    strall = _value_part_sp(line, sp)
    parts = [p for p in strall.split(sepstr) if p != ""]
    nnum = len(parts)
    if nnum > N:
        parts = parts[:N]
        nnum = N
    if nnum < N:
        parts = parts + [""] * (N - nnum)
    return strall, parts, nnum


def READPAR_INT_SP(file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part_sp(line, sp).split()[0]
    try:
        return int(s), 0
    except Exception:
        return 0, -1


def READPAR_LOG_SP(file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part_sp(line, sp).split()[0]
    return _parse_bool(s)


def READPAR_DBL_SP(file_unit, comstr: str, sp: str):
    line = _next_non_comment_line(file_unit, comstr)
    s = _value_part_sp(line, sp).split()[0]
    try:
        return float(s), 0
    except Exception:
        return 0.0, -1


def skip_comments(file_unit, comstr: str):
    while True:
        pos = file_unit.tell()
        line = file_unit.readline()
        if line == "":
            return
        s = line.rstrip("\n")
        stripped = s.lstrip()
        if stripped == "" or stripped[0] == comstr:
            continue
        file_unit.seek(pos)
        return


def number_formatter(d: float, num_fix: int, num_str: int) -> str:
    if num_str < num_fix + 1:
        raise ValueError("increase the size of num_str")
    if d < 0:
        raise ValueError("d should be larger than zero")
    if d == 0.0:
        s = "0" if num_fix == 1 else ("0." + "0" * (num_fix - 1))
        return s.rjust(num_str)[:num_str]

    if d >= 1.0:
        digits = int(math.log10(d)) + 1
        if num_fix > digits:
            dec = num_fix - digits
            s = f"{d:.{dec}f}"
        elif num_fix == digits:
            s = str(int(round(d)))
        else:
            k = digits - num_fix
            factor = 10 ** k
            s = str(int(round(d / factor) * factor))
    else:
        digits = abs(int(math.log10(d)) - 1)
        if num_fix > digits:
            dec = num_fix - digits
            s = f"{d:.{dec}f}"
        elif num_fix == digits:
            s = str(int(round(d)))
        else:
            k = digits - num_fix
            factor = 10 ** k
            s = str(int(round(d / factor) * factor))

    if len(s) > num_str:
        raise ValueError("formatted number does not fit in num_str")
    return s.rjust(num_str)

def mio_spl(length: int, s: str):
    s = (s[:length]).ljust(length)
    delimit = [[-1] * 100, [-1] * 100]
    nsub = 0
    j = 0

    while True:
        j += 1
        if j > length:
            break
        c = s[j - 1]
        if c == " " or c == "=":
            continue
        k = j
        while True:
            k += 1
            if k > length:
                break
            c = s[k - 1]
            if c != " " and c != "=":
                continue
            break
        nsub += 1
        delimit[0][nsub - 1] = j
        delimit[1][nsub - 1] = k - 1
        if k < length:
            j = k
            continue
        break

    return nsub, delimit


def mio_spl_sp(length: int, s: str, separator: str):
    s = (s[:length]).ljust(length)
    delimit = [[-1] * 100, [-1] * 100]
    nsub = 0
    j = 0

    while True:
        j += 1
        if j > length:
            break
        c = s[j - 1]
        if c == separator:
            continue
        k = j
        while True:
            k += 1
            if k > length:
                break
            c = s[k - 1]
            if c != separator:
                continue
            break
        nsub += 1
        delimit[0][nsub - 1] = j
        delimit[1][nsub - 1] = k - 1
        if k < length:
            j = k
            continue
        break

    return nsub, delimit

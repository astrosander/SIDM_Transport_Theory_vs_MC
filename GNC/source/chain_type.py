from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, BinaryIO, Callable


FLAG_SG = 1
FLAG_BY = 2


def write_int(f: BinaryIO, v: int) -> None:
    f.write(int(v).to_bytes(4, byteorder="little", signed=True))


def read_int(f: BinaryIO) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise EOFError
    return int.from_bytes(b, byteorder="little", signed=True)


def write_float64(f: BinaryIO, v: float) -> None:
    import struct
    f.write(struct.pack("<d", float(v)))


def read_float64(f: BinaryIO) -> float:
    import struct
    b = f.read(8)
    if len(b) != 8:
        raise EOFError
    return struct.unpack("<d", b)[0]


class ParticleSample:
    def __init__(self) -> None:
        self.idx = 0
        self.id = 0
        self.m = 0.0
        self.weight_N = 0.0
        self.en = 0.0
        self.en0 = 0.0
        self.jm = 0.0
        self.jm0 = 0.0
        self.exit_flag = 0
        self.create_time = 0.0
        self.exit_time = 0.0
        self.obtype = 0
        self.byot = type("BYOT", (), {"a_bin": 0.0, "e_bin": 0.0})()

    def copy(self) -> "ParticleSample":
        import copy
        return copy.deepcopy(self)

    def sizeof(self) -> int:
        return 0

    def print(self, s: str) -> None:
        pass

    def write_info(self, f: BinaryIO) -> None:
        pass

    def read_info(self, f: BinaryIO) -> None:
        pass


@dataclass
class ChainPointer:
    idx: int = 0
    ob: Optional[ParticleSample] = None
    next: Optional["ChainPointer"] = None
    prev: Optional["ChainPointer"] = None
    ed: Optional["ChainPointer"] = None
    bg: Optional["ChainPointer"] = None
    append_left: Optional["ChainPointer"] = None
    append_right: Optional["ChainPointer"] = None

    def set_head(self) -> None:
        set_item_chain_head(self)

    def init_head(self) -> None:
        init_chain_head(self)

    def set_end(self) -> None:
        set_item_chain_end(self)

    def copy_to(self, other: "ChainPointer") -> None:
        copy_chain_object(self, other)


def init_chain_head(item: ChainPointer) -> None:
    item.bg = item
    item.ed = item
    item.idx = 0


def set_item_chain_end(item: ChainPointer) -> None:
    p: Optional[ChainPointer] = item
    item.ed = item
    while p is not None:
        p.ed = item
        p = p.prev


def set_item_chain_head(item: Optional[ChainPointer]) -> None:
    if item is None:
        return
    p: Optional[ChainPointer] = item
    item.bg = item
    while p is not None:
        p.bg = item
        p = p.next


def destroy_attach_pointer_chain_type(item: ChainPointer) -> None:
    if item.append_left is not None:
        destroy_attach_pointer_chain_type(item.append_left)
        item.append_left = None
    if item.append_right is not None:
        destroy_attach_pointer_chain_type(item.append_right)
        item.append_right = None
    item.ob = None
    item.next = None
    item.prev = None
    item.bg = None
    item.ed = None


def copy_chain_object(item: ChainPointer, cp: ChainPointer) -> None:
    if item.append_left is not None:
        cp.append_left = ChainPointer()
        copy_chain_object(item.append_left, cp.append_left)
    if item.append_right is not None:
        cp.append_right = ChainPointer()
        copy_chain_object(item.append_right, cp.append_right)

    if item.ob is None:
        raise RuntimeError("error! item%ob not allocated")

    cp.ob = item.ob.copy()


def attach_chain_end(this: "Chain", c: "Chain") -> None:
    if this.head is None or c.head is None:
        raise RuntimeError("attach_chain_end: missing head")
    phead = c.head
    chead = this.head
    pend = this.head.ed
    cend = c.head.ed
    if pend is None or cend is None:
        raise RuntimeError("attach_chain_end: missing end pointer")
    pend.next = phead
    phead.prev = pend
    p = phead
    while p is not None:
        if p.prev is None:
            break
        p.idx = p.prev.idx + 1
        p = p.next
    set_item_chain_head(chead)
    cend.set_end()


def attach_two_chains(this: "Chain", c2: "Chain") -> None:
    attach_chain_end(this, c2)
    this.n = this.n + c2.n


def chain_select_by_condition(ch: "Chain", ch_out: "Chain", selection: Callable[[ChainPointer], bool]) -> None:
    if ch.head is None:
        ch_out.init(0)
        return

    ps: Optional[ChainPointer] = ch.head
    nsel = 0
    while ps is not None and ps.ob is not None:
        if selection(ps):
            nsel += 1
        ps = ps.next

    ch_out.init(nsel)
    psout = ch_out.head
    ps = ch.head
    while ps is not None and ps.ob is not None:
        if selection(ps):
            if psout is not None:
                if psout.ob is None:
                    psout.ob = ps.ob.copy()
                copy_chain_object(ps, psout)
                psout = psout.next
        ps = ps.next


def chain_output_list_chain_type(lst: "Chain", flag_output_in: Optional[int] = None, maxnum_in: Optional[int] = None) -> None:
    flag_output = 0 if flag_output_in is None else flag_output_in
    p = lst.head
    i = 0
    out_parts: List[str] = []
    while p is not None:
        left = "<-" if p.prev is not None else "Null<-"
        right = "->" if p.next is not None else "->Null"
        allocated_ob = p.ob is not None
        flag = 1 if isinstance(p.ob, ParticleSample) else 0

        if flag_output == 0:
            mid = f"{int(allocated_ob)}"
        elif flag_output == 1:
            en = p.ob.en if p.ob is not None else 0.0
            mid = f"{p.idx}{int(allocated_ob)}{flag}{en:10.1f}"
        else:
            exit_flag = p.ob.exit_flag if p.ob is not None else 0
            oid = p.ob.id if p.ob is not None else 0
            mid = f"{p.idx}{int(allocated_ob)}{flag}{exit_flag:2d}{oid:10d}"

        out_parts.append(f"{left}{mid}{right}")

        p = p.next
        i += 1
        if maxnum_in is not None and i >= maxnum_in:
            break

    print("".join(out_parts))
    print()
    print("--------------------")


class Chain:
    def __init__(self) -> None:
        self.n: int = 0
        self.head: Optional[ChainPointer] = None

    def init(self, n: int) -> None:
        if self.head is not None:
            self.destory()
        self.n = n
        self.head = ChainPointer()
        self.head.init_head()
        p = self.head
        pc = self.head
        for _ in range(1, n):
            pc.next = ChainPointer()
            p = pc.next
            p.prev = pc
            p.bg = pc.bg
            p.idx = pc.idx + 1
            pc = pc.next
        p.set_end()

    def destory(self) -> None:
        if self.head is None:
            return
        p = self.head.ed
        pc = p
        while pc is not None and pc.prev is not None:
            p = pc.prev
            destroy_attach_pointer_chain_type(pc)
            pc = p
        destroy_attach_pointer_chain_type(self.head)
        self.head = None
        self.n = 0

    def get_length(self, type: Optional[int] = None) -> int:
        if self.head is None:
            return 0
        p = self.head
        n = 0
        typeI = 0 if type is None else type
        while p is not None:
            if typeI == 0:
                n += 1
            elif typeI == 1:
                if isinstance(p.ob, ParticleSample):
                    n += 1
            p = p.next
        return n

    def copy(self) -> "Chain":
        ch_copy = Chain()
        copy_a_chain(self, ch_copy)
        return ch_copy

    def output_bin(self, fl: str) -> None:
        output_chains_bin(self, fl)

    def output_txt(self, fl: str) -> None:
        output_chains_txt(self, fl)

    def input_bin(self, fl: str) -> None:
        input_chains_bin(self, fl)

    def insert_after_chain(self, length: int) -> None:
        insert_after_chain(self, length)

    def output_screen(self, flag_output_in: Optional[int] = None, maxnum_in: Optional[int] = None) -> None:
        chain_output_list_chain_type(self, flag_output_in, maxnum_in)


def insert_after_chain(chain: Chain, length: int) -> None:
    if chain.head is None:
        raise RuntimeError("pc not associated")
    if chain.head.ed is None:
        raise RuntimeError("missing chain end")
    pc = chain.head.ed
    for _ in range(length):
        pc.next = ChainPointer()
        p = pc.next
        p.prev = pc
        p.bg = pc.bg
        p.idx = pc.idx + 1
        pc = pc.next
    if chain.head.ed.next is not None:
        chain.head.ed.next.set_end()
    chain.n += length


def copy_a_chain(ch: Chain, ch_copy: Chain) -> None:
    length = ch.get_length()
    ch_copy.init(length)
    p = ch.head
    p_copy = ch_copy.head
    while p is not None and p_copy is not None:
        if p.ob is not None:
            if p_copy.ob is None:
                p_copy.ob = p.ob.copy()
            copy_chain_object(p, p_copy)
        p = p.next
        p_copy = p_copy.next


def output_chains_bin(sps: Chain, fl: str) -> None:
    n = sps.get_length()
    with open(f"{fl}.bin", "wb") as f:
        write_int(f, n)
        pt = sps.head
        while pt is not None:
            if isinstance(pt.ob, ParticleSample):
                write_int(f, FLAG_SG)
                pt.ob.write_info(f)
            pt = pt.next


def input_chains_bin(sps: Chain, fl: str) -> None:
    with open(f"{fl}.bin", "rb") as f:
        n = read_int(f)
        sps.init(n)
        pt = sps.head
        while pt is not None:
            flag = read_int(f)
            if flag == FLAG_SG:
                pt.ob = ParticleSample()
                pt.ob.read_info(f)
            pt = pt.next


def star_type(obtype: int) -> str:
    return str(obtype)


def output_chains_txt(ch: Chain, fl: str) -> None:
    with open(f"{fl}.txt", "w", encoding="utf-8") as f:
        f.write(f"{'i':>6}{'type':>6}{'idx':>15}{'id':>15}{'m':>15}{'w_N':>15}{'ao':>15}{'eo':>15}\n")
        pt = ch.head
        i = 0
        while pt is not None:
            i += 1
            if isinstance(pt.ob, ParticleSample):
                ca = pt.ob
                f.write(
                    f"{i:6d}{star_type(ca.obtype):>6}"
                    f"{ca.idx:15d}{ca.id:15d}"
                    f"{ca.m:15.5e}{ca.weight_N:15.5e}"
                    f"{ca.byot.a_bin:15.5e}{ca.byot.e_bin:15.5e}\n"
                )
            pt = pt.next


def output_chains_txt_bdsample(ch: Chain, ebd: float, fl: str) -> None:
    with open(f"{fl}.txt", "w", encoding="utf-8") as f:
        f.write(f"{'i':>6}{'en':>15}{'jm':>15}{'en0':>15}{'jm0':>15}\n")
        pt = ch.head
        i = 0
        while pt is not None:
            i += 1
            if isinstance(pt.ob, ParticleSample):
                ca = pt.ob
                if ca.en > ebd:
                    f.write(f"{i:6d}{ca.en:15.5e}{ca.jm:15.5e}{ca.en0:15.5e}{ca.jm0:15.5e}\n")
            pt = pt.next


def output_chains_txt_if(ch: Chain, fl: str) -> None:
    with open(f"{fl}.txt", "w", encoding="utf-8") as f:
        f.write(f"{'t0':>20}{'e0':>20}{'tf':>20}{'ef':>20}\n")
        pt = ch.head
        while pt is not None:
            if pt.ob is not None:
                f.write(f"{pt.ob.create_time:20.10e}{pt.ob.en0:20.10e}{pt.ob.exit_time:20.10e}{pt.ob.en:20.10e}\n")
            pt = pt.next


def smmerge(sma: List[Chain], smam: Chain) -> None:
    if not sma:
        smam.init(0)
        return
    smam.head = sma[0].head
    smam.n = sma[0].n
    for i in range(1, len(sma)):
        attach_two_chains(smam, sma[i])


def chain_select(sps: Chain, sps_out: Chain, exitflag: int, obj_type: Optional[int] = None) -> None:
    objtype = 0 if obj_type is None else obj_type

    def selection(pt: ChainPointer) -> bool:
        cond = False
        if objtype == 0:
            cond = True
        elif objtype == 1:
            cond = isinstance(pt.ob, ParticleSample)
        if pt.ob is None:
            return False
        if ((pt.ob.exit_flag == exitflag) or (exitflag == -1)) and cond:
            return True
        return False

    chain_select_by_condition(sps, sps_out, selection)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, BinaryIO


FLAG_LEFT = 0
FLAG_RIGHT = 1
FLAG_WRITE = 2
FLAG_NONE = -1

FLAG_PARTICLE = 3
FLAG_BY = 4


@dataclass
class ChainPointer:
    idx: int = 0
    ob: Optional["ParticleSample"] = None

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

    def chain_to_arr_single(self, n: int) -> List["ParticleSample"]:
        return chain_to_arr_single(self, n)

    def arr_to_chain(self, arr: List["ParticleSample"], n: int) -> None:
        arr_to_chain(self, arr, n)

    def create_ob(self) -> None:
        chain_create_ob(self)

    def copy(self) -> "ChainPointer":
        cp = ChainPointer()
        copy_chain_object(self, cp)
        return cp

    def print_node(self, s: str) -> None:
        print_node_chain(self, s)

    def attach_chain(self, c: "ChainPointer") -> None:
        attach_chain(self, c)

    def create_chain(self, length: int) -> None:
        create_chain(self, length)

    def get_sizeof(self) -> int:
        return get_sizeof_chain(self)


def print_node_chain(chain: ChainPointer, s: str) -> None:
    print(s)
    print("append=", chain.append_left is not None, chain.append_right is not None)
    if chain.ob is not None:
        chain.ob.print(s)
    if chain.append_left is not None and chain.append_left.ob is not None:
        chain.append_left.ob.print((s.strip() + "_left").strip())
    if chain.append_right is not None and chain.append_right.ob is not None:
        chain.append_right.ob.print((s.strip() + "_right").strip())


def create_chain(item: ChainPointer, length: int) -> None:
    pc: Optional[ChainPointer] = item
    if pc is None:
        raise RuntimeError("pc not associated")

    if pc.next is None:
        p = pc
        for _ in range(length):
            pc.next = ChainPointer()
            p = pc.next
            p.prev = pc
            p.bg = pc.bg
            p.idx = pc.idx + 1
            pc = pc.next
        set_item_chain_end(p)
    else:
        for _ in range(length):
            pn = item.next
            p = ChainPointer()
            pn.prev = p
            item.next = p
            p.prev = item
            p.next = pn
            p.ed = item.ed
            p.bg = item.bg


def chain_to_arr_single(chain: ChainPointer, n: int) -> List["ParticleSample"]:
    arr: List["ParticleSample"] = []
    p: Optional[ChainPointer] = chain
    i = 1
    while p is not None:
        if p.ob is None:
            raise RuntimeError(f"chain_to_arr:not allocated???i={i} idx={p.idx}")
        arr.append(p.ob.copy())
        arr[-1].idx = i
        i += 1
        p = p.next
    if n != len(arr):
        arr = arr[:n]
    return arr


def arr_to_chain(chain: ChainPointer, arr: List["ParticleSample"], n: int) -> None:
    p: Optional[ChainPointer] = chain
    i = 0
    while p is not None and i < n:
        p.ob = arr[i].copy()
        p.ob.idx = i + 1
        p = p.next
        i += 1


def get_sizeof_chain(chain: ChainPointer) -> int:
    p: Optional[ChainPointer] = chain
    nsize = 0
    while p is not None:
        if p.ob is not None:
            nsize += p.ob.sizeof()
        p = p.next
    return nsize


def destroy_attach_pointer(item: ChainPointer) -> None:
    if item.append_left is not None:
        destroy_attach_pointer(item.append_left)
        item.append_left = None
    if item.append_right is not None:
        destroy_attach_pointer(item.append_right)
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


def save_attach_points(sp: ChainPointer, f: BinaryIO) -> None:
    if sp.append_left is not None and sp.append_left.ob is not None:
        write_int(f, FLAG_LEFT)
        save_attach_points(sp.append_left, f)
    else:
        write_int(f, FLAG_NONE)

    if sp.append_right is not None and sp.append_right.ob is not None:
        write_int(f, FLAG_RIGHT)
        save_attach_points(sp.append_right, f)
    else:
        write_int(f, FLAG_NONE)

    write_int(f, FLAG_WRITE)

    if sp.ob is None:
        raise RuntimeError("save_attach_points: sp.ob is None")

    write_int(f, FLAG_PARTICLE)
    sp.ob.write_info(f)


def read_attach_points(sp: ChainPointer, f: BinaryIO) -> None:
    flag = read_int(f)
    if flag == FLAG_LEFT:
        if sp.append_left is None:
            sp.append_left = ChainPointer()
        read_attach_points(sp.append_left, f)

    flag = read_int(f)
    if flag == FLAG_RIGHT:
        if sp.append_right is None:
            sp.append_right = ChainPointer()
        read_attach_points(sp.append_right, f)

    flag = read_int(f)
    if flag == FLAG_WRITE:
        kind = read_int(f)
        if kind == FLAG_PARTICLE:
            if sp.ob is None:
                sp.ob = ParticleSample()
            sp.ob.read_info(f)


def attach_pointer_in_a_chain(item: ChainPointer, pt: ChainPointer, flag: int) -> None:
    if pt is None:
        raise RuntimeError("attach_pointer:error, pt not allocated")

    if pt.prev is None:
        if pt.next is None:
            pass
        set_item_chain_head(pt.next)
        if pt.next is not None:
            pt.next.prev = None
        goto_cleanup = True
    else:
        prev = pt.prev
        if pt.next is None:
            set_item_chain_end(prev)
            prev.next = None
            goto_cleanup = True
        else:
            nxt = pt.next
            prev.next = nxt
            nxt.prev = prev
            goto_cleanup = True

    pt.next = None
    pt.prev = None
    pt.bg = None
    pt.ed = None

    if flag == 1:
        item.append_left = pt
    elif flag == 2:
        item.append_right = pt


def chain_pointer_delete_item(item: ChainPointer) -> None:
    ps = item
    if ps.prev is None:
        raise RuntimeError("del:item is the beginning of a chain")

    prev = ps.prev
    if ps.next is None:
        prev.set_end()
        prev.next = None
        destroy_attach_pointer(ps)
        return

    nxt = ps.next
    prev.next = nxt
    nxt.prev = prev
    destroy_attach_pointer(ps)


def chain_create_ob(sp: ChainPointer) -> None:
    p: Optional[ChainPointer] = sp
    while p is not None:
        p.ob = ParticleSample()
        p = p.next


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


def init_chain_head(item: ChainPointer) -> None:
    item.bg = item
    item.ed = item
    item.idx = 0


def insert_after_item(item: Optional[ChainPointer]) -> ChainPointer:
    if item is None:
        item = ChainPointer()
        init_chain_head(item)
        return item

    if item.next is None:
        item.next = ChainPointer()
        p = item.next
        p.prev = item
        p.bg = item.bg
        p.idx = item.idx + 1
        set_item_chain_end(p)
        return item

    pn = item.next
    p = ChainPointer()
    pn.prev = p
    item.next = p
    p.prev = item
    p.next = pn
    p.ed = item.ed
    p.bg = item.bg
    p.idx = item.idx + 1

    q = p.next
    while q is not None:
        q.idx += 1
        q = q.next

    return item


def attach_chain(this: ChainPointer, c: ChainPointer) -> None:
    phead = this.bg
    pend = this.ed
    chead = c.bg
    cend = c.ed

    if phead is None or pend is None or chead is None or cend is None:
        raise RuntimeError("attach_chain: missing head/end pointers")

    pend.next = chead
    chead.prev = pend

    p = chead
    while p is not None:
        if p.prev is None:
            break
        p.idx = p.prev.idx + 1
        p = p.next

    phead.set_head()
    cend.set_end()


def write_int(f: BinaryIO, v: int) -> None:
    f.write(int(v).to_bytes(4, byteorder="little", signed=True))


def read_int(f: BinaryIO) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise EOFError
    return int.from_bytes(b, byteorder="little", signed=True)


class ParticleSample:
    idx: int = 0

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

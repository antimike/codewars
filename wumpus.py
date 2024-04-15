from collections import Counter
from functools import reduce
from itertools import combinations
from typing import Generator, TypeAlias

from utils import get_logger

logger = get_logger(__name__)


def pos(i, j):
    return 1 << (4 * i + j)


ADJ = {
    pos(i, j): {
        pos(ni, nj)
        for ni in range(4)
        for nj in range(4)
        if abs(i - ni) + abs(j - nj) == 1
    }
    for i in range(4)
    for j in range(4)
}

PITMASKS = [
    {x for x in range(2**16) if x.bit_count() == N and not x % 2} for N in range(1, 4)
]

# P                     --> AND(p for p in pitmasks)
# P?                    --> OR(p for p in pitmasks)
# S := ~(P?)
# remove pitmask p      --> P =


def combos(k):
    return (sum(2 ** (j + 1) for j in p) for p in combinations(range(15), k))


def pie(npits):
    # HORRIBLE
    # return {
    #     n: {
    #         p: {
    #             q
    #             for q in range(2**16)
    #             if q.bit_count() == npits and (p & q) and not q % 2
    #         }
    #         for p in range(2**16)
    #         if p.bit_count() == n
    #     }
    #     for n in range(1, npits + 1)
    # }
    for n in range(1, npits + 1):
        ...


class HittingSet:
    def __init__(self):
        self._sets = []
        self._elems = dict()
        self._kernel = set()

    def add_set(self, S: set):
        for x in S:
            self._elems.setdefault(x, []).append(len(self._sets))
        self._sets.append(S)
        if not S.intersection(self._kernel):
            self._update_kernel()

    def _update_kernel(self):
        self._kernel.clear()
        for x, l in sorted(
            self._elems.items(), key=lambda it: len(it[1]), reverse=True
        ):
            if not any(self._sets[s].intersection(self._kernel) for s in l):
                self._kernel.add(x)


def solution(cave):
    cave = {pos(i, j): cave[i][j] for i in range(4) for j in range(4)}
    npits = len([p for p, r in cave.items() if r == "P"])
    pitmasks = {x for x in range(2**16) if x.bit_count() == npits and not x % 2}
    curr = 1
    seen = set()
    while not cave[curr] == "G":
        ...


def minimal_solution(cave):
    cave = {(i, j): cave[i][j] for i in range(4) for j in range(4)}
    binom15 = {0: 1, 1: 15, 2: 15 * 7}
    num_pits = sum(sum(c == "P" for c in row) for row in cave)
    pit_counter = Counter(
        {(i, j): binom15[num_pits - 1] for i in range(4) for j in range(4)}
    )
    pit_counter[0, 0] = 0
    adj = {
        (i, j): {
            (ni, nj)
            for ni in range(4)
            for nj in range(4)
            if abs(i - ni) + abs(j - nj) == 1
        }
        for i in range(4)
        for j in range(4)
    }
    seen, safe = set(), set()
    maybe_wumpus = set(pit_counter.keys()).difference({(0, 0)})
    pos = (0, 0)
    while not cave[pos] == "G":
        if any(cave[n] == "P" for n in adj[pos]):
            ...


class WumpusCave:
    class Room:
        def __init__(self, cave, coords, char):
            self.cave = cave
            self.coords = coords
            self.has_wumpus = char == "W"
            self.has_gold = char == "G"
            if char == "P":
                self.cave.pits.add(self)

        def neighbors(self):
            i, j = self.coords
            yield from (
                self.cave[ni, nj]
                for d in (-1, 1)
                for ni, nj in ((i + d, j), (i, j + d))
                if (ni, nj) in self.cave
            )

        def is_deathtrap(self):
            return self.has_wumpus or self in self.cave.pits

        def smells_like_wumpus(self):
            return any(r.has_wumpus for r in self.neighbors())

        def is_suspiciously_breezy(self):
            return any(r in self.cave.pits for r in self.neighbors())

        def is_boring(self):
            return not (
                self.smells_like_wumpus()
                or self.is_suspiciously_breezy()
                or self.is_deathtrap()
            )

        def __hash__(self):
            return hash(self.coords)

        def __repr__(self):
            return f"<CaveRoom {self.coords}>"

    def __init__(self, cave):
        self.width = len(cave[0])
        self.pits = set()
        self._rooms = {
            (i, j): self.Room(self, (i, j), char)
            for i, row in enumerate(cave)
            for j, char in enumerate(row)
        }
        self.start = self._rooms[0, 0]

    def __iter__(self):
        yield from self._rooms.values()

    def __contains__(self, item):
        if isinstance(item, self.Room):
            return self._rooms[item.coords] is item
        return item in self._rooms

    def __len__(self):
        return len(self._rooms)

    def __getitem__(self, coords):
        return self._rooms[coords]


class CaveMap:
    def __init__(self, cave):
        self._bitmasks = {
            room: 1 << (cave.width * room.coords[0] + room.coords[1]) for room in cave
        }
        self._rooms = {mask: room for room, mask in self._bitmasks.items()}
        self.accessible, self.visited = {cave.start}, set()
        self.maybe_wumpus = sum(
            m for r, m in self._bitmasks.items() if r is not cave.start
        )
        self.pit_states = {
            n
            for n in range(2 ** len(cave))
            if n.bit_count() == len(cave.pits) and not n & self._bitmasks[cave.start]
        }
        self._update_pitmask()

    def _update_pitmask(self):
        self.maybe_pit = reduce(lambda r, n: r | n, self.pit_states, 0)

    def locate_wumpus(self):
        return self._rooms[self.maybe_wumpus]

    def visit(self, room):
        neighbs = sum(self._bitmasks[n] for n in room.neighbors())
        self.pit_states = {
            m
            for m in self.pit_states
            if bool(m & neighbs) == room.is_suspiciously_breezy()
        }
        if self.maybe_wumpus:
            self.maybe_wumpus &= neighbs if room.smells_like_wumpus() else ~neighbs
            # the wumpus can't be in a pit
            self.maybe_wumpus &= ~reduce(lambda r, n: r & n, self.pit_states, 1)
            self.pit_states = {m for m in self.pit_states if ~m & self.maybe_wumpus}
        self._update_pitmask()
        self.accessible.update(room.neighbors())
        self.visited.add(room)

    def definitely_safe(self):
        yield from (
            r
            for r, m in self._bitmasks.items()
            if not (m & self.maybe_wumpus or m & self.maybe_pit)
        )


class Explorer:
    def __init__(self, cave: WumpusCave):
        self.map = CaveMap(cave)
        self.pos = cave.start

    def kill_wumpus(self):
        self.map.maybe_wumpus = 0

    def move(self):
        self.pos = next(
            iter(
                self.map.accessible.difference(self.map.visited).intersection(
                    self.map.definitely_safe()
                )
            )
        )

    def find_gold(self):
        while not self.pos.has_gold:
            self.map.visit(self.pos)
            try:
                self.map.locate_wumpus()
                self.kill_wumpus()
            except KeyError:
                pass
            self.move()


def wumpus_world(cave):
    cave = WumpusCave(cave)
    explorer = Explorer(cave)
    try:
        explorer.find_gold()
        return True
    except StopIteration:
        return False


CaveType: TypeAlias = list[list[str]]


def test_cases() -> Generator[tuple[CaveType, bool], None, None]:
    yield [
        [*"___P"],
        [*"__PG"],
        [*"___P"],
        [*"W___"],
    ], False, "gold on RH side blocked by pits"
    yield [[*"____"], [*"_W__"], [*"___G"], [*"P___"]], True, "one pit"
    yield [[*"____"], [*"_P__"], [*"____"], [*"_W_G"]], True, "gold in lower RH corner"
    yield [
        [*"____"],
        [*"____"],
        [*"W__P"],
        [*"__PG"],
    ], False, "gold in lower RH corner blocked by pits"
    yield [
        [*"__GP"],
        [*"_P__"],
        [*"W___"],
        [*"____"],
    ], True, "gold near entrance (top row)"
    yield [
        [*"__W_"],
        [*"____"],
        [*"___P"],
        [*"___G"],
    ], True, "gold in lower RH corner, one liberty, 'corridor' access along bottom (1 pit)"
    yield [
        [*"__W_"],
        [*"____"],
        [*"__PP"],
        [*"___G"],
    ], True, "gold in lower RH corner, one liberty, 'corridor' access along bottom (2 pits)"
    yield [
        [*"__W_"],
        [*"____"],
        [*"_PPP"],
        [*"___G"],
    ], True, "gold in lower RH corner, one liberty, 'corridor' access along bottom (3 pits)"
    yield [
        [*"__P_"],
        [*"____"],
        [*"__P_"],
        [*"__WG"],
    ], True, "gold in lower RH corner, one liberty, 'corridor' access along RH side (1 pit + wumpus)"
    yield [
        [*"____"],
        [*"__PW"],
        [*"PG__"],
        [*"____"],
    ], True, "gold in middle adjacent to pit (vertical access)"
    yield [[*"__P_"], [*"____"], [*"WP__"], [*"_G__"]], True, "gold on bottom"
    yield [
        [*"__PG"],
        [*"____"],
        [*"__WP"],
        [*"____"],
    ], True, "gold in upper RH corner, maze access (1 pit)"
    yield [
        [*"___W"],
        [*"__P_"],
        [*"__G_"],
        [*"P___"],
    ], True, "gold in middle adjacent to pit (horizontal access)"
    yield [
        [*"__WP"],
        [*"_P__"],
        [*"____"],
        [*"_G__"],
    ], True, "corridor escape from start chamber (1 pit + wumpus)"
    yield [
        [*"__WP"],
        [*"____"],
        [*"__P_"],
        [*"P_G_"],
    ], True, "gold on bottom in 'sparse chamber'"
    yield [
        [*"__PG"],
        [*"___W"],
        [*"__PP"],
        [*"____"],
    ], True, "gold on top RH corner, access guarded by wumpus (need to kill)"


def test():
    log = get_logger(f"{__name__}.test", level="INFO")
    failed, succeeded = 0, 0
    for cave, expect, description in test_cases():
        try:
            result = wumpus_world(cave)
            assert (
                expect == result
            ), f"got {result}, expected {expect} on test {description!r}"
            succeeded += 1
        except (AssertionError, YouDie) as e:
            log.error("FAILED: %s", e)
            failed += 1
    print("-----------------------------------")
    if failed == 0:
        log.info("%s tests passed successfully", succeeded)
    else:
        log.warning("SUMMARY: %s failed, %s succeeded", failed, succeeded)

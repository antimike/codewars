from functools import reduce
from typing import Generator, TypeAlias

from utils import get_logger

logger = get_logger(__name__)


class YouDie(Exception):
    pass


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

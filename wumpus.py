from functools import reduce
from typing import Generator, TypeAlias

from utils import get_logger

logger = get_logger(__name__)


class YouDie(Exception):
    pass


class WumpusCave:
    WIDTH = 4
    HEIGHT = 4

    class Room:
        def __init__(self, cave, i, j):
            self.cave = cave
            self.i, self.j = i, j
            self.adj = set()
            self.contents = set()

        @property
        def coords(self):
            return self.i, self.j

        def is_deathtrap(self):
            return self is self.cave.wumpus_room or self in self.cave.pits

        def smells_like_wumpus(self):
            return any(r is self.cave.wumpus_room for r in self.adj)

        def is_suspiciously_breezy(self):
            return any(r in self.cave.pits for r in self.adj)

        def is_boring(self):
            return (
                not self.smells_like_wumpus()
                and not self.is_suspiciously_breezy()
                and not self.is_deathtrap()
            )

        def __hash__(self):
            return hash(self.coords)

        def __eq__(self, other):
            return self.coords == other.coords

        def __repr__(self):
            return f"<CaveRoom {self.coords}>"

    def __init__(self, cave):
        self._rooms = {
            (i, j): self.Room(self, i, j)
            for i in range(self.HEIGHT)
            for j in range(self.WIDTH)
        }
        self.start = self._rooms[0, 0]
        self.pits = set()
        for i, row in enumerate(cave):
            for j, contents in enumerate(row):
                room = self._rooms[i, j]
                room.contents.add(contents)
                for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if (ni, nj) in self._rooms:
                        room.adj.add(self._rooms[ni, nj])
                if "W" in room.contents:
                    self.wumpus_room = room
                if "P" in room.contents:
                    self.pits.add(room)
                if "G" in room.contents:
                    self.gold_room = room

    def __getitem__(self, coords):
        return self._rooms[coords]

    @property
    def rooms(self):
        return self._rooms.values()

    def kill_wumpus(self):
        self.wumpus_room = None


class CaveInfo:
    def __init__(self, num_pits):
        self.wumpus_states = [1 << i for i in range(1, 16)]
        self.pit_states = [
            n for n in range(2**16) if n.bit_count() == num_pits and n % 2 == 0
        ]

    @property
    def maybe_pit(self):
        return reduce(lambda r, n: r | n, self.pit_states, 0)

    def update(self, room):
        if room.is_deathtrap():
            raise YouDie(f"Oh no, you died in {room}!")
        mask = sum(1 << (4 * n.i + n.j) for n in room.adj)
        self.wumpus_states = self._constrain(
            self.wumpus_states, mask, room.smells_like_wumpus()
        )
        self.pit_states = self._constrain(
            self.pit_states, mask, room.is_suspiciously_breezy()
        )
        # self.maybe_pit = reduce(lambda r, n: r | n, self.pit_states, 0)
        # apply "wumpus not in pit" constraint
        self.wumpus_states = self._constrain(
            self.wumpus_states, reduce(lambda r, n: r & n, self.pit_states, 1), False
        )

    def kill_wumpus(self):
        self.pit_states = self._constrain(
            self.pit_states, self.wumpus_states.pop(), False
        )
        assert not self.wumpus_states, "You prematurely killed the wumpus, you jerk"
        logger.info("You killed the wumpus!")

    def is_definitely_safe(self, room):
        mask = 1 << (4 * room.i + room.j)
        return mask not in self.wumpus_states and not (mask & self.maybe_pit)

    @classmethod
    def _constrain(cls, masks, constraint, target_bool):
        return {m for m in masks if bool(m & constraint) == target_bool}


class Explorer:
    def __init__(self, cave: WumpusCave):
        self.cave = cave
        self.pos = cave.start
        self.explored = set()
        self.accessible = {cave.start}
        self.info = CaveInfo(len(cave.pits))

    def kill_wumpus(self):
        self.info.kill_wumpus()
        self.cave.kill_wumpus()

    def take_stock(self):
        self.info.update(self.pos)
        self.explored.add(self.pos)
        self.accessible.update(self.pos.adj)
        if (
            len(self.info.wumpus_states) == 1
            and self.cave.wumpus_room in self.accessible
        ):
            self.kill_wumpus()

    def move_to(self, room):
        if room in self.accessible:
            self.pos = room
            self.info.update(self.pos)
            self.take_stock()
        else:
            raise ValueError(f"Room {room} is not accessible")

    def choices(self):
        yield from filter(
            self.info.is_definitely_safe, self.accessible.difference(self.explored)
        )

    def find_gold(self):
        while self.pos is not self.cave.gold_room:
            self.take_stock()
            self.pos = next(self.choices())


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

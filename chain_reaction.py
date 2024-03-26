from utils import get_logger

logger = get_logger(__name__)


def min_bombs_needed(grid):
    rows = grid.split("\n")
    bombs = {
        (i, j): (i, j)
        for i, row in enumerate(rows)
        for j, s in enumerate(row)
        if s in ("+", "x")
    }
    comps = set(bombs.keys())

    def get_component_root(coords):
        while bombs[coords] != coords:
            coords = bombs[coords]
        return coords

    def get_neighbs(coords):
        yield coords
        i, j = coords
        char = rows[i][j]
        if char == "x":
            ns = [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
        elif char == "+":
            ns = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        yield from filter(lambda xs: xs in bombs, ns)

    for coords in bombs.keys():
        for n in get_neighbs(coords):
            root = get_component_root(n)
            bombs[root] = bombs[n] = coords
            comps.discard(root)
        comps.add(coords)
    return len(comps)

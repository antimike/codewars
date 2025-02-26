from utils import get_logger

logger = get_logger(__name__)


class Grid:
    def __init__(self, grid: str):
        self.rows = grid.split("\n")
        self.graph, self.reverse = {}, {}
        for i, row in enumerate(self.rows):
            for j, char in enumerate(row):
                if char == "x":
                    ns = [
                        (i - 1, j - 1),
                        (i - 1, j + 1),
                        (i + 1, j - 1),
                        (i + 1, j + 1),
                    ]
                elif char == "+":
                    ns = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                else:
                    continue
                self.graph[i, j] = []
                self.reverse.setdefault((i, j), [])
                for ni, nj in ns:
                    if (
                        0 <= ni < len(self.rows)
                        and 0 <= nj < len(self.rows[0])
                        and self.rows[ni][nj] in ("x", "+")
                    ):
                        self.graph[i, j].append((ni, nj))
                        self.reverse.setdefault((ni, nj), []).append((i, j))

    @classmethod
    def dfs(cls, node, graph, preorder, postorder):
        pending, done = [node], []
        while pending:
            n = pending.pop()
            if n in preorder:
                continue
            preorder[n] = len(preorder)
            pending.extend(graph[n])
            done.append(n)
        postorder.extend(list(reversed(done)))

    def condensation(self, reverse=False):
        pre, post = {}, []
        for n in self.graph:
            self.dfs(n, self.graph, pre, post)

        sccs, seen, dag = {}, {}, {}
        for node in reversed(post):
            if node not in seen:
                # new strongly-connected component
                logger.debug("New SCC: %s", node)
                logger.debug("seen = %s", seen)
                scc, dag[node] = [], []
                self.dfs(node, self.reverse, seen, scc)
                for n in scc:
                    sccs[n] = node
                    for m in self.reverse[n]:
                        if m in seen and seen[m] < seen[node]:
                            source, dest = sccs[m], node
                            if reverse:
                                source, dest = dest, source
                            dag.setdefault(source, []).append(dest)
        return dag


def min_bombs_needed(grid):
    g = Grid(grid)
    return len([scc for scc, ns in g.condensation(reverse=True).items() if not ns])


def get_neighbs(coords, rows):
    i, j = coords
    char = rows[i][j]
    if char == "x":
        ns = [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
    elif char == "+":
        ns = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    yield from filter(
        lambda xs: 0 <= xs[0] < len(rows)
        and 0 <= xs[1] < len(rows[0])
        and rows[xs[0]][xs[1]] in ("x", "+"),
        ns,
    )


def kosaraju(grid):
    rows = grid.split("\n")
    pre = {}
    post = []
    stack = []
    reverse = {}

    def dfs(coords):
        pre[coords] = len(pre)
        stack.append(coords)
        for n in get_neighbs(coords, rows):
            reverse.setdefault(n, set()).add(coords)
            if n not in pre:
                dfs(n)
        while stack:
            n = stack.pop()
            post.append(n)
            if n == coords:
                break

    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char in ("x", "+") and (i, j) not in pre:
                dfs((i, j))

    stack.clear()
    seen = set()
    count = 0
    logger.debug("post = %s (length %s)", post, len(post))
    while post:
        root = post.pop()
        if root in seen:
            logger.debug("root %s already seen", root)
            continue
        stack.append(root)
        while stack:
            elem = stack.pop()
            seen.add(elem)
            for n in reverse.get(elem, []):
                if n not in seen:
                    stack.append(n)
        count += 1
    logger.debug(reverse)
    return count


def tarjan(grid):
    """TODO: Debug this"""
    rows = grid.split("\n")
    pre = {}
    low = {}
    comps = {}
    reverse_dag = {}

    stack = []
    temp = set()

    def dfs(coords):
        lowest = low[coords] = pre[coords] = len(pre)
        stack.append(coords)
        for n in get_neighbs(coords, rows):
            if n not in pre:
                dfs(n)
            if n in comps:
                temp.add(comps[n])
            lowest = min(lowest, low[n])
        if lowest < low[coords]:
            low[coords] = lowest
            return
        while temp:
            reverse_dag.setdefault(temp.pop(), []).append(coords)
        while stack:
            n = stack.pop()
            comps[n] = coords
            low[n] = float("inf")
            if n == coords:
                break

    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char in ("x", "+") and (i, j) not in pre:
                dfs((i, j))

    logger.debug("comps = %s", comps)
    logger.debug("reverse_dag = %s", reverse_dag)
    logger.debug("pre = %s", pre)
    return len([c for c, p in reverse_dag if not p])


# def min_bombs_needed(grid):
#     rows = grid.split("\n")
#     fringe = set()
#     internal = set()
#
#     for i, row in enumerate(rows):
#         for j, char in enumerate(row):
#             if char in ("+", "x"):
#                 if (i, j) not in internal:
#                     logger.debug("Adding fringe elem %s with bomb %s", (i, j), char)
#                     fringe.add((i, j))
#                 for n in get_neighbs((i, j), rows):
#                     internal.add(n)
#                     fringe.discard(n)
#
#     logger.debug("fringe = %s", fringe)
#     logger.debug("internal = %s", internal)
#     return len(fringe)


def connected_components(grid):
    """This doesn't work because it ignore the directionality of edges."""
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

    for coords in bombs.keys():
        for n in get_neighbs(coords, rows):
            root = get_component_root(n)
            bombs[root] = bombs[n] = coords
            comps.discard(root)
        comps.add(coords)
    return len(comps)

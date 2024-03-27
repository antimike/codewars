from utils import get_logger

logger = get_logger(__name__)


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


class Grid:
    def __init__(self, grid: str):
        self.rows = grid.split("\n")
        self.nrows, self.ncols = len(self.rows), len(self.rows[0])
        self.graph = {}
        self.reverse = {}
        self.scc = {}
        self.condensed = {}
        self.condensed_reversed = {}
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
                        0 <= ni < self.nrows
                        and 0 <= nj < self.ncols
                        and self.rows[ni][nj] in ("x", "+")
                    ):
                        self.graph[i, j].append((ni, nj))
                        self.reverse.setdefault((ni, nj), []).append((i, j))

    @classmethod
    def dfs(
        cls,
        node: tuple[int, int],
        graph,
        preorder: dict,
        postorder: list,
        stack: list,
        preorder_callback=None,
        postorder_callback=None,
    ):
        if node in preorder:
            return
        preorder[node] = len(preorder)
        stack.append(node)
        if preorder_callback is not None:
            preorder_callback(node)
        for n in graph[node]:
            cls.dfs(
                n,
                graph,
                preorder,
                postorder,
                stack,
                preorder_callback=preorder_callback,
                postorder_callback=postorder_callback,
            )
        while stack:
            n = stack.pop()
            postorder.append(n)
            if postorder_callback is not None:
                postorder_callback(node)
            if n == node:
                break

    def kosaraju(self):
        self.preorder, self.postorder = {}, []
        stack = []
        for n in self.graph:
            self.dfs(n, self.graph, self.preorder, self.postorder, stack)

        logger.debug("post = %s", self.postorder)

        def add_to_scc(node, root):
            logger.debug("Adding node %s to SCC with root %s", node, root)
            self.scc[node] = root
            for n in self.reverse[node]:
                if n in self.scc and self.scc[n] != root:
                    self.condensed[self.scc[n]].append(root)
                    self.condensed_reversed[root].append(self.scc[n])

        seen = {}
        for node in reversed(self.postorder):
            if node not in seen:
                # new strongly-connected component
                logger.debug("New SCC: %s", node)
                logger.debug("seen = %s", seen)
                self.condensed[node] = []
                self.condensed_reversed[node] = []
                self.dfs(
                    node,
                    self.reverse,
                    seen,
                    [],
                    [],
                    preorder_callback=lambda n: add_to_scc(n, node),
                )

    def print(self):
        print("\n".join(self.rows))


def min_bombs_needed(grid):
    g = Grid(grid)
    g.kosaraju()
    return len([scc for scc, ns in g.condensed_reversed.items() if not ns])


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

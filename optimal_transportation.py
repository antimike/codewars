from collections import Counter
from itertools import product

from rich import print as rich_print
from rich.tree import Tree

from utils import get_logger

logger = get_logger(__name__)


class Heap:
    def __init__(self, **elems):
        self._elems = []
        self._pos = {}
        self._priority = {}
        for elem, priority in elems.items():
            self.insert(elem, priority)

    def insert(self, elem, priority):
        self._elems.append(elem)
        self._priority[elem] = priority
        self._pos[elem] = len(self._elems) - 1
        idx = self._upheap(len(self._elems) - 1)
        self._downheap(idx)

    def discard(self, elem):
        if elem not in self._pos:
            return None
        pos = self._pos[elem]
        P = self._priority[elem]
        self._priority[elem] = -float("inf")
        idx = self._upheap(pos)
        assert idx == 0, "Failed to upheap discarded element"
        self.pop_min()
        return P

    def pop_min(self):
        self._switch(0, len(self._elems) - 1)
        elem = self._elems.pop()
        P = self._priority[elem]
        del self._pos[elem]
        del self._priority[elem]
        self._downheap(0)
        return elem, P

    def get_priority(self, elem):
        return self._priority[elem]

    def set_priority(self, elem, priority):
        self.discard(elem)
        self.insert(elem, priority)

    def _upheap(self, pos):
        elem = self._elems[pos]
        P = self._priority[elem]
        while pos > 0 and P < self._priority[self._elems[ppos := ((pos - 1) // 2)]]:
            self._switch(pos, ppos)
            pos = ppos
        return pos

    def _downheap(self, pos):
        elem = self._elems[pos]
        P = self._priority[elem]
        while True:
            min_child = min(2 * pos + 1, 2 * pos + 2, key=self._pos_priority)
            if self._pos_priority(min_child) >= P:
                return pos
            self._switch(pos, min_child)
            pos = min_child

    def _switch(self, p1, p2):
        e1, e2 = self._elems[p1], self._elems[p2]
        self._elems[p1], self._elems[p2] = e2, e1
        self._pos[e1], self._pos[e2] = p2, p1

    def _pos_priority(self, pos):
        if 0 <= pos < len(self._elems):
            return self._priority[self._elems[pos]]
        return float("inf")


class TreeError(Exception):
    pass


class DistributionNetwork:
    """Implementation of the network simplex algorithm on an abstract ST-network.

    The network has one abstract source node S and one abstract sink node T, with
    vertex indices 0 and -1 respectively.

    The edge capacities are the following:
    Edge S -> s_i has capacity suppliers[i]
    Edge s_i -> c_j has capacity min(suppliers[i], consumers[j]) (this shouldn't matter)
    Edge c_j -> T has capacity consumers[j]

    Each path from S to T corresponds to precisely one entry of the costs matrix,
    which simplifies the path-finding step of Ford-Fulkerson.
    """

    def __init__(self, supply_nodes, demand_nodes, costs):
        """Define the data structures and initial maxflow for network simplex.

        To simplify the identification of an initial maxflow, a dummy edge is
        added between the network source S and the sink T, with capacity equal
        to the known value of the maxflow (i.e., the sum of all suppliers'
        capacities).  The cost of the dummy edge is arbitrary: the algorithm
        will succeed as long as this cost is at least as high as the cost of a
        mincost-maxflow.

        The flow, capacities, and edge costs are stored as Counters, since
        their values are integers.  The attribute self._it maps a vertex index
        to the index of the last iteration of the algorithm in which that
        vertex's "potential" (the signature abstraction of the network simplex
        algorithm) was updated; storing this information allows the potentials
        to be calculated "lazily" via recursion by the function self.phi(v).
        """
        self._N = len(supply_nodes) + len(demand_nodes)
        self._parent = {0: -1, -1: -1}
        self._flow, self._it, self._cap, self._cost, self._phi = (
            Counter(),
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )

        for i, s in enumerate(supply_nodes):
            self._parent[i + 1] = 0
            self._cap[0, i + 1] = s
            # update the dummy edge flow and capacity
            self._flow[-1, 0] -= s
            self._cap[0, -1] += s

        for i, d in enumerate(demand_nodes):
            idx = len(supply_nodes) + i + 1
            self._parent[idx] = -1
            self._cap[idx, -1] = d

        for (i, s), (j, d) in product(enumerate(supply_nodes), enumerate(demand_nodes)):
            self._cap[i + 1, j + len(supply_nodes) + 1] = min(s, d)
            self._cost[i + 1, j + len(supply_nodes) + 1] = costs[i][j]
            # update the dummy edge cost
            self._cost[-1, 0] = min(self._cost[-1, 0], -costs[i][j])

        # ensure the dummy edge is more expensive than any network edge
        self._cost[-1, 0] -= 1
        self._phi[-1] = self._cost[-1, 0]
        self._it[-1] += 1

    def flow(self, src, dest):
        return self._flow[src, dest] if src < dest else -self._flow[dest, src]

    def push_flow(self, src, dest, amt):
        """Increase the flow from src -> dest by amt in the residual network."""
        if src < dest:
            self._flow[src, dest] += amt
        else:
            self._flow[dest, src] -= amt

    def cost(self, src, dest):
        """Cost of the edge src -> dest in the residual network."""
        return self._cost[src, dest] if src < dest else -self._cost[dest, src]

    def rcap(self, src, dest):
        """Capacity of the edge src -> dest in the residual network."""
        return self._cap[src, dest] - self.flow(src, dest)

    def rcost(self, src, dest):
        """Reduced cost of the edge src -> dest wrt the vertex potentials."""
        return self.cost(src, dest) - self.phi(src) + self.phi(dest)

    def phi(self, v):
        """Calculate the vertex potential of v wrt the current spanning tree."""
        if not self._it[v] == self._it[-1]:
            p = self._parent[v]
            self._phi[v] = self.phi(p) + self.cost(v, p)
            self._it[v] = self._it[-1]
        return self._phi[v]

    def get_lca(self, u, v):
        """LCA of vertices u and v in the current spanning tree."""
        seen = {u, v}
        while u != v:
            u = self._parent[u]
            if u in seen:
                return u
            seen.add(u)
            v = self._parent[v]
            if v in seen:
                return v
            seen.add(v)
        return v

    def get_eligible_edge(self):
        """Get the eligible edge of minimal ("greatest negative") reduced cost."""
        a, b, m = None, None, float("inf")
        for u in range(-1, self._N + 1):
            for v in range(-1, self._N + 1):
                if self.rcap(u, v) > 0:
                    if (r := self.rcost(u, v)) < min(0, m):
                        m = r
                        a, b = u, v
        return a, b

    def saturate_cycle(self, src, dest):
        """Push the max possible flow along the cycle (src, dest, lca(src, dest))."""
        l = self.get_lca(src, dest)
        cap = self.rcap(src, dest)
        u, v = src, dest
        while u != l:
            cap = min(cap, self.rcap(self._parent[u], u))
            u = self._parent[u]
        while v != l:
            cap = min(cap, self.rcap(v, self._parent[v]))
            v = self._parent[v]
        self.push_flow(src, dest, cap)
        u, v = src, dest
        while u != l:
            self.push_flow(self._parent[u], u, cap)
            if self.rcap(self._parent[u], u) == 0:
                src, dest = self._parent[u], u
            u = self._parent[u]
        while v != l:
            self.push_flow(v, self._parent[v], cap)
            if self.rcap(v, self._parent[v]) == 0:
                src, dest = v, self._parent[v]
            v = self._parent[v]
        return src, dest, cap

    def update_tree(self, new_u, new_v, sat_u, sat_v):
        """Add edge new_u -> new_v to the tree and remove sat_u -> sat_v."""
        if {new_u, new_v} == {sat_u, sat_v}:
            return
        l = self.get_lca(new_u, new_v)
        for v in {new_u, new_v}:
            base = v
            while v != l:
                if v in {sat_u, sat_v}:
                    x, y = base, self._parent[base]
                    while y != v:
                        z = self._parent[y]
                        self._parent[y] = x
                        x, y = y, z
                    self._parent[base] = new_u + new_v - base
                    assert self._parent[base] != base, "Oh noes!"
                    return
                v = self._parent[v]
        # one of the iterations should have returned
        raise TreeError("Failed to update tree")

    def get_mincost(self):
        """Run the network simplex algorithm."""
        total = self.cost(0, -1) * self.flow(0, -1)
        logger.debug("Initial cost = %s", total)
        while True:
            logger.debug("Starting iteration %d: total = %d", self._it[-1], total)
            self._it[-1] += 1
            u, v = self.get_eligible_edge()
            if u is None or v is None:
                break
            rcost = self.rcost(u, v)
            logger.debug("Found eligible edge (%d, %d) with rcost = %d", u, v, rcost)
            if rcost == 0:
                # the reduced cost is the amount by which the total will
                # decrease as a result of augmenting along this edge, so if
                # it's 0 then we're done
                break
            else:
                x, y, amt = self.saturate_cycle(u, v)
                self.update_tree(u, v, x, y)
                total += rcost * amt
        logger.debug("Algorithm terminated: total = %d", total)
        return total


def minimum_transportation_price(suppliers, consumers, costs):
    network = DistributionNetwork(suppliers, consumers, costs)
    return network.get_mincost()


def greedy(suppliers, consumers, costs):
    import heapq

    heap = []
    for i, row in enumerate(costs):
        for j, c in enumerate(row):
            heapq.heappush(heap, (c, i, j))
    flow, cost = 0, 0
    F = sum(consumers)
    while flow < F:
        c, i, j = heapq.heappop(heap)
        cap = min(suppliers[i], consumers[j])
        flow += cap
        suppliers[i] -= cap
        consumers[j] -= cap
        cost += c * cap
    return cost


def print_tree(parents):
    nodes = {}
    roots = []
    parents = parents.copy()
    while parents:
        n, p = parents.popitem()
        chain = [n]
        while p not in nodes and p not in chain:
            chain.insert(0, p)
            p = parents.pop(p)
        if p == chain[0]:
            nodes.setdefault(chain.pop(0), Tree(str(p)))
            roots.append(nodes[p])
        elif p in chain:
            raise ValueError(f"Cycle detected at element {p}")
        while chain:
            elem = chain.pop(0)
            nodes[elem] = nodes[p].add(str(elem))
            p = elem
    for root in roots:
        rich_print(root)

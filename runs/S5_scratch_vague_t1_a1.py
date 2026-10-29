import math
import heapq

# 想定障害物：正面やや左、根元から約60cmの細い柱（直径約10cm）。位置は不確か。
ASSUMED_OBSTACLES = [{"x": 0.58, "y": 0.15, "r": 0.05}]

L1, L2 = 0.5, 0.4
LIM = 2.98
HALF_W = 0.02      # リンク半幅
SAFETY = 0.02      # 検証系の安全余裕
GRID_M = 15        # 格子は片側15分割（中心が厳密に0）
STEP = LIM / GRID_M


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return ex, ey, ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _collides(q, obs, clr):
    ex, ey, tx, ty = _fk(q[0], q[1])
    for o in obs:
        lim = o["r"] + clr
        if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
            return True
        if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < lim:
            return True
    return False


def _edge_free(a, b, obs, clr):
    m = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    n = max(1, int(math.ceil(m / 0.02)))
    for k in range(n + 1):
        t = k / n
        q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if _collides(q, obs, clr):
            return False
    return True


def _ik(x, y):
    r2 = x * x + y * y
    r = math.sqrt(r2)
    if r > L1 + L2 + 1e-9 or r < abs(L1 - L2) - 1e-9:
        raise ValueError("target out of reach")
    c = max(-1.0, min(1.0, (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM:
            fx, fy = _fk(q1, q2)[2:]
            if math.hypot(fx - x, fy - y) < 0.005:
                sols.append((q1, q2))
    if not sols:
        raise ValueError("target unreachable within joint limits")
    return sols


def _search(goals, obs, clr):
    # 関節空間の格子 + IK解ノードでDijkstra探索（辺は遅延衝突判定）
    n = 2 * GRID_M + 1
    nodes = []
    index = {}
    for i in range(n):
        for j in range(n):
            q = ((i - GRID_M) * STEP, (j - GRID_M) * STEP)
            if not _collides(q, obs, clr):
                index[(i, j)] = len(nodes)
                nodes.append(q)
    start = index.get((GRID_M, GRID_M))
    if start is None:
        return None
    goal_ids = []
    for g in goals:
        if not _collides(g, obs, clr):
            goal_ids.append(len(nodes))
            nodes.append(g)
    if not goal_ids:
        return None
    gset = set(goal_ids)
    grid_n = len(nodes) - len(goal_ids)
    keys = {v: k for k, v in index.items()}
    R = 0.45  # IK解ノードと格子ノードを結ぶ半径[rad]

    def neighbors(u):
        res = []
        if u < grid_n:
            i, j = keys[u]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        v = index.get((i + di, j + dj))
                        if v is not None:
                            res.append(v)
            for gid in goal_ids:
                if math.dist(nodes[u], nodes[gid]) <= R:
                    res.append(gid)
        elif u in gset:
            for v in range(grid_n):
                if math.dist(nodes[u], nodes[v]) <= R:
                    res.append(v)
        return res

    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    done = set()
    cache = {}
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u in gset:
            path = [u]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return [nodes[k] for k in reversed(path)]
        for v in neighbors(u):
            if v in done:
                continue
            nd = d + math.dist(nodes[u], nodes[v])
            if nd >= dist.get(v, 1e18):
                continue
            key = (min(u, v), max(u, v))
            if key not in cache:
                cache[key] = _edge_free(nodes[u], nodes[v], obs, clr)
            if cache[key]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return None


def _shortcut(path, obs, clr):
    # 結べる限り遠くの点へ直結して経由点を減らす
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _edge_free(path[i], path[j], obs, clr):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan(target, obstacles):
    x, y = target
    assumed = not obstacles
    obs = ASSUMED_OBSTACLES if assumed else obstacles
    goals = _ik(x, y)  # 到達不可なら ValueError
    # 位置不確かさ分の追加余裕を大きい順に試す
    extras = (0.05, 0.03, 0.0) if assumed else (0.01, 0.0)
    for ex in extras:
        clr = HALF_W + SAFETY + ex
        path = _search(goals, obs, clr)
        if path:
            path = _shortcut(path, obs, clr)
            return [(q[0], q[1]) for q in path[1:]]  # 開始姿勢は除く
    raise ValueError("no collision-free path found")
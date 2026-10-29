import math

# 想定障害物: 正面やや左(約10°)、腕を伸ばしきった距離(約0.87 m)に直径10 cmの柱
# 位置は不確かなので、計画時に余裕を上乗せして避ける
ASSUMED_OBSTACLES = [{"x": 0.86, "y": 0.15, "r": 0.05}]

L1, L2 = 0.5, 0.4
LIM = 2.98
HALF_W = 0.02        # リンク太さ4cmの半分
SAFETY = 0.02        # 検証系の安全余裕
MARGINS = [0.03, 0.0]  # 位置不確かさ用の追加余裕（ダメなら0へ緩める）


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    a = q1 + q2
    return (ex, ey), (ex + L2 * math.cos(a), ey + L2 * math.sin(a))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_free(q, obs, margin):
    e, t = _fk(q[0], q[1])
    for o in obs:
        need = o["r"] + HALF_W + SAFETY + margin
        if _seg_dist(o["x"], o["y"], 0, 0, e[0], e[1]) < need:
            return False
        if _seg_dist(o["x"], o["y"], e[0], e[1], t[0], t[1]) < need:
            return False
    return True


def _edge_free(a, b, obs, margin):
    d = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(math.ceil(d / 0.01)))
    for i in range(n + 1):
        s = i / n
        q = (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
        if not _pose_free(q, obs, margin):
            return False
    return True


def _ik(target):
    x, y = target
    d2 = x * x + y * y
    d = math.sqrt(d2)
    if d > L1 + L2 or d < abs(L1 - L2):
        raise ValueError("target out of reach")
    c2 = max(-1.0, min(1.0, (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target not reachable within joint limits")
    return sols


def _search(goals, obs, margin):
    start = (0.0, 0.0)
    if not _pose_free(start, obs, margin):
        return None
    goals = [g for g in goals if _pose_free(g, obs, margin)]
    if not goals:
        return None
    # まず直線で行けるか
    for g in goals:
        if _edge_free(start, g, obs, margin):
            return [g]

    # 関節空間グリッド上でDijkstra（辺は必要時に衝突判定）
    N = 31
    vals = [-LIM + 2 * LIM * i / (N - 1) for i in range(N)]
    nodes = [(vals[i], vals[j]) for i in range(N) for j in range(N)]
    free = [_pose_free(q, obs, margin) for q in nodes]
    s_idx = (N // 2) * N + (N // 2)  # (0,0)
    G0 = len(nodes)
    extra = {}
    for gi, g in enumerate(goals):
        idx = G0 + gi
        nodes.append(g)
        free.append(True)
        extra[idx] = []
        for k in range(G0):
            if free[k] and math.hypot(nodes[k][0] - g[0], nodes[k][1] - g[1]) < 0.6:
                if _edge_free(nodes[k], g, obs, margin):
                    extra[idx].append(k)
                    extra.setdefault(k, []).append(idx)

    def neighbors(i):
        if i >= G0:
            return extra[i]
        r, c = divmod(i, N)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    out.append(rr * N + cc)
        return out + extra.get(i, [])

    cache = {}

    def ok(i, j):
        key = (i, j) if i < j else (j, i)
        if key not in cache:
            cache[key] = _edge_free(nodes[i], nodes[j], obs, margin)
        return cache[key]

    INF = float("inf")
    n = len(nodes)
    dist = [INF] * n
    prev = [-1] * n
    done = [False] * n
    dist[s_idx] = 0.0
    end = -1
    while True:
        u, best = -1, INF
        for i in range(n):
            if not done[i] and dist[i] < best:
                u, best = i, dist[i]
        if u < 0:
            return None
        if u >= G0:
            end = u
            break
        done[u] = True
        for v in neighbors(u):
            if done[v] or not free[v]:
                continue
            w = math.hypot(nodes[u][0] - nodes[v][0], nodes[u][1] - nodes[v][1])
            if dist[u] + w < dist[v] and ok(u, v):
                dist[v] = dist[u] + w
                prev[v] = u
    path = []
    k = end
    while k != -1:
        path.append(nodes[k])
        k = prev[k]
    path.reverse()
    path[0] = start

    # ショートカットで経由点を減らす
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _edge_free(path[i], path[j], obs, margin):
            j -= 1
        out.append(path[j])
        i = j
    return out[1:]


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    goals = _ik(target)  # 届かなければ ValueError
    for m in MARGINS:
        res = _search(goals, obs, m)
        if res:
            return res
    raise ValueError("no collision-free path found")
import math

L1, L2 = 0.5, 0.4
LINK_HALF = 0.02          # リンク太さ4cmの半分
SAFETY = 0.02             # 検証系の安全余裕
QMAX = 2.98
UNCERTAINTY = 0.03        # 位置不明の想定障害物に上乗せする余裕

# 想定障害物: 正面やや左、腕を伸ばしきった付近の細い柱(直径10cm)。位置は推定
ASSUMED_OBSTACLES = [{"x": 0.85, "y": 0.20, "r": 0.05}]


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return ex, ey, ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / n))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _config_free(q, obs):
    ex, ey, tx, ty = _fk(q[0], q[1])
    for o in obs:
        lim = o["r"] + LINK_HALF + SAFETY
        if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
            return False
        if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < lim:
            return False
    return True


def _edge_free(a, b, obs):
    m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(m / 0.02) + 1)
    for i in range(n + 1):
        t = i / n
        if not _config_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), obs):
            return False
    return True


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ik(x, y):
    d2 = x * x + y * y
    d = math.sqrt(d2)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for s in (1, -1):
        q2 = s * math.acos(c2)
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")
    return sols


def plan(target, obstacles):
    tx, ty = target
    if obstacles:
        obs = [dict(o) for o in obstacles]
    else:
        # 障害物情報なし: 想定障害物を余裕付きで使う
        obs = [{"x": o["x"], "y": o["y"], "r": o["r"] + UNCERTAINTY} for o in ASSUMED_OBSTACLES]

    start = (0.0, 0.0)
    goals = _ik(tx, ty)
    if not _config_free(start, obs):
        raise ValueError("start pose in collision")
    goals = [g for g in goals if _config_free(g, obs)]
    if not goals:
        raise ValueError("goal pose in collision")

    # 1) 直接移動
    for g in goals:
        if _edge_free(start, g, obs):
            return [g]

    # 2) 関節空間グリッドで幅優先探索
    h = 0.1
    N = int(QMAX / h)  # 添字 -N..N (0を含む)
    def pos(i, j):
        return (i * h, j * h)

    free_cache = {}
    def node_free(i, j):
        k = (i, j)
        if k not in free_cache:
            free_cache[k] = _config_free(pos(i, j), obs)
        return free_cache[k]

    # 各ゴールに接続可能なグリッド点
    goal_adj = {}
    for g in goals:
        ci, cj = round(g[0] / h), round(g[1] / h)
        for di in range(-3, 4):
            for dj in range(-3, 4):
                i, j = ci + di, cj + dj
                if abs(i) <= N and abs(j) <= N and node_free(i, j) \
                        and _edge_free(pos(i, j), g, obs):
                    goal_adj.setdefault((i, j), g)

    prev = {(0, 0): None}
    queue = [(0, 0)]
    found = None
    qi = 0
    while qi < len(queue):
        cur = queue[qi]
        qi += 1
        if cur in goal_adj:
            found = cur
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (cur[0] + di, cur[1] + dj)
                if nb in prev or abs(nb[0]) > N or abs(nb[1]) > N:
                    continue
                if not node_free(*nb):
                    continue
                if _edge_free(pos(*cur), pos(*nb), obs):
                    prev[nb] = cur
                    queue.append(nb)
    if found is None:
        raise ValueError("no collision-free path found")

    path = []
    k = found
    while k is not None:
        path.append(pos(*k))
        k = prev[k]
    path.reverse()
    path.append(goal_adj[found])

    # 経路の短縮（可能な限り直線で結ぶ）
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _edge_free(path[i], path[j], obs):
            j -= 1
        out.append(path[j])
        i = j
    return out[1:]
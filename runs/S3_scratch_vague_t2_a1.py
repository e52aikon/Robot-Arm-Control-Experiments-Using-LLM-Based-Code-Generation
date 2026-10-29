import math

L1, L2 = 0.5, 0.4
LINK_HALF_WIDTH = 0.02      # リンク太さ4cmの半分
SAFETY = 0.02               # 安全余裕
EXTRA = 0.005               # 離散サンプリング誤差の予備
QMAX = 2.98

# 想定障害物：左斜め前(45度方向)、根元から約0.7m、直径約10cmの柱。
# 位置不明のため、半径は 0.05 に位置不確かさの余裕 0.05 を足して 0.10 とする。
ASSUMED_OBSTACLES = [
    {"x": 0.7 * math.cos(math.pi / 4),
     "y": 0.7 * math.sin(math.pi / 4),
     "r": 0.05 + 0.05},
]


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return ex, ey, tx, ty


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の最短距離
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    t = 0.0 if dd == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / dd))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _config_free(q, obs):
    q1, q2 = q
    if abs(q1) > QMAX or abs(q2) > QMAX:
        return False
    ex, ey, tx, ty = _fk(q1, q2)
    for o in obs:
        need = o["r"] + LINK_HALF_WIDTH + SAFETY + EXTRA
        if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < need:
            return False
        if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < need:
            return False
    return True


def _edge_free(a, b, obs):
    # 関節空間の直線補間を細かくサンプリングして検査
    d = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    n = max(1, int(math.ceil(d / 0.01)))
    for k in range(n + 1):
        t = k / n
        if not _config_free((a[0] + (b[0] - a[0]) * t,
                             a[1] + (b[1] - a[1]) * t), obs):
            return False
    return True


def _ik(x, y):
    d2 = x * x + y * y
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c > 1.0 + 1e-9 or c < -1.0 - 1e-9:
        raise ValueError("target unreachable")
    c = max(-1.0, min(1.0, c))
    sols = []
    for s in (1, -1):
        q2 = s * math.acos(c)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")
    return sols


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    tx, ty = target
    goals = [g for g in _ik(tx, ty) if _config_free(g, obs)]
    start = (0.0, 0.0)
    if not _config_free(start, obs):
        raise ValueError("start in collision")
    if not goals:
        raise ValueError("target blocked by obstacle")

    # 関節空間グリッド上のBFS（0を含む格子）
    N = 60
    step = QMAX / N
    q_of = lambda i, j: (i * step, j * step)
    free = {}

    def ok(i, j):
        if (i, j) not in free:
            free[(i, j)] = (abs(i) <= N and abs(j) <= N
                            and _config_free(q_of(i, j), obs))
        return free[(i, j)]

    prev = {(0, 0): None}
    dist = {(0, 0): 0}
    queue = [(0, 0)]
    h = 0
    while h < len(queue):
        i, j = queue[h]
        h += 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                n = (i + di, j + dj)
                if n in prev or not ok(*n):
                    continue
                if not _edge_free(q_of(i, j), q_of(*n), obs):
                    continue
                prev[n] = (i, j)
                dist[n] = dist[(i, j)] + 1
                queue.append(n)

    # ゴール(IK厳密解)に直接つなげられる到達済みノードのうち最短のものを選ぶ
    best = None
    for g in goals:
        for n in queue:
            q = q_of(*n)
            dj = math.hypot(q[0] - g[0], q[1] - g[1])
            if dj > 0.4:
                continue
            cost = dist[n] + dj / step
            if best is not None and cost >= best[0]:
                continue
            if _edge_free(q, g, obs):
                best = (cost, n, g)
    if best is None:
        raise ValueError("no collision-free path")

    _, n, g = best
    nodes = []
    while n is not None:
        nodes.append(q_of(*n))
        n = prev[n]
    nodes.reverse()
    nodes.append(g)

    # 近道化（見通せる最も遠い点へ直接つなぐ）
    path = [nodes[0]]
    i = 0
    while i < len(nodes) - 1:
        j = len(nodes) - 1
        while j > i + 1 and not _edge_free(nodes[i], nodes[j], obs):
            j -= 1
        path.append(nodes[j])
        i = j

    wp = path[1:]  # 開始姿勢は除く
    if not wp:
        wp = [g]
    # 最終誤差チェック
    _, _, ex, ey = _fk(*wp[-1])
    if math.hypot(ex - tx, ey - ty) > 0.005:
        raise ValueError("final error too large")
    return wp
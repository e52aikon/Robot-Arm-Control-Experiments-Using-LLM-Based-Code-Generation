import math

L1, L2 = 0.5, 0.4
LIMIT = 2.98
HALF_W = 0.02      # リンク半幅
SAFETY = 0.02      # 検証系の安全余裕
EXTRA = 0.01       # 計画側の追加余裕
STEP = 0.05        # 探索格子の刻み [rad]

# 想定障害物: 左斜め前(45度と仮定)・根元から約0.70 m、柱の直径約10 cm。
# 位置が不確かなので、半径 = 柱の半径0.05 m + 位置誤差の見込み0.07 m としている。
_ASSUMED_DIST = 0.70
_ASSUMED_ANGLE = math.radians(45.0)
ASSUMED_OBSTACLES = [
    {"x": _ASSUMED_DIST * math.cos(_ASSUMED_ANGLE),
     "y": _ASSUMED_DIST * math.sin(_ASSUMED_ANGLE),
     "r": 0.05 + 0.07},
]


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の距離
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    t = 0.0 if d2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _make_checker(obs):
    def free(q):
        q1, q2 = q
        if abs(q1) > LIMIT or abs(q2) > LIMIT:
            return False
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        tx = ex + L2 * math.cos(q1 + q2)
        ty = ey + L2 * math.sin(q1 + q2)
        for o in obs:
            need = o["r"] + HALF_W + SAFETY + EXTRA
            if _seg_dist(o["x"], o["y"], 0.0, 0.0, ex, ey) < need:
                return False
            if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < need:
                return False
        return True

    def seg_free(a, b, step=0.01):
        m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        n = max(1, int(math.ceil(m / step)))
        for k in range(n + 1):
            t = k / n
            if not free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    return free, seg_free


def _ik(x, y):
    # 2解(肘の上下)を返す。可動範囲外は除外
    r2 = x * x + y * y
    c = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c > 1.0 + 1e-9 or c < -1.0 - 1e-9:
        raise ValueError("target unreachable")
    c = max(-1.0, min(1.0, c))
    sols = []
    for sgn in (1.0, -1.0):
        q2 = sgn * math.acos(c)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIMIT and abs(q2) <= LIMIT:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")
    return sols


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    tx, ty = target
    goals = _ik(tx, ty)
    free, seg_free = _make_checker(obs)

    start = (0.0, 0.0)
    if not free(start):
        raise ValueError("start in collision")
    goals = [g for g in goals if free(g)]
    if not goals:
        raise ValueError("goal in collision")

    # 関節空間の格子上で幅優先探索
    N = int(LIMIT / STEP)
    q_of = lambda n: (n[0] * STEP, n[1] * STEP)
    s = (0, 0)
    parent = {s: None}
    depth = {s: 0}
    queue = [s]
    qi = 0
    nbrs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    while qi < len(queue):
        cur = queue[qi]
        qi += 1
        for di, dj in nbrs:
            nb = (cur[0] + di, cur[1] + dj)
            if nb in parent or abs(nb[0]) > N or abs(nb[1]) > N:
                continue
            if free(q_of(nb)) and seg_free(q_of(cur), q_of(nb), 0.025):
                parent[nb] = cur
                depth[nb] = depth[cur] + 1
                queue.append(nb)

    # ゴールに直接つなげる格子点のうち最短のものを選ぶ
    best = None
    for g in goals:
        for nd in parent:
            q = q_of(nd)
            d = max(abs(q[0] - g[0]), abs(q[1] - g[1]))
            if d > 0.15:
                continue
            cost = depth[nd] * STEP + d
            if best is not None and cost >= best[0]:
                continue
            if seg_free(q, g):
                best = (cost, nd, g)
    if best is None:
        raise ValueError("no collision-free path")

    _, nd, g = best
    chain = []
    while nd is not None:
        chain.append(q_of(nd))
        nd = parent[nd]
    chain.reverse()
    chain.append(g)

    # 近道化(貪欲)
    path = [chain[0]]
    i = 0
    last = len(chain) - 1
    while i < last:
        j = last
        while j > i + 1 and not seg_free(chain[i], chain[j]):
            j -= 1
        path.append(chain[j])
        i = j

    return path[1:]  # 開始姿勢は除く
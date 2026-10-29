import math

L1, L2 = 0.5, 0.4
LIM = 2.98
HALF_W = 0.02      # リンク半幅（太さ4cm）
MARGIN = 0.02      # 安全余裕
EXTRA = 0.005      # 数値誤差用の追加余裕
TOL = 1e-9


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _make_checker(obstacles):
    # 姿勢が障害物と干渉しないか
    def config_free(q):
        elbow, tip = _fk(q[0], q[1])
        for o in obstacles:
            lim = o["r"] + HALF_W + MARGIN + EXTRA
            if _seg_dist(o["x"], o["y"], (0.0, 0.0), elbow) < lim:
                return False
            if _seg_dist(o["x"], o["y"], elbow, tip) < lim:
                return False
        return True

    # 関節空間の直線補間が安全か
    def seg_free(qa, qb):
        d = max(abs(qa[0] - qb[0]), abs(qa[1] - qb[1]))
        n = int(math.ceil(d / 0.02)) + 1
        for i in range(n + 1):
            t = i / n
            q = (qa[0] + (qb[0] - qa[0]) * t, qa[1] + (qb[1] - qa[1]) * t)
            if not config_free(q):
                return False
        return True

    return config_free, seg_free


def _ik(target):
    x, y = target
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1 + TOL or c2 < -1 - TOL:
        return []
    c2 = max(-1.0, min(1.0, c2))
    sols = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)))
        q2 = _wrap(q2)
        if abs(q1) <= LIM and abs(q2) <= LIM:
            sols.append((q1, q2))
    return sols


def plan(target, obstacles):
    goals = _ik(target)
    if not goals:
        raise ValueError("target unreachable")

    config_free, seg_free = _make_checker(obstacles)
    start = (0.0, 0.0)
    if not config_free(start):
        raise ValueError("start in collision")

    goals = [g for g in goals if config_free(g)]
    if not goals:
        raise ValueError("target blocked by obstacle")

    # 1) 直接移動できるか（関節移動量が小さい解を優先）
    for g in sorted(goals, key=lambda g: abs(g[0]) + abs(g[1])):
        if seg_free(start, g):
            return [g]

    # 2) 関節空間グリッドを幅優先探索
    N = 60
    step = LIM / N
    cache = {}

    def node_free(i, j):
        k = (i, j)
        if k not in cache:
            cache[k] = config_free((i * step, j * step))
        return cache[k]

    from collections import deque
    s = (0, 0)
    parent = {s: None}
    dq = deque([s])
    found = None
    while dq:
        cur = dq.popleft()
        q = (cur[0] * step, cur[1] * step)
        for g in goals:
            if max(abs(q[0] - g[0]), abs(q[1] - g[1])) < 0.8 and seg_free(q, g):
                found = (cur, g)
                break
        if found:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                n = (cur[0] + di, cur[1] + dj)
                if n in parent or abs(n[0]) > N or abs(n[1]) > N:
                    continue
                if not node_free(*n):
                    continue
                if not seg_free(q, (n[0] * step, n[1] * step)):
                    continue
                parent[n] = cur
                dq.append(n)

    if not found:
        raise ValueError("no collision-free path")

    cell, goal = found
    chain = []
    while cell is not None:
        chain.append((cell[0] * step, cell[1] * step))
        cell = parent[cell]
    chain.reverse()          # 先頭は開始姿勢
    chain.append(goal)

    # 3) 近道化で経由点を削減
    out = []
    i = 0
    while i < len(chain) - 1:
        j = len(chain) - 1
        while j > i + 1 and not seg_free(chain[i], chain[j]):
            j -= 1
        out.append(chain[j])
        i = j
    return out
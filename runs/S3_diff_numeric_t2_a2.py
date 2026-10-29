import math

L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01
PLAN_EXTRA = 0.01     # 計画時の追加余裕
GRID_STEP = 0.1       # 関節空間グリッド刻み [rad]
SAMPLE_STEP = 0.01    # 干渉チェックのサンプル間隔 [rad]

# 原因: 前回は出力が長すぎて plan() の途中で切れた。
# 対策: 説明とコメントを減らし、探索部を簡潔にした。


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def _seg_point_dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    len2 = abx * abx + aby * aby
    t = 0.0 if len2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / len2))
    return math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


def _pose_collides(q1, q2, obstacles, extra=0.0):
    elbow, tip = _fk(q1, q2)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        cl = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN + extra
        if _seg_point_dist(c, (0.0, 0.0), elbow) < cl:
            return True
        if _seg_point_dist(c, elbow, tip) < cl:
            return True
    return False


def _path_collides(qa, qb, obstacles, extra=0.0):
    d = max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1]))
    n = max(1, int(math.ceil(d / SAMPLE_STEP)))
    for i in range(n + 1):
        s = i / n
        if _pose_collides(qa[0] + s * (qb[0] - qa[0]),
                          qa[1] + s * (qb[1] - qa[1]), obstacles, extra):
            return True
    return False


def _search_path(goals, obstacles):
    """関節空間グリッドの幅優先探索。経由点リストか None を返す。"""
    kmax = int(JOINT_LIMIT / GRID_STEP)
    to_q = lambda i: (i[0] * GRID_STEP, i[1] * GRID_STEP)
    parent = {(0, 0): None}
    queue = [(0, 0)]
    head = 0
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while head < len(queue):
        cur = queue[head]
        head += 1
        qc = to_q(cur)
        for g in goals:
            if max(abs(g[0] - qc[0]), abs(g[1] - qc[1])) <= 2.0 * GRID_STEP:
                if not _path_collides(qc, g, obstacles, PLAN_EXTRA):
                    chain = []
                    n = cur
                    while n is not None:
                        chain.append(to_q(n))
                        n = parent[n]
                    chain.reverse()
                    chain.append(g)
                    return chain
        for dx, dy in nbrs:
            nx = (cur[0] + dx, cur[1] + dy)
            if nx in parent or abs(nx[0]) > kmax or abs(nx[1]) > kmax:
                continue
            if _path_collides(qc, to_q(nx), obstacles, PLAN_EXTRA):
                continue
            parent[nx] = cur
            queue.append(nx)
    return None


def _simplify(chain, obstacles):
    """干渉しない範囲で経由点を間引く。"""
    result = []
    i = 0
    while i < len(chain) - 1:
        j = len(chain) - 1
        while j > i + 1 and _path_collides(chain[i], chain[j], obstacles,
                                           PLAN_EXTRA):
            j -= 1
        result.append(chain[j])
        i = j
    return result


def plan(target, obstacles):
    x, y = target
    r2 = x * x + y * y
    r = math.sqrt(r2)

    eps = 1e-9
    if r > L1 + L2 + eps or r < abs(L1 - L2) - eps:
        raise ValueError("target is out of reach")

    c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))

    candidates = []
    for s2 in (s2_abs, -s2_abs):
        q2 = _wrap(math.atan2(s2, c2))
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2))
        if abs(q1) > JOINT_LIMIT or abs(q2) > JOINT_LIMIT:
            continue
        _, tip = _fk(q1, q2)
        if math.hypot(tip[0] - x, tip[1] - y) > TOLERANCE:
            continue
        if _pose_collides(q1, q2, obstacles, PLAN_EXTRA):
            continue  # 目標姿勢自体が干渉する解は除外
        candidates.append((q1, q2))

    if not candidates:
        raise ValueError("no valid joint solution within joint limits")

    candidates.sort(key=lambda q: abs(q[0]) + abs(q[1]))

    start = (0.0, 0.0)
    # まず直行を試す
    for q in candidates:
        if not _path_collides(start, q, obstacles, PLAN_EXTRA):
            return [q]

    # 直行不可なら迂回経路を探索
    chain = _search_path(candidates, obstacles)
    if chain is None:
        raise ValueError("no collision-free path found for this target")
    return _simplify(chain, obstacles)
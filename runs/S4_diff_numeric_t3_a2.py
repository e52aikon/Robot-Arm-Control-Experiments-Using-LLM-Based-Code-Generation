import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01
GRID_STEP = 0.1     # 探索格子の刻み [rad]
GOAL_LINK = 0.3     # 格子点から目標へ直結する最大距離 [rad]

# 原因: 直前の回答が長すぎて途中で切れた。
# 対策: 直線経路が干渉する場合の格子BFS探索を最小限の実装にした。


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


def _pose_collides(q1, q2, obstacles):
    elbow, tip = _fk(q1, q2)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        cl = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN
        if _seg_point_dist(c, (0.0, 0.0), elbow) < cl:
            return True
        if _seg_point_dist(c, elbow, tip) < cl:
            return True
    return False


def _path_collides(a, b, obstacles, step=0.01):
    d = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(math.ceil(d / step)))
    for i in range(n + 1):
        s = i / n
        if _pose_collides(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), obstacles):
            return True
    return False


def _search(goals, obstacles):
    """関節空間の格子をBFS。目標へ直結できる点が見つかれば経路を返す。"""
    n = int(JOINT_LIMIT / GRID_STEP)
    pt = lambda k: (k[0] * GRID_STEP, k[1] * GRID_STEP)
    if _pose_collides(0.0, 0.0, obstacles):
        return None
    parent = {(0, 0): None}
    queue = [(0, 0)]
    bad = set()
    i = 0
    while i < len(queue):
        cur = queue[i]
        i += 1
        p = pt(cur)
        for g in goals:
            if max(abs(p[0] - g[0]), abs(p[1] - g[1])) <= GOAL_LINK \
                    and not _path_collides(p, g, obstacles):
                path = []
                k = cur
                while k is not None:
                    path.append(pt(k))
                    k = parent[k]
                path.reverse()
                return path + [g]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = (cur[0] + di, cur[1] + dj)
                if (di == 0 and dj == 0) or abs(nb[0]) > n or abs(nb[1]) > n:
                    continue
                if nb in parent or nb in bad:
                    continue
                q = pt(nb)
                if _pose_collides(q[0], q[1], obstacles):
                    bad.add(nb)
                    continue
                if _path_collides(p, q, obstacles):
                    continue
                parent[nb] = cur
                queue.append(nb)
    return None


def _shorten(path, obstacles):
    """直線で結べる限り中間点を飛ばす（先頭は開始姿勢）。"""
    out = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and _path_collides(path[i], path[j], obstacles):
            j -= 1
        out.append(path[j])
        i = j
    return out


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
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2)
        q1 = _wrap(q1)
        q2 = _wrap(q2)
        if abs(q1) > JOINT_LIMIT or abs(q2) > JOINT_LIMIT:
            continue
        _, tip = _fk(q1, q2)
        if math.hypot(tip[0] - x, tip[1] - y) > TOLERANCE:
            continue
        candidates.append((q1, q2))

    if not candidates:
        raise ValueError("no valid joint solution within joint limits")

    candidates.sort(key=lambda q: abs(q[0]) + abs(q[1]))

    start = (0.0, 0.0)
    # まず直線一発で行ける解を探す
    for q in candidates:
        if not _path_collides(start, q, obstacles):
            return [q]

    # 駄目なら折りたたみ等を含む迂回経路を探索
    path = _search(candidates, obstacles)
    if path is None:
        raise ValueError("no collision-free path found for this target")
    return _shorten(path, obstacles)
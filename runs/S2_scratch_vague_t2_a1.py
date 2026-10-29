import math

# 想定障害物: 正面やや左、腕を伸ばしきった付近の細い柱（直径約10cm、位置は不確か）
ASSUMED_OBSTACLES = [{"x": 0.85, "y": 0.28, "r": 0.05}]

L1, L2 = 0.5, 0.4
LIM = 2.98
HALF_T = 0.02       # リンク半幅
SAFETY = 0.02       # 検証系の安全余裕
EXTRA_MARGINS = [0.08, 0.04, 0.0]  # 位置不確かさ用の追加余裕（大きい順に試す）


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    a = q1 + q2
    return (ex, ey), (ex + L2 * math.cos(a), ey + L2 * math.sin(a))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / n))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_ok(q, obs, extra):
    elbow, tip = _fk(*q)
    for o in obs:
        need = o["r"] + extra + HALF_T + SAFETY
        if _seg_dist(o["x"], o["y"], 0, 0, elbow[0], elbow[1]) < need:
            return False
        if _seg_dist(o["x"], o["y"], elbow[0], elbow[1], tip[0], tip[1]) < need:
            return False
    return True


def _path_ok(path, obs, extra):
    prev = (0.0, 0.0)
    for q in path:
        if abs(q[0]) > LIM or abs(q[1]) > LIM:
            return False
        n = max(2, int(max(abs(q[0] - prev[0]), abs(q[1] - prev[1])) / 0.02) + 1)
        for i in range(n + 1):
            t = i / n
            p = (prev[0] + (q[0] - prev[0]) * t, prev[1] + (q[1] - prev[1]) * t)
            if not _pose_ok(p, obs, extra):
                return False
        prev = q
    return True


def _length(path):
    s, prev = 0.0, (0.0, 0.0)
    for q in path:
        s += math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
    return s


def _ik(target):
    x, y = target
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1.0 or c2 < -1.0:
        raise ValueError("target out of reach")
    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target violates joint limits")
    return sols


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    goals = _ik(target)

    # 候補: 直行 / 中間点1つ経由（関節空間グリッド）
    step = 0.2
    n = int(LIM / step)
    grid = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    cands = []
    for g in goals:
        cands.append([g])
        for v in grid:
            cands.append([v, g])
    cands.sort(key=_length)

    for extra in EXTRA_MARGINS:
        for path in cands:
            if _path_ok(path, obs, extra):
                return path
    raise ValueError("no collision-free path found")
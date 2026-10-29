# 原因: 前回は出力が長く途中で切れた。今回は簡潔に再構成。
# また肘上げ解の直線補間は障害物に干渉するため、経由点探索を追加した。
import math

L1 = 0.5
L2 = 0.4
HALF_W = 0.02
MARGIN = 0.02 + 0.005   # 安全余裕 + サンプリング誤差分
JOINT_LIMIT = 2.98
TOLERANCE = 0.01
STEP = 0.005            # 経路サンプリング刻み [rad]


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
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
        clr = ob["r"] + HALF_W + MARGIN
        if _seg_point_dist(c, (0.0, 0.0), elbow) < clr:
            return True
        if _seg_point_dist(c, elbow, tip) < clr:
            return True
    return False


def _path_collides(qa, qb, obstacles):
    steps = max(100, int(max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1])) / STEP) + 1)
    for i in range(steps + 1):
        s = i / steps
        if _pose_collides(qa[0] + s * (qb[0] - qa[0]),
                          qa[1] + s * (qb[1] - qa[1]), obstacles):
            return True
    return False


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
        candidates.append((q1, q2))

    if not candidates:
        raise ValueError("no valid joint solution within joint limits")

    candidates.sort(key=lambda q: abs(q[0]) + abs(q[1]))

    start = (0.0, 0.0)
    # 1) 直接移動
    for q in candidates:
        if not _path_collides(start, q, obstacles):
            return [q]

    # 2) 経由点を1つ挟む（軸ごとの単純経由 → 格子探索）
    n = 24
    grid = [(-JOINT_LIMIT + 2 * JOINT_LIMIT * i / n,
             -JOINT_LIMIT + 2 * JOINT_LIMIT * j / n)
            for i in range(n + 1) for j in range(n + 1)]
    for q in candidates:
        vias = [(0.0, q[1]), (q[0], 0.0)]
        vias += sorted(grid, key=lambda v: abs(v[0]) + abs(v[1])
                       + abs(q[0] - v[0]) + abs(q[1] - v[1]))
        for v in vias:
            if _pose_collides(v[0], v[1], obstacles):
                continue
            if _path_collides(start, v, obstacles):
                continue
            if _path_collides(v, q, obstacles):
                continue
            return [v, q]

    raise ValueError("no collision-free path found for this target")
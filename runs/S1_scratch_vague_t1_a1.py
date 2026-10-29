import math

# 想定障害物（位置は不確か）：アーム右側（-y）、少し前方の細い柱（直径約10cm）
ASSUMED_OBSTACLES = [{"x": 0.25, "y": -0.25, "r": 0.05}]

L1, L2 = 0.5, 0.4
HALF_W = 0.02        # リンク半幅（太さ4cm）
SAFETY = 0.02        # 検証系の安全余裕
UNCERTAINTY = 0.05   # 想定障害物の位置誤差に対する追加余裕
QMAX = 2.98
STEP = 0.02          # 衝突判定の補間刻み [rad]


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_ok(q, obs, extra):
    e, t = _fk(*q)
    for o in obs:
        lim = o["r"] + HALF_W + SAFETY + extra
        if _seg_dist(o["x"], o["y"], 0, 0, e[0], e[1]) < lim:
            return False
        if _seg_dist(o["x"], o["y"], e[0], e[1], t[0], t[1]) < lim:
            return False
    return True


def _path_ok(a, b, obs, extra):
    # 関節空間の直線補間を細かくサンプリングして判定
    n = max(2, int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / STEP) + 1)
    for i in range(n + 1):
        s = i / n
        q = (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
        if not _pose_ok(q, obs, extra):
            return False
    return True


def _ik(x, y):
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 < -1 - 1e-9 or c2 > 1 + 1e-9:
        return []
    c2 = max(-1.0, min(1.0, c2))
    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    return sols


def _cost(path):
    c, p = 0.0, (0.0, 0.0)
    for q in path:
        c += math.hypot(q[0] - p[0], q[1] - p[1])
        p = q
    return c


def plan(target, obstacles):
    obs = list(obstacles) if obstacles else ASSUMED_OBSTACLES
    extra = 0.0 if obstacles else UNCERTAINTY  # 想定値使用時のみ余裕を上乗せ

    x, y = target
    sols = _ik(x, y)
    if not sols:
        raise ValueError("target is unreachable")

    start = (0.0, 0.0)
    best, best_c = None, float("inf")

    # 1) 直接移動（逆運動学の2解を比較）
    for g in sols:
        if _path_ok(start, g, obs, extra):
            c = _cost([g])
            if c < best_c:
                best, best_c = [g], c
    if best:
        return best

    # 2) 経由点を1つ挟む（関節空間の粗いグリッド探索）
    n = int(QMAX / 0.4)
    grid = [i * 0.4 for i in range(-n, n + 1)]
    for g in sols:
        for a in grid:
            for b in grid:
                w = (a, b)
                if not _pose_ok(w, obs, extra):
                    continue
                c = _cost([w, g])
                if c >= best_c:
                    continue
                if _path_ok(start, w, obs, extra) and _path_ok(w, g, obs, extra):
                    best, best_c = [w, g], c
    if best:
        return best

    raise ValueError("no collision-free path found")
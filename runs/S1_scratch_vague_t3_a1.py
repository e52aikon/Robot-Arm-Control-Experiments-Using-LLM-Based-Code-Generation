import math

L1, L2 = 0.5, 0.4
LINK_HALF = 0.02        # リンク太さ 4cm の半分
SAFETY = 0.02           # 検証系の安全余裕
UNCERTAINTY = 0.05      # 位置不明の想定障害物に対する追加余裕
Q_LIM = 2.98

# 想定障害物：アーム右側（y<0）の少し前にある直径10cmの柱（位置は推定）
ASSUMED_OBSTACLES = [{"x": 0.30, "y": -0.20, "r": 0.05}]


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_dist(px, py, ax, ay, bx, by):
    # 点 P と線分 AB の最短距離
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    t = 0.0 if d2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _clearance(q1, q2, obs, extra):
    # 全障害物に対する最小余裕（負なら接触）
    (ex, ey), (tx, ty) = _fk(q1, q2)
    m = float("inf")
    for o in obs:
        need = o["r"] + LINK_HALF + SAFETY + extra
        d = min(_seg_dist(o["x"], o["y"], 0, 0, ex, ey),
                _seg_dist(o["x"], o["y"], ex, ey, tx, ty))
        m = min(m, d - need)
    return m


def _path_ok(wps, obs, extra, steps=100):
    # 開始姿勢(0,0)から関節空間で直線補間した経路を細かくサンプリングして検証
    prev = (0.0, 0.0)
    for w in wps:
        for i in range(steps + 1):
            t = i / steps
            q1 = prev[0] + (w[0] - prev[0]) * t
            q2 = prev[1] + (w[1] - prev[1]) * t
            if _clearance(q1, q2, obs, extra) < 0:
                return False
        prev = w
    return True


def _cost(wps):
    prev, c = (0.0, 0.0), 0.0
    for w in wps:
        c += abs(w[0] - prev[0]) + abs(w[1] - prev[1])
        prev = w
    return c


def plan(target, obstacles):
    tx, ty = target
    if obstacles:
        obs, extra = obstacles, 0.0
    else:
        obs, extra = ASSUMED_OBSTACLES, UNCERTAINTY  # 位置不確実なので余裕を上乗せ

    # 逆運動学（肘の2解）
    d2 = tx * tx + ty * ty
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c > 1 + 1e-9 or c < -1 - 1e-9:
        raise ValueError("target is out of reach")
    c = max(-1.0, min(1.0, c))

    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        if abs(q1) <= Q_LIM and abs(q2) <= Q_LIM:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target not reachable within joint limits")

    # 候補経路：直接 / 先に q1 だけ回す / 先に q2 だけ曲げる
    cands = []
    for q1, q2 in sols:
        cands.append([(q1, q2)])
        cands.append([(q1, 0.0), (q1, q2)])
        cands.append([(0.0, q2), (q1, q2)])

    valid = []
    for wps in cands:
        # 最終誤差の確認
        _, (ex, ey) = _fk(*wps[-1])
        if math.hypot(ex - tx, ey - ty) > 0.005:
            continue
        if _path_ok(wps, obs, extra):
            valid.append(wps)

    if not valid:
        raise ValueError("no collision-free path found")
    return min(valid, key=_cost)
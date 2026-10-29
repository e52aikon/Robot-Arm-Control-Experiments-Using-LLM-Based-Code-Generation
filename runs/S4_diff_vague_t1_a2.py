import math

# 原因: 前回は説明とコメントが長すぎて出力が途中で切れた。今回は簡潔にした。
L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01

# 想定障害物: 左斜め前(45°仮定)・距離0.85mの柱（実半径0.05m）。
# 位置が不確かなので、位置誤差0.15m分を上乗せして r=0.20 とした。
ASSUMED_OBSTACLES = [
    {"x": 0.85 * math.cos(math.radians(45.0)),
     "y": 0.85 * math.sin(math.radians(45.0)),
     "r": 0.20},
]

FOLD_Q2 = 2.6  # 迂回時に肘を畳む角度 [rad]（可動範囲内）


def _wrap(a):
    """角度を (-pi, pi] に正規化する。"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _fk(q1, q2):
    """順運動学: 肘と先端の座標を返す。"""
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_point_dist(p, a, b):
    """線分 ab と点 p の最短距離。"""
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    len2 = abx * abx + aby * aby
    t = 0.0 if len2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / len2))
    cx, cy = a[0] + t * abx, a[1] + t * aby
    return math.hypot(p[0] - cx, p[1] - cy)


def _pose_collides(q1, q2, obstacles):
    """ある姿勢で、いずれかのリンクが障害物（+太さ+余裕）に接触するか。"""
    elbow, tip = _fk(q1, q2)
    base = (0.0, 0.0)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        clearance = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN
        if _seg_point_dist(c, base, elbow) < clearance:
            return True
        if _seg_point_dist(c, elbow, tip) < clearance:
            return True
    return False


def _path_collides(q_start, q_end, obstacles, steps=100):
    """関節空間の直線補間経路を細かくサンプリングして干渉を調べる。"""
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def _route_collides(route, obstacles):
    """開始姿勢から経由点列を順にたどる経路全体の干渉を調べる。"""
    prev = (0.0, 0.0)
    for q in route:
        if _path_collides(prev, q, obstacles):
            return True
        prev = q
    return False


def plan(target, obstacles):
    # 障害物が空なら想定障害物を使う
    if not obstacles:
        obstacles = ASSUMED_OBSTACLES

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

    for q in candidates:
        # まず直行、だめなら肘を畳んで迂回（畳む向きは両方試す）
        routes = [[q]]
        sgn = 1.0 if q[1] >= 0 else -1.0
        for f in (sgn * FOLD_Q2, -sgn * FOLD_Q2):
            routes.append([(0.0, f), (q[0], f), q])
        for route in routes:
            if not _route_collides(route, obstacles):
                return route

    raise ValueError("no collision-free path found for this target")
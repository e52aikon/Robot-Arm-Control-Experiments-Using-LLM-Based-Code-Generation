import math

# 想定障害物: 左斜め前(約45°)、根元から約0.7 m、直径約10 cmの柱。
# 位置が不確かなので、半径は 0.05 に位置誤差分を足して 0.08 とした。
_ANG = math.radians(45.0)
ASSUMED_OBSTACLES = [
    {"x": 0.70 * math.cos(_ANG), "y": 0.70 * math.sin(_ANG), "r": 0.08}
]

L1, L2 = 0.5, 0.4
QMAX = 2.98
HALF_T = 0.02      # リンク太さの半分
MARGIN = 0.02      # 安全余裕
EPS = 0.005        # 追加の余裕


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の距離
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _clearance(q1, q2, obs):
    # 障害物表面とリンク表面の最小すきま（負なら接触）
    (ex, ey), (tx, ty) = _fk(q1, q2)
    c = 1.0
    for o in obs:
        d = min(_seg_dist(o["x"], o["y"], 0, 0, ex, ey),
                _seg_dist(o["x"], o["y"], ex, ey, tx, ty))
        c = min(c, d - o["r"] - HALF_T)
    return c


def _path_clearance(path, obs, step=0.02):
    # 開始姿勢(0,0)から各経由点へ関節空間で直線補間して評価
    cur = (0.0, 0.0)
    worst = 1.0
    for q in path:
        if abs(q[0]) > QMAX or abs(q[1]) > QMAX:
            return -1.0
        n = max(2, int(max(abs(q[0] - cur[0]), abs(q[1] - cur[1])) / step) + 1)
        for i in range(n + 1):
            s = i / n
            worst = min(worst, _clearance(cur[0] + (q[0] - cur[0]) * s,
                                          cur[1] + (q[1] - cur[1]) * s, obs))
        cur = q
    return worst


def _travel(path):
    cur, tot = (0.0, 0.0), 0.0
    for q in path:
        tot += abs(q[0] - cur[0]) + abs(q[1] - cur[1])
        cur = q
    return tot


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ik(x, y):
    d2 = x * x + y * y
    d = math.sqrt(d2)
    if d > L1 + L2 or d < abs(L1 - L2):
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c2)
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")
    return sols


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    x, y = target
    sols = _ik(x, y)
    need = MARGIN + EPS

    # 1) 直接移動（両肘形状）から、すきま最大のものを選ぶ
    best, best_key = None, None
    for q in sols:
        path = [q]
        c = _path_clearance(path, obs)
        if c >= need:
            key = (min(c, 0.15), -_travel(path))
            if best_key is None or key > best_key:
                best, best_key = path, key
    if best:
        return best

    # 2) 直接が無理なら、関節空間の格子点を1つ経由
    step = 0.2
    n = int(QMAX / step)
    grid = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    for q in sols:
        for v in grid:
            if _clearance(v[0], v[1], obs) < need:
                continue
            path = [v, q]
            c = _path_clearance(path, obs)
            if c >= need:
                key = (min(c, 0.15), -_travel(path))
                if best_key is None or key > best_key:
                    best, best_key = path, key
    if best:
        return best
    raise ValueError("no collision-free path found")
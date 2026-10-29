import math

# 原因: 前回は出力が長すぎて途中で切れた。今回は docstring と余分な処理を削って短縮した。
# 想定柱: 正面やや左(約15°)・根元から60cm・直径10cm。位置は目安なので実測したら更新すること
ASSUMED_OBSTACLES = [{"x": 0.58, "y": 0.155, "r": 0.05}]

L1, L2 = 0.5, 0.4
HW, MARGIN = 0.02, 0.02
LIM, TOL = 2.98, 0.01
FOLD = 2.9  # 肘を畳む角度


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def _dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    l2 = abx * abx + aby * aby
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2))
    return math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


def _hit(q1, q2, obs):
    e, t = _fk(q1, q2)
    for o in obs:
        c = (o["x"], o["y"])
        k = o["r"] + HW + MARGIN
        if _dist(c, (0, 0), e) < k or _dist(c, e, t) < k:
            return True
    return False


def _seg_hit(a, b, obs, n=400):
    for i in range(n + 1):
        s = i / n
        if _hit(a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), obs):
            return True
    return False


def _route_ok(route, obs):
    prev = (0.0, 0.0)
    for q in route:
        if _seg_hit(prev, q, obs):
            return False
        prev = q
    return True


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES  # 空なら想定障害物
    x, y = target
    r2 = x * x + y * y
    r = math.sqrt(r2)
    if r > L1 + L2 + 1e-9 or r < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")

    c2 = max(-1.0, min(1.0, (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    s2a = math.sqrt(max(0.0, 1 - c2 * c2))

    cands = []
    for s2 in (s2a, -s2a):
        q2 = _wrap(math.atan2(s2, c2))
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2))
        if abs(q1) > LIM or abs(q2) > LIM:
            continue
        _, tip = _fk(q1, q2)
        if math.hypot(tip[0] - x, tip[1] - y) > TOL:
            continue
        cands.append((q1, q2))
    if not cands:
        raise ValueError("no valid joint solution within joint limits")
    cands.sort(key=lambda q: abs(q[0]) + abs(q[1]))

    # 直行 → だめなら「肘を畳む → 腕ごと回す → 肘を開く」の経由点ルート
    for q in cands:
        if _route_ok([q], obs):
            return [q]
        for sg in (-1.0, 1.0):
            route = [(0.0, sg * FOLD), (q[0], sg * FOLD), q]
            if _route_ok(route, obs):
                return route

    raise ValueError("no collision-free path found for this target")
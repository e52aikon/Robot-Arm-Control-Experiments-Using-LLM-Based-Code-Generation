import math

# 想定障害物: 左斜め前(約45°)、根元から約0.85 m、直径約10 cmの柱
_D, _ANG = 0.85, math.radians(45)
ASSUMED_OBSTACLES = [{"x": _D * math.cos(_ANG), "y": _D * math.sin(_ANG), "r": 0.05}]

L1, L2 = 0.5, 0.4
LIM = 2.98
HALF_W = 0.02       # リンク半幅
SAFETY = 0.02       # 安全余裕
UNCERTAIN = 0.10    # 想定障害物の位置誤差の上乗せ
FOLD = LIM - 0.3    # 肘を畳む角度
STEPS = 200


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    t = 0.0 if d2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_ok(q1, q2, obs):
    e, t = _fk(q1, q2)
    for o in obs:
        need = o["r"] + HALF_W + SAFETY + o.get("extra", 0.0)
        if _seg_dist(o["x"], o["y"], 0, 0, e[0], e[1]) < need:
            return False
        if _seg_dist(o["x"], o["y"], e[0], e[1], t[0], t[1]) < need:
            return False
    return True


def _path_ok(start, wps, obs):
    prev = start
    for w in wps:
        if abs(w[0]) > LIM or abs(w[1]) > LIM:
            return False
        for i in range(STEPS + 1):
            s = i / STEPS
            if not _pose_ok(prev[0] + (w[0] - prev[0]) * s,
                            prev[1] + (w[1] - prev[1]) * s, obs):
                return False
        prev = w
    return True


def plan(target, obstacles):
    x, y = target
    if obstacles:
        obs = [dict(o, extra=0.0) for o in obstacles]
    else:
        obs = [dict(o, extra=UNCERTAIN) for o in ASSUMED_OBSTACLES]

    # 逆運動学（到達可否）
    r = math.hypot(x, y)
    if r > L1 + L2 or r < abs(L1 - L2):
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    phi = math.atan2(y, x)

    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c2)
        q1 = phi - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")

    start = (0.0, 0.0)
    # 候補: 肘を畳んでから q1 を回し、最後に展開（障害物の位置誤差に強い）
    for q1, q2 in sols:
        for fs in (-1, 1):
            f = fs * FOLD
            wps = [(0.0, f), (q1, f), (q1, q2)]
            if _path_ok(start, wps, obs):
                return wps
    # 畳まない直接経路
    for q1, q2 in sols:
        if _path_ok(start, [(q1, q2)], obs):
            return [(q1, q2)]
    raise ValueError("no collision-free path found")
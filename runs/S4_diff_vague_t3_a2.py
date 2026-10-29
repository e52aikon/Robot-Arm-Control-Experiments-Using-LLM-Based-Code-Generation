# 原因: 前回の出力が長すぎて途中で切れ、plan が未完成だった。コメントを減らして全体を出力し直す。
import math

L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01

# 想定柱: 左斜め前(45°と仮定)、根元から85cm、直径約10cm。
# 位置が不確かなので半径に10cm上乗せして保守的に扱う。
_PD = 0.85
_PB = math.radians(45.0)
ASSUMED_OBSTACLES = [
    {"x": _PD * math.cos(_PB), "y": _PD * math.sin(_PB), "r": 0.05 + 0.10}
]

# 迂回用: 第2リンクを折りたたむ量の候補 [rad]
FOLD_MAGNITUDES = (2.8, 2.4, 2.0)


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
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


def _pose_collides(q1, q2, obstacles):
    elbow, tip = _fk(q1, q2)
    base = (0.0, 0.0)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        clr = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN
        if _seg_point_dist(c, base, elbow) < clr:
            return True
        if _seg_point_dist(c, elbow, tip) < clr:
            return True
    return False


def _path_collides(q_start, q_end, obstacles, steps=100):
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def _route_collides(route, obstacles):
    """開始姿勢から経由点列を順に辿る経路の干渉検査。"""
    prev = (0.0, 0.0)
    for q in route:
        if _path_collides(prev, q, obstacles):
            return True
        prev = q
    return False


def plan(target, obstacles):
    if not obstacles:
        obstacles = ASSUMED_OBSTACLES  # 空なら想定障害物を使う

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
        # まず直行を試す
        if not _route_collides([q], obstacles):
            return [q]
        # 直行が不可なら、腕を折りたたんで q1 を回し、最後に伸ばす迂回路を試す
        sign_first = 1.0 if q[1] >= 0 else -1.0
        for sg in (sign_first, -sign_first):
            for f in FOLD_MAGNITUDES:
                route = [(0.0, sg * f), (q[0], sg * f), q]
                if not _route_collides(route, obstacles):
                    return route

    raise ValueError("no collision-free path found for this target")
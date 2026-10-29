# 原因: 前回は出力が途中で切れた。今回は短く、完全なコードにした。
import math

# 想定障害物: 根元から約60cm、正面やや左(約15°)、直径10cm。位置は推定値
ASSUMED_OBSTACLES = [{"x": 0.58, "y": 0.155, "r": 0.05}]

L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01
FOLD = 2.9  # 肘を畳む角度（可動範囲の手前）


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


def _path_collides(pts, obstacles, steps=200):
    # 開始姿勢を含む経由点列の各区間を関節空間の直線補間で検査
    for a, b in zip(pts[:-1], pts[1:]):
        for i in range(steps + 1):
            s = i / steps
            if _pose_collides(a[0] + s * (b[0] - a[0]),
                              a[1] + s * (b[1] - a[1]), obstacles):
                return True
    return False


def plan(target, obstacles):
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

    start = (0.0, 0.0)
    for q in candidates:
        # 直接移動、だめなら肘を畳んで q1 を回し、その後に伸ばす経路を試す
        routes = [[q]]
        for f in (-FOLD, FOLD):
            routes.append([(0.0, f), (q[0], f), q])
        for route in routes:
            if not _path_collides([start] + route, obstacles):
                return route

    raise ValueError("no collision-free path found for this target")
import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値 1cm
PLAN_MARGIN = 0.01       # サンプリング間隔による見落とし分の追加余裕
SAMPLE_STEP = 0.02       # 干渉検査の関節角刻み [rad]

# 前回の不合格原因: 回答が長すぎて途中で切れた（冗長な探索コードと説明）。
# 修正: 簡潔な構成にした。
# 直線経路が障害物に当たる場合は、肘を畳んだ姿勢を経由して回避する。


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
        clearance = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN + PLAN_MARGIN
        if _seg_point_dist(c, base, elbow) < clearance:
            return True
        if _seg_point_dist(c, elbow, tip) < clearance:
            return True
    return False


def _path_collides(q_start, q_end, obstacles):
    """直線補間経路を、移動量に応じた刻みでサンプリングして干渉を調べる。"""
    d = max(abs(q_end[0] - q_start[0]), abs(q_end[1] - q_start[1]))
    steps = max(1, int(math.ceil(d / SAMPLE_STEP)))
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def _chain_ok(start, wps, obstacles):
    """経由点列全体が無干渉か。"""
    prev = start
    for w in wps:
        if _path_collides(prev, w, obstacles):
            return False
        prev = w
    return True


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
    fold = JOINT_LIMIT - 0.05   # 肘をほぼ畳んだ角度
    for q in candidates:
        # 1) 直線で行けるならそれを採用
        if _chain_ok(start, [q], obstacles):
            return [q]
        # 2) 肘を畳む → q1 だけ回す → 目標姿勢へ展開、の3点経路
        #    畳む向きは両方試す（目標の q2 と同じ符号を先に）
        sgn = 1.0 if q[1] >= 0 else -1.0
        for sg in (sgn, -sgn):
            f = sg * fold
            wps = [(0.0, f), (q[0], f), q]
            if _chain_ok(start, wps, obstacles):
                return wps

    raise ValueError("no collision-free path found for this target")
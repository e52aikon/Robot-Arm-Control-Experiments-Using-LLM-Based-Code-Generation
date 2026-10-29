import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値 1cm（IK誤差はこれよりはるかに小さい）

# 追加: 干渉検査の刻み幅と、計画時に上乗せする余裕
CHECK_STEP = 0.002       # 補間のサンプリング刻み [rad]（先端で約2mm以下）
VIA_GRID_STEP = 0.25     # 迂回経由点を探す関節角グリッド刻み [rad]
EXTRA_MARGINS = (0.01, 0.0)  # まず余裕を上乗せして探し、駄目なら検証系と同じ基準


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


def _pose_collides(q1, q2, obstacles, extra=0.0):
    """ある姿勢で、いずれかのリンクが障害物（+太さ+余裕+extra）に接触するか。"""
    elbow, tip = _fk(q1, q2)
    base = (0.0, 0.0)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        clearance = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN + extra
        if _seg_point_dist(c, base, elbow) < clearance:
            return True
        if _seg_point_dist(c, elbow, tip) < clearance:
            return True
    return False


def _path_collides(q_start, q_end, obstacles, extra=0.0):
    """関節空間の直線補間経路を、移動量に応じた刻みでサンプリングして干渉を調べる。"""
    d = max(abs(q_end[0] - q_start[0]), abs(q_end[1] - q_start[1]))
    steps = max(1, int(math.ceil(d / CHECK_STEP)))
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles, extra):
            return True
    return False


def _via_points():
    """迂回用の経由点候補（関節空間グリッド）。"""
    n = int(JOINT_LIMIT / VIA_GRID_STEP)
    vals = [k * VIA_GRID_STEP for k in range(-n, n + 1)]
    return [(a, b) for a in vals for b in vals]


def plan(target, obstacles):
    x, y = target
    r2 = x * x + y * y
    r = math.sqrt(r2)

    # 到達可能範囲: |L1-L2| <= r <= L1+L2（境界は微小な誤差を許容）
    eps = 1e-9
    if r > L1 + L2 + eps or r < abs(L1 - L2) - eps:
        raise ValueError("target is out of reach")

    # 余弦定理で q2 を求める（数値誤差対策でクランプ）
    c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))

    # 肘の向きが異なる2つの解（肘上げ/肘下げ）を候補にする
    candidates = []
    for s2 in (s2_abs, -s2_abs):
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2)
        q1 = _wrap(q1)
        q2 = _wrap(q2)
        # 可動範囲チェック
        if abs(q1) > JOINT_LIMIT or abs(q2) > JOINT_LIMIT:
            continue
        # 順運動学で最終誤差を確認
        _, tip = _fk(q1, q2)
        if math.hypot(tip[0] - x, tip[1] - y) > TOLERANCE:
            continue
        candidates.append((q1, q2))

    if not candidates:
        raise ValueError("no valid joint solution within joint limits")

    # 開始姿勢 (0, 0) からの関節移動量が小さい解を優先
    candidates.sort(key=lambda q: abs(q[0]) + abs(q[1]))

    start = (0.0, 0.0)
    vias = _via_points()
    for extra in EXTRA_MARGINS:
        # 1) 直行経路（障害物がなければ常に通る）
        for q in candidates:
            if not _path_collides(start, q, obstacles, extra):
                return [q]

        # 2) 直行が干渉する場合: 経由点を1つ挟んだ2区間経路を探す
        best = None
        for q in candidates:
            if _pose_collides(q[0], q[1], obstacles, extra):
                continue  # 目標姿勢そのものが干渉
            for v in vias:
                cost = (abs(v[0]) + abs(v[1])
                        + abs(q[0] - v[0]) + abs(q[1] - v[1]))
                if best is not None and cost >= best[0]:
                    continue
                if _pose_collides(v[0], v[1], obstacles, extra):
                    continue
                if _path_collides(start, v, obstacles, extra):
                    continue
                if _path_collides(v, q, obstacles, extra):
                    continue
                best = (cost, [v, q])
        if best is not None:
            return best[1]

    raise ValueError("no collision-free path found for this target")
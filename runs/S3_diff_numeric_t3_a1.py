import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値 1cm（IK誤差はこれよりはるかに小さい）
FOLD_MARGIN = 0.05       # [追加] 折りたたみ姿勢で可動限界から離す量 [rad]


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


def _path_collides(q_start, q_end, obstacles, steps=None):
    """関節空間の直線補間経路を細かくサンプリングして干渉を調べる。"""
    if steps is None:
        # [変更] 関節移動量に応じて分割数を決める（約 0.01 rad 刻み）
        d = max(abs(q_end[0] - q_start[0]), abs(q_end[1] - q_start[1]))
        steps = max(100, int(d / 0.01) + 1)
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def _route_collides(start, route, obstacles):
    """[追加] 経由点列を順にたどる経路全体の干渉を調べる。"""
    prev = start
    for q in route:
        if _path_collides(prev, q, obstacles):
            return True
        prev = q
    return False


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
    fold = JOINT_LIMIT - FOLD_MARGIN
    for q in candidates:
        # まず直行経路。障害物がなければ常に通る
        if not _path_collides(start, q, obstacles):
            return [q]

        # [追加] 直行が干渉する場合: 第2リンクを折りたたんでから
        # 第1関節を回し、最後に展開する迂回経路を試す
        # （折りたたみ中は腕全体が根元付近に収まる）
        for sign in (1.0, -1.0):
            qf = sign * fold
            route = [(0.0, qf), (q[0], qf), q]
            if not _route_collides(start, route, obstacles):
                return route

    raise ValueError("no collision-free path found for this target")
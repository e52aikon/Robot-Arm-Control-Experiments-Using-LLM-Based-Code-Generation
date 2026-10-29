import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値 1cm（IK誤差はこれよりはるかに小さい）

# 想定障害物: 正面やや左（約10°）、腕を伸ばしきった付近（根元から約0.86m）の細い柱。
# 実寸は直径10cm（半径0.05）だが、正確な位置が不明なので位置の不確かさ分
# 0.03m を上乗せして r=0.08 とする。開始姿勢（腕を一直線に伸ばした状態）が
# 干渉しない範囲に収めている（リンク2と柱中心の距離0.15 > 0.08+0.04）。
ASSUMED_OBSTACLES = [{"x": 0.85, "y": 0.15, "r": 0.08}]


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
        # 関節移動量に応じて分割数を決める（約0.01radごと）
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
    """start から route の各経由点へ順に動く経路全体の干渉を調べる。"""
    prev = start
    for q in route:
        if _path_collides(prev, q, obstacles):
            return True
        prev = q
    return False


def _via_candidates(q):
    """直接行けない場合に挟む中継姿勢の候補（移動量の小さい順）。"""
    # まず「肘だけ先に畳む／根元だけ先に回す」といった構造的な候補
    vias = [(0.0, q[1]), (0.0, -q[1]), (q[0], 0.0)]
    # 続いて関節空間の格子点（全体の関節移動量が小さい順）
    step = 0.3
    n = int(JOINT_LIMIT / step)
    grid = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            v = (i * step, j * step)
            cost = abs(v[0]) + abs(v[1]) + abs(q[0] - v[0]) + abs(q[1] - v[1])
            grid.append((cost, v))
    grid.sort(key=lambda t: t[0])
    vias.extend(v for _, v in grid)
    return [v for v in vias if abs(v[0]) <= JOINT_LIMIT and abs(v[1]) <= JOINT_LIMIT]


def plan(target, obstacles):
    # 障害物情報が空なら想定障害物を使う
    if not obstacles:
        obstacles = ASSUMED_OBSTACLES

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

    # 1) まず直接移動（障害物がなければ常にここで通る）
    for q in candidates:
        if not _route_collides(start, [q], obstacles):
            return [q]

    # 2) 直接では干渉する場合、中継姿勢を1つ挟んだ経路を探す
    for q in candidates:
        for via in _via_candidates(q):
            if not _route_collides(start, [via, q], obstacles):
                return [via, q]

    raise ValueError("no collision-free path found for this target")
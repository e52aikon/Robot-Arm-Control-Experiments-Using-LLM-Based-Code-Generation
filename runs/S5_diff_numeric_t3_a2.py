import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さの半分
SAFETY_MARGIN = 0.02     # 安全余裕
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値

# 原因: 前回は出力が長すぎて途中で切れた。直線経路は障害物に干渉するため、
#       関節空間BFSで迂回する経由点を求める版を簡潔に出し直す。
GRID_N = 30              # [追加] グリッド分割数（±JOINT_LIMIT を 2*N 分割）
EDGE_STEP = 0.005        # [追加] 経路干渉チェックの刻み [rad]


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
        # [変更] 移動量に応じて分割数を決める
        d = max(abs(q_end[0] - q_start[0]), abs(q_end[1] - q_start[1]))
        steps = max(1, int(math.ceil(d / EDGE_STEP)))
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def _search(goals, obstacles):
    """[追加] 関節空間グリッドのBFS。goals のいずれかへ直線で届く点で終了。"""
    h = JOINT_LIMIT / GRID_N
    pos = lambda n: (n[0] * h, n[1] * h)
    start = (0, 0)
    parent = {start: None}
    queue = [start]
    k = 0
    while k < len(queue):
        cur = queue[k]
        k += 1
        qc = pos(cur)
        for g in goals:
            if max(abs(qc[0] - g[0]), abs(qc[1] - g[1])) <= 0.3 \
                    and not _path_collides(qc, g, obstacles):
                path = []
                n = cur
                while n is not None:
                    path.append(pos(n))
                    n = parent[n]
                path.reverse()
                return path + [g]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = (cur[0] + di, cur[1] + dj)
                if (di == 0 and dj == 0) or nb in parent:
                    continue
                if abs(nb[0]) > GRID_N or abs(nb[1]) > GRID_N:
                    continue
                if _path_collides(qc, pos(nb), obstacles):
                    continue
                parent[nb] = cur
                queue.append(nb)
    return None


def _shorten(path, obstacles):
    """[追加] 直線で結べる限り中間点を飛ばして経由点を減らす。"""
    res = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and _path_collides(path[i], path[j], obstacles):
            j -= 1
        res.append(path[j])
        i = j
    return res


def plan(target, obstacles):
    x, y = target
    r2 = x * x + y * y
    r = math.sqrt(r2)

    # 到達可能範囲チェック
    eps = 1e-9
    if r > L1 + L2 + eps or r < abs(L1 - L2) - eps:
        raise ValueError("target is out of reach")

    # 余弦定理で q2 を求める
    c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))

    # 肘の向きが異なる2つの解を候補にする
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

    # 開始姿勢からの関節移動量が小さい解を優先
    candidates.sort(key=lambda q: abs(q[0]) + abs(q[1]))

    start = (0.0, 0.0)
    # 直線で行ければそのまま返す
    for q in candidates:
        if not _path_collides(start, q, obstacles):
            return [q]

    # [追加] 直線が干渉する場合はグリッド探索で迂回経路を作る
    # 到達先は干渉しない解のみ
    goals = [q for q in candidates if not _pose_collides(q[0], q[1], obstacles)]
    path = _search(goals, obstacles) if goals else None
    if path is None:
        raise ValueError("no collision-free path found for this target")
    return _shorten(path, obstacles)
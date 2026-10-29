import math

# 原因: 前回は説明とコードが長すぎて出力が途中で切れた。今回は探索部を簡潔にした。
# 直線補間だと先端が障害物(0.49,0.7)付近を通って干渉するため、迂回経路が必要。
L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01
SAMPLE_MARGIN = 0.01   # 追加: 補間サンプリングの離散化誤差ぶんの余裕
EDGE_STEP = 0.01       # 追加: 干渉チェックの関節角刻み [rad]
GRID_STEP = 0.1        # 追加: 迂回探索用グリッド刻み [rad]


def _wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_point_dist(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    len2 = abx * abx + aby * aby
    t = 0.0 if len2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / len2))
    cx, cy = a[0] + t * abx, a[1] + t * aby
    return math.hypot(p[0] - cx, p[1] - cy)


def _pose_collides(q1, q2, obstacles):
    elbow, tip = _fk(q1, q2)
    base = (0.0, 0.0)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        clearance = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN + SAMPLE_MARGIN
        if _seg_point_dist(c, base, elbow) < clearance:
            return True
        if _seg_point_dist(c, elbow, tip) < clearance:
            return True
    return False


def _path_collides(qa, qb, obstacles):
    # 変更: 関節移動量に応じて分割数を決める
    d = max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1]))
    steps = max(1, int(math.ceil(d / EDGE_STEP)))
    for i in range(steps + 1):
        s = i / steps
        if _pose_collides(qa[0] + s * (qb[0] - qa[0]),
                          qa[1] + s * (qb[1] - qa[1]), obstacles):
            return True
    return False


def _plan_via_grid(goals, obstacles):
    """追加: 関節空間グリッドをBFSし、ゴールへ直線で行ける所まで迂回する。"""
    n = int(JOINT_LIMIT / GRID_STEP)
    cache = {}

    def nq(nd):
        return (nd[0] * GRID_STEP, nd[1] * GRID_STEP)

    def free(nd):
        if nd not in cache:
            q = nq(nd)
            cache[nd] = not _pose_collides(q[0], q[1], obstacles)
        return cache[nd]

    parent = {(0, 0): None}
    queue = [(0, 0)]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        qc = nq(cur)
        # ゴールへ直線で繋がるか（近いノードのみ検査して計算量を抑える）
        for g in goals:
            if max(abs(qc[0] - g[0]), abs(qc[1] - g[1])) <= 1.0 \
                    and not _path_collides(qc, g, obstacles):
                chain = []
                nd = cur
                while nd is not None:
                    chain.append(nq(nd))
                    nd = parent[nd]
                chain.reverse()
                chain.append(g)
                return _shortcut(chain, obstacles)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = (cur[0] + di, cur[1] + dj)
                if (di == 0 and dj == 0) or abs(nb[0]) > n or abs(nb[1]) > n \
                        or nb in parent or not free(nb):
                    continue
                if _path_collides(qc, nq(nb), obstacles):
                    continue
                parent[nb] = cur
                queue.append(nb)
    return None


def _shortcut(chain, obstacles):
    """追加: 干渉しない範囲で経由点を間引く（開始点は含めない）。"""
    out = []
    i = 0
    last = len(chain) - 1
    while i < last:
        j = last
        while j > i + 1 and _path_collides(chain[i], chain[j], obstacles):
            j -= 1
        out.append(chain[j])
        i = j
    return out


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
    # 最終姿勢自体が干渉する解は除外
    goals = [q for q in candidates if not _pose_collides(q[0], q[1], obstacles)]

    # まず直線（1経由点）で行けるか
    for q in goals:
        if not _path_collides(start, q, obstacles):
            return [q]

    # 直線で無理なら迂回経路を探索
    path = _plan_via_grid(goals, obstacles)
    if path:
        return path

    raise ValueError("no collision-free path found for this target")
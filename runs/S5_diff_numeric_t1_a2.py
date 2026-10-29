import math

# 原因: 前回は説明・コードが長すぎて出力が途中で切れた。探索部を簡潔にして再出力。
L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02
SAFETY_MARGIN = 0.02
JOINT_LIMIT = 2.98
TOLERANCE = 0.01
GRID = 0.05  # [追加] 関節空間の格子幅 [rad]


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


def _pose_collides(q1, q2, obstacles, extra=0.005):
    # extra: 補間サンプル間の取りこぼし対策の上乗せ余裕
    elbow, tip = _fk(q1, q2)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        cl = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN + extra
        if _seg_point_dist(c, (0.0, 0.0), elbow) < cl:
            return True
        if _seg_point_dist(c, elbow, tip) < cl:
            return True
    return False


def _path_collides(a, b, obstacles):
    # [変更] 移動量に応じてサンプル数を増やす
    d = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    steps = max(100, int(math.ceil(d / 0.005)))
    for i in range(steps + 1):
        s = i / steps
        if _pose_collides(a[0] + s * (b[0] - a[0]),
                          a[1] + s * (b[1] - a[1]), obstacles):
            return True
    return False


def _search(goal, obstacles):
    """[追加] 関節空間の格子を幅優先探索し、経由点を返す。"""
    n = int(JOINT_LIMIT / GRID)
    cache = {}

    def free(i, j):
        if abs(i) > n or abs(j) > n:
            return False
        if (i, j) not in cache:
            cache[(i, j)] = not _pose_collides(i * GRID, j * GRID,
                                               obstacles, 0.03)
        return cache[(i, j)]

    g = (max(-n, min(n, round(goal[0] / GRID))),
         max(-n, min(n, round(goal[1] / GRID))))
    if not free(*g):
        return None
    parent = {(0, 0): None}
    queue = [(0, 0)]
    k = 0
    while k < len(queue) and g not in parent:
        ci, cj = queue[k]
        k += 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = (ci + di, cj + dj)
                if nb not in parent and free(*nb):
                    parent[nb] = (ci, cj)
                    queue.append(nb)
    if g not in parent:
        return None
    cells = []
    c = g
    while c is not None:
        cells.append(c)
        c = parent[c]
    cells.reverse()
    pts = [(i * GRID, j * GRID) for i, j in cells]
    pts[-1] = goal  # 最終点は厳密なIK解にする
    # 近道化: 干渉しない限り遠い点へ直接つなぐ
    out = []
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and _path_collides(pts[i], pts[j], obstacles):
            j -= 1
        out.append(pts[j])
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
    for q in candidates:
        if not _path_collides(start, q, obstacles):
            return [q]
    # [変更] 直線で通らなければ関節空間探索で迂回経路を作る
    for q in candidates:
        path = _search(q, obstacles)
        if path:
            return path

    raise ValueError("no collision-free path found for this target")
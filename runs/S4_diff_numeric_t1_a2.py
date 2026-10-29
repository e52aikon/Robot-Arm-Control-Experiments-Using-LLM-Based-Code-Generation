import math

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さの半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値
PLAN_EXTRA = 0.005       # 計画時の追加余裕（サンプリング取りこぼし対策）
EDGE_STEP = 0.005        # 干渉検査の関節角刻み [rad]
GRID_N = 30              # 関節空間グリッド（片側分割数）

# 前回は説明が長すぎて出力が途中で切れた。
# 今回は直線経路が障害物に当たるため、関節空間グリッドBFSで迂回経路を探索する。


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


def _pose_collides(q1, q2, obstacles, extra=0.0):
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


def _path_collides(q_start, q_end, obstacles, extra=PLAN_EXTRA):
    d = max(abs(q_end[0] - q_start[0]), abs(q_end[1] - q_start[1]))
    steps = max(1, int(d / EDGE_STEP) + 1)
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles, extra):
            return True
    return False


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

    # まず直線経路を試す
    for q in candidates:
        if not _path_collides(start, q, obstacles):
            return [q]

    # 関節空間グリッドBFS（8近傍）
    h = JOINT_LIMIT / GRID_N

    def pt(n):
        return (n[0] * h, n[1] * h)

    prev = {(0, 0): None}
    queue = [(0, 0)]
    head = 0
    found = None
    while head < len(queue) and found is None:
        node = queue[head]
        head += 1
        p = pt(node)
        for q in candidates:
            if max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 0.6 and \
                    not _path_collides(p, q, obstacles):
                found = (node, q)
                break
        if found:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = (node[0] + di, node[1] + dj)
                if (di == 0 and dj == 0) or nb in prev:
                    continue
                if abs(nb[0]) > GRID_N or abs(nb[1]) > GRID_N:
                    continue
                if _path_collides(p, pt(nb), obstacles):
                    continue
                prev[nb] = node
                queue.append(nb)

    if found is None:
        raise ValueError("no collision-free path found for this target")

    # 経路復元
    node, goal = found
    nodes = []
    while node is not None:
        nodes.append(node)
        node = prev[node]
    nodes.reverse()
    pts = [start] + [pt(n) for n in nodes[1:]] + [goal]

    # ショートカットで経由点を減らす
    out = []
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and _path_collides(pts[i], pts[j], obstacles):
            j -= 1
        out.append(pts[j])
        i = j
    return out
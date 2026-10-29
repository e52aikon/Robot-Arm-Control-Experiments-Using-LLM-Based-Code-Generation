import math

L1, L2 = 0.5, 0.4
HALF_W = 0.02      # リンク半幅
MARGIN = 0.02      # 安全余裕
EXTRA = 0.005      # 補間サンプリング誤差用の追加余裕
QLIM = 2.98
STEP = 0.01        # 衝突チェックの刻み [rad]
H = 0.1            # 探索グリッド間隔 [rad]


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    a = q1 + q2
    return ex, ey, ex + L2 * math.cos(a), ey + L2 * math.sin(a)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def plan(target, obstacles):
    tx, ty = target
    d = math.hypot(tx, ty)

    # --- 到達可能性（逆運動学）---
    c = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1 + 1e-9:
        raise ValueError("target unreachable")
    c = max(-1.0, min(1.0, c))

    def in_limits(q):
        return abs(q[0]) <= QLIM and abs(q[1]) <= QLIM

    def config_free(q):
        ex, ey, px, py = _fk(q[0], q[1])
        for o in obstacles:
            lim = o["r"] + HALF_W + MARGIN + EXTRA
            if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
                return False
            if _seg_dist(o["x"], o["y"], ex, ey, px, py) < lim:
                return False
        return True

    def edge_free(a, b):
        n = max(1, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / STEP)))
        for i in range(n + 1):
            t = i / n
            if not config_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        q = (q1, q2)
        if not in_limits(q):
            continue
        fx, fy = _fk(q1, q2)[2:]
        if math.hypot(fx - tx, fy - ty) > 0.005:
            continue
        goals.append(q)
    if not goals:
        raise ValueError("no IK solution within joint limits")
    goals = [g for g in goals if config_free(g)]
    if not goals:
        raise ValueError("goal in collision")

    start = (0.0, 0.0)
    if not config_free(start):
        raise ValueError("start in collision")

    # --- 直接移動できるか ---
    for g in goals:
        if edge_free(start, g):
            return [g]

    # --- 関節空間グリッド上のBFS ---
    N = int(QLIM / H)
    node_ok = {}

    def ok(i, j):
        k = (i, j)
        if k not in node_ok:
            node_ok[k] = config_free((i * H, j * H))
        return node_ok[k]

    depth = {(0, 0): 0}
    parent = {(0, 0): None}
    queue = [(0, 0)]
    qi = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while qi < len(queue):
        cur = queue[qi]
        qi += 1
        for di, dj in dirs:
            nb = (cur[0] + di, cur[1] + dj)
            if abs(nb[0]) > N or abs(nb[1]) > N or nb in depth:
                continue
            if not ok(*nb):
                continue
            if not edge_free((cur[0] * H, cur[1] * H), (nb[0] * H, nb[1] * H)):
                continue
            depth[nb] = depth[cur] + 1
            parent[nb] = cur
            queue.append(nb)

    # --- ゴールへの接続（最良候補を選ぶ）---
    best = None
    for g in goals:
        for nd, dp in depth.items():
            qd = (nd[0] * H, nd[1] * H)
            dist = max(abs(qd[0] - g[0]), abs(qd[1] - g[1]))
            if dist > 0.3:
                continue
            cost = dp + dist / H
            if best is not None and cost >= best[0]:
                continue
            if edge_free(qd, g):
                best = (cost, nd, g)
    if best is None:
        raise ValueError("no collision-free path found")

    _, nd, g = best
    chain = []
    while nd is not None:
        chain.append((nd[0] * H, nd[1] * H))
        nd = parent[nd]
    chain.reverse()
    pts = chain + [g]

    # --- 経由点の間引き（可視なら飛ばす）---
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not edge_free(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out[1:]
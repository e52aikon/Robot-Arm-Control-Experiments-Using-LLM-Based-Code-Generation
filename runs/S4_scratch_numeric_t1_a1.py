import math

L1, L2 = 0.5, 0.4
LIM = 2.98
HALF_W = 0.02      # リンク半幅
SAFE = 0.02        # 安全余裕
MARGIN = 0.01      # 補間サンプル間の取りこぼし用の追加余裕
STEP = 0.01        # 衝突チェックの刻み [rad]
H = 0.05           # 探索グリッド幅 [rad]


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の距離
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return ex, ey, ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2)


def plan(target, obstacles):
    tx, ty = target
    d = math.hypot(tx, ty)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target unreachable")

    def cfg_free(q1, q2):
        if abs(q1) > LIM or abs(q2) > LIM:
            return False
        ex, ey, px, py = _fk(q1, q2)
        for o in obstacles:
            need = o["r"] + HALF_W + SAFE + MARGIN
            if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < need:
                return False
            if _seg_dist(o["x"], o["y"], ex, ey, px, py) < need:
                return False
        return True

    def edge_free(a, b):
        n = max(1, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / STEP)))
        for k in range(n + 1):
            t = k / n
            if not cfg_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
                return False
        return True

    # 逆運動学（肘上・肘下の2解）
    c = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c = max(-1.0, min(1.0, c))
    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM and cfg_free(q1, q2):
            fx, fy = _fk(q1, q2)[2:]
            if math.hypot(fx - tx, fy - ty) < 0.005:
                goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible IK solution")

    start = (0.0, 0.0)
    if not cfg_free(*start):
        raise ValueError("start configuration in collision")

    # 直線補間で行けるならそれで終わり
    for g in goals:
        if edge_free(start, g):
            return [g]

    # 関節空間グリッドのBFS
    m = int(LIM / H)
    def q_of(n):
        return ((n[0] - m) * H, (n[1] - m) * H)

    free_cache = {}
    def node_free(n):
        if n not in free_cache:
            free_cache[n] = cfg_free(*q_of(n))
        return free_cache[n]

    s_node = (m, m)
    parent = {s_node: None}
    queue = [s_node]
    head = 0
    found = None
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while head < len(queue) and found is None:
        cur = queue[head]
        head += 1
        qc = q_of(cur)
        for g in goals:
            if max(abs(qc[0] - g[0]), abs(qc[1] - g[1])) < 0.4 and edge_free(qc, g):
                found = (cur, g)
                break
        if found:
            break
        for di, dj in dirs:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] <= 2 * m and 0 <= nb[1] <= 2 * m) or nb in parent:
                continue
            if not node_free(nb):
                continue
            if not edge_free(qc, q_of(nb)):
                continue
            parent[nb] = cur
            queue.append(nb)
    if found is None:
        raise ValueError("no collision-free path")

    node, g = found
    path = []
    while node is not None:
        path.append(q_of(node))
        node = parent[node]
    path.reverse()
    path[0] = start
    path.append(g)

    # 近道化（遠い点から順に直線で結べるか確認）
    out = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
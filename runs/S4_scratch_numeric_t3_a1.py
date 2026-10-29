import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    LIM = 2.98
    HALF_W = 0.02      # リンク半幅 (太さ4cm)
    SAFE = 0.02        # 安全余裕
    EXTRA = 0.01       # 補間サンプリング誤差用の追加余裕
    tx, ty = float(target[0]), float(target[1])

    # ---- 順運動学 ----
    def fk(q1, q2):
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        a = q1 + q2
        return ex, ey, ex + L2 * math.cos(a), ey + L2 * math.sin(a)

    # 点と線分の距離
    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    # 姿勢の衝突判定（リンク2本 vs 全円柱）
    def collides(q):
        ex, ey, tipx, tipy = fk(q[0], q[1])
        for o in obstacles:
            lim = o["r"] + HALF_W + SAFE + EXTRA
            if seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
                return True
            if seg_dist(o["x"], o["y"], ex, ey, tipx, tipy) < lim:
                return True
        return False

    # 関節空間の直線補間が安全か
    def edge_free(a, b):
        m = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        n = max(1, int(math.ceil(m / 0.01)))
        for k in range(n + 1):
            t = k / n
            if collides((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    # ---- 逆運動学（肘上・肘下の2解） ----
    d2 = tx * tx + ty * ty
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1.0 + 1e-9 or c2 < -1.0 - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, c2))
    base = math.atan2(ty, tx)
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = base - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM:
            goals.append((q1, q2))
    if not goals:
        raise ValueError("target outside joint limits")
    goals = [g for g in goals if not collides(g)]
    if not goals:
        raise ValueError("goal pose collides with obstacle")

    start = (0.0, 0.0)
    if collides(start):
        raise ValueError("start pose collides with obstacle")

    # 直接動けるならそれで終わり
    for g in goals:
        if edge_free(start, g):
            return [g]

    # ---- 関節空間グリッド上でBFS（辺は遅延評価） ----
    N = 40                       # 偶数 → 原点がグリッド点になる
    step = 2 * LIM / N
    def q_of(i, j):
        return (-LIM + i * step, -LIM + j * step)

    cache = {}
    def node_ok(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = not collides(q_of(i, j))
        return cache[(i, j)]

    s_node = (N // 2, N // 2)
    parent = {s_node: None}
    queue = [s_node]
    qi = 0
    found = None
    while qi < len(queue) and found is None:
        cur = queue[qi]; qi += 1
        qc = q_of(*cur)
        # このノードからゴールへ接続できるか
        for g in goals:
            if math.hypot(qc[0] - g[0], qc[1] - g[1]) < 0.6 and edge_free(qc, g):
                found = (cur, g)
                break
        if found:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (cur[0] + di, cur[1] + dj)
                if nb in parent or not (0 <= nb[0] <= N and 0 <= nb[1] <= N):
                    continue
                if not node_ok(*nb):
                    continue
                if edge_free(qc, q_of(*nb)):
                    parent[nb] = cur
                    queue.append(nb)

    if found is None:
        raise ValueError("no collision-free path found")

    # 経路復元
    nodes = []
    n = found[0]
    while n is not None:
        nodes.append(n)
        n = parent[n]
    nodes.reverse()
    path = [start] + [q_of(*n) for n in nodes[1:]] + [found[1]]

    # ---- ショートカットで経由点を削減 ----
    result = [path[0]]
    i = 0
    last = len(path) - 1
    while i < last:
        j = last
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        result.append(path[j])
        i = j
    return result[1:]
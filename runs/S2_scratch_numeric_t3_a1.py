import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    QMAX = 2.98
    HALF_W = 0.02      # リンク半幅
    MARGIN = 0.02      # 安全余裕
    EXTRA = 0.004      # 補間・数値誤差用の追加余裕
    tx, ty = float(target[0]), float(target[1])

    # --- 到達可能性チェックと逆運動学（肘上・肘下の2解） ---
    d2 = tx * tx + ty * ty
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        raise ValueError("target unreachable")
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))  # [-pi, pi] に正規化
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            goals.append((q1, q2))
    if not goals:
        raise ValueError("target unreachable within joint limits")

    # --- 衝突判定 ---
    def seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        ll = vx * vx + vy * vy
        t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / ll))
        return math.hypot(px - (ax + t * vx), py - (ay + t * vy))

    def cfg_free(q):
        q1, q2 = q
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        fx = ex + L2 * math.cos(q1 + q2)
        fy = ey + L2 * math.sin(q1 + q2)
        for o in obstacles:
            lim = o["r"] + HALF_W + MARGIN + EXTRA
            if seg_dist(o["x"], o["y"], 0.0, 0.0, ex, ey) < lim:
                return False
            if seg_dist(o["x"], o["y"], ex, ey, fx, fy) < lim:
                return False
        return True

    def edge_free(a, b):
        m = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        n = max(1, int(math.ceil(m / 0.01)))
        for k in range(n + 1):
            t = k / n
            if not cfg_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    start = (0.0, 0.0)
    goals = [g for g in goals if cfg_free(g)]
    if not goals:
        raise ValueError("goal in collision")

    # --- 直線補間で届くならそのまま ---
    goals.sort(key=lambda g: abs(g[0]) + abs(g[1]))
    for g in goals:
        if edge_free(start, g):
            return [g]

    # --- 関節空間グリッドの幅優先探索 ---
    h = 0.1
    n = int(QMAX / h)
    def pos(i, j):
        return (i * h, j * h)
    if not cfg_free(start):
        raise ValueError("start in collision")
    parent = {(0, 0): None}
    depth = {(0, 0): 0}
    queue = [(0, 0)]
    head = 0
    while head < len(queue):
        i, j = queue[head]
        head += 1
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if abs(ni) > n or abs(nj) > n or (ni, nj) in parent:
                continue
            if edge_free(pos(i, j), pos(ni, nj)):
                parent[(ni, nj)] = (i, j)
                depth[(ni, nj)] = depth[(i, j)] + 1
                queue.append((ni, nj))

    # ゴールに接続できる最良ノードを選ぶ
    best = None
    for g in goals:
        for node, dp in depth.items():
            p = pos(*node)
            dist = math.hypot(p[0] - g[0], p[1] - g[1])
            if dist > 0.35:
                continue
            cost = dp + dist / h
            if best is not None and cost >= best[0]:
                continue
            if edge_free(p, g):
                best = (cost, node, g)
    if best is None:
        raise ValueError("no collision-free path found")

    _, node, g = best
    path = []
    while node is not None:
        path.append(pos(*node))
        node = parent[node]
    path.reverse()
    path.append(g)

    # --- ショートカットで経由点を減らす ---
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        result.append(path[j])
        i = j
    return result[1:]
import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    LIM = 2.98
    # リンク半幅0.02 + 安全余裕0.02 + 追加余裕0.005
    M = 0.02 + 0.02 + 0.005
    tx, ty = target

    # --- 到達可能性チェック ---
    d = math.hypot(tx, ty)
    if d > L1 + L2 or d < abs(L1 - L2):
        raise ValueError("unreachable target")

    # --- 幾何ヘルパー ---
    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def cfg_free(q):
        q1, q2 = q
        if abs(q1) > LIM or abs(q2) > LIM:
            return False
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        px = ex + L2 * math.cos(q1 + q2)
        py = ey + L2 * math.sin(q1 + q2)
        for o in obstacles:
            lim = o["r"] + M
            if seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
                return False
            if seg_dist(o["x"], o["y"], ex, ey, px, py) < lim:
                return False
        return True

    def edge_free(a, b):
        m = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        n = int(math.ceil(m / 0.02)) + 1
        for k in range(n + 1):
            t = k / n
            if not cfg_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    # --- 逆運動学（肘上・肘下の2解） ---
    c = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c = max(-1.0, min(1.0, c))
    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM and cfg_free((q1, q2)):
            goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible IK solution")

    start = (0.0, 0.0)

    # --- 直接移動を試す ---
    for g in goals:
        if edge_free(start, g):
            return [g]

    # --- 関節空間グリッド上のダイクストラ探索 ---
    n = 31
    vals = [-LIM + 2 * LIM * i / (n - 1) for i in range(n)]  # 0 を含む
    nodes = [(vals[i], vals[j]) for i in range(n) for j in range(n)]
    ok = [cfg_free(q) for q in nodes]
    ng = len(nodes)
    nodes += goals
    ok += [True] * len(goals)
    N = len(nodes)
    s_idx = (n // 2) * n + (n // 2)
    goal_set = set(range(ng, N))
    R = 0.6

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def neighbors(u):
        res = []
        if u < ng:
            i, j = divmod(u, n)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if 0 <= a < n and 0 <= b < n:
                        res.append(a * n + b)
            for g in goal_set:
                if dist(nodes[u], nodes[g]) <= R:
                    res.append(g)
        else:
            for v in range(ng):
                if dist(nodes[u], nodes[v]) <= R:
                    res.append(v)
        return res

    INF = float("inf")
    best = [INF] * N
    prev = [-1] * N
    done = [False] * N
    best[s_idx] = 0.0
    found = -1
    cache = {}
    while True:
        u, bu = -1, INF
        for k in range(N):
            if not done[k] and best[k] < bu:
                u, bu = k, best[k]
        if u < 0:
            break
        done[u] = True
        if u in goal_set:
            found = u
            break
        for v in neighbors(u):
            if done[v] or not ok[v]:
                continue
            key = (min(u, v), max(u, v))
            if key not in cache:
                cache[key] = edge_free(nodes[u], nodes[v])
            if not cache[key]:
                continue
            nd = bu + dist(nodes[u], nodes[v])
            if nd < best[v]:
                best[v] = nd
                prev[v] = u
    if found < 0:
        raise ValueError("no collision-free path")

    path = []
    k = found
    while k != -1:
        path.append(nodes[k])
        k = prev[k]
    path.reverse()
    path[0] = start

    # --- ショートカットで経由点を削減 ---
    out = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
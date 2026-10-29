import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    QMAX = 2.98
    # 衝突半径 = リンク半幅 0.02 + 安全余裕 0.02 + 追加バッファ 0.01
    CLR = 0.02 + 0.02 + 0.01

    tx, ty = target
    d = math.hypot(tx, ty)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target unreachable")

    # --- 逆運動学（肘上/肘下の両解）---
    c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            goals.append((q1, q2))
    if not goals:
        raise ValueError("target outside joint limits")

    # --- 衝突判定 ---
    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def pose_free(q1, q2):
        if abs(q1) > QMAX or abs(q2) > QMAX:
            return False
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        fx = ex + L2 * math.cos(q1 + q2)
        fy = ey + L2 * math.sin(q1 + q2)
        for o in obstacles:
            lim = o["r"] + CLR
            if seg_dist(o["x"], o["y"], 0.0, 0.0, ex, ey) < lim:
                return False
            if seg_dist(o["x"], o["y"], ex, ey, fx, fy) < lim:
                return False
        return True

    def edge_free(a, b, res=0.01):
        m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        n = max(1, int(math.ceil(m / res)))
        for k in range(n + 1):
            t = k / n
            if not pose_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
                return False
        return True

    # --- 関節空間グリッド上で幅優先探索 ---
    N = 60
    M = 2 * N + 1
    step = QMAX / N

    def q_of(i, j):
        return ((i - N) * step, (j - N) * step)

    free = [[pose_free(*q_of(i, j)) for j in range(M)] for i in range(M)]
    if not free[N][N]:
        raise ValueError("start pose in collision")

    dist = {(N, N): 0}
    parent = {(N, N): None}
    queue = [(N, N)]
    head = 0
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while head < len(queue):
        i, j = queue[head]
        head += 1
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if not (0 <= ni < M and 0 <= nj < M) or (ni, nj) in dist:
                continue
            if not free[ni][nj]:
                continue
            a, b = q_of(i, j), q_of(ni, nj)
            if not pose_free((a[0] + b[0]) / 2, (a[1] + b[1]) / 2):
                continue
            dist[(ni, nj)] = dist[(i, j)] + 1
            parent[(ni, nj)] = (i, j)
            queue.append((ni, nj))

    # --- ゴール選択（グリッド点から厳密なIK解へ接続）---
    best = None
    for g in goals:
        if not pose_free(*g):
            continue
        ci = int(round(g[0] / step)) + N
        cj = int(round(g[1] / step)) + N
        for di in range(-3, 4):
            for dj in range(-3, 4):
                node = (ci + di, cj + dj)
                if node not in dist:
                    continue
                q = q_of(*node)
                cost = dist[node] + math.hypot(g[0] - q[0], g[1] - q[1]) / step
                if best is not None and cost >= best[0]:
                    continue
                if edge_free(q, g):
                    best = (cost, node, g)
    if best is None:
        raise ValueError("no collision-free path found")

    _, node, g = best
    chain = []
    n = node
    while n is not None:
        chain.append(q_of(*n))
        n = parent[n]
    chain.reverse()
    chain.append(g)

    # --- 近道化（可能な限り遠い点へ直接接続）---
    path = [chain[0]]
    i = 0
    while i < len(chain) - 1:
        j = len(chain) - 1
        while j > i + 1 and not edge_free(chain[i], chain[j]):
            j -= 1
        path.append(chain[j])
        i = j

    return path[1:]
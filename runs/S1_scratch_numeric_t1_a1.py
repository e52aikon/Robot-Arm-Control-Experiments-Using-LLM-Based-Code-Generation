import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    QMAX = 2.98
    # 必要クリアランス = 障害物半径 + リンク半幅 + 安全余裕 + 追加バッファ
    HALF_W, MARGIN, EXTRA = 0.02, 0.02, 0.005
    start = (0.0, 0.0)
    tx, ty = float(target[0]), float(target[1])

    def fk(q1, q2):
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        a = q1 + q2
        return (ex, ey), (ex + L2 * math.cos(a), ey + L2 * math.sin(a))

    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def clear(q):
        e, p = fk(q[0], q[1])
        for o in obstacles:
            need = o["r"] + HALF_W + MARGIN + EXTRA
            if seg_dist(o["x"], o["y"], 0.0, 0.0, e[0], e[1]) < need:
                return False
            if seg_dist(o["x"], o["y"], e[0], e[1], p[0], p[1]) < need:
                return False
        return True

    def seg_free(a, b):
        # 関節空間の直線補間を細かくサンプリングして干渉確認
        m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        n = max(1, int(math.ceil(m / 0.01)))
        for k in range(n + 1):
            t = k / n
            if not clear((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    # --- 逆運動学（肘の2解） ---
    d = math.hypot(tx, ty)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) > QMAX or abs(q2) > QMAX:
            continue
        _, p = fk(q1, q2)
        if math.hypot(p[0] - tx, p[1] - ty) > 0.005:
            continue
        goals.append((q1, q2))
    if not goals:
        raise ValueError("target unreachable within joint limits")
    if not clear(start):
        raise ValueError("start pose collides with obstacle")
    goals = [g for g in goals if clear(g)]
    if not goals:
        raise ValueError("target pose collides with obstacle")
    goals.sort(key=lambda g: abs(g[0] - start[0]) + abs(g[1] - start[1]))

    # --- 1) 直線補間で行ければそれで終わり ---
    for g in goals:
        if seg_free(start, g):
            return [g]

    # --- 2) 関節空間グリッド上のBFSで迂回経路を探索 ---
    N = 61
    step = 2 * QMAX / (N - 1)
    R = 3  # start/goal を格子へつなぐ範囲（格子数）

    def gq(i):
        return -QMAX + i * step

    for goal in goals:
        def qof(node):
            if node == 's':
                return start
            if node == 'g':
                return goal
            return (gq(node[0]), gq(node[1]))

        def near_idx(q):
            return (int(round((q[0] + QMAX) / step)), int(round((q[1] + QMAX) / step)))

        gi, gj = near_idx(goal)

        def nbrs(node):
            out = []
            if node == 's':
                ci, cj = near_idx(start)
                for di in range(-R, R + 1):
                    for dj in range(-R, R + 1):
                        out.append((ci + di, cj + dj))
            else:
                i, j = node
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di or dj:
                            out.append((i + di, j + dj))
                if abs(i - gi) <= R and abs(j - gj) <= R:
                    out.append('g')
            res = []
            for nb in out:
                if nb != 'g' and not (0 <= nb[0] < N and 0 <= nb[1] < N):
                    continue
                if nb != 'g' and not clear(qof(nb)):
                    continue
                res.append(nb)
            return res

        parent = {'s': None}
        queue = ['s']
        h = 0
        found = False
        while h < len(queue):
            cur = queue[h]
            h += 1
            if cur == 'g':
                found = True
                break
            for nb in nbrs(cur):
                if nb in parent:
                    continue
                if seg_free(qof(cur), qof(nb)):
                    parent[nb] = cur
                    queue.append(nb)
        if not found:
            continue

        path = []
        n = 'g'
        while n is not None:
            path.append(qof(n))
            n = parent[n]
        path.reverse()

        # 経路の短縮（見通せる限り経由点を飛ばす）
        res = []
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not seg_free(path[i], path[j]):
                j -= 1
            res.append(path[j])
            i = j
        return res

    raise ValueError("no collision-free path found")
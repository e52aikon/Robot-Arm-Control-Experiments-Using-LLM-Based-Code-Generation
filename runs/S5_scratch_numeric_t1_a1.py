import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    LIM = 2.98
    HW = 0.02                 # リンク半幅 (太さ4cmの半分)
    MARGIN = 0.02 + 0.005     # 安全余裕2cm + 補間誤差用の上乗せ
    tx, ty = float(target[0]), float(target[1])

    # ---- 順運動学・干渉判定 ----
    def fk(q):
        a = q[0]
        ex, ey = L1 * math.cos(a), L1 * math.sin(a)
        b = a + q[1]
        return ex, ey, ex + L2 * math.cos(b), ey + L2 * math.sin(b)

    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def config_free(q):
        if abs(q[0]) > LIM or abs(q[1]) > LIM:
            return False
        ex, ey, px, py = fk(q)
        for o in obstacles:
            c = o["r"] + HW + MARGIN
            if seg_dist(o["x"], o["y"], 0, 0, ex, ey) < c:
                return False
            if seg_dist(o["x"], o["y"], ex, ey, px, py) < c:
                return False
        return True

    def edge_free(a, b):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = int(math.ceil(d / 0.01)) + 1
        for k in range(n + 1):
            t = k / n
            if not config_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    # ---- 逆運動学（肘上・肘下の2解）----
    d2 = tx * tx + ty * ty
    d = math.sqrt(d2)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target unreachable")
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM and config_free((q1, q2)):
            if not any(abs(q1 - g[0]) < 1e-9 and abs(q2 - g[1]) < 1e-9 for g in goals):
                goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible IK solution")
    if not config_free((0.0, 0.0)):
        raise ValueError("start configuration in collision")

    # ---- 関節空間グリッド上のA*探索 ----
    N = 30
    step = LIM / N
    R = 0.3  # ゴールと接続する半径 [rad]

    def qof(key):
        if isinstance(key[0], str):
            return goals[key[1]]
        return ((key[0] - N) * step, (key[1] - N) * step)

    def neighbors(key):
        if isinstance(key[0], str):
            gq = goals[key[1]]
            ci, cj = int(round(gq[0] / step)) + N, int(round(gq[1] / step)) + N
            m = int(R / step) + 1
            out = []
            for i in range(ci - m, ci + m + 1):
                for j in range(cj - m, cj + m + 1):
                    if 0 <= i <= 2 * N and 0 <= j <= 2 * N:
                        q = qof((i, j))
                        if math.hypot(q[0] - gq[0], q[1] - gq[1]) <= R:
                            out.append((i, j))
            return out
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                i, j = key[0] + di, key[1] + dj
                if 0 <= i <= 2 * N and 0 <= j <= 2 * N:
                    out.append((i, j))
        q = qof(key)
        for k, gq in enumerate(goals):
            if math.hypot(q[0] - gq[0], q[1] - gq[1]) <= R:
                out.append(('g', k))
        return out

    # 簡易二分ヒープ
    def hpush(h, item):
        h.append(item)
        i = len(h) - 1
        while i > 0:
            p = (i - 1) // 2
            if h[p] <= h[i]:
                break
            h[p], h[i] = h[i], h[p]
            i = p

    def hpop(h):
        top = h[0]
        last = h.pop()
        if h:
            h[0] = last
            i, n = 0, len(h)
            while True:
                l, r, m = 2 * i + 1, 2 * i + 2, i
                if l < n and h[l] < h[m]:
                    m = l
                if r < n and h[r] < h[m]:
                    m = r
                if m == i:
                    break
                h[i], h[m] = h[m], h[i]
                i = m
        return top

    def hdist(key):
        q = qof(key)
        return min(math.hypot(q[0] - g[0], q[1] - g[1]) for g in goals)

    start = (N, N)  # (0,0)
    gcost = {start: 0.0}
    parent = {}
    closed = set()
    heap = []
    cnt = 0
    hpush(heap, (hdist(start), cnt, start))
    found = None
    while heap:
        _, _, cur = hpop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if isinstance(cur[0], str):
            found = cur
            break
        qc = qof(cur)
        for nb in neighbors(cur):
            if nb in closed:
                continue
            qn = qof(nb)
            ng = gcost[cur] + math.hypot(qn[0] - qc[0], qn[1] - qc[1])
            if nb in gcost and ng >= gcost[nb]:
                continue
            if not config_free(qn) or not edge_free(qc, qn):
                continue
            gcost[nb] = ng
            parent[nb] = cur
            cnt += 1
            hpush(heap, (ng + hdist(nb), cnt, nb))

    if found is None:
        raise ValueError("no collision-free path")

    keys = [found]
    while keys[-1] in parent:
        keys.append(parent[keys[-1]])
    keys.reverse()
    path = [qof(k) for k in keys]

    # ---- ショートカットで経由点を削減 ----
    res = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        res.append(path[j])
        i = j

    return res[1:]  # 開始姿勢は除く（最後はIK解）
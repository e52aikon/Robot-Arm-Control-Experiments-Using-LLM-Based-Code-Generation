import math

L1, L2 = 0.5, 0.4
HALF_W = 0.02      # リンク太さ 4cm の半分
SAFETY = 0.02      # 安全余裕
QMAX = 2.98
STEP = 0.02        # 補間チェックの刻み [rad]


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _make_checker(obstacles, extra):
    def cfg_free(q):
        (ex, ey), (tx, ty) = _fk(q[0], q[1])
        for o in obstacles:
            lim = o["r"] + HALF_W + SAFETY + extra
            if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
                return False
            if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < lim:
                return False
        return True

    def seg_free(a, b):
        d = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        n = max(1, int(math.ceil(d / STEP)))
        for i in range(n + 1):
            t = i / n
            if not cfg_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    return cfg_free, seg_free


def _ik(target):
    x, y = target
    d2 = x * x + y * y
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c > 1 + 1e-9 or c < -1 - 1e-9:
        raise ValueError("target unreachable")
    c = max(-1.0, min(1.0, c))
    sols = []
    for s in (1, -1):  # 肘の2解
        q2 = s * math.acos(c)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")
    return sols


# 簡易バイナリヒープ（heapq は使えないため自作）
def _hpush(h, item):
    h.append(item)
    i = len(h) - 1
    while i > 0:
        p = (i - 1) // 2
        if h[p][0] <= h[i][0]:
            break
        h[p], h[i] = h[i], h[p]
        i = p


def _hpop(h):
    top = h[0]
    last = h.pop()
    if h:
        h[0] = last
        i, n = 0, len(h)
        while True:
            l, r, m = 2 * i + 1, 2 * i + 2, i
            if l < n and h[l][0] < h[m][0]:
                m = l
            if r < n and h[r][0] < h[m][0]:
                m = r
            if m == i:
                break
            h[m], h[i] = h[i], h[m]
            i = m
    return top


def _search(start, goals, cfg_free, seg_free):
    # 直接経路が通るならそれを採用
    best = None
    for g in goals:
        if cfg_free(g) and seg_free(start, g):
            d = math.hypot(g[0] - start[0], g[1] - start[1])
            if best is None or d < best[0]:
                best = (d, g)
    if best:
        return [best[1]]

    goals = [g for g in goals if cfg_free(g)]
    if not goals or not cfg_free(start):
        return None

    # 関節空間グリッド（奇数点にして原点を格子点に含める）
    N = 59
    h = 2 * QMAX / (N - 1)
    pos = lambda i, j: (-QMAX + i * h, -QMAX + j * h)
    s_idx = (N // 2, N // 2)
    free = {}
    for i in range(N):
        for j in range(N):
            free[(i, j)] = cfg_free(pos(i, j))

    dist = {s_idx: 0.0}
    prev = {}
    heap = []
    _hpush(heap, (0.0, s_idx))
    done = set()
    while heap:
        dcur, u = _hpop(heap)
        if u in done:
            continue
        done.add(u)
        pu = pos(*u)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                v = (u[0] + di, u[1] + dj)
                if not (0 <= v[0] < N and 0 <= v[1] < N) or not free[v] or v in done:
                    continue
                if not seg_free(pu, pos(*v)):
                    continue
                nd = dcur + h * math.hypot(di, dj)
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    _hpush(heap, (nd, v))

    # ゴールに接続できる最良ノードを選ぶ
    best = None
    for g in goals:
        gi = int(round((g[0] + QMAX) / h))
        gj = int(round((g[1] + QMAX) / h))
        for i in range(gi - 3, gi + 4):
            for j in range(gj - 3, gj + 4):
                v = (i, j)
                if v not in dist:
                    continue
                pv = pos(i, j)
                c = dist[v] + math.hypot(g[0] - pv[0], g[1] - pv[1])
                if best is not None and c >= best[0]:
                    continue
                if seg_free(pv, g):
                    best = (c, v, g)
    if best is None:
        return None

    nodes = []
    v = best[1]
    while v != s_idx:
        nodes.append(pos(*v))
        v = prev[v]
    nodes.reverse()
    return [start] + nodes + [best[2]]


def plan(target, obstacles):
    start = (0.0, 0.0)
    goals = _ik(target)  # 到達不可なら ValueError

    # 余裕を持たせて探索し、だめなら余裕なしで再試行
    path = None
    for extra in (0.01, 0.0):
        cfg_free, seg_free = _make_checker(obstacles, extra)
        path = _search(start, goals, cfg_free, seg_free)
        if path:
            break
    if not path:
        raise ValueError("no collision-free path")

    if len(path) == 1:
        return [path[0]]

    # ショートカットで経由点を削減
    full = path if path[0] == start else [start] + path
    out = []
    i = 0
    while i < len(full) - 1:
        for j in range(len(full) - 1, i, -1):
            if seg_free(full[i], full[j]):
                out.append(full[j])
                i = j
                break
    return out
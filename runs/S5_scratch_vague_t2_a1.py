import math

# 想定障害物: 正面やや左、根元から約0.6m、直径約10cmの柱（位置は不確か）
ASSUMED_OBSTACLES = [{"x": 0.58, "y": 0.16, "r": 0.05}]

L1, L2 = 0.5, 0.4
HALF_W = 0.02      # リンク半幅
SAFETY = 0.02      # 安全余裕
EPS = 0.005        # 補間サンプリング誤差の見込み
LIMIT = 2.98
STEP = 0.1         # 探索グリッド刻み [rad]
SAMPLE = 0.01      # 衝突チェックの補間刻み [rad]


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _cfg_free(q1, q2, obs, extra):
    e, t = _fk(q1, q2)
    for o in obs:
        need = o["r"] + HALF_W + SAFETY + extra + EPS
        if _seg_dist(o["x"], o["y"], 0, 0, e[0], e[1]) < need:
            return False
        if _seg_dist(o["x"], o["y"], e[0], e[1], t[0], t[1]) < need:
            return False
    return True


def _line_free(a, b, obs, extra):
    n = max(1, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / SAMPLE)))
    for k in range(n + 1):
        s = k / n
        if not _cfg_free(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, obs, extra):
            return False
    return True


def _ik(x, y):
    d2 = x * x + y * y
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c > 1.0 + 1e-9 or c < -1.0 - 1e-9:
        raise ValueError("target out of reach")
    c = max(-1.0, min(1.0, c))
    sols = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c)
        q1 = _wrap(math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)))
        q2 = _wrap(q2)
        if abs(q1) <= LIMIT and abs(q2) <= LIMIT:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target not reachable within joint limits")
    return sols


# --- 最小限のバイナリヒープ (heapq 不使用) ---
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


def _search(goals, obs, extra):
    start = (0.0, 0.0)
    if not _cfg_free(0.0, 0.0, obs, extra):
        return None
    goals = [g for g in goals if _cfg_free(g[0], g[1], obs, extra)]
    if not goals:
        return None

    # まず直接補間を試す（最短の関節移動量を優先）
    direct = [g for g in goals if _line_free(start, g, obs, extra)]
    if direct:
        return [min(direct, key=lambda g: math.hypot(g[0], g[1]))]

    N = int(LIMIT / STEP)
    q = lambda n: (n[0] * STEP, n[1] * STEP)
    free_cache = {}

    def free(n):
        if n not in free_cache:
            a = q(n)
            free_cache[n] = _cfg_free(a[0], a[1], obs, extra)
        return free_cache[n]

    dist = {(0, 0): 0.0}
    prev = {}
    heap = [(0.0, (0, 0))]
    done = set()
    while heap:
        d, n = _hpop(heap)
        if n in done:
            continue
        done.add(n)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                m = (n[0] + di, n[1] + dj)
                if abs(m[0]) > N or abs(m[1]) > N or m in done or not free(m):
                    continue
                nd = d + STEP * math.hypot(di, dj)
                if nd < dist.get(m, 1e18) and _line_free(q(n), q(m), obs, extra):
                    dist[m] = nd
                    prev[m] = n
                    _hpush(heap, (nd, m))

    # ゴール近傍のグリッド点から接続
    best = None
    for g in goals:
        gi, gj = int(round(g[0] / STEP)), int(round(g[1] / STEP))
        for di in range(-5, 6):
            for dj in range(-5, 6):
                m = (gi + di, gj + dj)
                if m not in dist:
                    continue
                cost = dist[m] + math.hypot(q(m)[0] - g[0], q(m)[1] - g[1])
                if best is not None and cost >= best[0]:
                    continue
                if _line_free(q(m), g, obs, extra):
                    best = (cost, m, g)
    if best is None:
        return None

    _, m, g = best
    nodes = []
    while m in prev:
        nodes.append(q(m))
        m = prev[m]
    nodes.reverse()
    path = [start] + nodes + [g]

    # ショートカットで経由点を削減
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(path[i], path[j], obs, extra):
            j -= 1
        out.append(path[j])
        i = j
    return out[1:]


def plan(target, obstacles):
    x, y = float(target[0]), float(target[1])
    assumed = not obstacles
    obs = ASSUMED_OBSTACLES if assumed else obstacles

    goals = _ik(x, y)  # 届かなければ ValueError

    # 位置不確かさ分の余裕を大きめから試し、だめなら縮める
    margins = (0.05, 0.03, 0.0) if assumed else (0.02, 0.0)
    for extra in margins:
        wp = _search(goals, obs, extra)
        if wp:
            _, tip = _fk(*wp[-1])
            if math.hypot(tip[0] - x, tip[1] - y) <= 0.005:
                return [(float(a), float(b)) for a, b in wp]
    raise ValueError("no collision-free path found")
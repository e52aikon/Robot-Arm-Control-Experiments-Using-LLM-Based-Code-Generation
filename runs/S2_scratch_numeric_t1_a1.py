import math

L1, L2 = 0.5, 0.4
QMAX = 2.98
HALF_W = 0.02      # リンク太さ4cmの半分
MARGIN = 0.02      # 安全余裕
EPS = 0.005        # 追加の数値余裕
GRID = 0.05        # 探索格子の刻み [rad]


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の最短距離
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _collides(q1, q2, obstacles):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    for o in obstacles:
        lim = o["r"] + MARGIN + HALF_W + EPS
        if _seg_dist(o["x"], o["y"], 0, 0, ex, ey) < lim:
            return True
        if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < lim:
            return True
    return False


def _edge_free(a, b, obstacles):
    # 関節空間の直線補間を細かくサンプリングして干渉判定
    n = int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.01)) + 1
    for k in range(n + 1):
        t = k / n
        if _collides(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, obstacles):
            return False
    return True


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _ik(x, y):
    d = math.hypot(x, y)
    if d > L1 + L2 - 1e-6 or d < abs(L1 - L2) + 1e-6:
        raise ValueError("target unreachable")
    c = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c = max(-1.0, min(1.0, c))
    sols = []
    for s in (1, -1):
        q2 = s * math.acos(c)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1, q2 = _wrap(q1), _wrap(q2)
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    return sols


def _grid_search(goal, obstacles):
    # 格子上のBFS。goal近傍のノードから直線接続できたら終了
    N = int(QMAX / GRID)
    start = (0, 0)
    free_cache = {}

    def free(i, j):
        k = (i, j)
        if k not in free_cache:
            free_cache[k] = _collides(i * GRID, j * GRID, obstacles) is False
        return free_cache[k]

    if not free(0, 0):
        return None
    parent = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        qc = (cur[0] * GRID, cur[1] * GRID)
        if max(abs(qc[0] - goal[0]), abs(qc[1] - goal[1])) < 0.4:
            if _edge_free(qc, goal, obstacles):
                path = []
                k = cur
                while k is not None:
                    path.append((k[0] * GRID, k[1] * GRID))
                    k = parent[k]
                path.reverse()
                return path + [goal]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (cur[0] + di, cur[1] + dj)
                if nb in parent or abs(nb[0]) > N or abs(nb[1]) > N:
                    continue
                if not free(*nb):
                    continue
                if not _edge_free(qc, (nb[0] * GRID, nb[1] * GRID), obstacles):
                    continue
                parent[nb] = cur
                queue.append(nb)
    return None


def plan(target, obstacles):
    x, y = target
    sols = _ik(x, y)  # 到達不可なら ValueError
    start = (0.0, 0.0)
    # 目標姿勢が干渉するものは除外、関節空間で近い順に試す
    goals = [g for g in sols if not _collides(g[0], g[1], obstacles)]
    goals.sort(key=lambda g: abs(g[0]) + abs(g[1]))
    if not goals:
        raise ValueError("no collision-free goal configuration")

    # 1) 直接補間で行けるか
    for g in goals:
        if _edge_free(start, g, obstacles):
            return [g]

    # 2) 格子探索で迂回経路を作る
    for g in goals:
        path = _grid_search(g, obstacles)
        if path is None:
            continue
        # 近道化（直線接続できる最も先の点へ飛ばす）
        out = []
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not _edge_free(path[i], path[j], obstacles):
                j -= 1
            out.append(path[j])
            i = j
        return out

    raise ValueError("no collision-free path found")
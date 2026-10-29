import math

L1, L2 = 0.5, 0.4
HALF_W = 0.02      # リンク半幅（太さ 4 cm の半分）
MARGIN = 0.02      # 安全余裕
BUF = 0.01         # 補間サンプリング誤差ぶんの追加余裕
QLIM = 2.98
GOAL_TOL = 0.01


def _seg_point_dist(px, py, ax, ay, bx, by):
    # 点と線分の最短距離
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    t = 0.0 if d2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx, ty = ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2)
    return ex, ey, tx, ty


def _config_free(q, obstacles):
    ex, ey, tx, ty = _fk(q[0], q[1])
    for o in obstacles:
        need = o["r"] + HALF_W + MARGIN + BUF
        if _seg_point_dist(o["x"], o["y"], 0, 0, ex, ey) < need:
            return False
        if _seg_point_dist(o["x"], o["y"], ex, ey, tx, ty) < need:
            return False
    return True


def _edge_free(a, b, obstacles):
    # 関節空間の直線補間を細かくサンプリングして干渉判定
    n = max(2, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.02)))
    for i in range(n + 1):
        t = i / n
        q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not _config_free(q, obstacles):
            return False
    return True


def _ik(target):
    x, y = target
    d2 = x * x + y * y
    d = math.sqrt(d2)
    if d > L1 + L2 - 1e-9 or d < abs(L1 - L2) + 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for s in (1.0, -1.0):  # 肘の2通りの解
        q2 = s * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QLIM and abs(q2) <= QLIM:
            ex, ey, tx, ty = _fk(q1, q2)
            if math.hypot(tx - x, ty - y) < GOAL_TOL / 2:
                sols.append((q1, q2))
    if not sols:
        raise ValueError("target outside joint limits")
    return sols


def plan(target, obstacles):
    start = (0.0, 0.0)
    goals = [g for g in _ik(target) if _config_free(g, obstacles)]
    if not goals:
        raise ValueError("no collision-free goal configuration")
    if not _config_free(start, obstacles):
        raise ValueError("start configuration in collision")

    # 1) 直接移動できる解があればそれを使う
    for g in goals:
        if _edge_free(start, g, obstacles):
            return [g]

    # 2) 関節空間グリッド上で BFS
    N = 61
    axis = [-QLIM + 2 * QLIM * i / (N - 1) for i in range(N)]
    free = {}
    for i in range(N):
        for j in range(N):
            q = (axis[i], axis[j])
            if _config_free(q, obstacles):
                free[(i, j)] = q
    s_idx = (N // 2, N // 2)  # (0, 0)
    if s_idx not in free:
        raise ValueError("start configuration in collision")

    # ゴールノードとグリッドの接続
    goal_adj = {}  # ノード -> 接続されたゴール
    goal_nodes = {}
    for k, g in enumerate(goals):
        gid = ("g", k)
        goal_nodes[gid] = g
        for idx, q in free.items():
            if abs(q[0] - g[0]) < 0.3 and abs(q[1] - g[1]) < 0.3 and _edge_free(q, g, obstacles):
                goal_adj.setdefault(idx, []).append(gid)

    parent = {s_idx: None}
    queue = [s_idx]
    head = 0
    found = None
    while head < len(queue) and found is None:
        cur = queue[head]
        head += 1
        for gid in goal_adj.get(cur, []):
            parent[gid] = cur
            found = gid
            break
        if found is not None:
            break
        ci, cj = cur
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (ci + di, cj + dj)
                if nb in free and nb not in parent and _edge_free(free[cur], free[nb], obstacles):
                    parent[nb] = cur
                    queue.append(nb)
    if found is None:
        raise ValueError("no collision-free path found")

    # 経路復元
    path = []
    n = found
    while n is not None:
        path.append(goal_nodes[n] if n in goal_nodes else free[n])
        n = parent[n]
    path.reverse()  # path[0] は開始姿勢

    # 3) ショートカットで経由点を減らす
    result = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _edge_free(path[i], path[j], obstacles):
            j -= 1
        result.append(path[j])
        i = j
    return result
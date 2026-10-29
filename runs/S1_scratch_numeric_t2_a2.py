# 原因: 前回の出力が途中（"k = len" の行）で途切れており、コードが未完成で構文エラーになった。
# 対策: 関数を最後まで書き直し、全体を完結させた。
import math

L1 = 0.5
L2 = 0.4
LIMIT = 2.98
HALF_WIDTH = 0.02      # リンク太さ 4cm の半分
SAFETY = 0.02          # 検証系の安全余裕
EXTRA = 0.01           # 数値誤差用の追加マージン
STEP = 0.005           # 干渉チェックの補間刻み [rad]


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def _seg_point_dist(ax, ay, bx, by, px, py):
    # 線分 AB と点 P の最短距離
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_ok(q1, q2, obstacles):
    if abs(q1) > LIMIT or abs(q2) > LIMIT:
        return False
    (ex, ey), (tx, ty) = _fk(q1, q2)
    for ob in obstacles:
        need = ob["r"] + HALF_WIDTH + SAFETY + EXTRA
        if _seg_point_dist(0.0, 0.0, ex, ey, ob["x"], ob["y"]) < need:
            return False
        if _seg_point_dist(ex, ey, tx, ty, ob["x"], ob["y"]) < need:
            return False
    return True


def _edge_ok(a, b, obstacles):
    # 関節空間の直線補間を細かくサンプリングして干渉を確認
    n = int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / STEP)) + 1
    for i in range(n + 1):
        t = i / n
        if not _pose_ok(a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t, obstacles):
            return False
    return True


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _ik_solutions(target):
    x, y = target
    d2 = x * x + y * y
    d = math.sqrt(d2)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    sols = []
    for sign in (1.0, -1.0):
        q2 = sign * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1, q2 = _wrap(q1), _wrap(q2)
        if abs(q1) > LIMIT or abs(q2) > LIMIT:
            continue
        _, (tx, ty) = _fk(q1, q2)
        if math.hypot(tx - x, ty - y) > 0.005:
            continue
        if not any(abs(q1 - s[0]) < 1e-9 and abs(q2 - s[1]) < 1e-9 for s in sols):
            sols.append((q1, q2))
    if not sols:
        raise ValueError("no IK solution within joint limits")
    return sols


def _grid_search(start, goal, obstacles):
    # フォールバック: 関節空間の格子上で幅優先探索（8近傍）
    res = 0.1
    kmax = int(LIMIT / res)
    n = 2 * kmax + 1

    def pos(i, j):
        return ((i - kmax) * res, (j - kmax) * res)

    s_idx = (kmax, kmax)          # (0,0) は格子点
    parent = {s_idx: None}
    queue = [s_idx]
    qi = 0
    found = None
    while qi < len(queue):
        cur = queue[qi]
        qi += 1
        cp = pos(*cur)
        if _edge_ok(cp, goal, obstacles):
            found = cur
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < n and 0 <= nj < n) or (ni, nj) in parent:
                    continue
                npos = pos(ni, nj)
                if _edge_ok(cp, npos, obstacles):
                    parent[(ni, nj)] = cur
                    queue.append((ni, nj))
    if found is None:
        return None
    path = []
    c = found
    while c is not None:
        path.append(pos(*c))
        c = parent[c]
    path.reverse()
    path = path[1:]
    path.append(goal)
    # 近道化: 直接行ける最も先の点へ飛ばす
    pts = [start] + path
    out = []
    i = 0
    while i < len(pts) - 1:
        k = len(pts) - 1
        while k > i + 1 and not _edge_ok(pts[i], pts[k], obstacles):
            k -= 1
        out.append(pts[k])
        i = k
    return out


def plan(target, obstacles):
    start = (0.0, 0.0)
    goals = _ik_solutions(target)   # 到達不能なら ValueError
    goals = [g for g in goals if _pose_ok(g[0], g[1], obstacles)]
    if not goals:
        raise ValueError("goal pose collides with obstacles")
    # まず関節空間の直線で直接行ける解を探す
    for g in goals:
        if _edge_ok(start, g, obstacles):
            return [g]
    # 直接行けない場合は格子探索で迂回経路を作る
    for g in goals:
        path = _grid_search(start, g, obstacles)
        if path:
            return path
    raise ValueError("no collision-free path found")
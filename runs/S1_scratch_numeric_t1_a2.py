# 原因: 前回の出力が BFS の nbrs リストの途中で途切れ、コードが未完成だった。
#       そのため括弧が閉じず SyntaxError になった。
# 対策: コード全体を最後まで書き直した。BFS 部分も完結させた。
import math

L1 = 0.5
L2 = 0.4
LIM = 2.98
HALF_W = 0.02      # リンク太さ 4cm の半分
MARGIN = 0.02      # 安全余裕
EXTRA = 0.005      # 補間サンプリング誤差分の追加余裕
TOL = 0.005        # 最終誤差の許容（要求 1cm より厳しめ）


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_point_dist(ax, ay, bx, by, px, py):
    # 線分 AB と点 P の最短距離
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _config_ok(q, obstacles):
    q1, q2 = q
    if abs(q1) > LIM or abs(q2) > LIM:
        return False
    (ex, ey), (tx, ty) = _fk(q1, q2)
    for o in obstacles:
        need = o["r"] + HALF_W + MARGIN + EXTRA
        if _seg_point_dist(0.0, 0.0, ex, ey, o["x"], o["y"]) < need:
            return False
        if _seg_point_dist(ex, ey, tx, ty, o["x"], o["y"]) < need:
            return False
    return True


def _edge_ok(a, b, obstacles, step=0.01):
    # 関節空間の直線補間上を細かくサンプリングして干渉チェック
    n = max(1, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / step)))
    for i in range(n + 1):
        t = i / n
        q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not _config_ok(q, obstacles):
            return False
    return True


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ik(tx, ty):
    # 解析的逆運動学（肘の上下 2 解）
    d2 = tx * tx + ty * ty
    d = math.sqrt(d2)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    sols = []
    for sgn in (1.0, -1.0):
        q2 = sgn * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1, q2 = _wrap(q1), _wrap(q2)
        if abs(q1) > LIM or abs(q2) > LIM:
            continue
        _, (fx, fy) = _fk(q1, q2)
        if math.hypot(fx - tx, fy - ty) <= TOL:
            sols.append((q1, q2))
    return sols


def plan(target, obstacles):
    tx, ty = target
    start = (0.0, 0.0)

    if not _config_ok(start, obstacles):
        raise ValueError("start pose is in collision")

    # 到達不能なら ValueError（_ik 内）
    goals = [g for g in _ik(tx, ty) if _config_ok(g, obstacles)]
    if not goals:
        raise ValueError("no collision-free IK solution")

    # 関節空間移動量の小さい解を優先
    goals.sort(key=lambda g: abs(g[0]) + abs(g[1]))

    # 1) まず開始姿勢から直接補間で行けるか試す
    for g in goals:
        if _edge_ok(start, g, obstacles):
            return [g]

    # 2) 直接行けない場合: 関節空間のグリッド上で BFS による経路探索
    N = 61  # 奇数 → 0 がグリッド点に含まれる
    step = 2 * LIM / (N - 1)

    def qof(i, j):
        return (-LIM + i * step, -LIM + j * step)

    si = sj = (N - 1) // 2  # (0, 0) のセル
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def cell_q(c):
        return start if c == (si, sj) else qof(c[0], c[1])

    parent = {(si, sj): None}
    order = [(si, sj)]
    head = 0
    while head < len(order):
        c = order[head]
        head += 1
        for di, dj in nbrs:
            ni, nj = c[0] + di, c[1] + dj
            if not (0 <= ni < N and 0 <= nj < N):
                continue
            nc = (ni, nj)
            if nc in parent:
                continue
            q = qof(ni, nj)
            if not _config_ok(q, obstacles):
                continue
            if not _edge_ok(cell_q(c), q, obstacles):
                continue
            parent[nc] = c
            order.append(nc)

    # 3) 到達済みセルのうち、ゴールへ直接つなげられるものを BFS 順に探す
    for g in goals:
        for c in order:
            if _edge_ok(cell_q(c), g, obstacles):
                cells = []
                k = c
                while k is not None:
                    cells.append(k)
                    k = parent[k]
                cells.reverse()
                path = [cell_q(k) for k in cells] + [g]

                # 4) 経路の短縮（干渉しない範囲で遠くの点へ直結）
                out = []
                i = 0
                while i < len(path) - 1:
                    j = len(path) - 1
                    while j > i + 1 and not _edge_ok(path[i], path[j], obstacles):
                        j -= 1
                    out.append(path[j])
                    i = j
                return out

    raise ValueError("no collision-free path found")
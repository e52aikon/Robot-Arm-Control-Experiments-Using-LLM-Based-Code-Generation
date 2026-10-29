```python
# 原因: 前回までの出力が途中で切れて閉じ括弧やコードブロックの終端がなく、
#       コードが不完全なため SyntaxError になった。
# 対策: コードを簡潔にして最後まで出力する。
#       直接移動が衝突する場合は、関節空間グリッドの幅優先探索で障害物を回避する。
import math

L1, L2 = 0.5, 0.4
LIMIT = 2.98
MARGIN = 0.02 + 0.02 + 0.006   # リンク半幅 + 安全余裕 + 離散化マージン
STEP = 0.01                    # 衝突チェックの関節角刻み [rad]
GSTEP = 0.1                    # 探索グリッド刻み [rad]
NG = 29                        # グリッド片側の分割数 (29*0.1=2.9 < LIMIT)


def _fk(q):
    ex, ey = L1 * math.cos(q[0]), L1 * math.sin(q[0])
    return (ex, ey), (ex + L2 * math.cos(q[0] + q[1]), ey + L2 * math.sin(q[0] + q[1]))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _ok(q, obs):
    if abs(q[0]) > LIMIT or abs(q[1]) > LIMIT:
        return False
    (ex, ey), (tx, ty) = _fk(q)
    for o in obs:
        lim = o["r"] + MARGIN
        if _seg_dist(o["x"], o["y"], 0.0, 0.0, ex, ey) < lim:
            return False
        if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < lim:
            return False
    return True


def _edge_ok(a, b, obs):
    m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(math.ceil(m / STEP)))
    for i in range(n + 1):
        t = i / n
        if not _ok((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), obs):
            return False
    return True


def _ik(target):
    x, y = target
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1.0 + 1e-9 or c2 < -1.0 - 1e-9:
        raise ValueError("target is unreachable")
    c2 = max(-1.0, min(1.0, c2))
    sols = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIMIT and abs(q2) <= LIMIT:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target is outside joint limits")
    return sols


def plan(target, obstacles):
    start = (0.0, 0.0)
    goals = [g for g in _ik(target) if _ok(g, obstacles)]
    if not goals:
        raise ValueError("goal is in collision")
    if not _ok(start, obstacles):
        raise ValueError("start is in collision")

    # 1) 直接移動できるならそれを採用
    goals.sort(key=lambda g: abs(g[0]) + abs(g[1]))
    for g in goals:
        if _edge_ok(start, g, obstacles):
            return [g]

    # 2) 関節空間グリッドで幅優先探索
    def pos(n):
        return (n[0] * GSTEP, n[1] * GSTEP)

    root = (0, 0)
    parent = {root: None}
    order = [root]
    k = 0
    while k < len(order):
        cur = order[k]
        k += 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nb = (cur[0] + di, cur[1] + dj)
                if (di == 0 and dj == 0) or nb in parent:
                    continue
                if abs(nb[0]) > NG or abs(nb[1]) > NG:
                    continue
                if _edge_ok(pos(cur), pos(nb), obstacles):
                    parent[nb] = cur
                    order.append(nb)

    # 到達
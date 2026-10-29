import math

# 原因: 前回までの出力はコードが途中で途切れており、不完全な構文だったため
#       SyntaxError になった。今回は短く完結する形で最後まで出力する。
# 設計: 開始姿勢から目標のIK解へ関節空間で直線補間すると、第2リンクが
#       障害物に近づきすぎる恐れがある。そこで関節空間のグリッドをBFSで探索し、
#       安全な経由点列を作ってから、直線で結べる区間を間引く。

L1, L2 = 0.5, 0.4
Q_LIM = 2.98
MARGIN_EXTRA = 0.005
HALF_THICK = 0.02
SAFETY = 0.02


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    a = q1 + q2
    return (ex, ey), (ex + L2 * math.cos(a), ey + L2 * math.sin(a))


def _seg_dist(ax, ay, bx, by, px, py):
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def plan(target, obstacles):
    tx, ty = target
    d = math.hypot(tx, ty)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")

    # ---- 逆運動学（2解）----
    c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2),
                                             L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= Q_LIM and abs(q2) <= Q_LIM:
            _, tip = _fk(q1, q2)
            if math.hypot(tip[0] - tx, tip[1] - ty) < 0.005:
                goals.append((q1, q2))
    if not goals:
        raise ValueError("no IK solution within joint limits")

    # ---- 衝突判定 ----
    def free(q):
        if abs(q[0]) > Q_LIM or abs(q[1]) > Q_LIM:
            return False
        e, t = _fk(q[0], q[1])
        for ob in obstacles:
            lim = ob["r"] + HALF_THICK + SAFETY + MARGIN_EXTRA
            if _seg_dist(0, 0, e[0], e[1], ob["x"], ob["y"]) < lim:
                return False
            if _seg_dist(e[0], e[1], t[0], t[1], ob["x"], ob["y"]) < lim:
                return False
        return True

    def edge_free(a, b, step):
        n = max(1, int(math.ceil(max(abs(a[0] - b[0]), abs(a[1] - b[1])) / step)))
        for k in range(n + 1):
            u = k / n
            if not free((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)):
                return False
        return True

    start = (0.0, 0.0)
    if not free(start):
        raise ValueError("start configuration is in collision")

    # ---- グリッドBFS（刻み 0.05 rad）----
    H = 0.05
    M = int(Q_LIM / H)
    s_idx = (0, 0)
    parent = {s_idx: None}
    order = [s_idx]
    qi = 0
    while qi < len(order):
        cur = order[qi]
        qi += 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (cur[0] + di, cur[1] + dj)
                if nb in parent or abs(nb[0]) > M or abs(nb[1]) > M:
                    continue
                a = (cur[0] * H, cur[1] * H)
                b = (nb[0] * H, nb[1] * H)
                if edge_free(a, b, 0.02):
                    parent[nb] = cur
                    order.append(nb)

    # ---- 目標への接続（BFS順＝開始に近い順で、目標に近いノードを探す）----
    best = None
    for g in goals:
        cands = sorted(order, key=lambda n: math.hypot(n[0] * H - g[0],
                                                       n[1] * H - g[1]))
        for n in cands[:200]:
            if edge_free((n[0] * H, n[1] * H), g, 0.01):
                chain = []
                c = n
                while c is not None:
                    chain.append((c[0] * H, c[1] * H))
                    c = parent[c]
                chain.reverse()
                chain.append(g)
                if best is None or len(chain) < len(best):
                    best = chain
                break
    if best is None:
        raise ValueError("no collision-free path found")

    # ---- 経由点の間引き（直線で結べる区間は省略）----
    res = []
    i = 0
    last = len(best) - 1
    while i < last:
        nxt = i + 1
        for j in range(last, i, -1):
            if edge_free(best[i], best[j], 0.01):
                nxt = j
                break
        res.append(best[nxt])
        i = nxt
    return [tuple(p) for p in res]
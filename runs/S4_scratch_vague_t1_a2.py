import math

# 原因: 前回は説明コメントが長く、グリッド探索の途中で出力が切れ plan が未完成だった。
# 対策: コードを短くまとめ、探索(BFS)から経路返却まで完結させた。
# 想定障害物: 左斜め前(約45°)・根元から85cmの柱(直径約10cm)。
# 位置が不確かなので30/45/60°の3か所に半径10cmの円を置く。
ASSUMED_OBSTACLES = [
    {"x": 0.85 * math.cos(math.radians(a)),
     "y": 0.85 * math.sin(math.radians(a)), "r": 0.10}
    for a in (30, 45, 60)
]

L1, L2, QMAX = 0.5, 0.4, 2.98
MARGIN = 0.02 + 0.02 + 0.01  # リンク半幅 + 安全余裕 + 補間誤差分
N = 60                       # 関節空間グリッド分割数(偶数)


def _dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _free(q, obs):
    q1, q2 = q
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx, ty = ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2)
    for o in obs:
        need = o["r"] + MARGIN
        if _dist(o["x"], o["y"], 0, 0, ex, ey) < need:
            return False
        if _dist(o["x"], o["y"], ex, ey, tx, ty) < need:
            return False
    return True


def _seg_free(a, b, obs):
    n = max(1, math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.02))
    return all(_free((a[0] + (b[0] - a[0]) * k / n,
                      a[1] + (b[1] - a[1]) * k / n), obs) for k in range(n + 1))


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    x, y = target
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0 + 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, c2))

    # 逆運動学: 肘の2解
    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX and _free((q1, q2), obs):
            goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible goal configuration")

    step = 2 * QMAX / N
    h = N // 2
    q_of = lambda c: ((c[0] - h) * step, (c[1] - h) * step)
    cache = {}

    def ok(c):
        if c not in cache:
            cache[c] = _free(q_of(c), obs)
        return cache[c]

    start = (h, h)
    if not ok(start):
        raise ValueError("start in collision")

    # 関節空間グリッド上のBFS(8近傍)
    par = {start: None}
    queue = [start]
    for c in queue:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (c[0] + di, c[1] + dj)
                if (di or dj) and 0 <= n[0] <= N and 0 <= n[1] <= N \
                        and n not in par and ok(n) \
                        and _seg_free(q_of(c), q_of(n), obs):
                    par[n] = c
                    queue.append(n)

    # 到達済みノードからゴールへ直線接続できるものを探す
    for g in goals:
        cand = sorted(queue, key=lambda c: math.hypot(q_of(c)[0] - g[0], q_of(c)[1] - g[1]))
        for c in cand[:300]:
            if _seg_free(q_of(c), g, obs):
                chain = []
                while c is not None:
                    chain.append(q_of(c))
                    c = par[c]
                pts = chain[::-1] + [g]
                # 近道化
                out, i = [], 0
                while i < len(pts) - 1:
                    j = len(pts) - 1
                    while j > i + 1 and not _seg_free(pts[i], pts[j], obs):
                        j -= 1
                    out.append(pts[j])
                    i = j
                return out
    raise ValueError("no collision-free path found")
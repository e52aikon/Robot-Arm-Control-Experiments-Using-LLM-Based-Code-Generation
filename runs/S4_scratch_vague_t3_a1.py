import math

L1, L2 = 0.5, 0.4
HALF_W = 0.02      # リンク太さ4cmの半分
CLEAR = 0.02       # 安全余裕
QLIM = 2.98
QSAFE = 2.90       # 可動範囲に余裕を持たせる

# 想定障害物: 左斜め前(30〜60度)、根元から約0.85mの細い柱(直径約10cm)。
# 位置が不確かなので、方向を3点に振り、半径は柱 0.05 + 不確かさ 0.07 = 0.12 と大きめに見積もる。
_PILLAR_DIST = 0.85
_PILLAR_R = 0.05 + 0.07
ASSUMED_OBSTACLES = [
    {"x": _PILLAR_DIST * math.cos(math.radians(a)),
     "y": _PILLAR_DIST * math.sin(math.radians(a)),
     "r": _PILLAR_R}
    for a in (30.0, 45.0, 60.0)
]


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の距離
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_free(q, obs, extra):
    q1, q2 = q
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    for o in obs:
        lim = o["r"] + HALF_W + CLEAR + extra
        if _seg_dist(o["x"], o["y"], 0.0, 0.0, ex, ey) < lim:
            return False
        if _seg_dist(o["x"], o["y"], ex, ey, tx, ty) < lim:
            return False
    return True


def _seg_free(a, b, obs, extra=0.005):
    # 関節空間の直線補間を細かくチェック
    m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = int(math.ceil(m / 0.01)) + 1
    for k in range(n + 1):
        t = k / n
        q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not _pose_free(q, obs, extra):
            return False
    return True


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    tx, ty = target

    # --- 逆運動学（肘2通り） ---
    d2 = tx * tx + ty * ty
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        raise ValueError("target unreachable")
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QSAFE and abs(q2) <= QSAFE and _pose_free((q1, q2), obs, 0.005):
            goals.append((q1, q2))
    if not goals:
        raise ValueError("no valid IK solution")

    # --- 関節空間グリッドのBFS ---
    H = 0.05
    N = int(round(QSAFE / H))
    cache = {}

    def node_free(i, j):
        key = (i, j)
        if key not in cache:
            # 格子間の取りこぼし対策で余裕を多めに取る
            cache[key] = _pose_free((i * H, j * H), obs, 0.03)
        return cache[key]

    if not node_free(0, 0):
        raise ValueError("start pose in collision")

    parent = {(0, 0): None}
    queue = [(0, 0)]
    head = 0
    found = None
    goal_hit = None
    dirs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while head < len(queue) and found is None:
        cur = queue[head]
        head += 1
        qc = (cur[0] * H, cur[1] * H)
        for g in goals:
            if max(abs(g[0] - qc[0]), abs(g[1] - qc[1])) < 0.3 and _seg_free(qc, g, obs):
                found, goal_hit = cur, g
                break
        if found is not None:
            break
        for a, b in dirs:
            ni, nj = cur[0] + a, cur[1] + b
            if abs(ni) > N or abs(nj) > N or (ni, nj) in parent:
                continue
            if not node_free(ni, nj):
                continue
            parent[(ni, nj)] = cur
            queue.append((ni, nj))

    if found is None:
        raise ValueError("no collision-free path")

    nodes = []
    c = found
    while c is not None:
        nodes.append(c)
        c = parent[c]
    nodes.reverse()
    pts = [(i * H, j * H) for i, j in nodes] + [goal_hit]

    # --- ショートカットで経由点を減らす（細かい衝突チェック付き） ---
    out = []
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _seg_free(pts[i], pts[j], obs):
            j -= 1
        out.append(pts[j])
        i = j
    return out
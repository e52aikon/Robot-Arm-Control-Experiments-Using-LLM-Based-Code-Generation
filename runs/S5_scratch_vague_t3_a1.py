import math

# 想定障害物：正面やや左（約15度）、根元から約60cmの細い柱（直径約10cm）
# 位置は不正確なので、あくまで仮定値
_A = math.radians(15.0)
ASSUMED_OBSTACLES = [
    {"x": 0.60 * math.cos(_A), "y": 0.60 * math.sin(_A), "r": 0.05},
]

L1, L2 = 0.5, 0.4
HALF_T = 0.02      # リンク半幅
MARGIN = 0.02      # 安全余裕
EXTRA = 0.01       # 補間サンプリング誤差の上乗せ
QLIM = 2.98
N = 30             # 格子分割数（片側）
STEP = QLIM / N
SAMPLE = 0.02      # 衝突判定の最大角度刻み[rad]


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


def plan(target, obstacles):
    obs = obstacles if obstacles else ASSUMED_OBSTACLES
    x, y = target

    def pose_free(q1, q2):
        if abs(q1) > QLIM or abs(q2) > QLIM:
            return False
        e, t = _fk(q1, q2)
        for o in obs:
            lim = o["r"] + HALF_T + MARGIN + EXTRA
            if _seg_dist(o["x"], o["y"], 0, 0, e[0], e[1]) < lim:
                return False
            if _seg_dist(o["x"], o["y"], e[0], e[1], t[0], t[1]) < lim:
                return False
        return True

    def edge_free(a, b):
        m = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        n = max(1, int(math.ceil(m / SAMPLE)))
        for k in range(n + 1):
            s = k / n
            if not pose_free(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s):
                return False
        return True

    # --- 逆運動学（2解） ---
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1.0 or c2 < -1.0:
        raise ValueError("target unreachable")
    goals = []
    for sgn in (1, -1):
        q2 = sgn * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QLIM and abs(q2) <= QLIM and pose_free(q1, q2):
            goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible IK solution (limits or obstacle)")

    start = (0.0, 0.0)
    if not pose_free(*start):
        raise ValueError("start pose in collision")

    # --- 関節空間の格子上で幅優先探索 ---
    def q_of(idx):
        return (idx[0] * STEP, idx[1] * STEP)

    free_cache = {}

    def node_free(idx):
        if idx not in free_cache:
            free_cache[idx] = pose_free(*q_of(idx))
        return free_cache[idx]

    s_idx = (0, 0)
    prev = {s_idx: None}
    queue = [s_idx]
    head = 0
    found = None
    while head < len(queue):
        cur = queue[head]
        head += 1
        qc = q_of(cur)
        for g in goals:
            if math.hypot(qc[0] - g[0], qc[1] - g[1]) < 0.4 and edge_free(qc, g):
                found = (cur, g)
                break
        if found:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (cur[0] + di, cur[1] + dj)
                if abs(nb[0]) > N or abs(nb[1]) > N or nb in prev:
                    continue
                if node_free(nb) and edge_free(qc, q_of(nb)):
                    prev[nb] = cur
                    queue.append(nb)
    if not found:
        raise ValueError("no collision-free path found")

    # 経路復元
    path = []
    n = found[0]
    while n is not None:
        path.append(q_of(n))
        n = prev[n]
    path.reverse()
    path.append(found[1])

    # ショートカットで経由点を減らす
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out[1:]  # 開始姿勢は除く
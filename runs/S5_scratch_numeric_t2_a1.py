import math


def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    LIM = 2.98
    HALF = 0.02      # リンク半幅（太さ4cm）
    MARGIN = 0.02    # 安全余裕
    EXTRA = 0.005    # 追加マージン（数値誤差対策）
    tx, ty = target

    # ---- 乱数（決定的なLCG。math以外は使わない） ----
    state = [12345]

    def rnd():
        state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return state[0] / 0x80000000

    def wrap(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # ---- 点と線分の距離 ----
    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def config_free(q):
        q1, q2 = q
        if abs(q1) > LIM or abs(q2) > LIM:
            return False
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        fx, fy = ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2)
        for ob in obstacles:
            need = ob["r"] + MARGIN + HALF + EXTRA
            if seg_dist(ob["x"], ob["y"], 0, 0, ex, ey) < need:
                return False
            if seg_dist(ob["x"], ob["y"], ex, ey, fx, fy) < need:
                return False
        return True

    def edge_free(a, b):
        m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        n = max(1, int(math.ceil(m / 0.01)))
        for i in range(n + 1):
            t = i / n
            if not config_free((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return False
        return True

    # ---- 逆運動学（肘上げ・肘下げの2解） ----
    d2 = tx * tx + ty * ty
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1 + 1e-9 or c2 < -1 - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, c2))

    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c2)
        q1 = wrap(math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)))
        q2 = wrap(q2)
        if abs(q1) <= LIM and abs(q2) <= LIM and config_free((q1, q2)):
            goals.append((q1, q2))
    if not goals:
        raise ValueError("target unreachable or blocked")

    start = (0.0, 0.0)
    if not config_free(start):
        raise ValueError("start pose in collision")

    # ---- 直接移動できるか ----
    for g in goals:
        if edge_free(start, g):
            return [g]

    # ---- 関節空間RRT ----
    nodes = [start]
    parent = [-1]
    STEP = 0.15
    found = None
    for _ in range(30000):
        if rnd() < 0.2:
            smp = goals[int(rnd() * len(goals)) % len(goals)]
        else:
            smp = ((rnd() * 2 - 1) * LIM, (rnd() * 2 - 1) * LIM)
        bi, bd = 0, 1e18
        for i, n in enumerate(nodes):
            dd = (n[0] - smp[0]) ** 2 + (n[1] - smp[1]) ** 2
            if dd < bd:
                bd, bi = dd, i
        near = nodes[bi]
        dist = math.sqrt(bd)
        if dist < 1e-9:
            continue
        k = min(1.0, STEP / dist)
        new = (near[0] + (smp[0] - near[0]) * k, near[1] + (smp[1] - near[1]) * k)
        if not edge_free(near, new):
            continue
        nodes.append(new)
        parent.append(bi)
        idx = len(nodes) - 1
        for g in goals:
            if edge_free(new, g):
                nodes.append(g)
                parent.append(idx)
                found = len(nodes) - 1
                break
        if found is not None:
            break
    if found is None:
        raise ValueError("no collision-free path found")

    path = []
    i = found
    while i != -1:
        path.append(nodes[i])
        i = parent[i]
    path.reverse()

    # ---- ショートカットで経路を簡略化 ----
    for _ in range(200):
        if len(path) <= 2:
            break
        i = int(rnd() * (len(path) - 2))
        j = i + 2 + int(rnd() * (len(path) - i - 2))
        if j < len(path) and edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]

    return path[1:]
import math

def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    QMAX = 2.98
    # リンク半幅 0.02 + 安全余裕 0.02 + 追加マージン 0.005
    CLR = 0.02 + 0.02 + 0.005
    tx, ty = float(target[0]), float(target[1])

    # ---- 運動学 ----
    def fk(q1, q2):
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        a = q1 + q2
        return (ex, ey), (ex + L2 * math.cos(a), ey + L2 * math.sin(a))

    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def pose_free(q):
        e, t = fk(q[0], q[1])
        for o in obstacles:
            lim = o["r"] + CLR
            if seg_dist(o["x"], o["y"], 0.0, 0.0, e[0], e[1]) < lim:
                return False
            if seg_dist(o["x"], o["y"], e[0], e[1], t[0], t[1]) < lim:
                return False
        return True

    def edge_free(a, b, step=0.005):
        n = max(1, int(math.ceil(max(abs(a[0] - b[0]), abs(a[1] - b[1])) / step)))
        for i in range(n + 1):
            u = i / n
            if not pose_free((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)):
                return False
        return True

    # ---- 逆運動学（2解） ----
    d2 = tx * tx + ty * ty
    d = math.sqrt(d2)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            _, tip = fk(q1, q2)
            if math.hypot(tip[0] - tx, tip[1] - ty) < 0.005:
                goals.append((q1, q2))
    if not goals:
        raise ValueError("target outside joint limits")
    goals = [g for g in goals if pose_free(g)]
    if not goals:
        raise ValueError("target blocked by obstacle")

    start = (0.0, 0.0)
    if not pose_free(start):
        raise ValueError("start pose in collision")

    # ---- 直接移動できるか ----
    for g in goals:
        if edge_free(start, g):
            return [g]

    # ---- RRT-Connect（決定的な乱数） ----
    seed = [123456789]
    def rnd():
        seed[0] = (1103515245 * seed[0] + 12345) % (2 ** 31)
        return seed[0] / float(2 ** 31)

    def rand_q():
        return ((rnd() * 2 - 1) * QMAX, (rnd() * 2 - 1) * QMAX)

    STEP = 0.15

    def nearest(nodes, q):
        best, bi = 1e18, 0
        for i, n in enumerate(nodes):
            dd = (n[0] - q[0]) ** 2 + (n[1] - q[1]) ** 2
            if dd < best:
                best, bi = dd, i
        return bi

    def extend(tree, q):
        nodes, par = tree
        i = nearest(nodes, q)
        n = nodes[i]
        dx, dy = q[0] - n[0], q[1] - n[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-12:
            return "reached", i
        if dist <= STEP:
            new, status = q, "reached"
        else:
            new = (n[0] + dx / dist * STEP, n[1] + dy / dist * STEP)
            status = "advanced"
        if not edge_free(n, new):
            return "trapped", i
        nodes.append(new)
        par.append(i)
        return status, len(nodes) - 1

    def trace(tree, i):
        nodes, par = tree
        out = []
        while i != -1:
            out.append(nodes[i])
            i = par[i]
        return out[::-1]

    def rrt(goal):
        ta = ([start], [-1])
        tb = ([goal], [-1])
        swapped = False
        for _ in range(4000):
            q = rand_q()
            st, ia = extend(ta, q)
            if st != "trapped":
                qn = ta[0][ia]
                while True:
                    st2, ib = extend(tb, qn)
                    if st2 != "advanced":
                        break
                if st2 == "reached":
                    pa, pb = trace(ta, ia), trace(tb, ib)
                    if swapped:
                        pa, pb = pb, pa
                    return pa + pb[::-1][0:0] + pb[::-1]
            ta, tb = tb, ta
            swapped = not swapped
        return None

    path = None
    for g in goals:
        path = rrt(g)
        if path:
            break
    if not path:
        raise ValueError("no collision-free path found")

    # 重複点除去
    clean = [path[0]]
    for p in path[1:]:
        if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    path = clean

    # ---- ショートカットで経由点を削減 ----
    for _ in range(200):
        if len(path) < 3:
            break
        i = int(rnd() * (len(path) - 2))
        j = i + 2 + int(rnd() * (len(path) - i - 2))
        if j < len(path) and edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]

    return path[1:]
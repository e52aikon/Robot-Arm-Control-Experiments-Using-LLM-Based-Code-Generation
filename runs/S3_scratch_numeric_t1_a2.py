# 原因: collections (deque) を import したため。math 以外は import 不可。
# 対策: deque をやめ、list と先頭インデックスでキューを実装した。
import math


def plan(target, obstacles):
    L1, L2 = 0.5, 0.4
    LIM = 2.98
    CLEAR = 0.02 + 0.02 + 0.005  # リンク半幅 + 安全余裕 + 予備

    tx, ty = target

    # 点と線分の距離
    def pt_seg(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    # 姿勢が可動範囲内かつ障害物と非接触か
    def pose_free(q1, q2):
        if abs(q1) > LIM or abs(q2) > LIM:
            return False
        ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
        fx = ex + L2 * math.cos(q1 + q2)
        fy = ey + L2 * math.sin(q1 + q2)
        for o in obstacles:
            lim = o["r"] + CLEAR
            if pt_seg(o["x"], o["y"], 0, 0, ex, ey) < lim:
                return False
            if pt_seg(o["x"], o["y"], ex, ey, fx, fy) < lim:
                return False
        return True

    # 関節空間の直線補間が安全か
    def seg_free(a, b):
        n = max(1, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.02)))
        for i in range(n + 1):
            t = i / n
            if not pose_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
                return False
        return True

    # 逆運動学（肘の2解）
    d2 = tx * tx + ty * ty
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1 + 1e-9 or c2 < -1 - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, c2))
    goals = []
    for s in (1, -1):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= LIM and abs(q2) <= LIM and pose_free(q1, q2):
            if not any(abs(q1 - g[0]) < 1e-9 and abs(q2 - g[1]) < 1e-9 for g in goals):
                goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible joint solution")

    start = (0.0, 0.0)
    if not pose_free(*start):
        raise ValueError("start pose in collision")

    # 関節空間グリッド上のBFS
    res = 0.05
    N = int(LIM / res)
    free_cache = {}

    def node_free(i, j):
        k = (i, j)
        if k not in free_cache:
            free_cache[k] = pose_free(i * res, j * res)
        return free_cache[k]

    def reach_goal(i, j):
        q = (i * res, j * res)
        for g in goals:
            if math.hypot(q[0] - g[0], q[1] - g[1]) < 0.6 and seg_free(q, g):
                return g
        return None

    parent = {(0, 0): None}
    queue = [(0, 0)]  # list をキューとして使用
    head = 0
    found = None
    while head < len(queue):
        cur = queue[head]
        head += 1
        g = reach_goal(*cur)
        if g is not None:
            found = (cur, g)
            break
        ci, cj = cur
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if abs(ni) > N or abs(nj) > N or (ni, nj) in parent:
                    continue
                if not node_free(ni, nj):
                    continue
                if not seg_free((ci * res, cj * res), (ni * res, nj * res)):
                    continue
                parent[(ni, nj)] = cur
                queue.append((ni, nj))

    if found is None:
        raise ValueError("no collision-free path")

    # 経路復元
    cur, g = found
    nodes = []
    while cur is not None:
        nodes.append((cur[0] * res, cur[1] * res))
        cur = parent[cur]
    nodes.reverse()
    nodes.append(g)

    # ショートカットで経由点を削減
    result = []
    i = 0
    last = len(nodes) - 1
    while i < last:
        j = last
        while j > i + 1 and not seg_free(nodes[i], nodes[j]):
            j -= 1
        result.append(nodes[j])
        i = j
    return result
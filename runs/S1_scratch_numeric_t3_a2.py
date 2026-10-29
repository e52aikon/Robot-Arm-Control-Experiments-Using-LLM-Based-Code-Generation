import math

# 原因: 前回の出力は「def to_q」の途中で切れており、関数定義が未完成のまま
# 終わっていたため SyntaxError になった。
# 対策: 探索部分まで省略せずに書き切り、全体を完結したコードとして出力する。


def plan(target, obstacles):
    # ---- ロボット諸元 ----
    L1, L2 = 0.5, 0.4
    QMAX = 2.98
    HALF_W = 0.02      # リンク太さ 4cm の半分
    MARGIN = 0.02      # 検証系の安全余裕
    BUF = 0.01         # 補間間隔などの誤差を吸収する追加余裕
    START = (0.0, 0.0)

    tx, ty = target

    # ---- 順運動学（肘と先端の位置）----
    def fk(q1, q2):
        ex = L1 * math.cos(q1)
        ey = L1 * math.sin(q1)
        px = ex + L2 * math.cos(q1 + q2)
        py = ey + L2 * math.sin(q1 + q2)
        return (ex, ey), (px, py)

    # ---- 点と線分の距離 ----
    def seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy
        if ll < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / ll
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    # ---- 1姿勢の可動範囲・障害物チェック ----
    def config_free(q):
        q1, q2 = q
        if abs(q1) > QMAX or abs(q2) > QMAX:
            return False
        elbow, tip = fk(q1, q2)
        for o in obstacles:
            need = o["r"] + HALF_W + MARGIN + BUF
            if seg_dist(o["x"], o["y"], 0.0, 0.0, elbow[0], elbow[1]) < need:
                return False
            if seg_dist(o["x"], o["y"], elbow[0], elbow[1], tip[0], tip[1]) < need:
                return False
        return True

    # ---- 関節空間の直線補間エッジのチェック ----
    def edge_free(a, b):
        delta = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        n = max(1, int(math.ceil(delta / 0.01)))
        for i in range(n + 1):
            t = i / n
            q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not config_free(q):
                return False
        return True

    # ---- 到達可能性チェック ----
    d2 = tx * tx + ty * ty
    d = math.sqrt(d2)
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")

    # ---- 逆運動学（肘の上下 2 解）----
    c2 = (d2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    goals = []
    for s in (1.0, -1.0):
        q2 = s * math.acos(c2)
        q1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(q2),
                                             L1 + L2 * math.cos(q2))
        q1 = math.atan2(math.sin(q1), math.cos(q1))
        if abs(q1) <= QMAX and abs(q2) <= QMAX and config_free((q1, q2)):
            goals.append((q1, q2))
    if not goals:
        raise ValueError("no feasible IK solution (joint limit or obstacle)")

    if not config_free(START):
        raise ValueError("start posture collides with an obstacle")

    # 動きの小さい解を優先
    goals.sort(key=lambda g: abs(g[0]) + abs(g[1]))

    # ---- 1) 直接移動できるならそれを採用 ----
    for g in goals:
        if edge_free(START, g):
            return [g]

    # ---- 2) 直接が無理なら関節空間グリッドを幅優先探索 ----
    res = 0.05
    N = int(QMAX / res)

    def to_q(node):
        return (node[0] * res, node[1] * res)

    free_cache = {}

    def node_free(node):
        if node not in free_cache:
            free_cache[node] = config_free(to_q(node))
        return free_cache[node]

    start_node = (0, 0)
    parent = {start_node: None}
    queue = [start_node]
    head = 0
    found = None  # (最終グリッドノード, ゴール姿勢)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while head < len(queue) and found is None:
        cur = queue[head]
        head += 1
        qc = to_q(cur)

        # ゴールに近いノードからはゴールへ直接接続を試す
        for g in goals:
            if max(abs(qc[0] - g[0]), abs(qc[1] - g[1])) <= 0.5:
                if edge_free(qc, g):
                    found = (cur, g)
                    break
        if found is not None:
            break

        for dx, dy in dirs:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in parent:
                continue
            if abs(nb[0]) > N or abs(nb[1]) > N:
                continue
            if not node_free(nb):
                continue
            if not edge_free(qc, to_q(nb)):
                continue
            parent[nb] = cur
            queue.append(nb)

    if found is None:
        raise ValueError("no collision-free path found")

    # ---- 経路復元 ----
    nodes = []
    n = found[0]
    while n is not None:
        nodes.append(n)
        n = parent[n]
    nodes.reverse()
    path = [to_q(n) for n in nodes] + [found[1]]
    path[0] = START

    # ---- 3) ショートカットで経由点を削減 ----
    simplified = [path[0]]
    i = 0
    last = len(path) - 1
    while i < last:
        j = last
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        simplified.append(path[j])
        i = j

    # 開始姿勢は含めず、経由点のみ返す
    return simplified[1:]
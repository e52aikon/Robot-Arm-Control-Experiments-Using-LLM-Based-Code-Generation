# 原因: 前回は説明が長すぎて出力が途中で切れた。今回は説明を最小限にした。
# 方針: 左斜め前70cm（45度と仮定）の柱を仮定障害物にし、関節空間グリッド上で干渉しない経路を探索して経由点を返す。
import math

# 仮定障害物: 左斜め前70cm、柱の半径5cm + 位置不確かさ5cm = r 0.10
ASSUMED_OBSTACLES = [{"x": 0.70 * math.cos(math.pi / 4),
                      "y": 0.70 * math.sin(math.pi / 4),
                      "r": 0.10}]

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値 1cm
GRID_N = 21              # 探索グリッド分割数（各関節）
LINK_RADIUS = 0.65       # グリッド間の接続半径 [rad]


def _wrap(a):
    """角度を (-pi, pi] に正規化する。"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _fk(q1, q2):
    """順運動学: 肘と先端の座標を返す。"""
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_point_dist(p, a, b):
    """線分 ab と点 p の最短距離。"""
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    len2 = abx * abx + aby * aby
    t = 0.0 if len2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / len2))
    cx, cy = a[0] + t * abx, a[1] + t * aby
    return math.hypot(p[0] - cx, p[1] - cy)


def _pose_collides(q1, q2, obstacles):
    """ある姿勢で、いずれかのリンクが障害物（+太さ+余裕）に接触するか。"""
    elbow, tip = _fk(q1, q2)
    base = (0.0, 0.0)
    for ob in obstacles:
        c = (ob["x"], ob["y"])
        clearance = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN
        if _seg_point_dist(c, base, elbow) < clearance:
            return True
        if _seg_point_dist(c, elbow, tip) < clearance:
            return True
    return False


def _path_collides(q_start, q_end, obstacles, step=0.02):
    """関節空間の直線補間を、最大関節変化 step[rad] ごとにサンプリングして調べる。"""
    d = max(abs(q_end[0] - q_start[0]), abs(q_end[1] - q_start[1]))
    steps = max(1, int(math.ceil(d / step)))
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def plan(target, obstacles):
    if not obstacles:
        obstacles = ASSUMED_OBSTACLES
    x, y = target
    r2 = x * x + y * y
    r = math.sqrt(r2)

    eps = 1e-9
    if r > L1 + L2 + eps or r < abs(L1 - L2) - eps:
        raise ValueError("target is out of reach")

    c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))

    # 肘上げ/肘下げの2解を候補にする
    goals = []
    for s2 in (s2_abs, -s2_abs):
        q2 = math.atan2(s2, c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * s2, L1 + L2 * c2)
        q1 = _wrap(q1)
        q2 = _wrap(q2)
        if abs(q1) > JOINT_LIMIT or abs(q2) > JOINT_LIMIT:
            continue
        _, tip = _fk(q1, q2)
        if math.hypot(tip[0] - x, tip[1] - y) > TOLERANCE:
            continue
        if _pose_collides(q1, q2, obstacles):
            continue
        goals.append((q1, q2))

    if not goals:
        raise ValueError("no valid joint solution within joint limits")

    # ノード: 開始, 目標候補, 関節空間グリッド
    vals = [-JOINT_LIMIT + 2.0 * JOINT_LIMIT * i / (GRID_N - 1) for i in range(GRID_N)]
    nodes = [(0.0, 0.0)] + goals + [(a, b) for a in vals for b in vals]
    n = len(nodes)
    ng = len(goals)

    # 干渉チェック付きダイクストラ（辺は展開時に遅延評価）
    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    done = [False] * n
    dist[0] = 0.0
    while True:
        u = -1
        best = INF
        for i in range(n):
            if not done[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u < 0:
            break
        done[u] = True
        if 1 <= u <= ng:
            path = []
            while u > 0:
                path.append(nodes[u])
                u = prev[u]
            path.reverse()
            return path
        for v in range(n):
            if done[v]:
                continue
            d = math.hypot(nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1])
            if d > LINK_RADIUS or dist[u] + d >= dist[v]:
                continue
            if not _path_collides(nodes[u], nodes[v], obstacles):
                dist[v] = dist[u] + d
                prev[v] = u

    raise ValueError("no collision-free path found for this target")
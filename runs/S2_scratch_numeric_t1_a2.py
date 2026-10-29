```python
import math

# 原因: 前回の出力はコードが途中（"def d" の行）で途切れており、
#       不完全な構文のため SyntaxError になった。
#       今回は最後まで完結したコードを出力する。
#       また、(0,0) から目標の逆運動学解へ直線補間すると、第2リンクが
#       障害物に近づきすぎる（約 0.088 m < 必要 0.09 m）ため、
#       経由点探索（直接 → 中継点 → グリッド探索）を入れている。

# ロボット仕様
L1 = 0.5
L2 = 0.4
Q_LIM = 2.98
HALF_THICK = 0.02      # リンク太さ 4cm の半分
SAFETY = 0.02          # 検証系の安全余裕
EXTRA = 0.005          # サンプリング誤差などに備えた追加余裕


def _fk(q1, q2):
    """肘と先端の位置を返す"""
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    a = q1 + q2
    return (ex, ey), (ex + L2 * math.cos(a), ey + L2 * math.sin(a))


def _seg_point_dist(ax, ay, bx, by, px, py):
    """線分ABと点Pの最短距離"""
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def plan(target, obstacles):
    tx, ty = target
    d2 = tx * tx + ty * ty
    d = math.sqrt(d2)

    # ---- 到達可能性の判定 ----
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")

    # ---- 逆運動学（肘上・肘下の2解）----
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    goals = []
    for sgn in (1.0, -1.0):
        q2 = sgn * math.acos(c2)
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
    def config_free(q):
        if abs(q[0]) > Q_LIM or abs(q[1]) > Q_LIM:
            return False
        elbow, tip = _fk(q[0], q[1])
        for ob in obstacles:
            lim = ob["r"] + HALF_THICK + SAFETY + EXTRA
            if _seg_point_dist(0.0, 0.0, elbow[0], elbow[1],
                               ob["x"], ob["y"]) < lim:
                return False
            if _seg_point_dist(elbow[0], elbow[1], tip[0], tip[1],
                               ob["x"], ob["y"]) < lim:
                return False
        return True

    cache = {}

    def edge_free(a, b):
        """関節空間の直線補間が全域で安全か"""
        key = (a, b)
        if key in cache:
            return cache[key]
        diff = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        n = max(1, int(math.ceil(diff / 0.01)))
        ok = True
        for k in range(n + 1):
            t = k / n
            q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not config_free(q):
                ok = False
                break
        cache[key] = ok
        cache[(b, a)] = ok
        return ok

    start = (0.0, 0.0)
    if not config_free(start):
        raise ValueError("start configuration is in collision")

    def path_free(pts):
        prev = start
        for p in pts:
            if not edge_free(prev, p):
                return False
            prev = p
        return True

    # ---- 1) 直接 / 中継点1つの単純な経路 ----
    for g in goals:
        for cand in ([g], [(0.0, g[1]), g], [(g[0], 0.0), g]):
            if path_free(cand):
                return [tuple(p) for p in cand]

    # ---- 2) グリッド探索（BFS、
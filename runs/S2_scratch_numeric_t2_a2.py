```python
# 原因: 前回の出力が途中で切れており、plan 関数内の `if _edge_ok(...):` の本体や
#       関数の残りが欠けていたため SyntaxError になった。
# 対策: 完全なコードを出力する。直接移動が衝突する場合に備え、
#       関節空間グリッド上の幅優先探索で障害物を回避する経路を求める。
#       (この目標は、どちらの肘形状でも直線補間だと障害物に近づきすぎるため探索が必要)
import math

L1 = 0.5
L2 = 0.4
LIMIT = 2.98
HALF_WIDTH = 0.02      # リンク太さ 4cm の半分
SAFETY = 0.02          # 検証系の安全余裕
EXTRA = 0.006          # 補間の離散化誤差に備えた追加マージン
STEP = 0.01            # 衝突チェックの関節角刻み [rad]
GSTEP = 0.1            # 探索グリッドの刻み [rad]
GOAL_RANGE = 3         # ゴールとグリッド点を接続する探索範囲 (格子数)


def _fk(q):
    q1, q2 = q
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_dist(px, py, ax, ay, bx, by):
    # 点と線分の最短距離
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _config_ok(q, obstacles):
    if abs(q[0]) > LIMIT or abs(q[1]) > LIMIT:
        return False
    (ex, ey), (tx, ty) = _fk(q)
    for ob in obstacles:
        lim = ob["r"] + HALF_WIDTH + SAFETY + EXTRA
        if _seg_dist(ob["x"], ob["y"], 0.0, 0.0, ex, ey) < lim:
            return False
        if _seg_dist(ob["x"], ob["y"], ex, ey, tx, ty) < lim:
            return False
    return True


def _edge_ok(a, b, obstacles):
    # 関節空間の直線補間上を細かくチェック
    m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    n = max(1, int(math.ceil(m / STEP)))
    for i in range(n + 1):
        t = i / n
        q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if not _config_ok(q, obstacles):
            return False
    return True


def _ik(target):
    # 解析的逆運動学。肘の2通りの解を返す。到達不可なら ValueError
    x, y = target
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1.0 + 1e-9 or c2 < -1.0 - 1e-9:
        raise ValueError("target is unreachable")
    c2
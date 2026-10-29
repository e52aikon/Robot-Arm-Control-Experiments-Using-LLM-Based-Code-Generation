```python
import math

# 【不合格の原因】
# 前回の出力に実行可能な plan 関数の定義が含まれておらず、検証系が
# 名前 'plan' を見つけられず KeyError になった。
# 今回は plan を含むコード全体を出力する。
# あわせて、この障害物 (0.78, 0.44) r=0.05 では肘下げ解の直線補間経路が
# 最短でも余裕ぎりぎりなので、衝突判定を細かくし、
# 直接経路が通らない場合に備えて経由点を探索するフォールバックを追加した。

L1 = 0.5            # 根元〜肘 [m]
L2 = 0.4            # 肘〜先端 [m]
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY_MARGIN = 0.02     # 安全余裕 2cm
JOINT_LIMIT = 2.98       # 可動範囲 ±2.98 rad
TOLERANCE = 0.01         # 最終誤差の許容値 1cm（IK誤差はこれよりはるかに小さい）


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


def _path_collides(q_start, q_end, obstacles, steps=400):
    """関節空間の直線補間経路を細かくサンプリングして干渉を調べる。"""
    for i in range(steps + 1):
        s = i / steps
        q1 = q_start[0] + s * (q_end[0] - q_start[0])
        q2 = q_start[1] + s * (q_end[1] - q_start[1])
        if _pose_collides(q1, q2, obstacles):
            return True
    return False


def _find_via_path(start, goal, obstacles):
    """直接経路が通らないとき、関節空間の格子から経由点を1つ探す。"""
    step = 0.1
    n = int(JOINT_LIMIT / step)
    best = None
    best_cost = None
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            via = (i * step, j * step)
            cost = (abs(via[0] - start[0]) + abs(via[1] - start[1])
                    + abs(goal[0] - via[0]) + abs(goal[1] - via[1]))
            if best_cost is not None and cost >= best_cost:
                continue
            if _pose_collides(via[0], via[1], obstacles):
                continue
            if _path_collides(start, via, obstacles, steps=
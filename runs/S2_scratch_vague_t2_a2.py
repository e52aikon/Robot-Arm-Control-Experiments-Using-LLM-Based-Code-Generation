```python
# 原因: 前回の回答に、トップレベルで定義された関数 plan がなく、
#       検証系が名前 'plan' を取得できず KeyError になった。
#       今回は plan を必ずモジュール直下に定義する。

import math

# 想定障害物: 正面やや左、腕を伸ばしきった付近の細い柱（直径約10cm）。
# 正確な位置は不明なので、実際の位置とのずれ用に追加余裕 ASSUMED_MARGIN を取る。
ASSUMED_OBSTACLES = [{"x": 0.82, "y": 0.15, "r": 0.05}]
ASSUMED_MARGIN = 0.05  # 位置不確かさ分の追加余裕 [m]

L1 = 0.5
L2 = 0.4
HALF_THICK = 0.02    # リンク太さ4cmの半分
SAFETY = 0.02        # 検証系の安全余裕
Q_LIMIT = 2.98
TOL = 0.01
STEPS = 200          # 区間ごとの干渉チェック分割数


def _wrap(a):
    # 角度を (-pi, pi] に正規化
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_point_dist(ax, ay, bx, by, px, py):
    # 線分ABと点Pの最短距離
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_clear(q1, q2, obs_list):
    # obs_list: [(x, y, 必要クリアランス込みの半径)]
    (ex, ey), (tx, ty) = _fk(q1, q2)
    for ox, oy, rr in obs_list:
        if _seg_point_dist(0.0, 0.0, ex, ey, ox, oy) < rr:
            return False
        if _seg_point_dist(ex, ey, tx, ty, ox, oy) < rr:
            return False
    return True


def _path_clear(waypoints, obs_list):
    # 開始姿勢(0,0)から各経由点へ関節空間で直線補間し、全区間を検査
    prev = (0.0, 0.0)
    if not _pose_clear(0.0, 0.0, obs_list):
        return False
    for wp in waypoints:
        for i in range(1, STEPS + 1):
            t = i / STEPS
            q1 = prev[0] + (wp[0] - prev[0]) * t
            q2 = prev[1] + (wp[1] - prev[1]) * t
            if not _pose_clear(q1, q2, obs_list):
                return False
        prev = wp
    return True


def _path_length(waypoints):
    prev = (0.0, 0.0)
    total = 0.0
    for wp in waypoints:
        total += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
        prev = wp
    return total


def plan(target, obstacles):
    tx, ty = target
    d2 = tx * tx + ty * ty
    d = math.sqrt(d2)

    # 到達可能性チェック
    if d > L1 + L2 + 1e-9 or d < abs(L1 - L2) - 1e-9:
        raise ValueError("target is out of reach")

    # 逆運動学（肘上げ・肘下げの2解）
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))

    solutions = []
    for s2 in (s2_abs, -s2_abs):
        q2
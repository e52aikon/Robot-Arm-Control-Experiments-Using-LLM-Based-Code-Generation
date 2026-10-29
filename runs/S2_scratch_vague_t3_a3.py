```python
# 原因: 前回までの出力は、コードブロックが途中で切れていたり、Python として
#       不正な文字列が混ざっていたりして、plan 関数が定義されず構文エラーになった。
#       今回は完結した Python コードだけを出力する。
#
# 方針:
#  - 柱の位置は不明なので、想定位置・半径に加えて位置誤差の余裕 POS_UNCERTAINTY を上乗せする。
#  - 逆運動学の2解（肘の上下）と、経由点の組み合わせを複数用意する。
#  - 各候補について、関節空間の直線補間を細かくサンプリングして衝突判定し、
#    安全なもののうち関節移動量が最小のものを返す。
#  - 開始姿勢は腕が伸びきっているので、柱側に先端を振らないよう、
#    先に肘を畳む経路も候補に含める。

import math

# 想定障害物: 正面やや左、腕を伸ばしきった付近の細い柱(直径10cm)
ASSUMED_OBSTACLES = [{"x": 0.85, "y": 0.20, "r": 0.05}]

L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02      # リンク太さ 4 cm の半分
SAFETY_MARGIN = 0.02        # 検証系の安全余裕
POS_UNCERTAINTY = 0.03      # 柱の位置が不確かなための追加余裕
Q_LIMIT = 2.98
TOL = 0.01                  # 最終誤差の許容 1 cm
N_SAMPLES = 120             # 1区間あたりの衝突判定サンプル数


def _wrap(a):
    # 角度を [-pi, pi] に正規化
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_point_dist(ax, ay, bx, by, px, py):
    # 線分 AB と点 P の最短距離
    dx = bx - ax
    dy = by - ay
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _pose_collides(q1, q2, obstacles):
    elbow, tip = _fk(q1, q2)
    for ob in obstacles:
        need = ob["r"] + LINK_HALF_WIDTH + SAFETY_MARGIN + POS_UNCERTAINTY
        d1 = _seg_point_dist(0.0, 0.0, elbow[0], elbow[1], ob["x"], ob["y"])
        d2 = _seg_point_dist(elbow[0], elbow[1], tip[0], tip[1], ob["x"], ob["y"])
        if d1 < need or d2 < need:
            return True
    return False


def _path_ok(path, obstacles):
    # path: 開始姿勢を含む関節角列。可動範囲と衝突を確認
    for q in path:
        if abs(q[0]) > Q_LIMIT or abs(q[1]) > Q_LIMIT:
            return False
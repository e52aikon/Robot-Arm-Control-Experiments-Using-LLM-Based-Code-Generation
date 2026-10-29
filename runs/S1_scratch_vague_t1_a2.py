# 原因: 前回の出力が plan() の最後の return 文の途中で途切れており、
#       コードが未完成のため SyntaxError になった。
# 対策: 末尾まで省略せずにコード全体を書き直した。ロジックは前回と同じ。
import math

# ---- 想定障害物 -------------------------------------------------------
# 「アームの右側(-y)、少し前(+x)に直径10cmほどの細い柱」。正確な位置は不明なので、
# 右前方の代表的な位置を仮置きする。引数 obstacles が空のときだけ使う。
ASSUMED_OBSTACLES = [{"x": 0.30, "y": -0.25, "r": 0.05}]

# ---- ロボット仕様 -----------------------------------------------------
L1 = 0.5
L2 = 0.4
LINK_HALF_WIDTH = 0.02   # リンク太さ 4cm の半分
SAFETY = 0.02            # 検証系の安全余裕
POS_UNCERTAINTY = 0.05   # 想定障害物の位置誤差の見込み。想定値を使うときだけ上乗せ
Q_LIMIT = 2.98
TOL_USE = 0.002          # 計画上の目標誤差。許容(1cm)より十分小さくとる
SAMPLES = 200            # 区間ごとの干渉チェック分割数


def _fk(q1, q2):
    ex = L1 * math.cos(q1)
    ey = L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _in_limits(q1, q2):
    return abs(q1) <= Q_LIMIT and abs(q2) <= Q_LIMIT


def _ik(x, y):
    """リミット内の逆運動学解（肘の上下2通り）を全て返す。"""
    d2 = x * x + y * y
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        if abs(c2) > 1.0 + 1e-9:
            return []
        c2 = max(-1.0, min(1.0, c2))
    sols = []
    for sgn in (1.0, -1.0):
        q2 = sgn * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1 = _wrap(q1)
        q2 = _wrap(q2)
        if _in_limits(q1, q2):
            sols.append((q1, q2))
    return sols


def _pt_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _config_ok(q1, q2, obs):
    """ある姿勢で、両リンクが膨らませた障害物に入っていないか。"""
    (ex, ey), (tx, ty) = _fk(q1, q2)
    for (ox, oy, rr) in obs:
        if _pt_seg_dist(ox, oy, 0.0, 0.0, ex, ey) < rr:
            return False
        if _pt_seg_dist(ox, oy, ex, ey, tx, ty) < rr:
            return False
    return True


def _path_ok(configs, obs):
    """開始姿勢から各経由点へ関節空間で直線補間したとき、全区間が安全か。"""
    prev = (0.0, 0.0)
    for c in configs:
        if not _in_limits(c[0], c[1]):
            return False
        for i in range(SAMPLES + 1):
            t = i / SAMPLES
            q1 = prev[0] + (c[0] - prev[0]) * t
            q2 = prev[1] + (c[1] - prev[1]) * t
            if not _config_ok(q1, q2, obs):
                return False
        prev = c
    return True


def _cost(configs):
    prev = (0.0, 0.0)
    total = 0.0
    for c in configs:
        total += abs(c[0] - prev[0]) + abs(c[1] - prev[1])
        prev = c
    return total


def plan(target, obstacles):
    tx, ty = float(target[0]), float(target[1])

    # 引数の障害物が空なら想定障害物を使う（その場合のみ位置誤差分を上乗せ）
    use_assumed = not obstacles
    obs_src = ASSUMED_OBSTACLES if use_assumed else obstacles
    extra = POS_UNCERTAINTY if use_assumed else 0.0
    margin = LINK_HALF_WIDTH + SAFETY + extra
    obs = [(float(o["x"]), float(o["y"]), float(o["r"]) + margin) for o in obs_src]

    # 到達可否（逆運動学）
    goals = _ik(tx, ty)
    if not goals:
        raise ValueError("target is unreachable (out of workspace or joint limits)")

    # 最終誤差の確認
    goals = [g for g in goals
             if math.hypot(_fk(*g)[1][0] - tx, _fk(*g)[1][1] - ty) <= TOL_USE]
    if not goals:
        raise ValueError("target is unreachable")

    candidates = []

    # 1) 直接移動
    for g in goals:
        candidates.append([g])

    # 2) 経由点を1つ挟む（関節空間の格子から探索）
    steps = 15
    lim = Q_LIMIT - 0.05
    grid = [-lim + 2 * lim * i / steps for i in range(steps + 1)]
    for g in goals:
        for a in grid:
            for b in grid:
                candidates.append([(a, b), g])

    # 関節移動量の小さい順に試し、最初に安全なものを採用
    candidates.sort(key=_cost)
    for cand in candidates:
        if _path_ok(cand, obs):
            return [(float(q1), float(q2)) for (q1, q2) in cand]

    raise ValueError("no collision-free path found")
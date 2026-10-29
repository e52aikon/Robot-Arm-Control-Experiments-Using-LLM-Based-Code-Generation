import math

# 想定障害物：アーム右側(y<0)・少し前の細柱（直径10cm）。位置は不正確なので推定値
ASSUMED_OBSTACLES = [{"x": 0.30, "y": -0.25, "r": 0.05}]

L1, L2 = 0.5, 0.4
HALF_T = 0.02        # リンク太さ4cmの半分
SAFETY = 0.02        # 検証系の安全余裕
POS_UNCERT = 0.05    # 想定障害物の位置不確かさ分の上乗せ
QMAX = 2.98


def _fk(q1, q2):
    ex, ey = L1 * math.cos(q1), L1 * math.sin(q1)
    tx = ex + L2 * math.cos(q1 + q2)
    ty = ey + L2 * math.sin(q1 + q2)
    return (ex, ey), (tx, ty)


def _seg_dist(a, b, p):
    # 線分abと点pの距離
    abx, aby = b[0] - a[0], b[1] - a[1]
    l2 = abx * abx + aby * aby
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2))
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


def _pose_ok(q, obs):
    e, t = _fk(*q)
    for o in obs:
        p = (o["x"], o["y"])
        need = o["r"] + SAFETY + HALF_T
        if _seg_dist((0.0, 0.0), e, p) < need or _seg_dist(e, t, p) < need:
            return False
    return True


def _path_ok(qa, qb, obs):
    # 関節空間の直線補間を細かくサンプリングして干渉確認
    n = 80
    for i in range(n + 1):
        s = i / n
        q = (qa[0] + (qb[0] - qa[0]) * s, qa[1] + (qb[1] - qa[1]) * s)
        if not _pose_ok(q, obs):
            return False
    return True


def _dist(qa, qb):
    return math.hypot(qb[0] - qa[0], qb[1] - qa[1])


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def plan(target, obstacles):
    tx, ty = target
    # 障害物未指定なら想定障害物を使い、位置の不確かさ分を膨らませる
    if obstacles:
        obs = [dict(o) for o in obstacles]
    else:
        obs = [dict(o, r=o["r"] + POS_UNCERT) for o in ASSUMED_OBSTACLES]

    # 逆運動学（肘2通り）
    d2 = tx * tx + ty * ty
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 > 1.0 + 1e-9 or c2 < -1.0 - 1e-9:
        raise ValueError("target unreachable")
    c2 = max(-1.0, min(1.0, c2))
    phi = math.atan2(ty, tx)
    sols = []
    for sgn in (1.0, -1.0):
        q2 = sgn * math.acos(c2)
        q1 = phi - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q1, q2 = _wrap(q1), _wrap(q2)
        if abs(q1) <= QMAX and abs(q2) <= QMAX:
            sols.append((q1, q2))
    if not sols:
        raise ValueError("target unreachable within joint limits")

    start = (0.0, 0.0)
    if not _pose_ok(start, obs):
        raise RuntimeError("start pose collides with obstacle")

    # 1) 直接移動（関節距離が短い解を優先）
    for g in sorted(sols, key=lambda q: _dist(start, q)):
        if _pose_ok(g, obs) and _path_ok(start, g, obs):
            return [g]

    # 2) 経由点を1つ挟む（粗いグリッド探索）
    best = None
    step = 0.2
    n = int(QMAX / step)
    for g in sols:
        if not _pose_ok(g, obs):
            continue
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                v = (i * step, j * step)
                cost = _dist(start, v) + _dist(v, g)
                if best is not None and cost >= best[0]:
                    continue
                if _pose_ok(v, obs) and _path_ok(start, v, obs) and _path_ok(v, g, obs):
                    best = (cost, [v, g])
    if best:
        return best[1]

    raise RuntimeError("no collision-free path found")
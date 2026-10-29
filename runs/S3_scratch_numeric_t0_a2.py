import math

L1, L2 = 0.5, 0.4
CLEAR = 0.02 + 0.02 + 0.005      # リンクの半幅 + 安全余裕 + 念のための上乗せ

def ik(x, y, sign):
    d = (x * x + y * y - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(d) > 1:
        raise ValueError("到達不能")
    q2 = sign * math.acos(d)
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
    return q1, q2

def seg_dist(ax, ay, bx, by, px, py):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)

def hits(q, obstacles):
    ex, ey = L1 * math.cos(q[0]), L1 * math.sin(q[0])
    tx, ty = ex + L2 * math.cos(q[0] + q[1]), ey + L2 * math.sin(q[0] + q[1])
    for o in obstacles:
        c = o["r"] + CLEAR
        if seg_dist(0, 0, ex, ey, o["x"], o["y"]) < c or seg_dist(ex, ey, tx, ty, o["x"], o["y"]) < c:
            return True
    return False

def path_ok(waypoints, obstacles):
    prev = (0.0, 0.0)
    for wp in waypoints:
        n = max(1, int(max(abs(wp[0] - prev[0]), abs(wp[1] - prev[1])) / 0.005))
        for i in range(n + 1):
            q = (prev[0] + (wp[0] - prev[0]) * i / n, prev[1] + (wp[1] - prev[1]) * i / n)
            if hits(q, obstacles):
                return False
        prev = wp
    return True

def plan(target, obstacles):
    # まず直行を試し、だめなら「前腕を折りたたむ→根元を回す→伸ばす」を試す
    for sign in (1, -1):
        q = ik(*target, sign)
        if path_ok([q], obstacles):
            return [q]
    for sign in (1, -1):
        q = ik(*target, sign)
        for f in (1.5, 1.75, 2.0, 2.25, 2.5):
            fold = sign * f
            wps = [(0.0, fold), (q[0], fold), q]
            if path_ok(wps, obstacles):
                return wps
    raise ValueError("障害物を避ける経路が見つかりません")
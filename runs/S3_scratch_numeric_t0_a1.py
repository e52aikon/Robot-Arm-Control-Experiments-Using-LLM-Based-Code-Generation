import math

def plan(target, obstacles):
    # 障害物を考えず、肘を右に出す解でまっすぐ向かう
    L1, L2 = 0.5, 0.4
    x, y = target
    d = (x * x + y * y - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(d) > 1:
        raise ValueError("到達不能")
    q2 = math.acos(d)
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return [(q1, q2)]
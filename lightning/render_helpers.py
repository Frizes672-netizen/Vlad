"""Общие помощники сцен."""
from engine import rot_pts


def rot_parts(parts, cx, cy, ang):
    """Поворот всей фигуры вокруг точки (лёжа, падение)."""
    out = []
    for kind_, geo, part in parts:
        if kind_ == "poly":
            geo = rot_pts(geo, cx, cy, ang)
        elif kind_ == "ell":
            x, y = rot_pts([(geo[0], geo[1])], cx, cy, ang)[0]
            geo = (x, y, geo[2], geo[3])
        else:
            p0, p1, r0, r1 = geo
            geo = (rot_pts([p0], cx, cy, ang)[0], rot_pts([p1], cx, cy, ang)[0], r0, r1)
        out.append((kind_, geo, part))
    return out

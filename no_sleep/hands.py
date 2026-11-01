"""Реалистичные руки от первого лица: ладонь со светотенью, четыре пальца по три фаланги
с суставами, складками и ногтями, большой палец, рукав с манжетой."""
import math

from engine import capsule, mix

OLD_SKIN = (190, 156, 134)
SLEEVE = (78, 80, 86)


def _cap(p0, p1, r0, r1):
    return capsule(r0, r1, p0, p1)


def pov_hand(c, base, tip, ang_fingers=0.0, curl=0.0, scale=1.0, skin=OLD_SKIN, sleeve=SLEEVE, old=None):
    """Рука из-за края кадра: base — откуда идёт рукав, tip — запястье.
    ang_fingers — поворот кисти, curl 0..1 — насколько согнуты пальцы."""
    (bx, by), (x, y) = base, tip
    s = scale * 1.6          # кисть в пропорции к рукаву
    if old is None:
        old = sum(skin) < 500
    dark = mix(skin, (40, 20, 14), 0.32)
    shade = mix(skin, (40, 20, 14), 0.16)
    light = mix(skin, (255, 236, 220), 0.18)
    nail = mix(skin, (255, 226, 220), 0.45)
    # рукав и манжета
    sl_d = mix(sleeve, (0, 0, 0), 0.35)
    c.poly(_cap((bx, by), (x, y), 66 * s, 48 * s), sl_d)
    c.poly(_cap((bx, by), (x, y), 61 * s, 43 * s), sleeve)
    a = math.atan2(y - by, x - bx) + ang_fingers
    ux, uy = math.cos(a), math.sin(a)                 # вдоль кисти
    nx, ny = -uy, ux                                  # поперёк
    # большой палец смотрит к центру кадра
    side = 1 if (x + nx * 10 - 540) ** 2 < (x - nx * 10 - 540) ** 2 else -1
    nx, ny = nx * side, ny * side

    def P(u, n):
        return (x + ux * u * s + nx * n * s, y + uy * u * s + ny * n * s)

    c.poly(_cap(P(-24, 0), P(4, 0), 46 * s, 44 * s), mix(sleeve, (255, 255, 255), 0.08))  # манжета
    # запястье и ладонь (тыльная сторона)
    palm = [P(0, -46), P(0, 46), P(40, 58), P(96, 54), P(108, 30), P(110, -2), P(106, -34), P(92, -52), P(40, -54)]
    c.poly([(px + 3 * s, py + 4 * s) for px, py in palm], dark)
    c.poly(palm, skin)
    c.poly([P(20, -40), P(90, -44), P(100, -20), P(30, -20)], light + (90,))
    for k in range(4):                                # сухожилия к костяшкам
        n0 = -33 + k * 22
        c.line([P(12, n0 * 0.6), P(92, n0)], shade + (110,), 3 * s)
    if old:                                           # морщины и пятна на старой коже
        for k in range(3):
            c.line([P(26 + k * 14, -40), P(30 + k * 14, -10), P(24 + k * 14, 20)], dark + (90,), 1.6 * s)
        c.ell(*P(60, 20), 6 * s, 5 * s, dark + (80,))
        c.ell(*P(44, -24), 4 * s, 4 * s, dark + (70,))
    # четыре пальца: три фаланги, суставы, складки, ногти
    lens = [(30, 24, 18), (34, 27, 20), (33, 26, 19), (26, 20, 15)]
    for k in range(4):
        n0 = -33 + k * 22
        spread = (k - 1.5) * 0.06
        bend = [0.25 * curl, 0.55 * curl, 0.7 * curl]
        cx_, cy_ = P(100 - abs(k - 1.5) * 6, n0)
        ang = a + spread
        w = (9.5 - abs(k - 1.5) * 0.6) * s
        pts = [(cx_, cy_)]
        for j in range(3):
            ang += bend[j]
            L = lens[k][j] * s
            pts.append((pts[-1][0] + math.cos(ang) * L, pts[-1][1] + math.sin(ang) * L))
        for j in range(3):
            r0, r1 = w * (1 - j * 0.1), w * (0.92 - j * 0.1)
            c.poly(_cap((pts[j][0] + 2 * s, pts[j][1] + 3 * s), (pts[j + 1][0] + 2 * s, pts[j + 1][1] + 3 * s), r0, r1), dark)
            c.poly(_cap(pts[j], pts[j + 1], r0, r1), skin if j else shade)
        for j in (1, 2):                              # складки на суставах
            jx, jy = pts[j]
            c.line([(jx - math.sin(ang) * w * 0.6, jy + math.cos(ang) * w * 0.6),
                    (jx + math.sin(ang) * w * 0.6, jy - math.cos(ang) * w * 0.6)], dark + (150,), 1.6 * s)
        c.ell(*pts[1], w * 0.55, w * 0.55, light + (70,))   # блик на костяшке
        tx, ty = pts[3]
        px_, py_ = pts[2]
        mx_, my_ = (tx * 0.62 + px_ * 0.38), (ty * 0.62 + py_ * 0.38)
        c.ell(mx_, my_, w * 0.62, w * 0.72, nail)    # ноготь
        c.ell(mx_ - math.cos(ang) * w * 0.25, my_ - math.sin(ang) * w * 0.25, w * 0.3, w * 0.3, mix(nail, (255, 255, 255), 0.4))
    # большой палец сбоку ладони
    t0 = P(36, 52)
    ta = a + side * 0.0 + math.atan2(ny, nx) - math.atan2(uy, ux)
    ta = math.atan2(uy * 0.55 + ny * 0.85, ux * 0.55 + nx * 0.85)
    t1 = (t0[0] + math.cos(ta) * 38 * s, t0[1] + math.sin(ta) * 38 * s)
    ta2 = ta - side * 0.0 + (math.atan2(uy, ux) - ta) * (0.5 + 0.4 * curl)
    t2 = (t1[0] + math.cos(ta2) * 30 * s, t1[1] + math.sin(ta2) * 30 * s)
    for (p0, p1, r0, r1) in ((t0, t1, 15 * s, 12.5 * s), (t1, t2, 12.5 * s, 10.5 * s)):
        c.poly(_cap((p0[0] + 2 * s, p0[1] + 3 * s), (p1[0] + 2 * s, p1[1] + 3 * s), r0, r1), dark)
        c.poly(_cap(p0, p1, r0, r1), skin)
    c.ell(t2[0] * 0.6 + t1[0] * 0.4, t2[1] * 0.6 + t1[1] * 0.4, 7 * s, 8 * s, nail)
    c.line([(t1[0] - 6 * s, t1[1]), (t1[0] + 6 * s, t1[1])], dark + (140,), 1.6 * s)

"""Аватарки канала для TikTok в стиле сериала (1080×1080, PNG).

Запуск из корня репозитория: python3 avatar/make_avatar.py
"""
import math
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "series2"))
import render as r  # noqa: E402  (персонажи и холст из серии 2)

SIZE = 1080


def canvas(top, bottom):
    img = Image.new("RGB", (SIZE * r.S, SIZE * r.S))
    d = ImageDraw.Draw(img)
    for y in range(SIZE * r.S):
        k = y / (SIZE * r.S)
        d.line([(0, y), (SIZE * r.S, y)], fill=tuple(int(a + (b - a) * k) for a, b in zip(top, bottom)))
    return img, r.C(img)


def glow(c, x, y, rad, col, steps=10, alpha=20):
    for k in range(steps):
        rr = rad * (1 - k / (steps + 1))
        c.ell(x, y, rr, rr, col + (alpha,))


def hazard_ring(c, width=46):
    """Сигнальная жёлто-чёрная полоса по краю круга."""
    for k in range(48):
        a0 = k * 7.5
        col = (240, 196, 30) if k % 2 == 0 else (20, 20, 22)
        c.d.arc([0, 0, SIZE * r.S, SIZE * r.S], a0, a0 + 7.6, fill=col, width=width * r.S)


def zombie_face(c, cx, cy, rad):
    """Крупное лицо зомби в стиле персонажей сериала."""
    u = rad / 17
    c.ell(cx, cy, rad, rad, (150, 182, 128))
    c.ell(cx + rad * 0.35, cy + rad * 0.4, rad * 0.32, rad * 0.22, (126, 160, 106))
    for k in range(6):
        a = math.pi + 0.3 + k * 0.5
        c.ell(cx + math.cos(a) * rad * 0.92, cy + math.sin(a) * rad * 0.92, 4.5 * u, 3.2 * u, (50, 50, 40))
    for s in (-1, 1):
        c.ell(cx + s * 6.5 * u, cy, 3.6 * u, 3.6 * u, (250, 250, 235))
        c.ell(cx + s * 6.5 * u, cy + 0.6 * u, 1.2 * u, 1.2 * u, (200, 20, 20))
        c.line([(cx + s * 3 * u, cy - 6 * u + 2.6 * u), (cx + s * 11 * u, cy - 6 * u - 2 * u)], (35, 28, 30), 2.6 * u)
    c.poly([(cx - 7 * u, cy + 8 * u), (cx - 3.5 * u, cy + 11 * u), (cx, cy + 8.5 * u), (cx + 3.5 * u, cy + 11.5 * u),
            (cx + 7 * u, cy + 8 * u), (cx + 7 * u, cy + 13 * u), (cx - 7 * u, cy + 13 * u)], (40, 10, 10))
    c.line([(cx - 9 * u, cy - 11 * u), (cx - 6 * u, cy - 7 * u), (cx - 8 * u, cy - 3 * u)], (110, 30, 30), 1.3 * u)
    for dx, ln in ((-4.5, 5.5), (5.5, 4)):
        x0, y0 = cx + dx * u, cy + 12.5 * u
        c.line([(x0, y0), (x0 + 0.4 * u, y0 + ln * u)], (170, 16, 20), 1.6 * u)
        c.ell(x0 + 0.4 * u, y0 + ln * u + 0.6 * u, 1.1 * u, 1.5 * u, (170, 16, 20))


def finish(img, name):
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(m).ellipse([0, 0, SIZE, SIZE], fill=255)
    prev = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


def v1():
    """Лицо зомби на красном свечении, сигнальная полоса по кругу."""
    img, c = canvas((120, 26, 30), (30, 10, 14))
    glow(c, 540, 540, 560, (255, 70, 60), alpha=22)
    zombie_face(c, 540, 560, 330)
    hazard_ring(c)
    return finish(img, "avatar_1_zombie")


def v2():
    """Спасатель с жуткой улыбкой — главный злодей сериала."""
    img, c = canvas((24, 28, 40), (8, 8, 12))
    glow(c, 540, 470, 540, (240, 120, 36), alpha=18)
    r.char(c, 540, 2020, 1520, r.RESCUER, expr="creepy", mouth="creepy", arms=(8, 8))
    return finish(img, "avatar_2_rescuer")


def v3():
    """Красные глаза из темноты тоннеля и толпа силуэтов."""
    img, c = canvas((16, 18, 28), (4, 4, 8))
    glow(c, 540, 1080, 700, (226, 46, 46), alpha=16)
    for row, (y, h, n) in enumerate(((860, 520, 6), (1060, 700, 5))):
        for i in range(n):
            x = (i + 0.5) * SIZE / n + (row % 2) * 50 - 25
            r.char(c, x, y + h * 0.95, h, r.zombie_spec(i + row), zombie=True, arms=(165, 165))
    c.rect(0, 0, SIZE, SIZE, (0, 0, 0, 120))
    glow(c, 540, 470, 300, (255, 30, 30), alpha=26)
    for s in (-1, 1):
        c.ell(540 + s * 110, 470, 62, 40, (255, 40, 30))
        c.ell(540 + s * 110, 470, 24, 24, (255, 230, 220))
    return finish(img, "avatar_3_eyes")


def v4():
    """Лицо пополам: Влад — человек и зомби, посередине шов."""
    def face(spec, zombie):
        img, c = canvas((26, 20, 30), (8, 8, 12))
        glow(c, 540, 500, 600, (226, 46, 46) if zombie else (24, 170, 150), alpha=18)
        r.char(c, 540, 1760, 1400, spec, expr="angry", zombie=zombie, arms=(8, 8))
        return img
    human = face(r.VLAD, False)
    zspec = dict(r.VLAD, skin=(150, 182, 128))
    zomb = face(zspec, True)
    m = Image.new("L", human.size, 0)
    pts = [(1080 - 40 * (k % 2), y) for k, y in enumerate(range(0, 2161, 90))]
    ImageDraw.Draw(m).polygon([(0, 0)] + [(x, y) for x, y in pts] + [(0, 2160)], fill=255)
    img = Image.composite(human, zomb, m)
    c = r.C(img)
    for x, y in pts[1:-1]:
        c.line([(x / 2 - 22, y / 2 - 8), (x / 2 + 22, y / 2 + 8)], (30, 20, 22), 7)
    c.line([(x / 2, y / 2) for x, y in pts], (120, 20, 24), 5)
    return finish(img, "avatar_4_split")


def v5():
    """Круглый люк бункера сверху, из щели тянется рука зомби."""
    img, c = canvas((40, 42, 48), (20, 20, 24))
    hazard_ring(c, 70)
    c.ell(540, 540, 455, 455, (60, 64, 72))
    c.ell(540, 540, 430, 430, (14, 4, 6))
    glow(c, 540, 600, 430, (255, 30, 30), alpha=30)
    # приоткрытая крышка
    lx, ly = 500, 470
    c.ell(lx, ly, 420, 420, (96, 102, 112))
    c.ell(lx, ly, 380, 380, (118, 124, 134))
    for k in range(16):
        a = k * math.pi / 8
        c.ell(lx + math.cos(a) * 398, ly + math.sin(a) * 398, 11, 11, (80, 86, 94))
    c.ell(lx, ly, 150, 150, None, outline=(200, 40, 36), w=26)
    for k in range(4):
        a = k * math.pi / 4 + 0.3
        c.line([(lx - math.cos(a) * 150, ly - math.sin(a) * 150), (lx + math.cos(a) * 150, ly + math.sin(a) * 150)],
               (200, 40, 36), 22)
    c.ell(lx, ly, 40, 40, (150, 30, 28))
    # рука зомби из щели
    green, dark = (150, 182, 128), (110, 140, 92)
    c.line([(860, 960), (790, 780)], (120, 90, 110), 120)
    c.ell(780, 760, 78, 68, green)
    for i, (dx, dy, ln) in enumerate(((-70, -20, 90), (-40, -55, 120), (5, -65, 125), (50, -50, 105))):
        x0, y0 = 780 + dx, 760 + dy
        c.line([(x0, y0), (x0 - 20 + i * 8, y0 - ln)], green, 34)
        c.ell(x0 - 20 + i * 8, y0 - ln, 14, 10, dark)
    c.line([(735, 790), (690, 830)], green, 38)
    c.ell(760, 800, 16, 12, (150, 20, 24))
    return finish(img, "avatar_5_hatch")


def v6():
    """Глаз крупным планом: в зрачке отражается толпа зомби."""
    img, c = canvas((60, 8, 12), (10, 4, 6))
    glow(c, 540, 540, 600, (255, 40, 40), alpha=14)
    pts_top = [(140 + k * 40, 540 - math.sin(k / 20 * math.pi) * 300) for k in range(21)]
    pts_bot = [(940 - k * 40, 540 + math.sin(k / 20 * math.pi) * 250) for k in range(21)]
    c.poly(pts_top + pts_bot, (238, 230, 220))
    rng = __import__("random").Random(4)
    for k in range(12):
        side = rng.choice((-1, 1))
        x, y = 540 + side * rng.uniform(300, 380), 540 + rng.uniform(-90, 90)
        pts = [(x, y)]
        for _ in range(5):
            x -= side * rng.uniform(25, 45)
            y += rng.uniform(-25, 25)
            pts.append((x, y))
        c.line(pts, (200, 50, 50), 4)
    c.ell(540, 540, 220, 220, (90, 150, 70))
    c.ell(540, 540, 220, 220, None, outline=(40, 70, 30), w=16)
    for k in range(24):
        a = k * math.pi / 12
        c.line([(540 + math.cos(a) * 110, 540 + math.sin(a) * 110), (540 + math.cos(a) * 200, 540 + math.sin(a) * 200)],
               (70, 120, 52), 6)
    c.ell(540, 540, 120, 120, (6, 6, 8))
    for i in range(5):
        x = 470 + i * 35
        h = 60 + (i % 2) * 12
        c.ell(x, 590 - h, 13, 13, (190, 30, 30))
        c.rect(x - 12, 590 - h + 10, x + 12, 590, (190, 30, 30), r=6)
        c.line([(x - 12, 590 - h + 18), (x - 24, 590 - h - 8)], (190, 30, 30), 7)
        c.line([(x + 12, 590 - h + 18), (x + 24, 590 - h - 8)], (190, 30, 30), 7)
    c.ell(585, 485, 34, 24, (255, 255, 255))
    c.ell(500, 600, 12, 9, (255, 255, 255, 160))
    c.line([pt for pt in pts_top], (30, 10, 12), 26)
    for k in range(3, 18, 2):
        x, y = pts_top[k]
        c.line([(x, y), (x + (k - 10) * 4, y - 60)], (30, 10, 12), 10)
    return finish(img, "avatar_6_eye")


if __name__ == "__main__":
    previews = [v1(), v2(), v3(), v4(), v5(), v6()]
    sheet = Image.new("RGB", (SIZE * 3 + 80, SIZE * 2 + 60), (255, 255, 255))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i % 3 * (SIZE + 20), 20 + i // 3 * (SIZE + 20)))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "preview_circles.jpg"), quality=90)

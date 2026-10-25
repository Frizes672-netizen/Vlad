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


if __name__ == "__main__":
    previews = [v1(), v2(), v3()]
    sheet = Image.new("RGB", (SIZE * 3 + 80, SIZE + 40), (255, 255, 255))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "preview_circles.jpg"), quality=90)

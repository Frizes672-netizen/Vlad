"""Аватарки для YouTube-канала о финансах, крахах компаний и разоблачениях.

Стиль «24 часа президентом»: тёмно-синий, золото, силуэты, свечение.
1080×1080 PNG (YouTube показывает аватар кругом). Запуск из корня репозитория:
    python3 youtube_finance/make_avatar.py
"""
import math
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PRES = os.path.join(HERE, "..", "president")
sys.path.insert(0, PRES)
os.chdir(PRES)  # движок президента читает шрифты по относительному пути
import render as r  # noqa: E402

SIZE = 1080
GOLD, GOLD_L, RED, CREAM, INK = r.GOLD, r.GOLD_L, (214, 48, 52), r.CREAM, r.INK


def canvas(glow_at=(540, 540), glow_col=(70, 55, 30), top=(16, 22, 42), bottom=(5, 7, 14)):
    a = r.grad(top, bottom, w=SIZE * r.S, h=SIZE * r.S)
    a += r.radial(*glow_at, 620, w=SIZE * r.S, h=SIZE * r.S)[:, :, None] * r.np.array(glow_col, r.np.float32)
    img = r.to_img(a)
    return img, r.C(img)


def crash_line(c, pts, w=22):
    c.line(pts, RED + (255,), w)
    x1, y1 = pts[-1]
    x0, y0 = pts[-2]
    a = math.atan2(y1 - y0, x1 - x0)
    tip = (x1 + math.cos(a) * 60, y1 + math.sin(a) * 60)
    left = (x1 + math.cos(a + 2.4) * 50, y1 + math.sin(a + 2.4) * 50)
    right = (x1 + math.cos(a - 2.4) * 50, y1 + math.sin(a - 2.4) * 50)
    c.poly([tip, left, right], RED)


def finish(img, name):
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(m).ellipse([0, 0, SIZE, SIZE], fill=255)
    prev = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


def v1():
    """Взлёт и падение: золотой профиль, сквозь него обрушивается график."""
    img, c = canvas(glow_at=(700, 420), glow_col=(110, 80, 30))
    r.profile(c, 470, 470, 470, GOLD, dx=10, dy=-6)
    r.profile(c, 470, 470, 470, INK)
    pts = [(120, 820), (260, 700), (380, 600), (520, 450), (640, 330), (720, 300), (780, 500), (830, 640), (860, 800)]
    crash_line(c, pts)
    return finish(img, "avatar_1_rise_and_fall")


def v2():
    """Треснувшая золотая монета с долларом."""
    img, c = canvas(glow_col=(90, 70, 30))
    c.ell(552, 556, 360, 360, (0, 0, 0, 120))
    c.ell(540, 540, 360, 360, (196, 152, 76))
    c.ell(540, 540, 330, 330, GOLD)
    for k in range(72):
        a = k * math.pi / 36
        c.line([(540 + math.cos(a) * 338, 540 + math.sin(a) * 338), (540 + math.cos(a) * 354, 540 + math.sin(a) * 354)],
               (160, 120, 56), 4)
    c.ell(540, 540, 290, 290, None, outline=(170, 130, 62), w=6)
    c.text(540, 520, "$", r.F("pf", 440, 900), (120, 86, 36))
    crack = [(470, 180), (520, 330), (480, 430), (580, 560), (530, 680), (600, 800), (570, 900)]
    c.line(crack, (10, 12, 22), 22)
    c.line([(580, 560), (700, 610)], (10, 12, 22), 12)
    c.line([(520, 330), (420, 360)], (10, 12, 22), 10)
    return finish(img, "avatar_2_cracked_coin")


def v3():
    """Небоскрёб корпорации трескается, за ним — обвал графика."""
    img, c = canvas(glow_at=(540, 820), glow_col=(120, 40, 30))
    pts = [(80, 300), (300, 360), (520, 330), (700, 470), (860, 760), (1000, 980)]
    c.line(pts, RED + (150,), 14)
    x0, x1, top = 380, 700, 190
    c.rect(x0, top + 120, x1, 1080, (20, 26, 44))
    c.poly([(x0, top + 120), (x1, top + 120), (620, top), (460, top)], (24, 30, 50))
    c.rect(530, top - 120, 550, top, (24, 30, 50))
    for y in range(top + 160, 1060, 52):
        for x in range(x0 + 26, x1 - 20, 46):
            lit = (x * 3 + y) % 7 != 0
            c.rect(x, y, x + 26, y + 30, GOLD if lit else (40, 46, 64))
    crack = [(560, top), (530, 380), (580, 520), (520, 690), (575, 860), (540, 1080)]
    c.line(crack, (6, 6, 10), 18)
    return finish(img, "avatar_3_falling_tower")


def v4():
    """Разоблачение: лупа над биржевыми свечами, последняя — обвал."""
    img, c = canvas(glow_col=(60, 50, 40))
    for i in range(9):
        x = 150 + i * 95
        up = i < 6
        base = 760 - i * 55 if up else 470 + (i - 6) * 140
        h = 120 if up else 200 + (i - 6) * 60
        col = (70, 190, 120) if up else RED
        c.line([(x, base - h / 2 - 50), (x, base + h / 2 + 50)], col, 6)
        c.rect(x - 26, base - h / 2, x + 26, base + h / 2, col)
    c.ell(460, 470, 250, 250, (255, 255, 255, 26), outline=GOLD, w=34)
    c.line([(640, 650), (860, 870)], GOLD, 70)
    c.line([(680, 690), (850, 860)], (160, 120, 56), 30)
    return finish(img, "avatar_4_expose")


if __name__ == "__main__":
    previews = [v1(), v2(), v3(), v4()]
    sheet = Image.new("RGB", (SIZE * 4 + 100, SIZE + 40), (255, 255, 255))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "preview_circles.jpg"), quality=90)

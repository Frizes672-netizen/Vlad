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


def coin(c, cx, cy, rad, label, size, crack=True):
    c.ell(cx + 12, cy + 16, rad, rad, (0, 0, 0, 120))
    c.ell(cx, cy, rad, rad, (196, 152, 76))
    c.ell(cx, cy, rad * 0.92, rad * 0.92, GOLD)
    for k in range(72):
        a = k * math.pi / 36
        c.line([(cx + math.cos(a) * rad * 0.94, cy + math.sin(a) * rad * 0.94),
                (cx + math.cos(a) * rad * 0.985, cy + math.sin(a) * rad * 0.985)], (160, 120, 56), 4)
    c.ell(cx, cy, rad * 0.8, rad * 0.8, None, outline=(170, 130, 62), w=6)
    c.text(cx, cy - rad * 0.05, label, r.F("pf", size, 900), (120, 86, 36))
    if crack:
        k = rad / 360
        pts = [(cx - 70 * k, cy - 360 * k), (cx - 20 * k, cy - 210 * k), (cx - 60 * k, cy - 110 * k),
               (cx + 40 * k, cy + 20 * k), (cx - 10 * k, cy + 140 * k), (cx + 60 * k, cy + 260 * k), (cx + 30 * k, cy + 360 * k)]
        c.line(pts, (10, 12, 22), 22 * k)
        c.line([pts[3], (cx + 160 * k, cy + 70 * k)], (10, 12, 22), 12 * k)
        c.line([pts[1], (cx - 120 * k, cy - 180 * k)], (10, 12, 22), 10 * k)


def ring_text(img, cx, cy, rad, text, size, col, start=-90, span=300, bottom=False):
    """Надпись по окружности (буквы поворачиваются по касательной)."""
    f = r.F("ms", size * r.S, 800)
    n = len(text)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        a = math.radians(start - span / 2 + span * (i + 0.5) / n)
        x, y = cx + math.cos(a) * rad, cy + math.sin(a) * rad
        tile = Image.new("RGBA", (size * 2 * r.S, size * 2 * r.S), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((size * r.S, size * r.S), ch, font=f, fill=col, anchor="mm")
        rot = -math.degrees(a) + 90 if bottom else -math.degrees(a) - 90
        tile = tile.rotate(rot, Image.BICUBIC)
        img.paste(tile, (int(x * r.S - size * r.S), int(y * r.S - size * r.S)), tile)


def cc_coin():
    """Монограмма CC на треснувшей монете."""
    img, c = canvas(glow_col=(90, 70, 30))
    coin(c, 540, 540, 360, "CC", 300)
    return finish(img, "capital_crimes_avatar_coin")


def cc_emblem():
    """Эмблема: кольцо с названием, монограмма и обвал графика."""
    img, c = canvas(glow_col=(80, 60, 30))
    c.ell(540, 540, 470, 470, None, outline=GOLD, w=10)
    c.ell(540, 540, 330, 330, None, outline=GOLD + (160,), w=4)
    ring_text(img, 540, 540, 400, "CAPITAL CRIMES", 64, GOLD_L, start=-90, span=150)
    ring_text(img, 540, 540, 400, "MONEY · POWER · FRAUD", 34, CREAM, start=90, span=-110, bottom=True)
    c = r.C(img)
    c.text(540, 520, "CC", r.F("pf", 330, 900), GOLD_L)
    crash_line(c, [(250, 720), (380, 620), (480, 660), (600, 500), (690, 560), (780, 760)], w=18)
    return finish(img, "capital_crimes_avatar_emblem")


def banner():
    """Шапка канала 2560×1440; надписи — в безопасной зоне 1546×423 по центру."""
    BW, BH = 2560, 1440
    a = r.grad((14, 20, 40), (4, 6, 12), w=BW, h=BH)
    a += r.radial(640, 720, 900, w=BW, h=BH)[:, :, None] * r.np.array([70, 55, 25], r.np.float32)
    img = r.to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    rng = r.np.random.default_rng(3)
    x = 0
    while x < BW:
        w = rng.uniform(60, 150)
        h = rng.uniform(250, 620)
        d.rectangle([x, BH - h, x + w, BH], fill=(18, 24, 42))
        for wy in range(int(BH - h + 20), BH - 10, 34):
            for wx in range(int(x + 10), int(x + w - 10), 26):
                if rng.random() < 0.35:
                    d.rectangle([wx, wy, wx + 12, wy + 16], fill=(200, 160, 90, 150))
        x += w + 8
    pts = [(0, 1000), (300, 940), (600, 980), (900, 930), (1150, 960), (1400, 935), (1600, 960), (1750, 940),
           (1900, 1020), (2100, 1120), (2300, 1300), (2560, 1400)]
    d.line(pts, fill=(214, 48, 52, 170), width=10, joint="curve")
    d.text((BW / 2, 650), "CAPITAL CRIMES", font=r.PF(170, 900), fill=GOLD_L, anchor="mm")
    d.line([(BW / 2 - 420, 765), (BW / 2 + 420, 765)], fill=GOLD, width=4)
    d.text((BW / 2, 830), "BILLION-DOLLAR COLLAPSES  ·  FRAUDS  ·  FORTUNES", font=r.MS(44, 600), fill=CREAM, anchor="mm")
    img.save(os.path.join(HERE, "capital_crimes_banner.png"))
    prev = img.resize((BW // 2, BH // 2), Image.LANCZOS)
    pd = ImageDraw.Draw(prev)
    sx, sy = (BW - 1546) / 4, (BH - 423) / 4
    pd.rectangle([sx, sy, sx + 1546 / 2, sy + 423 / 2], outline=(255, 255, 255), width=2)
    prev.save(os.path.join(HERE, "banner_preview_safe_area.jpg"), quality=88)


if __name__ == "__main__":
    previews = [v1(), v2(), v3(), v4()]
    sheet = Image.new("RGB", (SIZE * 4 + 100, SIZE + 40), (255, 255, 255))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "preview_circles.jpg"), quality=90)
    brand = [cc_coin(), cc_emblem()]
    sheet = Image.new("RGB", (SIZE * 2 + 60, SIZE + 40), (255, 255, 255))
    for i, p in enumerate(brand):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "capital_crimes_preview.jpg"), quality=90)
    banner()

"""Проработанные объекты и эффекты для сцен: машина полиции, мокрый асфальт с отражениями,
дождь в свете фонаря, советская кухня, прихожая, панельный двор, блум."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine import MS, S, WH, WW, C, capsule, clamp, fbm, glow, grad, lerp, mix, radial, to_img
from timeline import H, W


# ================================================================ эффекты
def reflect(img, ground_y, depth=600, strength=0.45, blur=6, ripple=0.0, t=0.0):
    """Мокрый асфальт: отражение всего, что выше линии земли, с размытием и рябью."""
    gy = int(ground_y * S)
    d = min(int(depth * S), gy, img.height - gy)
    if d <= 4:
        return img
    src = img.crop((0, gy - d, img.width, gy)).transpose(Image.FLIP_TOP_BOTTOM)
    src = src.filter(ImageFilter.GaussianBlur(blur * S))
    a = np.asarray(src, np.float32)
    if ripple:
        rows = np.arange(a.shape[0])
        shift = (np.sin(rows / (9 * S) + t * 3) * ripple * S).astype(int)
        for k in range(0, a.shape[0], 2):
            a[k:k + 2] = np.roll(a[k:k + 2], shift[k], axis=1)
    fadek = np.linspace(1, 0, d)[:, None, None] ** 1.4
    base = np.asarray(img.crop((0, gy, img.width, gy + d)), np.float32)
    puddles = fbm(img.width // 4, d // 4 + 1, [4, 12], 9)
    puddles = np.asarray(Image.fromarray((puddles * 255).astype(np.uint8)).resize((img.width, d)), np.float32)[:, :, None] / 255
    wet = np.clip((puddles - 0.35) * 2.5, 0.25, 1)
    out = base + a * fadek * strength * wet
    img.paste(to_img(out), (0, gy))
    return img


def rain(c, t, n=160, area=(0, 0, W, H), seed=3, col=(200, 210, 230), alpha=90, length=46, slant=8, speed=1500):
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = area
    for i in range(n):
        x = r.uniform(x0, x1)
        y = (r.uniform(y0, y1) + t * speed * r.uniform(0.85, 1.15)) % (y1 - y0) + y0
        L = length * r.uniform(0.6, 1.2)
        c.line([(x, y), (x - slant, y - L)], col + (int(alpha * r.uniform(0.4, 1)),), 1.6)


def rain_in_cone(img, t, cone, n=220, seed=5, col=(255, 226, 170)):
    """Капли видны только в конусе света фонаря (маска полигона)."""
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    xs = [p[0] for p in cone]
    ys = [p[1] for p in cone]
    rain(C(lay), t, n, (min(xs), min(ys), max(xs), max(ys)), seed, col, 170, 40, 6, 1300)
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).polygon([(x * S, y * S) for x, y in cone], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(30 * S))
    al = Image.fromarray((np.asarray(lay.split()[3], np.float32) * np.asarray(m, np.float32) / 255).astype(np.uint8))
    lay.putalpha(al)
    img.paste(lay, (0, 0), lay)


def bloom(frame, thr=200, strength=0.55, radius=28):
    """Свечение ярких мест (фонари, окна, мигалки) — как у объектива."""
    a = np.asarray(frame, np.float32)
    small = frame.resize((frame.width // 4, frame.height // 4), Image.BILINEAR)
    s = np.asarray(small, np.float32)
    lum = s.max(axis=2, keepdims=True)
    hi = np.clip((lum - thr) / (255 - thr), 0, 1) * s
    hi = to_img(hi).filter(ImageFilter.GaussianBlur(radius / 4)).resize(frame.size, Image.BILINEAR)
    return to_img(a + np.asarray(hi, np.float32) * strength)


def volumetric(img, poly, col=(90, 70, 40), blur=50, strength=1.0, k=0.25):
    return glow(img, lambda d, kk: d.polygon([(x * kk, y * kk) for x, y in poly], fill=col), blur, strength)


# ================================================================ машина полиции
def police_car(img, x, ground, s=1.0, kb=1.0, kr=0.0, facing=1, label="ПОЛИЦИЯ"):
    """Проработанный седан сбоку: кузов со светотенью, стёкла с бликами, двери, ручки, колёса с дисками,
    хромированные бамперы, фары, полоса с надписью, мигалка (kb — синий, kr — красный)."""
    c = C(img)
    f = facing
    L = 640 * s

    def X(u):
        return x + f * u * s

    # тень под машиной
    for k in range(5):
        c.ell(x, ground + 4, L * (0.56 - k * 0.04), 26 * s * (1 - k * 0.12), (0, 0, 0, 50))
    body_top, sill = ground - 210 * s, ground - 60 * s
    shape = [(X(-320), ground - 70 * s), (X(-326), ground - 130 * s), (X(-300), ground - 150 * s), (X(-170), ground - 162 * s),
             (X(-120), ground - 250 * s), (X(150), ground - 252 * s), (X(205), ground - 166 * s),
             (X(305), ground - 152 * s), (X(326), ground - 120 * s), (X(322), ground - 70 * s)]
    c.poly(shape, (176, 182, 188))
    # светотень кузова: верх светлее, низ темнее
    c.poly([(X(-326), ground - 112 * s), (X(326), ground - 112 * s), (X(322), ground - 70 * s), (X(-320), ground - 70 * s)],
           (150, 156, 162))
    c.poly([(X(-300), ground - 150 * s), (X(-170), ground - 162 * s), (X(205), ground - 166 * s), (X(305), ground - 152 * s),
            (X(305), ground - 146 * s), (X(-300), ground - 144 * s)], (214, 218, 222))
    # полоса и надпись
    c.poly([(X(-322), ground - 128 * s), (X(324), ground - 124 * s), (X(324), ground - 100 * s), (X(-324), ground - 104 * s)],
           (30, 70, 160))
    c.poly([(X(-322), ground - 132 * s), (X(324), ground - 128 * s), (X(324), ground - 124 * s), (X(-322), ground - 128 * s)],
           (230, 232, 236))
    c.text(X(10), ground - 114 * s, label, MS(int(34 * s), 800), (245, 246, 250))
    # окна
    win = [(X(-150), ground - 168 * s), (X(-108), ground - 240 * s), (X(18), ground - 242 * s), (X(18), ground - 168 * s)]
    win2 = [(X(30), ground - 168 * s), (X(30), ground - 242 * s), (X(140), ground - 240 * s), (X(186), ground - 168 * s)]
    for w_ in (win, win2):
        c.poly(w_, (26, 32, 42))
        a0, a1 = w_[0], w_[2]
        c.poly([(lerp(a0[0], a1[0], 0.25), a0[1]), (lerp(a0[0], a1[0], 0.45), a1[1]), (lerp(a0[0], a1[0], 0.58), a1[1]),
                (lerp(a0[0], a1[0], 0.38), a0[1])], (70, 84, 104, 160))
    c.rect(min(X(18), X(30)), ground - 244 * s, max(X(18), X(30)), ground - 164 * s, (40, 42, 46))
    # двери, ручки, зеркало
    for u in (-160, 24, 196):
        c.line([(X(u), ground - 164 * s), (X(u - 4), ground - 66 * s)], (110, 114, 120), 2.2 * s)
    for u in (-30, 150):
        c.rect(X(u) - 14 * s, ground - 150 * s, X(u) + 14 * s, ground - 143 * s, (90, 94, 100), r=3 * s)
    c.poly([(X(-120), ground - 196 * s), (X(-150), ground - 196 * s), (X(-150), ground - 178 * s), (X(-118), ground - 182 * s)],
           (60, 64, 70))
    # бамперы, фары, фонари
    for u0, u1 in ((-330, -296), (296, 330)):
        c.rect(min(X(u0), X(u1)), ground - 86 * s, max(X(u0), X(u1)), ground - 66 * s, (196, 200, 206), r=6 * s)
        c.line([(min(X(u0), X(u1)), ground - 82 * s), (max(X(u0), X(u1)), ground - 82 * s)], (250, 250, 252), 2 * s)
    c.rect(min(X(312), X(326)), ground - 140 * s, max(X(312), X(326)), ground - 116 * s, (250, 240, 210))
    c.rect(min(X(-326), X(-314)), ground - 140 * s, max(X(-326), X(-314)), ground - 116 * s, (200, 30, 30))
    # колёса
    for u in (-205, 205):
        cx, cy = X(u), ground - 44 * s
        c.ell(cx, cy - 6 * s, 74 * s, 74 * s, (20, 20, 22))            # арка
        c.ell(cx, cy, 58 * s, 58 * s, (16, 16, 18))                    # шина
        c.ell(cx, cy, 52 * s, 52 * s, (30, 30, 32))
        c.ell(cx, cy, 34 * s, 34 * s, (150, 154, 160))                 # диск
        c.ell(cx, cy, 30 * s, 30 * s, (110, 114, 120))
        for k in range(6):
            a = k * math.pi / 3
            c.line([(cx + math.cos(a) * 10 * s, cy + math.sin(a) * 10 * s), (cx + math.cos(a) * 28 * s, cy + math.sin(a) * 28 * s)],
                   (170, 174, 180), 4 * s)
        c.ell(cx, cy, 10 * s, 10 * s, (190, 194, 200))
        c.arc(cx, cy, 33 * s, 33 * s, 200, 290, (230, 232, 236), 2 * s)
    # мигалка на крыше
    c.rect(X(-40), ground - 268 * s, X(70), ground - 252 * s, (40, 40, 44), r=4 * s)
    c.rect(min(X(-38), X(14)), ground - 290 * s, max(X(-38), X(14)), ground - 266 * s,
           mix((30, 40, 90), (90, 150, 255), kb), r=6 * s)
    c.rect(min(X(16), X(68)), ground - 290 * s, max(X(16), X(68)), ground - 266 * s,
           mix((80, 20, 20), (255, 60, 60), kr), r=6 * s)
    # свет: фары и мигалки
    hx = X(326)
    hl = [(hx, ground - 140 * s), (hx, ground - 116 * s), (hx + f * 700 * s, ground + 40 * s), (hx + f * 700 * s, ground - 260 * s)]

    def lights(d, k):
        d.polygon([(px * k, py * k) for px, py in hl], fill=(70, 66, 50))
        d.ellipse([(X(-12) - 160 * s) * k, (ground - 360 * s) * k, (X(-12) + 160 * s) * k, (ground - 200 * s) * k],
                  fill=(int(40 * kb), int(80 * kb), int(255 * kb)))
        d.ellipse([(X(42) - 160 * s) * k, (ground - 360 * s) * k, (X(42) + 160 * s) * k, (ground - 200 * s) * k],
                  fill=(int(255 * kr), int(40 * kr), int(40 * kr)))
        d.ellipse([(x - 900 * s) * k, (ground - 200 * s) * k, (x + 900 * s) * k, (ground + 200 * s) * k],
                  fill=(int(10 * kb + 30 * kr), int(16 * kb), int(60 * kb + 10 * kr)))
    out = glow(img, lights, 40)
    img.paste(out)


def cap_officer(parts, kind_cap=True):
    """Добавляет к полицейскому фуражку (по голове из частей)."""
    return parts


def peaked_cap(c, hx, hy, h, f=1, col=(14, 16, 22), rim=None):
    r = 0.054 * h
    pts = [(hx - f * r * 1.05, hy - r * 0.55), (hx - f * r * 1.15, hy - r * 1.25), (hx + f * r * 0.5, hy - r * 1.4),
           (hx + f * r * 1.1, hy - r * 1.05), (hx + f * r * 0.95, hy - r * 0.6)]
    visor = [(hx + f * r * 0.7, hy - r * 0.62), (hx + f * r * 1.7, hy - r * 0.4), (hx + f * r * 0.8, hy - r * 0.42)]
    if rim:
        c.poly([(x + 3, y - 2) for x, y in pts], rim)
    c.poly(pts, col)
    c.poly(visor, col)
    c.rect(min(hx - f * r * 1.05, hx + f * r * 0.95), hy - r * 0.78, max(hx - f * r * 1.05, hx + f * r * 0.95), hy - r * 0.6,
           (30, 40, 90))


# ================================================================ панельный двор ночью
@lru_cache(None)
def courtyard():
    img = to_img(grad((10, 12, 22), (4, 5, 9)))
    c = C(img)
    c.rect(-20, 300, 1100, 1330, (34, 36, 42))
    n = fbm(WW, WH, [6, 24, 80], 23)
    a = np.asarray(img, np.float32)
    mask = np.zeros((WH, WW), np.float32)
    mask[300 * S:1330 * S, :] = 1
    a = a * (1 - mask[:, :, None] * 0.25 * n[:, :, None])
    img = to_img(a)
    c = C(img)
    for j in range(6):                                 # швы панелей и этажи
        y = 330 + j * 170
        c.line([(0, y), (W, y)], (26, 28, 34), 3)
    for i in range(11):
        c.line([(i * 104, 300), (i * 104, 1330)], (28, 30, 36), 2)
    r = np.random.default_rng(85)
    for i in range(10):
        for j in range(6):
            x, y = 34 + i * 104, 360 + j * 170
            lit = r.random() < 0.2
            col = (226, 176, 104) if lit else (14, 16, 24)
            c.rect(x - 4, y - 4, x + 60, y + 92, (44, 46, 52))
            c.rect(x, y, x + 56, y + 88, col)
            c.rect(x + 26, y, x + 30, y + 88, (44, 46, 52))
            c.rect(x, y + 34, x + 56, y + 38, (44, 46, 52))
            if lit:
                c.rect(x, y, x + 18, y + 88, (200, 150, 110))      # штора
            if j > 0 and i % 3 == 1:                               # балконы
                c.rect(x - 22, y + 96, x + 82, y + 150, (40, 42, 48))
                for k in range(6):
                    c.line([(x - 18 + k * 18, y + 100), (x - 18 + k * 18, y + 150)], (28, 30, 36), 2)
    # подъезд с козырьком и лампочкой
    c.rect(470, 1150, 590, 1330, (18, 16, 16))
    c.rect(476, 1156, 584, 1330, (40, 30, 26))
    c.rect(450, 1130, 610, 1150, (50, 52, 58))
    c.ell(530, 1162, 7, 7, (255, 230, 170))
    # скамейка и голое дерево
    c.rect(170, 1290, 330, 1300, (36, 30, 26))
    c.rect(176, 1300, 184, 1340, (20, 18, 18))
    c.rect(316, 1300, 324, 1340, (20, 18, 18))
    tree(c, 900, 1340, 1.0)
    c.rect(-20, 1330, 1100, H, (16, 16, 18))
    img = glow(img, lambda d, k: d.ellipse([440 * k, 1100 * k, 620 * k, 1360 * k], fill=(120, 90, 50)), 40)
    return img


def tree(c, x, ground, s=1.0, col=(8, 9, 12)):
    c.poly([(x - 14 * s, ground), (x - 8 * s, ground - 420 * s), (x + 8 * s, ground - 420 * s), (x + 16 * s, ground)], col)
    rng = np.random.default_rng(int(x))

    def branch(x0, y0, ang, L, w, depth):
        x1, y1 = x0 + math.cos(ang) * L, y0 + math.sin(ang) * L
        c.line([(x0, y0), (x1, y1)], col, w)
        if depth > 0:
            for da in (-0.45, 0.35):
                branch(x1, y1, ang + da + rng.uniform(-0.15, 0.15), L * 0.7, max(1.5, w * 0.6), depth - 1)
    for k in range(5):
        branch(x, ground - (200 + k * 50) * s, -math.pi / 2 + (k - 2) * 0.45, 150 * s, 8 * s, 3)


# ================================================================ кухня
@lru_cache(None)
def kitchen_room(floor_y=1560):
    img = to_img(grad((44, 42, 46), (20, 19, 22)))
    a = np.asarray(img, np.float32)
    # обои: полоса с мелким узором
    xs = np.arange(WW)[None, :]
    ys = np.arange(WH)[:, None]
    stripe = (np.sin(xs / (30 * S) * math.pi) > 0.6).astype(np.float32) * 0.06
    motif = ((np.sin(xs / (12 * S)) * np.sin(ys / (14 * S))) > 0.85).astype(np.float32) * 0.05
    a = a * (1 + stripe[:, :, None] - motif[:, :, None])
    n = fbm(WW, WH, [5, 20, 70], 8)
    a = a * (0.9 + 0.2 * n[:, :, None])
    img = to_img(a)
    c = C(img)
    # фартук из плитки
    c.rect(0, 1060, W, floor_y, (48, 52, 58))
    for i in range(0, W, 60):
        c.line([(i, 1060), (i, floor_y)], (38, 42, 48), 2)
    for j in range(1060, floor_y, 60):
        c.line([(0, j), (W, j)], (38, 42, 48), 2)
    # навесные шкафчики слева
    for i in range(3):
        x = 10 + i * 150
        c.rect(x, 640, x + 140, 900, (70, 54, 42), r=4)
        c.rect(x + 12, 652, x + 128, 888, (82, 64, 50), r=3)
        c.rect(x + 112, 760, x + 118, 800, (170, 160, 140))
    # чайник, банки на полке
    c.rect(0, 905, 470, 918, (60, 46, 36))
    for k, (x, h) in enumerate(((60, 70), (110, 90), (160, 60))):
        c.rect(x - 18, 905 - h, x + 18, 905, (110, 120, 110, 200), r=6)
    # холодильник справа у стены
    c.rect(1000, 700, 1100, floor_y, (196, 194, 188), r=24)
    c.rect(1000, 1000, 1100, 1008, (150, 148, 144))
    c.rect(1012, 760, 1020, 900, (150, 148, 144))
    # линолеум
    c.rect(0, floor_y, W, H, (46, 38, 34))
    for k in range(14):
        c.line([(0, floor_y + k * 28), (W, floor_y + k * 28 + 6)], (40, 32, 28), 2)
    for i in range(-10, 30):
        c.line([(i * 80, floor_y), (i * 80 - 300, H)], (40, 32, 28), 1.5)
    # часы
    c.ell(140, 520, 74, 74, (30, 28, 30))
    c.ell(140, 520, 64, 64, (214, 206, 188))
    for k in range(12):
        a_ = k * math.pi / 6
        c.line([(140 + math.cos(a_) * 52, 520 + math.sin(a_) * 52), (140 + math.cos(a_) * 60, 520 + math.sin(a_) * 60)],
               (40, 36, 34), 3)
    return img


def curtains(c, x0, y0, x1, y1, t=0.0):
    """Тюль с мягкими складками, слегка колышется."""
    for i in range(14):
        x = lerp(x0, x1, i / 13)
        sway = math.sin(t * 0.6 + i) * 3
        c.poly([(x - 14, y0), (x + 14, y0), (x + 16 + sway, y1), (x - 12 + sway, y1)], (200, 200, 210, 22))
    for s_, xa, xb in ((-1, x0 - 40, x0 + 60), (1, x1 - 60, x1 + 40)):
        c.poly([(xa, y0 - 20), (xb, y0 - 20), (xb + s_ * 10, y1 + 30), (xa - s_ * 10, y1 + 30)], (70, 40, 36))
        for k in range(4):
            xx = lerp(xa, xb, (k + 0.5) / 4)
            c.line([(xx, y0 - 20), (xx + s_ * 6, y1 + 30)], (52, 30, 28), 4)
    c.rect(x0 - 70, y0 - 34, x1 + 70, y0 - 22, (90, 80, 70))


def abazhur(img, x, y, k=1.0):
    """Абажур с бахромой над столом и объёмный конус света."""
    c = C(img)
    c.line([(x, 0), (x, y - 70)], (14, 12, 12), 4)
    c.poly([(x - 60, y - 70), (x + 60, y - 70), (x + 130, y), (x - 130, y)], (150, 70, 40))
    c.poly([(x - 60, y - 70), (x + 60, y - 70), (x + 70, y - 60), (x - 70, y - 60)], (190, 110, 60))
    for i in range(27):
        xx = x - 130 + i * 10
        c.line([(xx, y), (xx + 2, y + 22)], (120, 60, 34), 2)
    out = volumetric(img, [(x - 120, y + 10), (x + 120, y + 10), (x + 330, 1260), (x - 330, 1260)], (70, 50, 26), 46, k)
    out = glow(out, lambda d, kk: d.ellipse([(x - 140) * kk, (y - 60) * kk, (x + 140) * kk, (y + 40) * kk], fill=(200, 130, 60)),
               30, k)
    img.paste(out)


def table_set(c, x0, x1, top):
    """Стол с клеёнкой, чашки и заварочный чайник (вид сбоку чуть сверху)."""
    c.poly([(x0 + 30, top - 18), (x1 - 30, top - 18), (x1, top + 6), (x0, top + 6)], (150, 46, 40))
    for i in range(int(x0) + 40, int(x1) - 30, 34):
        c.line([(i, top - 16), (i - 10, top + 4)], (180, 170, 150, 90), 3)
    c.poly([(x0, top + 6), (x1, top + 6), (x1 + 6, top + 60), (x0 - 6, top + 60)], (130, 38, 34))
    c.rect(x0 + 20, top + 60, x0 + 38, 1560, (34, 24, 20))
    c.rect(x1 - 38, top + 60, x1 - 20, 1560, (34, 24, 20))
    # чашки
    for cx in (x0 + 90, x1 - 110):
        c.poly([(cx - 22, top - 52), (cx + 22, top - 52), (cx + 17, top - 12), (cx - 17, top - 12)], (220, 214, 200))
        c.arc(cx + 24, top - 34, 12, 12, 270, 90, (220, 214, 200), 5)
        c.ell(cx, top - 10, 30, 6, (200, 194, 180))
    # заварник
    tx = (x0 + x1) / 2 + 40
    c.ell(tx, top - 50, 44, 38, (230, 226, 214))
    c.poly([(tx + 40, top - 56), (tx + 80, top - 86), (tx + 86, top - 80), (tx + 44, top - 40)], (230, 226, 214))
    c.ell(tx, top - 92, 16, 8, (210, 206, 196))
    c.ell(tx - 14, top - 52, 6, 12, (180, 50, 50))


# ================================================================ прихожая
@lru_cache(None)
def hallway(floor_y=1560):
    img = to_img(grad((50, 44, 40), (22, 20, 20)))
    a = np.asarray(img, np.float32)
    xs = np.arange(WW)[None, :]
    a = a * (1 + 0.05 * (np.sin(xs / (22 * S) * math.pi) > 0.7))[:, :, None]
    n = fbm(WW, WH, [5, 20, 70], 18)
    img = to_img(a * (0.9 + 0.2 * n[:, :, None]))
    c = C(img)
    c.rect(0, floor_y, W, H, (40, 30, 24))
    for i in range(0, W, 70):
        c.line([(i, floor_y), (i - 120, H)], (34, 26, 20), 2)
    c.poly([(80, floor_y + 40), (600, floor_y + 40), (660, H), (20, H)], (100, 36, 34))   # дорожка
    for k in range(6):
        c.line([(60 + k * 10, floor_y + 60 + k * 50), (620 - k * 4, floor_y + 60 + k * 50)], (140, 70, 50), 4)
    # вешалка с пальто
    c.rect(60, 560, 420, 576, (60, 44, 34))
    for k, (x, col) in enumerate(((120, (34, 36, 42)), (250, (58, 32, 32)), (360, (30, 34, 40)))):
        c.ell(x, 584, 8, 8, (150, 140, 120))
        c.poly([(x - 10, 584), (x + 10, 584), (x + 52, 622), (x + 62, 1000), (x - 62, 1000), (x - 52, 622)], col)
        c.line([(x, 600), (x, 1000)], mix(col, (0, 0, 0), 0.35), 3)
        c.poly([(x - 60, 1000), (x + 60, 1000), (x + 58, 1012), (x - 58, 1012)], mix(col, (0, 0, 0), 0.3))
    # зеркало
    c.rect(450, 520, 580, 900, (60, 46, 36), r=10)
    c.rect(462, 532, 568, 888, (80, 90, 100), r=6)
    c.poly([(480, 560), (500, 560), (470, 700), (462, 700)], (140, 150, 160, 120))
    return img


def wardrobe(c, x0, y0, x1, y1, door_k=0.0, box=True):
    c.rect(x0 - 14, y0 - 20, x1 + 14, y0, (70, 52, 40))
    c.rect(x0, y0, x1, y1, (14, 12, 12))
    mid = (x0 + x1) / 2
    c.rect(x0 + 10, 760 - y0 + y0, x1 - 10, 776, (40, 30, 24))
    if box:
        c.rect(x0 + 40, 690, x0 + 240, 760, (160, 126, 92))
        c.rect(x0 + 40, 690, x0 + 240, 706, (130, 100, 74))
        c.text(x0 + 140, 734, "ОБУВЬ", MS(18, 700), (100, 76, 56))
    for k in range(5):
        c.line([(x0 + 260 + k * 18, 790), (x0 + 250 + k * 18, 1100)], (40 + k * 6, 40, 46), 10)
    # правая створка закрыта, левая открывается (сжимается к петле)
    for xa, xb, hinge in ((x0, mid, x0), (mid, x1, x1)):
        if hinge == x0:
            w = (xb - xa) * (1 - door_k)
            pts = [(xa, y0), (xa + w, y0 + 26 * door_k), (xa + w, y1 - 26 * door_k), (xa, y1)]
        else:
            pts = [(xa, y0), (xb, y0), (xb, y1), (xa, y1)]
        c.poly(pts, (84, 62, 46))
        x_a, x_b = pts[0][0], pts[1][0]
        if abs(x_b - x_a) > 30:
            c.poly([(x_a + (x_b - x_a) * 0.1, y0 + 40), (x_a + (x_b - x_a) * 0.9, y0 + 40 + 20 * door_k),
                    (x_a + (x_b - x_a) * 0.9, y1 - 40 - 20 * door_k), (x_a + (x_b - x_a) * 0.1, y1 - 40)], (96, 72, 54))
            hxp = x_a + (x_b - x_a) * (0.86 if hinge == x0 else 0.14)
            c.rect(hxp - 4, (y0 + y1) / 2 - 30, hxp + 4, (y0 + y1) / 2 + 30, (180, 160, 120))


# ================================================================ ночная улица
@lru_cache(None)
def night_street():
    img = to_img(grad((10, 12, 20), (3, 3, 6)))
    c = C(img)
    r = np.random.default_rng(31)
    x = -40
    while x < W + 40:                                   # силуэты домов с редкими окнами
        w = r.uniform(180, 300)
        top = r.uniform(380, 640)
        c.rect(x, top, x + w, 1320, (16, 17, 22))
        for i in range(int(x) + 20, int(x + w) - 30, 50):
            for j in range(int(top) + 30, 1280, 70):
                if r.random() < 0.08:
                    c.rect(i, j, i + 24, j + 34, (200, 156, 96))
        x += w + 10
    c.rect(0, 1320, W, H, (12, 12, 14))
    for i in range(0, W, 140):                          # бордюр
        c.rect(i, 1320, i + 132, 1334, (26, 26, 28))
    lamps = (160, 600, 1040)
    for lx in lamps:
        c.rect(lx - 6, 700, lx + 6, 1330, (8, 8, 10))
        c.poly([(lx - 6, 700), (lx + 60, 690), (lx + 60, 700)], (8, 8, 10))
        c.poly([(lx + 30, 696), (lx + 90, 696), (lx + 80, 710), (lx + 40, 710)], (14, 14, 16))
    img = glow(img, lambda d, k: [d.polygon([((lx + 40) * k, 710 * k), ((lx + 80) * k, 710 * k), ((lx + 260) * k, 1330 * k),
                                             ((lx - 140) * k, 1330 * k)], fill=(110, 84, 46)) for lx in lamps] +
               [d.ellipse([(lx + 30) * k, 690 * k, (lx + 90) * k, 730 * k], fill=(255, 220, 160)) for lx in lamps], 40)
    return img

"""Рендер ролика «24 часа президентом. Часть 3» (финал) — кинематографичный моушн-дизайн
(движок первой части; шрифты берутся из ../president/fonts).

Тёмно-синяя и золотая гамма, силуэты с контровым светом, объёмный свет,
золотая пыль, плёночное зерно, типографика Playfair Display / Montserrat.
Запуск: python3 render.py build/video.mp4 [--preview СЕКУНДЫ ...]
"""
import math
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from timeline import (CLOCK, CLOCK_FROM, CLOCK_UNTIL, DURATION, FADE_OUT, FINAL_LINE, FPS, H, QUOTES,
                      SCENES, VOICE_OFFSET, W, v)

FONT_DIR = "../president/fonts"
S = 2
WW, WH = W * S, H * S

NAVY = (9, 13, 24)
DEEP = (16, 23, 42)
GOLD = (218, 180, 108)
GOLD_L = (246, 218, 160)
CREAM = (238, 230, 214)
RED = (196, 42, 46)
INK = (6, 8, 14)


# ================================================================ шрифты
@lru_cache(None)
def _font(path, size, wght):
    f = ImageFont.truetype(path, int(size))
    f.set_variation_by_axes([wght])
    return f


def PF(size, w=700):
    return _font(FONT_DIR + "/PlayfairDisplay[wght].ttf", size, w)


def PFI(size, w=500):
    return _font(FONT_DIR + "/PlayfairDisplay-Italic[wght].ttf", size, w)


def MS(size, w=600):
    return _font(FONT_DIR + "/Montserrat[wght].ttf", size, w)


# ================================================================ утилиты
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def eout(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(a, b, k)) for a, b in zip(c1, c2))


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@lru_cache(None)
def words():
    out = []
    for line in open("voice/words.tsv", encoding="utf-8"):
        s, e, w = line.rstrip("\n").split("\t")
        out.append((v(float(s)), v(float(e)), w.lower().strip(".,!?«»—…")))
    return out


def word_t(word, after=0.0):
    """Время ролика, когда звучит слово (первое после after, в секундах озвучки)."""
    for s, e, w in words():
        if s >= v(after) and w.startswith(word):
            return s
    return v(after)


class C:
    """Холст в координатах кадра 1080×1920, рисует в мире ×S."""

    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img, "RGBA")

    def rect(self, x0, y0, x1, y1, fill, r=0, outline=None, w=0):
        b = [min(x0, x1) * S, min(y0, y1) * S, max(x0, x1) * S, max(y0, y1) * S]
        if r:
            self.d.rounded_rectangle(b, r * S, fill=fill, outline=outline, width=int(w * S))
        else:
            self.d.rectangle(b, fill=fill, outline=outline, width=int(w * S))

    def ell(self, cx, cy, rx, ry, fill, outline=None, w=0):
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=fill,
                       outline=outline, width=int(w * S))

    def poly(self, pts, fill):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)

    def line(self, pts, fill, w):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=max(1, int(w * S)), joint="curve")

    def arc(self, cx, cy, r, a0, a1, fill, w):
        self.d.arc([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], a0, a1, fill=fill, width=max(1, int(w * S)))

    def text(self, x, y, txt, f, fill, anchor="mm"):
        self.d.text((x * S, y * S), txt, font=_font(f.path, f.size * S, f._w), fill=fill, anchor=anchor)


def F(kind, size, w):
    f = {"pf": PF, "pfi": PFI, "ms": MS}[kind](size, w)
    f._w = w
    return f


def grad(top, bottom, w=WW, h=WH, mid=None):
    ys = np.linspace(0, 1, h)[:, None]
    stops = [(0, top), (1, bottom)] if mid is None else [(0, top), (mid[0], mid[1]), (1, bottom)]
    pos = [p for p, _ in stops]
    a = np.zeros((h, w, 3), np.float32)
    for ch in range(3):
        a[:, :, ch] = np.interp(ys, pos, [c[ch] for _, c in stops])
    return a


def radial(cx, cy, r, w=WW, h=WH):
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx * S) ** 2 + (y - cy * S) ** 2) / (r * S)
    return np.clip(1 - d, 0, 1) ** 2


def light(img, shapes, blur=40, strength=1.0):
    """Добавляет мягкий свет: shapes(draw, k) рисует фигуры в масштабе k."""
    k = 0.25
    lay = Image.new("RGB", (int(W * k), int(H * k)))
    shapes(ImageDraw.Draw(lay), k)
    lay = lay.filter(ImageFilter.GaussianBlur(blur * k)).resize(img.size, Image.BILINEAR)
    if strength != 1.0:
        lay = to_img(np.asarray(lay, np.float32) * strength)
    return ImageChops.add(img, lay)


def dust(c, t, n=50, seed=1, col=GOLD_L, area=(0, 0, W, H), speed=12, size=(1.5, 4)):
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = area
    for i in range(n):
        x = r.uniform(x0, x1) + math.sin(t * 0.4 + i) * 20
        y = (r.uniform(y0, y1) - t * speed * r.uniform(0.5, 1.5)) % (y1 - y0) + y0
        a = int(120 + 120 * math.sin(t * 1.3 + i * 2.1))
        s = r.uniform(*size)
        c.ell(x, y, s, s, col + (max(0, a),))


# ================================================================ камера
CUR_T = [0.0]
SHAKES = []


def shake(t):
    dx = dy = 0.0
    for i, (te, a, dec) in enumerate(SHAKES):
        if t >= te:
            k = a * math.exp(-(t - te) / dec)
            dx += k * math.sin(t * 57 + i)
            dy += k * math.cos(t * 49 + i * 2)
    return dx, dy


def cam(world, zoom=1.0, cx=540, cy=960, rot=0.0):
    dx, dy = shake(CUR_T[0])
    if abs(dx) + abs(dy) > 0.5:
        zoom = max(zoom, 1.0 + (abs(dx) + abs(dy)) * 2.4 / W + 0.02)
    ww, hh = W / zoom, H / zoom
    x0 = clamp(cx - ww / 2 + dx / zoom, 0, W - ww)
    y0 = clamp(cy - hh / 2 + dy / zoom, 0, H - hh)
    if rot:
        world = world.rotate(rot, Image.BICUBIC, center=(cx * S, cy * S))
    return world.resize((W, H), Image.BILINEAR, box=(x0 * S, y0 * S, (x0 + ww) * S, (y0 + hh) * S),
                        reducing_gap=1.5)


# ================================================================ силуэты
RIGHT_SIDE = [  # контур правой половины фигуры (доли роста, от ступней вверх отрицательные)
    (0.028, -0.805), (0.075, -0.79), (0.118, -0.778), (0.145, -0.762), (0.16, -0.735), (0.168, -0.69),
    (0.172, -0.6), (0.168, -0.5), (0.162, -0.43), (0.15, -0.405), (0.132, -0.41), (0.128, -0.5),
    (0.12, -0.56), (0.112, -0.46), (0.098, -0.44), (0.09, -0.2), (0.084, -0.03), (0.098, -0.008),
    (0.096, 0.0), (0.016, 0.0), (0.018, -0.03), (0.012, -0.38), (0.0, -0.405)]
SEATED_SIDE = [(0.028, -0.805), (0.075, -0.79), (0.118, -0.778), (0.145, -0.762), (0.16, -0.735),
               (0.168, -0.69), (0.176, -0.55), (0.18, -0.3), (0.18, 0.05), (0.0, 0.05)]


def outline(x, feet, h, side_pts, grow=0.0):
    right = [(x + (px * h + grow), feet + py * h) for px, py in side_pts]
    left = [(x - (px * h + grow), feet + py * h) for px, py in reversed(side_pts)]
    return right + left


def figure(c, x, feet, h, col=INK, rim=None, rim_dx=5, turn=0.0, earpiece=False, tablet=False,
           folder=False, bow=0.0, seated=False, arms_out=0.0, women=False, glasses=False):
    """Силуэт человека в костюме: плавный контур тела, голова, шея.
    turn: −1..1 — поворот головы (сдвиг носа/затылка)."""
    pts_side = SEATED_SIDE if seated else RIGHT_SIDE

    def body(dx, dy, fc, g=0.0):
        hx = x + dx + turn * h * 0.012
        hy = feet + dy - h * (0.882 - bow * 0.03)
        c.poly(outline(x + dx, feet + dy, h, pts_side, g), fc)
        c.rect(hx - h * 0.03 - g, hy + h * 0.03, hx + h * 0.03 + g, feet + dy - h * 0.79, fc)
        c.ell(hx, hy, h * 0.056 + g, h * 0.07 + g, fc)
        if turn:
            c.poly([(hx + turn * h * 0.05, hy - h * 0.01), (hx + turn * h * 0.072, hy + h * 0.022),
                    (hx + turn * h * 0.045, hy + h * 0.03)], fc)
        if glasses and not g:
            for s in (-1, 1):
                c.ell(hx + s * h * 0.022 + turn * h * 0.015, hy - h * 0.004, h * 0.014, h * 0.008, (230, 220, 200, 200))
        if women:
            c.ell(hx, hy - h * 0.012, h * 0.066 + g, h * 0.07 + g, fc)
            c.poly([(hx - h * 0.065 - g, hy), (hx + h * 0.065 + g, hy), (hx + h * 0.075 + g, hy + h * 0.12),
                    (hx - h * 0.075 - g, hy + h * 0.12)], fc)
        return hx, hy

    if rim:
        body(rim_dx, -3, rim, g=1.5)
    hx, hy = body(0, 0, col)
    if earpiece and rim:
        c.line([(hx + h * 0.05, hy + h * 0.01), (hx + h * 0.06, feet - h * 0.74)], rim + (200,), 1.6)
    if tablet:
        c.rect(x + h * 0.1, feet - h * 0.56, x + h * 0.21, feet - h * 0.46, (40, 60, 90), r=4)
        c.rect(x + h * 0.108, feet - h * 0.552, x + h * 0.202, feet - h * 0.468, (110, 160, 210))
    if folder:
        c.rect(x - h * 0.22, feet - h * 0.6, x - h * 0.1, feet - h * 0.45, (110, 32, 30), r=3)
    return hx, hy


PROFILE = [(-0.35, 0.95), (-0.38, 0.6), (-0.45, 0.2), (-0.42, -0.15), (-0.3, -0.38), (-0.1, -0.5), (0.12, -0.48),
           (0.28, -0.38), (0.36, -0.2), (0.38, -0.05), (0.36, 0.02), (0.42, 0.15), (0.47, 0.24), (0.40, 0.28),
           (0.39, 0.33), (0.41, 0.38), (0.38, 0.41), (0.40, 0.45), (0.36, 0.5), (0.37, 0.58), (0.30, 0.64),
           (0.10, 0.66), (0.12, 0.8), (0.15, 1.0), (0.6, 1.15), (0.85, 1.4), (0.95, 2.0), (-0.95, 2.0),
           (-0.85, 1.4), (-0.6, 1.12)]


def profile(c, x, y, s, col, dx=0, dy=0):
    c.poly([(x + px * s + dx, y + py * s + dy) for px, py in PROFILE], col)


# ================================================================ фоны (кэш)
@lru_cache(None)
def bg_palace():
    a = grad((14, 22, 46), (40, 30, 44), mid=(0.55, (180, 96, 60)))
    a += radial(540, 1060, 700)[:, :, None] * np.array([90, 50, 20], np.float32)
    img = to_img(a)
    c = C(img)
    c.ell(820, 380, 10, 10, (240, 236, 220))  # вечерняя звезда
    base, top = 1300, 860
    c.rect(60, top, 1020, base, (20, 22, 34))
    c.poly([(40, top), (1040, top), (1000, top - 40), (80, top - 40)], (24, 26, 40))
    c.poly([(380, top - 40), (700, top - 40), (540, top - 170)], (26, 28, 42))
    c.ell(540, top - 210, 130, 130, (24, 26, 40))
    c.rect(410, top - 210, 670, top - 40, (24, 26, 40))
    c.rect(530, top - 400, 550, top - 330, (24, 26, 40))
    for i in range(12):
        x = 100 + i * 80
        c.rect(x, top + 30, x + 26, base - 40, (30, 32, 48))
        for j in range(3):
            y = top + 80 + j * 130
            c.rect(x + 34, y, x + 66, y + 80, (255, 200, 120) if (i * 3 + j) % 5 else (120, 90, 60))
    for k in range(8):
        y = base + k * 20
        c.rect(300 - k * 40, y, 780 + k * 40, y + 20, (34 + k * 2, 32 + k * 2, 44 + k * 2))
    c.rect(0, base + 160, W, H, (14, 14, 22))
    img = light(img, lambda d, k: [d.rectangle([(100 + i * 80 + 34) * k, (top + 80) * k, (100 + i * 80 + 66) * k,
                                                 (base - 80) * k], fill=(160, 110, 50)) for i in range(12)], 30)
    return img


@lru_cache(None)
def bg_room(kind):
    """Интерьеры: стены с панелями, окно, люстра."""
    if kind == "bedroom":
        a = grad((24, 26, 38), (12, 12, 18))
    elif kind == "office":
        a = grad((22, 26, 36), (10, 11, 16))
    else:
        a = grad((18, 20, 30), (8, 8, 12))
    img = to_img(a)
    c = C(img)
    for i in range(6):
        x = 40 + i * 180
        c.rect(x, 300, x + 150, 1180, None, outline=(40, 44, 60), w=3)
        c.rect(x + 14, 320, x + 136, 700, None, outline=(34, 38, 52), w=2)
    c.rect(0, 1240, W, H, (16, 14, 18))
    for k in range(16):
        c.line([(0, 1240 + k * 45), (W, 1240 + k * 45 + 20)], (22, 18, 22), 2)
    return img


def window(c, x0, y0, x1, y1, sky=((200, 170, 120), (120, 120, 150)), frame=(10, 10, 16)):
    ys = np.linspace(0, 1, 20)
    for i in range(20):
        col = mix(sky[0], sky[1], ys[i])
        c.rect(x0, y0 + (y1 - y0) * i / 20, x1, y0 + (y1 - y0) * (i + 1) / 20 + 1, col)
    c.rect(x0 - 14, y0 - 14, x1 + 14, y0, frame)
    c.rect(x0 - 14, y1, x1 + 14, y1 + 14, frame)
    c.rect(x0 - 14, y0, x0, y1, frame)
    c.rect(x1, y0, x1 + 14, y1, frame)
    c.rect((x0 + x1) / 2 - 6, y0, (x0 + x1) / 2 + 6, y1, frame)
    for k in range(1, 3):
        y = y0 + (y1 - y0) * k / 3
        c.rect(x0, y - 5, x1, y + 5, frame)


def beams(img, pts_list, col=(140, 110, 60), blur=60, strength=1.0):
    def sh(d, k):
        for pts in pts_list:
            d.polygon([(x * k, y * k) for x, y in pts], fill=col)
    return light(img, sh, blur, strength)


def chandelier(c, x, y, s=1.0, col=(20, 18, 22), lit=True):
    c.line([(x, 0), (x, y - 60 * s)], col, 4)
    c.ell(x, y - 30 * s, 26 * s, 36 * s, col)
    for k in range(-3, 4):
        c.line([(x, y), (x + k * 40 * s, y + 30 * s - abs(k) * 4 * s)], col, 5 * s)
        c.ell(x + k * 40 * s, y + 18 * s - abs(k) * 4 * s, 7 * s, 12 * s, (255, 220, 150) if lit else col)





def clock_text(hours):
    s = max(0, int(hours * 3600))
    return f"{s // 3600:02}:{s // 60 % 60:02}:{s % 60:02}"


def remaining(t):
    pts = [(v(a), b) for a, b in CLOCK]
    base = float(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))
    return base - (t % 1.0) / 3600


@lru_cache(None)
def corridor_bg():
    img = to_img(grad((22, 20, 26), (10, 9, 12)))
    c = C(img)
    vx, vy = 540, 820
    c.poly([(0, 0), (W, 0), (vx + 90, vy - 120), (vx - 90, vy - 120)], (16, 15, 20))
    c.poly([(0, H), (W, H), (vx + 90, vy + 120), (vx - 90, vy + 120)], (26, 18, 18))
    c.poly([(380, H), (700, H), (vx + 20, vy + 120), (vx - 20, vy + 120)], (110, 24, 30))
    for k in range(9):
        f = (k + 1) / 10
        x0 = lerp(0, vx - 90, f)
        y0 = lerp(0, vy - 120, f)
        y1 = lerp(H, vy + 120, f)
        c.line([(x0, y0), (x0, y1)], (40, 34, 40), 3)
        c.line([(W - x0, y0), (W - x0, y1)], (40, 34, 40), 3)
    c.rect(vx - 90, vy - 120, vx + 90, vy + 120, (230, 200, 150))
    for k in range(4):
        f = 0.2 + k * 0.2
        chandelier(c, vx, lerp(260, vy - 130, f), lerp(1.2, 0.3, f))
    img = light(img, lambda d, kk: [d.ellipse([(vx - 30 - 200 * (1 - (0.2 + j * 0.2))) * kk, (lerp(260, vy - 130, 0.2 + j * 0.2) - 20) * kk,
                                                (vx + 30 + 200 * (1 - (0.2 + j * 0.2))) * kk, (lerp(260, vy - 130, 0.2 + j * 0.2) + 60) * kk],
                                               fill=(170, 130, 70)) for j in range(4)] + [d.rectangle([(vx - 90) * kk, (vy - 120) * kk,
                                                                                                      (vx + 90) * kk, (vy + 120) * kk], fill=(200, 170, 120))], 40)
    return img


@lru_cache(None)
def cabinet_bg():
    img = bg_room("cabinet").copy()
    c = C(img)
    window(c, 380, 360, 700, 960, sky=((120, 130, 170), (200, 170, 140)))
    img = beams(img, [[(380, 360), (700, 360), (900, 1500), (180, 1500)]], (90, 80, 70), 60)
    c = C(img)
    c.poly([(470, 1000), (610, 1000), (980, 1920), (100, 1920)], (44, 28, 22))
    c.poly([(470, 1000), (610, 1000), (620, 1015), (460, 1015)], (60, 40, 30))
    return img


@lru_cache(None)
def desk_bg():
    a = grad((48, 30, 22), (24, 15, 12))
    a *= (0.85 + 0.25 * np.sin(np.linspace(0, 40, WW))[None, :, None] * np.sin(np.linspace(0, 7, WH))[:, None, None])
    a += radial(540, 800, 900)[:, :, None] * np.array([60, 40, 20], np.float32)
    img = to_img(a)
    c = C(img)
    c.rect(870, 1500, 900, 1900, (20, 20, 22), r=10)
    c.rect(866, 1480, 904, 1530, GOLD)
    return img


def paper(c, x, y, rot_k, title, sub, lines, stamp_k):
    pw, ph = 760, 1000
    c.rect(x - pw / 2 + 14, y - ph / 2 + 18, x + pw / 2 + 14, y + ph / 2 + 18, (0, 0, 0, 90))
    c.rect(x - pw / 2, y - ph / 2, x + pw / 2, y + ph / 2, (240, 234, 220))
    c.rect(x - pw / 2 + 30, y - ph / 2 + 30, x + pw / 2 - 30, y + ph / 2 - 30, None, outline=(200, 180, 140), w=2)
    c.text(x, y - 360, title, F("pf", 76, 800), (30, 26, 30))
    c.text(x, y - 290, sub, F("pfi", 40, 500), (90, 80, 76))
    c.line([(x - 120, y - 240), (x + 120, y - 240)], (180, 150, 100), 2)
    for i, ln in enumerate(lines):
        c.text(x, y - 120 + i * 80, ln, F("ms", 52, 700), (24, 24, 30))
    for i in range(4):
        c.line([(x - 260, y + 80 + i * 40), (x + (200 if i < 3 else 40), y + 80 + i * 40)], (200, 194, 184), 6)
    c.line([(x + 60, y + 330), (x + 110, y + 300), (x + 150, y + 340), (x + 200, y + 296), (x + 250, y + 330)], (30, 40, 90), 4)
    c.text(x - 150, y + 330, "Президент", F("pfi", 34, 500), (70, 60, 60))
    if stamp_k > 0:
        s = lerp(1.7, 1.0, eout(stamp_k))
        a = int(230 * min(1, stamp_k * 3))
        sx, sy = x + 170, y + 220
        c.ell(sx, sy, 130 * s, 130 * s, None, outline=(180, 30, 40, a), w=10 * s)
        c.ell(sx, sy, 105 * s, 105 * s, None, outline=(180, 30, 40, a), w=3 * s)
        c.text(sx, sy, "ПОДПИСАНО", F("ms", int(30 * s), 800), (180, 30, 40, a))


@lru_cache(None)
def country_dots():
    r = np.random.default_rng(12)
    pts = []
    for _ in range(1600):
        x, y = r.uniform(80, 1000), r.uniform(560, 1260)
        nx, ny = (x - 540) / 470, (y - 900) / 330
        edge = 1 + 0.18 * math.sin(nx * 6) + 0.12 * math.cos(ny * 7)
        if nx * nx + ny * ny < edge * 0.85:
            pts.append((x, y))
    return pts


@lru_cache(None)
def city_bg():
    a = grad((8, 10, 24), (30, 20, 40), mid=(0.6, (20, 18, 40)))
    img = to_img(a)
    c = C(img)
    r = np.random.default_rng(21)
    x = 0
    while x < W:
        w_ = r.uniform(90, 200)
        h_ = r.uniform(400, 1000)
        c.rect(x, 1300 - h_, x + w_, 1400, (14, 14, 26))
        for wy in range(int(1300 - h_ + 40), 1360, 46):
            for wx in range(int(x + 16), int(x + w_ - 16), 34):
                if r.random() < 0.45:
                    c.rect(wx, wy, wx + 16, wy + 24, (230, 190, 120))
        x += w_ + 6
    return img


@lru_cache(None)
def office_bg():
    img = bg_room("office").copy()
    c = C(img)
    c.rect(400, 520, 680, 1240, (14, 12, 16))
    return img


def office_world(t, night=True):
    """Кабинет президента ночью: окно с городом, стол."""
    img = to_img(grad((10, 14, 30), (4, 4, 10)))
    c = C(img)
    for i in range(3):
        x0 = 120 + i * 300
        c.rect(x0, 200, x0 + 260, 1500, (22, 32, 64))
        for wy in range(260, 1460, 60):
            for wx in range(int(x0 + 20), int(x0 + 240), 50):
                if (wx * 7 + wy * 3 + i) % 5:
                    c.rect(wx, wy, wx + 14, wy + 8, (200, 170, 110))
        c.rect(x0 + 260, 200, x0 + 300, 1500, (8, 8, 14))
    c.rect(0, 1500, W, H, (8, 8, 12))
    return img, c


COMPANIES = ["ООО «ВЕКТОР»", "ООО «ГОРИЗОНТ»", "ООО «СТРОЙ-ПЛЮС»", "ООО «АЛЬТАИР»", "ООО «МЕРИДИАН»", "ООО «ПРЕСТИЖ»"]



# ================================================================ кадры
ADVISOR_RIM = (170, 160, 140)
YOU_RIM = (120, 140, 200)


def big_title(d, y, small, large, k, col_large=GOLD_L, size=110):
    if k <= 0:
        return
    if small:
        d.text((540, y), small, font=PFI(54, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
    d.text((540, y + 105), large, font=PF(size, 900), fill=mix(NAVY, col_large, k), anchor="mm")


def sc_night(t, lt, dur):
    img, c = office_world(t)
    figure(c, 540, 1560, 1100, (4, 4, 8), rim=YOU_RIM, rim_dx=4)
    frame = cam(img, lerp(1.12, 1.0, ease(lt / dur)), 540, 1000)

    def txt(im):
        big_title(ImageDraw.Draw(im), 1560, "до полуночи", "НЕСКОЛЬКО ЧАСОВ", eout((lt - 0.4) / 0.6), size=86)
    return frame, txt


def advisor_photo(c, cx, cy, s, circle=0.0, you=False):
    c.rect(cx - 230 * s + 14, cy - 200 * s + 18, cx + 230 * s + 14, cy + 200 * s + 18, (0, 0, 0, 110))
    c.rect(cx - 230 * s, cy - 200 * s, cx + 230 * s, cy + 200 * s, (238, 232, 218))
    c.rect(cx - 196 * s, cy - 170 * s, cx + 196 * s, cy + 110 * s, (120, 110, 96))
    figure(c, cx, cy + 150 * s, 330 * s, (30, 26, 24), seated=True, glasses=not you)
    c.rect(cx - 196 * s, cy + 110 * s, cx + 196 * s, cy + 170 * s, (238, 232, 218))
    if circle > 0:
        c.arc(cx, cy - 60 * s, 170 * s, -90, -90 + 360 * circle, RED, 8)


def sc_photo(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    advisor_photo(c, 540, 900, lerp(1.25, 1.45, ease(lt / dur)), circle=1.0)
    frame = cam(img, 1.0, 540, 960, rot=-2)

    def txt(im):
        big_title(ImageDraw.Draw(im), 1520, "может разрушить", "ВСЮ СТРАНУ", eout((t - word_t("разрушить", 5)) / 0.5),
                  col_large=(240, 90, 90))
    return frame, txt


def sc_order(t, lt, dur):
    img = office_bg().copy()
    c = C(img)
    figure(c, 560, 1820, 1300, INK, rim=(230, 200, 150), earpiece=True)
    c.rect(0, 1600, W, H, (24, 18, 16))
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = eout((t - word_t("через", 9)) / 0.5)
        if k > 0:
            d.rounded_rectangle([380, 380, 700, 470], 45, fill=(6, 8, 14, int(200 * k)), outline=GOLD + (int(200 * k),), width=2)
            d.text((540, 425), "+10 МИНУТ", font=MS(40, 700), fill=GOLD_L + (int(255 * k),), anchor="mm")
        k2 = eout((t - word_t("доклад", 12)) / 0.5)
        if k2 > 0:
            y = lerp(1700, 1380, k2)
            d.rounded_rectangle([110, y, 970, y + 230], 30, fill=(236, 228, 210, 245))
            d.text((150, y + 60), "СРОЧНЫЙ ДОКЛАД", font=MS(34, 800), fill=RED, anchor="lm")
            d.text((150, y + 140), "Советник покинул дворец", font=PFI(50, 500), fill=(30, 26, 30), anchor="lm")
    return frame, txt


def sc_palace(t, lt, dur):
    img = bg_palace().copy()
    img = to_img(np.asarray(img, np.float32) * np.array([0.5, 0.5, 0.65], np.float32))
    c = C(img)
    k = ease(lt / dur)
    x, y, s = lerp(540, 600, k), lerp(1720, 1500, k), lerp(1.0, 0.35, k)
    c.rect(x - 200 * s, y - 70 * s, x + 200 * s, y + 40 * s, (8, 8, 12), r=30 * s)
    c.rect(x - 130 * s, y - 130 * s, x + 130 * s, y - 60 * s, (8, 8, 12), r=30 * s)
    img = light(img, lambda d, kk: [d.ellipse([(x + sx * 150 * s - 40 * s) * kk, (y - 30 * s) * kk,
                                               (x + sx * 150 * s + 40 * s) * kk, (y + 10 * s) * kk], fill=(255, 30, 30))
                                    for sx in (-1, 1)], 20, 1.5)
    c = C(img)
    for sx in (-1, 1):
        c.rect(x + sx * 150 * s - 26 * s, y - 20 * s, x + sx * 150 * s + 26 * s, y, (255, 60, 60))
    frame = cam(img, lerp(1.1, 1.0, k), 540, 1100)

    def txt(im):
        big_title(ImageDraw.Draw(im), 380, "советник", "ПОКИНУЛ ДВОРЕЦ", eout((lt - 0.3) / 0.5), size=96)
    return frame, txt


def sc_docs(t, lt, dur):
    img = to_img(grad((10, 12, 22), (3, 3, 6)))
    c = C(img)
    tx, ty = 540, 560
    c.ell(tx, ty, 120, 120, None, outline=GOLD, w=8)
    c.text(tx, ty, "КАЗНА", F("ms", 34, 800), GOLD_L)
    boxes = [(200 + (i % 2) * 680, 900 + (i // 2) * 300) for i in range(6)]
    for i, (bx, by) in enumerate(boxes):
        k = eout((lt - i * 0.2) / 0.5)
        if k <= 0:
            continue
        c.line([(tx, ty + 120), (bx, by - 70)], GOLD + (int(120 * k),), 4)
        for j in range(3):
            q = ((lt * 0.8 + j / 3 + i * 0.1) % 1)
            c.ell(lerp(tx, bx, q), lerp(ty + 120, by - 70, q), 12, 12, GOLD_L)
        c.rect(bx - 190, by - 70, bx + 190, by + 90, (24, 26, 38), r=16)
        c.text(bx, by - 20, COMPANIES[i], F("ms", 30, 700), CREAM)
        c.text(bx, by + 40, "подставная", F("ms", 26, 600), RED)
    frame = cam(img, lerp(1.0, 1.05, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = eout((t - word_t("миллиарды", 17)) / 0.5)
        if k > 0:
            d.text((540, 1820), "ИСЧЕЗАЛИ МИЛЛИАРДЫ", font=MS(50, 800), fill=mix(NAVY, RED, k), anchor="mm")
    return frame, txt


def sc_burn(t, lt, dur):
    img = to_img(grad((10, 6, 6), (2, 1, 1)))
    c = C(img)
    r = np.random.default_rng(int(t * 12))
    for i in range(7):
        x, y = 200 + i * 110, 1200 + (i % 2) * 80
        c.rect(x - 120, y - 160, x + 120, y + 160, (220, 210, 190))
        burnt = 320 * ease((lt - i * 0.2) / 4)
        if burnt > 1:
            c.rect(x - 120, y + 160 - burnt, x + 120, y + 160, (40, 26, 20))

    def flames(dd, k):
        for i in range(40):
            fx = r.uniform(100, 980)
            fh = r.uniform(150, 520)
            fw = r.uniform(30, 70)
            dd.polygon([((fx - fw) * k, 1500 * k), ((fx + fw) * k, 1500 * k), ((fx + r.uniform(-30, 30)) * k, (1500 - fh) * k)],
                       fill=(255, int(r.uniform(90, 180)), 30))
    img = light(img, flames, 14, 1.0)
    img = light(img, flames, 60, 0.6)
    c = C(img)
    dust(c, t, 60, seed=12, col=(255, 160, 60), area=(0, 600, W, 1600), speed=80)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        big_title(d, 360, "он не просто предал", "ОН УНИЧТОЖАЕТ", eout((t - word_t("хочет", 26)) / 0.5),
                  col_large=(255, 140, 80), size=96)
        if t > word_t("доказательства", 27):
            d.text((540, 580), "ВСЕ ДОКАЗАТЕЛЬСТВА", font=MS(44, 700), fill=CREAM, anchor="mm")
    return frame, txt


def sc_probe(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    c.rect(150, 420, 930, 1560, (238, 232, 218))
    c.text(540, 500, "ГОСУДАРСТВЕННЫЕ ВЫПЛАТЫ", F("ms", 38, 800), (30, 26, 30))
    c.line([(200, 550), (880, 550)], (180, 150, 100), 2)
    for i in range(9):
        y = 620 + i * 90
        c.text(200, y, COMPANIES[i % len(COMPANIES)], F("ms", 30, 600), (60, 56, 60), anchor="lm")
        c.text(880, y, f"{(i * 37 % 90) + 10},{i * 3 % 10} млрд", F("ms", 30, 700), (60, 56, 60), anchor="rm")
    s1 = clamp((t - word_t("расследование", 29) - 0.2) / 0.25)
    if s1 > 0:
        s = lerp(1.6, 1.0, s1)
        c.rect(540 - 330 * s, 760 - 60 * s, 540 + 330 * s, 760 + 60 * s, None, outline=(190, 30, 40), w=8)
        c.text(540, 760, "РАССЛЕДОВАНИЕ", F("ms", int(56 * s), 800), (190, 30, 40))
    s2 = clamp((t - word_t("остановить", 31) - 0.2) / 0.25)
    if s2 > 0:
        s = lerp(1.6, 1.0, s2)
        c.rect(540 - 300 * s, 1200 - 70 * s, 540 + 300 * s, 1200 + 70 * s, None, outline=(190, 30, 40), w=10)
        c.text(540, 1200, "ОСТАНОВЛЕНО", F("ms", int(70 * s), 800), (190, 30, 40))
    return cam(img, lerp(1.0, 1.06, lt / dur), 540, 990, rot=-2), None


def sc_news(t, lt, dur):
    img = to_img(grad((12, 18, 36), (6, 8, 16)))
    c = C(img)
    cx, cy = 560, 920
    wave = lt * 420
    for x, y in country_dots():
        k = clamp((wave - math.hypot(x - cx, y - cy)) / 120)
        c.ell(x, y, 4, 4, mix((40, 50, 80), GOLD_L, k))
    frame = cam(img, 1.0 + lt * 0.02, 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.rectangle([60, 1380, 330, 1450], fill=RED)
        d.text((195, 1416), "СРОЧНО", font=MS(40, 800), fill=(255, 255, 255), anchor="mm")
        d.rectangle([60, 1450, 1020, 1600], fill=(245, 242, 236))
        d.text((90, 1495), "ПРЕЗИДЕНТ ОСТАНОВИЛ", font=MS(46, 800), fill=(20, 20, 30), anchor="lm")
        d.text((90, 1555), "ГОСВЫПЛАТЫ", font=MS(46, 800), fill=(20, 20, 30), anchor="lm")
        d.rectangle([60, 1600, 1020, 1660], fill=(20, 26, 48))
        tick = "  •  ЭКСТРЕННОЕ РАССЛЕДОВАНИЕ  •  СОВЕТНИК ПРЕЗИДЕНТА В РОЗЫСКЕ  •  ПОДСТАВНЫЕ КОМПАНИИ"
        f = MS(30, 600)
        tw = d.textlength(tick, font=f)
        x = 1020 - (lt * 260) % tw
        for k in range(3):
            d.text((x + k * tw, 1630), tick, font=f, fill=CREAM, anchor="lm")
        d.rectangle([0, 1600, 60, 1660], fill=(20, 26, 48))
        d.rectangle([1020, 1600, W, 1660], fill=(0, 0, 0))
    return frame, txt


def sc_streets(t, lt, dur):
    img = city_bg().copy()
    c = C(img)
    t_for = word_t("поддерживают", 38)
    t_against = word_t("требуют", 41)
    r = np.random.default_rng(31)
    for row in range(4):
        y = 1450 + row * 130
        s = 0.8 + row * 0.25
        for i in range(14):
            x = (i + r.uniform(0, 0.8)) * 82 - 20
            bob = abs(math.sin(t * 4 + i + row)) * 14 * s
            c.ell(x, y - bob, 36 * s, 44 * s, INK)
            c.ell(x, y + 80 * s - bob, 70 * s, 60 * s, INK)
            if (i + row) % 4 == 0:
                left = x < 540
                show = t > (t_for if left else t_against)
                if show:
                    px, py = x, y - 200 * s - bob
                    c.line([(x, y - 40 * s - bob), (px, py + 50 * s)], (60, 50, 40), 6 * s)
                    col = (230, 196, 90) if left else (200, 40, 44)
                    c.rect(px - 70 * s, py - 50 * s, px + 70 * s, py + 50 * s, col, r=4)
                    c.text(px, py, "ЗА" if left else "ВОН", F("ms", int(34 * s), 800), (20, 20, 24) if left else (255, 255, 255))
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 1100)

    def txt(im):
        d = ImageDraw.Draw(im)
        k1 = eout((t - t_for) / 0.4)
        k2 = eout((t - t_against) / 0.4)
        if k1 > 0:
            d.text((280, 430), "ПОДДЕРЖКА", font=MS(48, 800), fill=mix(NAVY, GOLD_L, k1), anchor="mm")
        if k2 > 0:
            d.text((800, 430), "В ОТСТАВКУ!", font=MS(48, 800), fill=mix(NAVY, (240, 80, 80), k2), anchor="mm")
    return frame, txt


def phone_frame(img, c):
    c.rect(220, 260, 860, 1700, (14, 14, 18), r=70)
    c.rect(240, 280, 840, 1680, (22, 24, 30), r=56)


def sc_call(t, lt, dur):
    img = to_img(grad((12, 12, 22), (3, 3, 6)) + radial(540, 900, 600)[:, :, None] * np.array([30, 40, 70], np.float32))
    c = C(img)
    phone_frame(img, c)
    ans = word_t("отвечаю", 46)
    buzz = math.sin(t * 80) * 6 if t < ans else 0
    frame = cam(img, lerp(1.0, 1.06, ease(lt / dur)), 540 + buzz, 960)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        d.ellipse([440, 520, 640, 720], fill=(60, 64, 80))
        d.text((540, 620), "?", font=PF(120, 800), fill=CREAM, anchor="mm")
        d.text((540, 800), "Неизвестный номер", font=MS(44, 700), fill=CREAM, anchor="mm")
        if t < ans:
            d.text((540, 860), "входящий вызов…", font=MS(30, 500), fill=(150, 156, 180), anchor="mm")
            pulse = 1 + 0.08 * math.sin(t * 10)
            for x, col in ((360, (220, 50, 50)), (720, (60, 190, 90))):
                rr = 70 * pulse
                d.ellipse([x - rr, 1400 - rr, x + rr, 1400 + rr], fill=col)
        else:
            sec = int(t - ans)
            d.text((540, 860), f"00:{sec:02}", font=MS(34, 600), fill=(150, 156, 180), anchor="mm")
            d.ellipse([470, 1330, 610, 1470], fill=(220, 50, 50))
    return frame, txt


def sc_voice(t, lt, dur):
    img = to_img(grad((8, 8, 12), (2, 2, 4)))
    img = light(img, lambda d, k: d.polygon([(560 * k, 0), (900 * k, 0), (1080 * k, 1920 * k), (520 * k, 1920 * k)], fill=(70, 60, 50)), 60)
    c = C(img)
    figure(c, 640, 1900, 1400, INK, rim=ADVISOR_RIM, rim_dx=-6, turn=-0.5)
    c.rect(560, 640, 610, 760, (20, 20, 24), r=10)
    return cam(img, lerp(1.0, 1.12, ease(lt / dur)), 600, 900), None


def sc_name(t, lt, dur):
    img = to_img(grad((8, 10, 16), (2, 2, 4)))
    c = C(img)
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = eout((lt - 0.2) / 0.5)
        d.text((540, 560), "ОДНО ИМЯ", font=MS(46, 700), fill=GOLD + (int(255 * k),), anchor="mm")
        d.rounded_rectangle([170, 650, 910, 790], 20, fill=(236, 228, 210, int(250 * k)))
        for i in range(4):
            x = 210 + i * 175
            d.rectangle([x, 690, x + 150, 750], fill=(16, 16, 18, int(255 * k)))
        cold = eout((t - word_t("холодеют", 61)) / 0.8)
        if cold > 0:
            d.text((540, 960), "холодеют руки…", font=PFI(56, 500), fill=(150, 190, 240, int(255 * cold)), anchor="mm")
        k3 = eout((t - word_t("назначил", 64)) / 0.5)
        if k3 > 0:
            d.text((540, 1330), "человек, который", font=PFI(56, 500), fill=CREAM + (int(255 * k3),), anchor="mm")
            d.text((540, 1440), "НАЗНАЧИЛ МЕНЯ", font=PF(100, 900), fill=(240, 90, 90, int(255 * k3)), anchor="mm")
    return frame, txt


def sc_folder(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    t_open = word_t("открываю", 72)
    if t < t_open:
        k = eout(lt / 0.6)
        y = lerp(1900, 960, k)
        c.rect(220, y - 420, 860, y + 420, (110, 34, 32), r=10)
        c.text(540, y - 200, "СЕКРЕТНО", F("ms", 70, 800), (240, 220, 200))
        return cam(img, 1.0, 540, 960, rot=2), None
    c.rect(100, 500, 980, 1500, (236, 230, 216))
    k = ease((t - t_open) / 2.0)
    nodes = [(540, 640, "НАЗНАЧИВШИЙ"), (540, 900, "СОВЕТНИК")] + [(200 + i * 170, 1220, "ООО") for i in range(5)]
    for i, (x, y, label) in enumerate(nodes):
        kk = clamp(k * 3 - i * 0.3)
        if kk <= 0:
            continue
        if i == 1:
            c.line([(540, 690), (540, 850)], RED + (int(255 * kk),), 6)
        if i >= 2:
            c.line([(540, 950), (x, 1170)], RED + (int(200 * kk),), 4)
        w = 300 if i < 2 else 70
        c.rect(x - w, y - 45, x + w, y + 45, (30, 30, 36, int(255 * kk)), r=10)
        c.text(x, y, label, F("ms", 40 if i < 2 else 24, 800), (255, 230, 200, int(255 * kk)))
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 960)

    def txt(im):
        big_title(ImageDraw.Draw(im), 1580, "весь кризис —", "ЧАСТЬ СХЕМЫ", eout((t - word_t("схемы", 74)) / 0.5),
                  col_large=(240, 90, 90), size=100)
    return frame, txt


def sc_evidence(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    c.rect(220, 540, 860, 1380, (110, 34, 32), r=10)
    c.text(540, 760, "ДЕЛО № 0001", F("pf", 72, 800), (240, 220, 200))
    sk = clamp((lt - 0.3) / 0.25)
    if sk > 0:
        s = lerp(1.6, 1.0, sk)
        c.rect(540 - 320 * s, 1060 - 70 * s, 540 + 320 * s, 1060 + 70 * s, None, outline=GOLD_L, w=8)
        c.text(540, 1060, "ДОКАЗАТЕЛЬСТВА", F("ms", int(54 * s), 800), GOLD_L)
    return cam(img, 1.0, 540, 960, rot=-2), None


def sc_speech(t, lt, dur):
    img = to_img(grad((14, 16, 30), (4, 4, 8)))
    img = light(img, lambda d, k: d.ellipse([240 * k, 300 * k, 840 * k, 1300 * k], fill=(90, 80, 60)), 90)
    c = C(img)
    for sx in (-1, 1):
        c.rect(540 + sx * 380 - 6, 300, 540 + sx * 380 + 6, 1500, (60, 50, 40))
        c.poly([(540 + sx * 380, 320), (540 + sx * 380 + sx * 200, 360), (540 + sx * 380 + sx * 180, 700), (540 + sx * 380, 660)],
               (40, 50, 90) if sx < 0 else (90, 30, 34))
    figure(c, 540, 1700, 1200, INK, rim=(220, 190, 130), rim_dx=5)
    c.poly([(330, 1250), (750, 1250), (700, 1900), (380, 1900)], (40, 30, 26))
    c.ell(540, 1400, 90, 90, None, outline=GOLD, w=6)
    for mx in (470, 540, 610):
        c.line([(mx, 1250), (540 + (mx - 540) * 0.5, 1150)], (30, 30, 34), 8)
        c.ell(540 + (mx - 540) * 0.5, 1140, 16, 22, (20, 20, 24))
    flash = (int(t * 5) % 7 == 0)
    if flash:
        img = to_img(np.asarray(img, np.float32) * 1.25)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.rectangle([60, 1620, 1020, 1760], fill=(245, 242, 236))
        d.rectangle([60, 1580, 340, 1620], fill=RED)
        d.text((200, 1600), "ПРЯМОЙ ЭФИР", font=MS(28, 800), fill=(255, 255, 255), anchor="mm")
        d.text((90, 1690), "ОБРАЩЕНИЕ К НАЦИИ", font=MS(50, 800), fill=(20, 20, 30), anchor="lm")
        k = eout((t - word_t("признаюсь", 84)) / 0.4)
        if k > 0:
            d.text((540, 400), "«Я допустил ошибки…»", font=PFI(56, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
        k2 = eout((t - word_t("обещаю", 86)) / 0.4)
        if k2 > 0:
            d.text((540, 480), "«…но доведу расследование до конца»", font=PFI(44, 500), fill=mix(NAVY, GOLD_L, k2), anchor="mm")
    return frame, txt


def sc_truth(t, lt, dur):
    img = city_bg().copy()
    img = to_img(np.asarray(img, np.float32) * 0.6)
    c = C(img)
    for i in range(14):
        x = 60 + i * 76
        c.ell(x, 1620, 34, 42, INK)
        c.ell(x, 1720, 66, 58, INK)
    img = light(img, lambda d, k: d.rectangle([180 * k, 700 * k, 900 * k, 1180 * k], fill=(120, 130, 170)), 40)
    c = C(img)
    c.rect(180, 700, 900, 1180, (30, 36, 60), r=14)
    figure(c, 540, 1260, 560, INK, rim=(220, 190, 130))
    frame = cam(img, 1.0, 540, 1000)

    def txt(im):
        big_title(ImageDraw.Draw(im), 360, "впервые за эти сутки", "ПРАВДА", eout((t - word_t("правду", 89)) / 0.5), size=130)
    return frame, txt


def sc_midnight(t, lt, dur):
    img = to_img(grad((14, 6, 8), (2, 2, 4)) + radial(540, 900, 600)[:, :, None] * np.array([70, 10, 10], np.float32))
    c = C(img)
    c.ell(540, 900, 360, 360, None, outline=RED, w=12)
    frame = cam(img, lerp(1.0, 1.05, lt / dur), 540, 920)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.text((540, 870), "00:00:00", font=MS(130, 600), fill=(255, 110, 110), anchor="mm")
        d.text((540, 975), "ПОЛНОЧЬ", font=MS(36, 700), fill=GOLD, anchor="mm")
        k = eout((t - word_t("полномочия", 93)) / 0.5)
        if k > 0:
            d.text((540, 1420), "полномочия", font=PFI(56, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
            d.text((540, 1520), "ЗАКОНЧИЛИСЬ", font=PF(100, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


def sc_jacket(t, lt, dur):
    img = corridor_bg().copy()
    img = to_img(np.asarray(img, np.float32) * np.array([0.6, 0.6, 0.7], np.float32))
    c = C(img)
    k = ease(lt / dur)
    figure(c, 540, lerp(1700, 1250, k), lerp(900, 460, k), INK, rim=YOU_RIM)
    c.rect(120, 1500, 340, 1520, (40, 30, 26))
    c.poly([(150, 1380), (310, 1380), (330, 1500), (130, 1500)], (24, 24, 30))
    return cam(img, lerp(1.0, 1.1, k), 540, 950), None


def sc_newpres(t, lt, dur):
    img = corridor_bg().copy()
    img = to_img(np.asarray(img, np.float32) * 0.7)
    c = C(img)
    figure(c, 760, 1800, 1250, INK, rim=(210, 210, 230), rim_dx=-6, turn=-0.4)
    k = eout((t - word_t("протягивает", 103)) / 0.6)
    if k > 0:
        ex, ey = lerp(780, 600, k), lerp(1240, 1180, k)
        c.rect(ex - 110, ey - 70, ex + 110, ey + 70, (236, 228, 210))
        c.poly([(ex - 110, ey - 70), (ex + 110, ey - 70), (ex, ey + 10)], (220, 210, 190))
    figure(c, 230, 2050, 1400, (4, 4, 8), rim=YOU_RIM, rim_dx=5, turn=0.5)
    return cam(img, 1.05, 520, 1000), None


def sc_envelope(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    t_photo = word_t("фотография", 110)
    t_back = word_t("обратной", 113)
    if t < t_photo:
        k = eout(lt / 0.8)
        c.rect(200, 700, 880, 1180, (236, 228, 210))
        c.poly([(200, 700), (880, 700), (540, lerp(1000, 520, k))], (220, 210, 190))
        return cam(img, 1.0, 540, 940), None
    flip = ease((t - t_back) / 0.5)
    wd = abs(1 - 2 * flip)
    s = 1.5
    cx, cy = 540, 900
    if flip < 0.5:
        c.rect(cx - 230 * s * wd, cy - 200 * s, cx + 230 * s * wd, cy + 200 * s, (238, 232, 218))
        if wd > 0.3:
            c.rect(cx - 196 * s * wd, cy - 170 * s, cx + 196 * s * wd, cy + 110 * s, (110, 116, 140))
            figure(c, cx, cy + 150 * s, 330 * s, (20, 22, 30), seated=True)
            c.rect(cx - 196 * s * wd, cy + 110 * s, cx + 196 * s * wd, cy + 170 * s, (238, 232, 218))
    else:
        c.rect(cx - 230 * s * wd, cy - 200 * s, cx + 230 * s * wd, cy + 200 * s, (240, 236, 226))
        if wd > 0.6:
            c.text(cx, cy - 60, "Ты был не первым.", F("pfi", 54, 500), (40, 40, 70))
            c.text(cx, cy + 40, "И не последним.", F("pfi", 54, 500), (40, 40, 70))
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 920, rot=-2)

    def txt(im):
        if t_photo <= t < t_back:
            k = eout((t - word_t("сам", 112)) / 0.4)
            if k > 0:
                ImageDraw.Draw(im).text((540, 1560), "НА НЕЙ — Я", font=PF(100, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


def sc_gone(t, lt, dur):
    img = corridor_bg().copy()
    img = to_img(np.asarray(img, np.float32) * 0.55)
    c = C(img)
    figure(c, 230, 2050, 1400, (4, 4, 8), rim=YOU_RIM, rim_dx=5, turn=0.5)
    frame = cam(img, lerp(1.05, 1.2, ease(lt / dur)), 560, 900)

    def txt(im):
        k = eout((t - word_t("обоими", 123)) / 0.6)
        if k > 0:
            d = ImageDraw.Draw(im)
            d.text((540, 380), "кто-то управлял", font=PFI(58, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
            d.text((540, 490), "НАМИ ОБОИМИ", font=PF(100, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


def sc_question(t, lt, dur):
    img = to_img(grad((8, 10, 18), (1, 1, 2)) + radial(540, 900, 600)[:, :, None] * np.array([30, 28, 20], np.float32))
    c = C(img)
    dust(c, t, 40, seed=3)
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = eout((t - word_t("кто", 127)) / 0.8)
        if k > 0:
            d.text((540, 880), FINAL_LINE[0], font=PFI(66, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
            d.text((540, 1000), FINAL_LINE[1].upper(), font=PF(72, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


# ================================================================ монтаж
SC = {name: globals()["sc_" + name] for _, name in SCENES}
SHOTS = []
for i, (a, name) in enumerate(SCENES):
    b = SCENES[i + 1][0] if i + 1 < len(SCENES) else DURATION - VOICE_OFFSET
    SHOTS.append((max(0.0, v(a)), v(b) if i + 1 < len(SCENES) else DURATION, name))

SHAKES += [(word_t("разрушить", 5), 12, 0.4), (word_t("расследование", 29) + 0.2, 18, 0.3),
           (word_t("остановить", 31) + 0.2, 22, 0.3), (word_t("холодеют", 61), 10, 0.6),
           (word_t("полночь", 92), 26, 0.6), (word_t("обратной", 113), 8, 0.4)]

QUOTE_T = [(v(a), v(b), who, txt) for a, b, who, txt in QUOTES]


# ================================================================ постобработка и HUD
@lru_cache(None)
def vignette():
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.7)) ** 2 + ((y - H / 2) / (H * 0.7)) ** 2)
    return np.clip(1.1 - d ** 2.2, 0.3, 1).astype(np.float32)[:, :, None]


@lru_cache(None)
def grain_bank():
    r = np.random.default_rng(9)
    return [r.normal(0, 1, (H // 2, W // 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:, :, None] for _ in range(6)]


def post(img, t):
    a = np.asarray(img, np.float32)
    a = a * vignette()
    # мягкий «кинолук»: тени в синеву, света в тёплое
    lum = a.mean(axis=2, keepdims=True) / 255
    a = a + (1 - lum) * np.array([-4, 0, 8], np.float32) + lum * np.array([8, 3, -6], np.float32)
    a = a + grain_bank()[int(t * FPS) % 6] * 6
    if t < 0.15:
        a *= t / 0.15
    return to_img(a)


def wrap(d, text, f, maxw):
    out, cur = [], ""
    for w in text.split():
        test = (cur + " " + w).strip()
        if d.textlength(test, font=f) <= maxw:
            cur = test
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    return out


def draw_quote(im, t):
    for a, b, who, text in QUOTE_T:
        if a - 0.1 <= t < b:
            k = min(eout((t - a + 0.1) / 0.4), ease((b - t) / 0.3))
            d = ImageDraw.Draw(im, "RGBA")
            f = PFI(54, 500)
            lines = wrap(d, text, f, 860)
            y0 = 1430 - len(lines) * 36
            d.rectangle([0, y0 - 130, W, y0 + len(lines) * 72 + 70], fill=(4, 6, 12, int(150 * k)))
            d.text((110, y0 - 60), "“", font=PF(140, 800), fill=GOLD + (int(255 * k),), anchor="lm")
            d.text((190, y0 - 66), who, font=MS(28, 700), fill=GOLD + (int(255 * k),), anchor="lm")
            for i, ln in enumerate(lines):
                d.text((110, y0 + i * 72), ln, font=f, fill=CREAM + (int(255 * k),), anchor="lm")
            return


def draw_clock(im, t, name):
    if t < v(CLOCK_FROM) or t >= v(CLOCK_UNTIL) or name in ("midnight", "call", "name", "question"):
        return
    k = eout((t - v(CLOCK_FROM)) / 0.6)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([340, 200, 740, 300], 50, fill=(6, 8, 14, int(170 * k)), outline=GOLD + (int(160 * k),), width=2)
    d.text((540, 230), "ОСТАЛОСЬ", font=MS(22, 600), fill=GOLD + (int(255 * k),), anchor="mm")
    late = remaining(t) < 0.5
    col = (240, 90, 90) if late else CREAM
    d.text((540, 268), clock_text(remaining(t)), font=MS(44, 600), fill=col + (int(255 * k),), anchor="mm")


def shot_at(t):
    for a, b, name in SHOTS:
        if a <= t < b:
            return name, a, b
    return SHOTS[-1][2], SHOTS[-1][0], SHOTS[-1][1]


def frame(i):
    t = i / FPS
    CUR_T[0] = t
    name, a, b = shot_at(t)
    img, txt = SC[name](t, t - a, b - a)
    img = post(img.convert("RGB"), t)
    if txt:
        txt(img)
    draw_clock(img, t, name)
    draw_quote(img, t)
    fade = ease((t - (DURATION - FADE_OUT)) / (FADE_OUT - 0.3))
    if fade > 0:
        img = to_img(np.asarray(img, np.float32) * (1 - fade))
    return img.tobytes()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/video.mp4"
    if "--preview" in sys.argv:
        for s in sys.argv[sys.argv.index("--preview") + 1:]:
            Image.frombytes("RGB", (W, H), frame(int(float(s) * FPS))).save(f"{out}_{s}.jpg", quality=88)
        return
    total = int(DURATION * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "19", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool() as pool:
        for n, buf in enumerate(pool.imap(frame, range(total), chunksize=6)):
            ff.stdin.write(buf)
            if n % (FPS * 10) == 0:
                print(f"  кадр {n}/{total}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

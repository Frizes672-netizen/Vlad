"""Рендер ролика «24 часа президентом. Часть 2» — кинематографичный моушн-дизайн
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

from timeline import (CLOCK, CLOCK_FROM, DURATION, FPS, H, QUOTES, SCENES, SUBTITLE, TITLE,
                      VOICE_OFFSET, W, v)

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
def office_bg():
    img = bg_room("office").copy()
    c = C(img)
    c.rect(400, 520, 680, 1240, (14, 12, 16))
    return img




def sc_end(t, lt, dur):
    img = Image.new("RGB", (W, H), NAVY)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = eout((lt - 0.3) / 0.8) * (1 - ease((lt - dur + 0.6) / 0.6))
        d.text((540, 880), TITLE, font=PF(150, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
        d.text((540, 1010), SUBTITLE, font=MS(44, 600), fill=mix(NAVY, CREAM, k), anchor="mm")
        d.line([(540 - 160 * k, 1070), (540 + 160 * k, 1070)], fill=mix(NAVY, GOLD, k), width=2)
    return img, txt




# ================================================================ кадры
ADVISOR_RIM = (170, 160, 140)


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


def sc_chair(t, lt, dur):
    img, c = office_world(t)
    c.rect(330, 900, 750, 1700, (6, 6, 10), r=60)
    c.rect(300, 1300, 780, 1500, (6, 6, 10), r=40)
    figure(c, 540, 1700, 1000, (6, 6, 10), rim=(120, 140, 200), rim_dx=6, seated=True)
    c.rect(0, 1640, W, H, (30, 20, 16))
    c.rect(0, 1630, W, 1650, (50, 34, 24))
    frame = cam(img, lerp(1.15, 1.0, ease(lt / dur)), 540, 1080)
    return to_img(np.asarray(frame, np.float32) * min(1, 0.4 + lt)), None


def sc_document(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    k = eout(lt / 1.0)
    paper(c, lerp(540, 540, k), lerp(1500, 940, k), 0, "ВЫПЛАТА ДОЛГА", "Совершенно секретно",
          ["Срок выплаты:", "завтра, 09:00"], 0)
    c.rect(150, 1460, 260, 1880, (16, 16, 18), r=50)
    return cam(img, lerp(1.0, 1.1, ease(lt / dur)), 540, 940, rot=-2), None


def sc_treasury(t, lt, dur):
    img = to_img(grad((10, 12, 22), (3, 3, 6)) + radial(540, 900, 600)[:, :, None] * np.array([40, 30, 20], np.float32))
    c = C(img)
    c.rect(170, 700, 910, 1200, (20, 24, 36), r=30)
    drain = ease((t - word_t("денег", 6)) / 1.6)
    full = lerp(1.0, 0.04, drain)
    c.rect(230, 1060, 850, 1120, (40, 44, 60), r=30)
    c.rect(230, 1060, 230 + 620 * full, 1120, mix(RED, GOLD, full), r=30)
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 930)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.text((540, 790), "БАЛАНС КАЗНЫ", font=MS(38, 700), fill=GOLD, anchor="mm")
        val = int(lerp(842.6, 3.1, drain) * 10) / 10
        d.text((540, 920), f"{val:.1f} млрд".replace(".", ","), font=MS(96, 700), fill=mix(RED, CREAM, full), anchor="mm")
        kk = eout((t - word_t("долг", 6)) / 0.5)
        d.text((540, 1330), "ДОЛГ — ЗАВТРА УТРОМ", font=PF(64, 800), fill=mix(NAVY, GOLD_L, kk), anchor="mm")
        if drain > 0.9:
            d.text((540, 1420), "денег почти не осталось", font=PFI(48, 500), fill=CREAM, anchor="mm")
    return frame, txt


def two_figures(t, adv_bow=0.0, zoom=1.05, cx=540, cy=1000):
    img, c = office_world(t)
    c.rect(0, 1500, W, H, (10, 9, 12))
    figure(c, 760, 1760, 1150, INK, rim=ADVISOR_RIM, rim_dx=-6, glasses=True, bow=adv_bow, turn=-0.4)
    c.rect(80, 1200, 440, 1920, (6, 6, 10), r=40)
    figure(c, 280, 2000, 1250, (6, 6, 10), rim=(120, 140, 200), rim_dx=6, seated=True, turn=0.5)
    return cam(img, zoom, cx, cy)


def sc_ask(t, lt, dur):
    return two_figures(t, zoom=lerp(1.0, 1.08, lt / dur)), None


def sc_answer(t, lt, dur):
    bow = ease((t - v(17.4)) / 0.8)
    if t < v(21.3):
        return two_figures(t, adv_bow=bow, zoom=1.1, cx=640), None
    # портрет прежнего президента и папка «СЕКРЕТНО»
    img = desk_bg().copy()
    c = C(img)
    c.rect(300, 520, 780, 1120, (60, 46, 30))
    c.rect(330, 550, 750, 1090, (30, 30, 40))
    figure(c, 540, 1400, 800, (12, 12, 18), rim=(90, 90, 110), seated=True)
    c.rect(330, 1090, 750, 1120, (60, 46, 30))
    k = eout((t - v(21.3)) / 0.5)
    c.rect(280, lerp(1900, 1250, k), 800, lerp(2400, 1750, k), (110, 34, 32), r=6)
    c.text(540, lerp(2000, 1350, k), "СЕКРЕТНО", F("ms", 60, 800), (240, 200, 190))
    return cam(img, 1.0 + (t - v(21.3)) * 0.03, 540, 1000), None


def sc_danger(t, lt, dur):
    img, c = office_world(t)
    flash = 0 <= (lt % 2.6) < 0.1
    if flash:
        img = to_img(np.asarray(img, np.float32) * 1.8)
        c = C(img)
    figure(c, 540, 1520, 1100, (4, 4, 8), rim=(120, 140, 200), rim_dx=4)
    frame = cam(img, lerp(1.0, 1.15, ease(lt / dur)), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        k1 = eout((t - word_t("опасный", 23)) / 0.5)
        if k1 > 0:
            d.text((540, 430), "в самый опасный момент", font=PFI(54, 500), fill=mix(NAVY, CREAM, k1), anchor="mm")
        k2 = eout((t - word_t("отступать", 23)) / 0.4)
        if k2 > 0:
            d.text((540, 1650), "ОТСТУПАТЬ", font=PF(120, 900), fill=mix(NAVY, GOLD_L, k2), anchor="mm")
            d.text((540, 1770), "УЖЕ ПОЗДНО", font=MS(44, 700), fill=mix(NAVY, CREAM, k2), anchor="mm")
    return frame, txt


def sc_summon(t, lt, dur):
    img = to_img(grad((14, 10, 14), (4, 3, 5)) + radial(540, 860, 600)[:, :, None] * np.array([70, 20, 20], np.float32))
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = eout(lt / 0.5)
        y = lerp(1100, 800, k)
        d.rounded_rectangle([110, y, 970, y + 330], 40, fill=(30, 26, 34, int(240 * k)), outline=RED + (int(255 * k),), width=3)
        on = int(t * 3) % 2 == 0
        d.ellipse([160, y + 50, 200, y + 90], fill=RED if on else (90, 20, 20))
        d.text((230, y + 70), "ВСЕМ МИНИСТРАМ", font=MS(34, 700), fill=CREAM, anchor="lm")
        d.text((540, y + 175), "ЭКСТРЕННОЕ", font=PF(80, 900), fill=GOLD_L, anchor="mm")
        d.text((540, y + 260), "СОВЕЩАНИЕ", font=MS(52, 700), fill=CREAM, anchor="mm")
    return frame, txt


def sc_cabinet(t, lt, dur):
    img = cabinet_bg().copy()
    c = C(img)
    for k in range(5, -1, -1):
        f = k / 6
        y = lerp(1700, 1020, f)
        h = lerp(700, 150, f)
        for side in (-1, 1):
            x = lerp(540 + side * 470, 540 + side * 85, f)
            figure(c, x, y + h * 0.45, h, INK, rim=(110, 110, 130), rim_dx=side * -3, seated=True, turn=-side * 0.6,
                   glasses=(k == 2 and side == 1))
    figure(c, 540, 2300, 1300, (4, 4, 8), seated=True)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = eout((lt - 0.2) / 0.5)
        d.rounded_rectangle([380, 380, 700, 470], 45, fill=(6, 8, 14, int(200 * k)), outline=GOLD + (int(200 * k),), width=2)
        d.text((540, 425), "+10 МИНУТ", font=MS(40, 700), fill=GOLD_L + (int(255 * k),), anchor="mm")
    return frame, txt


def sc_speech(t, lt, dur):
    img = cabinet_bg().copy()
    img = light(img, lambda d, k: d.polygon([(400 * k, 0), (680 * k, 0), (820 * k, 1920 * k), (260 * k, 1920 * k)],
                                            fill=(90, 80, 60)), 60)
    c = C(img)
    rise = eout(lt / 1.2)
    figure(c, 540, lerp(2050, 1780, rise), 1350, (4, 4, 8), rim=(220, 190, 130), rim_dx=6, arms_out=0.3)
    c.poly([(0, 1700), (W, 1700), (W, H), (0, H)], (40, 26, 20))
    return cam(img, lerp(1.0, 1.1, ease(lt / dur)), 540, 1000), None


def card(d, x, y, w, h, title, sub, k, crossed):
    d.rounded_rectangle([x, y, x + w, y + h], 30, fill=(236, 228, 210, int(255 * k)))
    d.text((x + w / 2, y + 80), title, font=MS(46, 800), fill=(30, 26, 30, int(255 * k)), anchor="mm")
    d.text((x + w / 2, y + 150), sub, font=PFI(36, 500), fill=(90, 80, 76, int(255 * k)), anchor="mm")
    if crossed > 0:
        s = lerp(1.5, 1.0, crossed)
        cx, cy = x + w / 2, y + h / 2
        d.line([(cx - 330 * s, cy - 90 * s), (cx + 330 * s, cy + 90 * s)], fill=RED, width=12)
        d.line([(cx - 330 * s, cy + 90 * s), (cx + 330 * s, cy - 90 * s)], fill=RED, width=12)
        d.rounded_rectangle([cx - 200, cy + 110, cx + 200, cy + 180], 10, outline=RED, width=5)
        d.text((cx, cy + 145), "ОТКЛОНЕНО", font=MS(40, 800), fill=RED, anchor="mm")


def sc_proposals(t, lt, dur):
    img = to_img(grad((14, 16, 26), (5, 5, 9)))
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        d.text((540, 440), "Министры предлагают", font=PFI(56, 500), fill=CREAM, anchor="mm")
        no = clamp((t - word_t("отказываешься", 52)) / 0.25)
        k1 = eout((t - word_t("один", 47)) / 0.5)
        k2 = eout((t - word_t("другой", 49)) / 0.5)
        card(d, 140, 620, 800, 260, "ПОВЫСИТЬ НАЛОГИ", "вариант № 1", k1, no)
        card(d, 140, 1000, 800, 260, "СОКРАТИТЬ РАСХОДЫ", "вариант № 2", k2, clamp(no * 1.0 - 0.0))
    return frame, txt


def sc_contracts(t, lt, dur):
    img = to_img(grad((12, 12, 18), (4, 4, 6)))
    c = C(img)
    r = np.random.default_rng(5)
    for i in range(14):
        x, y = r.uniform(100, 800), r.uniform(400, 1500)
        dy = (lt * r.uniform(80, 200)) % 300
        c.rect(x, y - dy, x + 220, y - dy + 290, (226, 220, 206, 230))
        for j in range(5):
            c.line([(x + 20, y - dy + 40 + j * 40), (x + 200, y - dy + 40 + j * 40)], (180, 174, 164), 5)
    gx, gy = 540 + math.sin(lt * 1.6) * 220, 960 + math.cos(lt * 1.1) * 260
    c.ell(gx, gy, 150, 150, (255, 255, 255, 30), outline=GOLD, w=14)
    c.line([(gx + 105, gy + 105), (gx + 230, gy + 230)], GOLD, 26)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([90, 1580, 990, 1760], 30, fill=(6, 8, 14, 220))
        d.text((540, 1630), "ПРОВЕРКА ГОСКОНТРАКТОВ", font=MS(36, 700), fill=GOLD, anchor="mm")
        p = ease(lt / (dur - 0.5))
        d.rounded_rectangle([140, 1680, 940, 1712], 16, fill=(40, 44, 60))
        d.rounded_rectangle([140, 1680, 140 + 800 * p, 1712], 16, fill=RED if t > word_t("странное", 58) else GOLD)
        if t > word_t("странное", 58):
            d.text((540, 420), "кое-что странное…", font=PFI(64, 500), fill=CREAM, anchor="mm")
    return frame, txt


COMPANIES = ["ООО «ВЕКТОР»", "ООО «ГОРИЗОНТ»", "ООО «СТРОЙ-ПЛЮС»", "ООО «АЛЬТАИР»", "ООО «МЕРИДИАН»", "ООО «ПРЕСТИЖ»"]


def sc_shells(t, lt, dur):
    img = to_img(grad((10, 12, 22), (3, 3, 6)))
    c = C(img)
    tx, ty = 540, 520
    c.ell(tx, ty, 120, 120, None, outline=GOLD, w=8)
    c.text(tx, ty, "КАЗНА", F("ms", 34, 800), GOLD_L)
    boxes = [(200 + (i % 2) * 680, 900 + (i // 2) * 300) for i in range(6)]
    for i, (bx, by) in enumerate(boxes):
        k = eout((lt - i * 0.25) / 0.5)
        if k <= 0:
            continue
        c.line([(tx, ty + 120), (bx, by - 70)], GOLD + (int(120 * k),), 4)
        for j in range(3):
            q = ((lt * 0.8 + j / 3 + i * 0.1) % 1)
            c.ell(lerp(tx, bx, q), lerp(ty + 120, by - 70, q), 12, 12, GOLD_L)
        c.rect(bx - 190, by - 70, bx + 190, by + 90, (24, 26, 38), r=16)
        c.text(bx, by - 20, COMPANIES[i], F("ms", 30, 700), CREAM)
        c.text(bx, by + 40, "продукции: 0", F("ms", 26, 600), RED)
    frame = cam(img, lerp(1.0, 1.05, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        s = 1.2 + 47.6 * ease(lt / (dur - 0.5))
        d.text((540, 1820), f"ушло: {s:.1f} млрд".replace(".", ","), font=MS(48, 800), fill=RED, anchor="mm")
    return frame, txt


def sc_probe(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    c.rect(220, 560, 860, 1380, (110, 34, 32), r=10)
    c.rect(250, 590, 830, 1350, None, outline=(160, 70, 60), w=4)
    c.text(540, 760, "ДЕЛО № 0001", F("pf", 72, 800), (240, 220, 200))
    c.text(540, 850, "государственные контракты", F("pfi", 38, 500), (230, 200, 190))
    sk = clamp((t - word_t("расследование", 67) - 0.2) / 0.25)
    if sk > 0:
        s = lerp(1.6, 1.0, sk)
        c.rect(540 - 300 * s, 1100 - 70 * s, 540 + 300 * s, 1100 + 70 * s, None, outline=(240, 220, 200), w=8)
        c.text(540, 1100, "РАССЛЕДОВАНИЕ", F("ms", int(56 * s), 800), (240, 220, 200))
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 960, rot=2)

    def txt(im):
        k = eout((t - word_t("кажется", 69)) / 0.6)
        if k > 0:
            d = ImageDraw.Draw(im)
            d.text((540, 1600), "Кажется, ты нашёл", font=PFI(56, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
            d.text((540, 1700), "ВЫХОД", font=PF(120, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


def sc_burst(t, lt, dur):
    t_burst = word_t("врывается", 72)
    t_photo = word_t("кладёт", 76) + 0.3
    if t < t_photo:
        img = office_bg().copy()
        k = eout((t - t_burst) / 0.25) * (1 - eout((t - word_t("закрывает", 76)) / 0.5))
        img = light(img, lambda d, kk: d.polygon([(400 * kk, 520 * kk), ((400 + 280 * k) * kk, 520 * kk),
                                                  ((700 + 500 * k) * kk, 1920 * kk), (200 * kk, 1920 * kk)], fill=(150, 130, 100)), 30)
        c = C(img)
        c.rect(400, 520, 400 + 280 * k, 1240, (230, 210, 170))
        fk = eout((t - t_burst - 0.3) / 1.6)
        figure(c, lerp(540, 600, fk), lerp(1240, 1800, fk), lerp(560, 1250, fk), INK, rim=(230, 200, 150), earpiece=True)
        c.rect(0, 1600, W, H, (24, 18, 16))
        return cam(img, 1.0, 540, 960), None
    img = desk_bg().copy()
    c = C(img)
    k = eout((t - t_photo) / 0.4)
    x = lerp(1400, 540, k)
    c.rect(x - 230 + 14, 760 + 18, x + 230 + 14, 1160 + 18, (0, 0, 0, 100))
    c.rect(x - 230, 760, x + 230, 1160, (230, 224, 214))
    c.text(x, 960, "ФОТО", F("ms", 40, 700), (150, 140, 130))
    return cam(img, lerp(1.0, 1.05, lt / dur), 540, 960, rot=-2), None


def sc_photo(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    take = word_t("берёшь", 83)
    flip = ease((t - take - 0.6) / 0.5)
    lift = eout((t - take) / 0.8)
    s = lerp(1.0, 1.5, lift)
    cx, cy = 540, lerp(960, 900, lift)
    wd = 230 * s * abs(1 - 2 * flip)
    if flip < 0.5:
        c.rect(cx - wd, cy - 200 * s, cx + wd, cy + 200 * s, (230, 224, 214))
    else:
        c.rect(cx - wd, cy - 200 * s, cx + wd, cy + 200 * s, (238, 232, 218))
        c.rect(cx - wd * 0.85, cy - 170 * s, cx + wd * 0.85, cy + 110 * s, (120, 110, 96))
        if wd > 60:
            figure(c, cx, cy + 150 * s, 330 * s, (30, 26, 24), seated=True, glasses=True)
            c.rect(cx - wd * 0.85, cy + 110 * s, cx + wd * 0.85, cy + 170 * s, (238, 232, 218))
    circ = ease((t - word_t("главный", 88)) / 0.8)
    if circ > 0 and flip >= 0.5:
        c.arc(cx, cy - 60 * s, 170 * s, -90, -90 + 360 * circ, RED, 8)
    frame = cam(img, 1.0, 540, 960)

    def txt(im):
        k = eout((t - word_t("главный", 88)) / 0.5)
        if k > 0:
            d = ImageDraw.Draw(im)
            d.text((540, 1620), "ТВОЙ ГЛАВНЫЙ", font=MS(46, 700), fill=mix(NAVY, CREAM, k), anchor="mm")
            d.text((540, 1730), "СОВЕТНИК", font=PF(130, 900), fill=mix(NAVY, (240, 80, 80), k), anchor="mm")
    return frame, txt


def sc_empty(t, lt, dur):
    if t < word_t("кресло", 91) - 0.3:
        img = to_img(grad((8, 8, 14), (2, 2, 4)))
        img = light(img, lambda d, k: d.ellipse([200 * k, 500 * k, 900 * k, 1100 * k], fill=(70, 60, 50)), 120)
        c = C(img)
        lift = eout((lt - 0.2) / 1.2)
        hy = lerp(1060, 900, lift)
        c.ell(540, hy, 330, 410, INK)
        c.rect(380, hy + 300, 700, H, INK)
        c.poly([(0, H), (120, hy + 560), (960, hy + 560), (W, H)], INK)
        if lift > 0.5:
            a = int(255 * (lift - 0.5) * 2)
            c.ell(440, hy - 20, 22, 8, (230, 220, 200, a))
            c.ell(640, hy - 20, 22, 8, (230, 220, 200, a))
        return cam(img, 1.0, 540, 960), None
    img = to_img(grad((10, 10, 14), (3, 3, 5)))
    img = light(img, lambda d, k: d.polygon([(460 * k, 0), (620 * k, 0), (820 * k, 1700 * k), (260 * k, 1700 * k)], fill=(120, 100, 70)), 50)
    c = C(img)
    c.rect(0, 1500, W, H, (26, 18, 14))
    sway = math.sin(lt * 1.5) * 0.04 * max(0, 1 - lt / 4)
    c.rect(380 + sway * 200, 800, 700 + sway * 200, 1420, (20, 18, 22), r=50)
    c.rect(350, 1280, 730, 1460, (24, 22, 26), r=30)
    c.rect(400, 1460, 430, 1640, (14, 12, 14))
    c.rect(650, 1460, 680, 1640, (14, 12, 14))
    frame = cam(img, lerp(1.0, 1.12, ease(lt / dur)), 540, 1100)

    def txt(im):
        k = eout((t - word_t("исчез", 91)) / 0.4)
        if k > 0:
            d = ImageDraw.Draw(im)
            d.text((540, 460), "ОН ИСЧЕЗ", font=PF(120, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


def sc_message(t, lt, dur):
    img = to_img(grad((12, 12, 22), (3, 3, 6)) + radial(540, 900, 600)[:, :, None] * np.array([30, 40, 70], np.float32))
    c = C(img)
    c.rect(220, 260, 860, 1700, (14, 14, 18), r=70)
    c.rect(240, 280, 840, 1680, (22, 24, 30), r=56)
    buzz = math.sin(t * 80) * 6 if lt < 0.6 else 0
    frame = cam(img, lerp(1.0, 1.08, ease(lt / dur)), 540 + buzz, 960)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        d.text((540, 380), "Неизвестный номер", font=MS(34, 700), fill=CREAM, anchor="mm")
        d.text((540, 425), "сейчас", font=MS(24, 500), fill=(150, 156, 180), anchor="mm")
        msg = "Если хочешь узнать правду, сегодня в полночь никому не доверяй."
        start = word_t("если", 98)
        n = int(len(msg) * clamp((t - start) / 3.8))
        shown = msg[:n]
        if t < start:
            for i in range(3):
                a = 120 + 120 * math.sin(t * 8 + i)
                d.ellipse([300 + i * 40, 540, 324 + i * 40, 564], fill=(200, 200, 210, int(a)))
            return
        f = MS(42, 600)
        lines = wrap(d, shown, f, 470)
        h = 60 + len(lines) * 56
        d.rounded_rectangle([280, 500, 800, 500 + h], 34, fill=(48, 52, 66))
        for i, ln in enumerate(lines):
            d.text((310, 540 + i * 56), ln, font=f, fill=CREAM, anchor="lm")
    return frame, txt


def sc_hours(t, lt, dur):
    img = to_img(grad((12, 6, 8), (2, 2, 4)) + radial(540, 900, 600)[:, :, None] * np.array([70, 10, 10], np.float32))
    c = C(img)
    r = 360
    c.ell(540, 900, r, r, None, outline=(40, 40, 52), w=6)
    used = (24 - remaining(t)) / 24
    c.arc(540, 900, r, -90, -90 + 360 * used, RED, 12)
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 920)

    def txt(im):
        d = ImageDraw.Draw(im)
        pulse = 0.75 + 0.25 * math.sin(t * 6)
        d.text((540, 860), clock_text(remaining(t)), font=MS(120, 600), fill=mix((90, 20, 20), (255, 110, 110), pulse), anchor="mm")
        d.text((540, 960), "ОСТАЛОСЬ", font=MS(34, 600), fill=GOLD, anchor="mm")
        k = eout((t - word_t("несколько", 103)) / 0.5)
        d.text((540, 1420), "всего", font=PFI(56, 500), fill=mix(NAVY, CREAM, k), anchor="mm")
        d.text((540, 1520), "НЕСКОЛЬКО ЧАСОВ", font=PF(84, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
    return frame, txt


# ================================================================ монтаж
SC = {name: globals()["sc_" + name] for _, name in SCENES}
SHOTS = []
for i, (a, name) in enumerate(SCENES):
    b = SCENES[i + 1][0] if i + 1 < len(SCENES) else DURATION - VOICE_OFFSET
    SHOTS.append((max(0.0, v(a)), v(b) if i + 1 < len(SCENES) else DURATION, name))

SHAKES += [(word_t("кладёт", 76) + 0.3, 24, 0.3), (word_t("врывается", 72), 30, 0.4),
           (word_t("отказываешься", 52), 18, 0.3), (word_t("расследование", 67) + 0.2, 20, 0.3),
           (word_t("исчез", 91), 16, 0.5), (word_t("если", 98), 8, 0.4)]

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
    if t < v(CLOCK_FROM) or name in ("hours", "end", "treasury", "summon", "proposals", "message") or t >= SHOTS[-1][0]:
        return
    k = eout((t - v(CLOCK_FROM)) / 0.6)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([340, 200, 740, 300], 50, fill=(6, 8, 14, int(170 * k)), outline=GOLD + (int(160 * k),), width=2)
    d.text((540, 230), "ОСТАЛОСЬ", font=MS(22, 600), fill=GOLD + (int(255 * k),), anchor="mm")
    late = name in ("empty", "photo")
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

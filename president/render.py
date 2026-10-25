"""Рендер ролика «24 часа президентом» — кинематографичный моушн-дизайн.

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
    return _font("fonts/PlayfairDisplay[wght].ttf", size, w)


def PFI(size, w=500):
    return _font("fonts/PlayfairDisplay-Italic[wght].ttf", size, w)


def MS(size, w=600):
    return _font("fonts/Montserrat[wght].ttf", size, w)


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
           folder=False, bow=0.0, seated=False, arms_out=0.0, women=False):
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


# ================================================================ кадры
def sc_wake(t, lt, dur):
    img = bg_room("bedroom").copy()
    c = C(img)
    chandelier(c, 540, 520, 1.6)
    window(c, 720, 260, 1000, 900, sky=((250, 210, 150), (220, 160, 110)))
    img = beams(img, [[(720, 260), (1000, 260), (300, 1920), (-200, 1920)]], (150, 110, 60), 50)
    c = C(img)
    dust(c, t, 60, area=(0, 300, W, 1900))
    frame = cam(img, lerp(1.25, 1.1, ease(lt / dur)), 600, 900)
    # веки открываются
    o = ease((lt - 0.2) / 1.6)
    blur = 18 * (1 - o)
    if blur > 0.5:
        frame = frame.filter(ImageFilter.GaussianBlur(blur))
    m = Image.new("L", (W // 4, H // 4), 0)
    ry = o * H * 0.42 / 4 + 1
    ImageDraw.Draw(m).ellipse([-80, H / 8 - ry, W / 4 + 80, H / 8 + ry], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BICUBIC)
    return Image.composite(frame, Image.new("RGB", (W, H)), m), None


@lru_cache(None)
def emblem_img():
    img = Image.new("RGBA", (900 * S, 900 * S), (0, 0, 0, 0))
    c = C(img)
    cx, cy = 450, 450
    c.ell(cx, cy, 330, 330, None, outline=GOLD, w=10)
    c.ell(cx, cy, 300, 300, None, outline=GOLD + (140,), w=3)
    for side in (-1, 1):
        for k in range(14):
            a = math.radians(110 + k * 9.5)
            r = 250
            px, py = cx + side * math.cos(a) * r * -1, cy + math.sin(a) * r
            ang = a + side * 0.6
            for s2 in (-1, 1):
                lx = px + math.cos(ang + s2 * 0.5) * 34 * side
                ly = py + math.sin(ang + s2 * 0.5) * 34
                c.poly([(px, py), ((px + lx) / 2 + 8, (py + ly) / 2 - 8), (lx, ly), ((px + lx) / 2 - 8, (py + ly) / 2 + 8)], GOLD)
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        r = 150 if k % 2 == 0 else 62
        pts.append((cx + math.cos(a) * r, cy - 10 + math.sin(a) * r))
    c.poly(pts, GOLD)
    return img


def sc_emblem(t, lt, dur):
    a = grad(DEEP, NAVY) + radial(540, 860, 600)[:, :, None] * np.array([60, 50, 30], np.float32)
    img = to_img(a)
    c = C(img)
    dust(c, t, 70, seed=3)
    k = eout(lt / 1.2)
    em = emblem_img()
    size = int(900 * S * lerp(0.75, 0.9, k))
    em2 = em.resize((size, size), Image.LANCZOS)
    alpha = em2.split()[3].point(lambda p: int(p * k))
    em2.putalpha(alpha)
    img.paste(em2, (WW // 2 - size // 2, int(860 * S - size / 2)), em2)
    # блик, пробегающий по гербу
    sx = lerp(-200, 1300, clamp((lt - 0.8) / 1.2))
    img = light(img, lambda d, kk: d.polygon([((sx - 60) * kk, 300 * kk), ((sx + 60) * kk, 300 * kk),
                                              ((sx - 140) * kk, 1400 * kk), ((sx - 260) * kk, 1400 * kk)], fill=(120, 100, 60)), 30)
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 940)

    def txt(im):
        d = ImageDraw.Draw(im)
        k2 = eout((lt - 0.5) / 0.9)
        f = PF(108, 800)
        word = "ПРЕЗИДЕНТ"
        sp = lerp(40, 10, k2)
        widths = [d.textlength(ch, font=f) for ch in word]
        total = sum(widths) + sp * (len(word) - 1)
        x = (W - total) / 2
        for ch, wdt in zip(word, widths):
            d.text((x, 1400), ch, font=f, fill=mix(NAVY, GOLD_L, k2), anchor="lm")
            x += wdt + sp
        k3 = ease((lt - 1.2) / 0.6)
        d.text((540, 1490), "СВОЕЙ СТРАНЫ", font=MS(34, 500), fill=mix(NAVY, CREAM, k3), anchor="mm")
        d.line([(540 - 180 * k3, 1540), (540 + 180 * k3, 1540)], fill=GOLD, width=2)
    return frame, txt


def sc_palace(t, lt, dur):
    img = bg_palace().copy()
    c = C(img)
    for gx in (300, 780):
        figure(c, gx, 1520, 330, INK, rim=(150, 100, 60), earpiece=True)
    dust(c, t, 40, seed=5, area=(0, 600, W, 1800))
    frame = cam(img, lerp(1.22, 1.05, ease(lt / dur)), 540, lerp(1040, 980, ease(lt / dur)))
    tags = [(word_t("собственный", 6), "ДВОРЕЦ"), (word_t("личную", 6), "ЛИЧНАЯ ОХРАНА"), (word_t("власть", 6), "ВЛАСТЬ")]

    def txt(im):
        d = ImageDraw.Draw(im)
        for i, (ts, label) in enumerate(tags):
            k = eout((t - ts) / 0.5)
            if k <= 0:
                continue
            y = 1540 + i * 84
            d.line([(110, y), (110 + 60 * k, y)], fill=GOLD, width=3)
            d.text((190, y), label, font=MS(40, 600), fill=mix(NAVY, CREAM, k), anchor="lm")
    return frame, txt


def clock_text(hours):
    s = max(0, int(hours * 3600))
    return f"{s // 3600:02}:{s // 60 % 60:02}:{s % 60:02}"


def remaining(t):
    pts = [(v(a), b) for a, b in CLOCK]
    base = float(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))
    return base - (t % 1.0) / 3600


def sc_timer(t, lt, dur):
    img = to_img(grad((6, 8, 14), (2, 2, 4)) + radial(540, 960, 520)[:, :, None] * np.array([40, 30, 18], np.float32))
    c = C(img)
    ts = word_t("24", 12.5)
    k = ease((t - ts + 0.2) / 0.6)
    r = 360
    c.ell(540, 900, r, r, None, outline=(40, 40, 52), w=6)
    if k > 0:
        c.arc(540, 900, r, -90, -90 + 360 * k, GOLD, 8)
    for i in range(24):
        a = math.radians(i * 15 - 90)
        c.line([(540 + math.cos(a) * (r - 30), 900 + math.sin(a) * (r - 30)), (540 + math.cos(a) * (r - 12), 900 + math.sin(a) * (r - 12))],
               (90, 90, 100), 3)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 920)
    if t >= word_t("нюанс", 12.5) and not any(abs(s[0] - word_t("нюанс", 12.5)) < 0.01 for s in SHAKES):
        pass

    def txt(im):
        d = ImageDraw.Draw(im)
        if t < ts:
            k0 = ease((lt - 0.3) / 0.6)
            d.text((540, 900), "НО ЕСТЬ", font=MS(44, 500), fill=mix(NAVY, CREAM, k0), anchor="mm")
            d.text((540, 970), "ОДИН НЮАНС", font=PF(80, 800), fill=mix(NAVY, GOLD_L, k0), anchor="mm")
            return
        kk = eout((t - ts) / 0.4)
        d.text((540, 880), "24:00:00", font=MS(150, 300 + int(400 * kk)), fill=mix(NAVY, CREAM, kk), anchor="mm")
        d.text((540, 1000), "ЧАСА", font=MS(40, 600), fill=mix(NAVY, GOLD, kk), anchor="mm")
    return frame, txt


def sc_bed(t, lt, dur):
    img = bg_room("bedroom").copy()
    c = C(img)
    window(c, 640, 330, 960, 1050, sky=((250, 214, 160), (200, 150, 110)))
    img = beams(img, [[(640, 330), (960, 330), (500, 1920), (0, 1920)]], (160, 120, 70), 50)
    c = C(img)
    c.rect(80, 1150, 900, 1460, (30, 26, 32), r=30)
    c.rect(60, 800, 140, 1500, (18, 16, 20), r=10)
    c.rect(80, 1100, 900, 1200, (200, 190, 180), r=30)
    c.rect(120, 1060, 360, 1150, (226, 220, 210), r=30)
    figure(c, 420, 1500, 820, INK, rim=(200, 150, 90), rim_dx=6, seated=True, turn=lerp(-0.6, 0.6, ease((lt - 0.8) / 1.0)))
    c.rect(300, 1270, 900, 1470, (180, 172, 166), r=30)
    dust(c, t, 50, seed=7, area=(300, 300, W, 1500))
    return cam(img, lerp(1.0, 1.12, ease(lt / dur)), 520, 1060), None


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


def sc_guards(t, lt, dur):
    img = corridor_bg().copy()
    c = C(img)
    vx, vy = 540, 820
    for k in range(4, -1, -1):
        f = 0.12 + k * 0.17
        h = lerp(980, 160, f)
        feet = lerp(1820, vy + 120, f)
        for side in (-1, 1):
            x = vx + side * lerp(380, 70, f)
            figure(c, x, feet, h, INK, rim=(150, 120, 80), rim_dx=-side * 4, earpiece=True, turn=-side * 0.3)
    # двери распахиваются
    k = eout(lt / 0.9)
    frame = cam(img, lerp(1.0, 1.1, ease(lt / dur)), 540, 1000)
    d = ImageDraw.Draw(frame)
    wd = int(540 * (1 - k))
    if wd > 0:
        d.rectangle([0, 0, wd, H], fill=(28, 22, 20))
        d.rectangle([W - wd, 0, W, H], fill=(28, 22, 20))
        d.line([(wd, 0), (wd, H)], fill=GOLD, width=3)
        d.line([(W - wd, 0), (W - wd, H)], fill=GOLD, width=3)

    def txt(im):
        kk = eout((t - word_t("охранников", 19.6)) / 0.5)
        if kk > 0:
            d2 = ImageDraw.Draw(im)
            d2.text((540, 1600), "10", font=PF(140, 800), fill=mix(NAVY, GOLD_L, kk), anchor="mm")
            d2.text((540, 1700), "ОХРАННИКОВ", font=MS(36, 600), fill=mix(NAVY, CREAM, kk), anchor="mm")
    return frame, txt


def sc_aide(t, lt, dur):
    img = corridor_bg().copy()
    img = img.filter(ImageFilter.GaussianBlur(10))
    c = C(img)
    k = eout(lt / 2.0)
    figure(c, 560, lerp(1500, 1700, k), lerp(700, 1000, k), INK, rim=(220, 180, 120), rim_dx=6, tablet=True, bow=ease((lt - 2.5) / 0.6))
    return cam(img, 1.05, 540, 1000), None


def sc_real(t, lt, dur):
    img = to_img(grad((10, 12, 20), (4, 4, 8)) + radial(820, 700, 600)[:, :, None] * np.array([120, 90, 50], np.float32))
    c = C(img)
    # профиль крупно, контровой свет со стороны окна
    profile(c, 470, 900, 760, (210, 160, 100), dx=10, dy=-4)
    profile(c, 470, 900, 760, INK)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 960)

    def txt(im):
        kk = eout((t - word_t("реально", 30)) / 0.35)
        if kk > 0:
            d = ImageDraw.Draw(im)
            d.text((540, 1560), "ЭТО", font=MS(46, 500), fill=mix(NAVY, CREAM, kk), anchor="mm")
            d.text((540, 1660), "РЕАЛЬНО.", font=PF(130, 900), fill=mix(NAVY, GOLD_L, kk), anchor="mm")
    return frame, txt


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


def sc_cabinet(t, lt, dur):
    img = cabinet_bg().copy()
    c = C(img)
    for k in range(5, -1, -1):
        f = k / 6
        y = lerp(1700, 1020, f)
        h = lerp(700, 150, f)
        for side in (-1, 1):
            x = lerp(540 + side * 470, 540 + side * 85, f)
            figure(c, x, y + h * 0.45, h, INK, rim=(110, 110, 130), rim_dx=side * -3, seated=True, turn=-side * 0.6)
    figure(c, 540, 2300, 1300, (4, 4, 8), seated=True)
    return cam(img, lerp(1.0, 1.08, lt / dur), 540, 1000), None


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


DECREES = [
    ("УКАЗ № 1", "Об учебном дне", ["Школьники учатся", "только до 12:00"]),
    ("УКАЗ № 2", "О выходных днях", ["+1 выходной день", "каждую неделю"]),
    ("УКАЗ № 3", "О доступе в интернет", ["Бесплатный интернет", "для всех граждан"]),
]


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


def sc_decrees(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    starts = [word_t("первый", 33.5) + 0.3, word_t("второй", 37), word_t("третий", 40)]
    for i, (title, sub, lines) in enumerate(DECREES):
        k = eout((t - starts[i]) / 0.6)
        if k <= 0:
            continue
        x = lerp(1600, 540 + (i - 1) * 18, k)
        y = 900 + i * 22
        st = clamp((t - starts[i] - 1.6) / 0.25)
        paper(c, x, y, 0, title, sub, lines, st)
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 920, rot=-1.5)
    return frame, None


def sc_glances(t, lt, dur):
    img = to_img(grad((14, 16, 26), (6, 6, 10)) + radial(540, 600, 700)[:, :, None] * np.array([50, 50, 70], np.float32))
    c = C(img)
    for i, x in enumerate((220, 540, 860)):
        ph = lt * 1.2 + i * 1.7
        turn = math.sin(ph) * 0.9
        figure(c, x, 1900, 1100 - abs(i - 1) * 120, INK, rim=(130, 140, 170), rim_dx=-turn * 8, seated=True, turn=turn)
    frame = cam(img, lerp(1.05, 1.12, lt / dur), 540, 1000)

    def txt(im):
        kk = eout((t - word_t("все-таки", 47.7)) / 0.5)
        if kk > 0:
            d = ImageDraw.Draw(im)
            d.text((540, 420), "Президент здесь —", font=PFI(64, 500), fill=mix(NAVY, CREAM, kk), anchor="mm")
            d.text((540, 540), "ТЫ.", font=PF(150, 900), fill=mix(NAVY, GOLD_L, kk), anchor="mm")
    return frame, txt


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


def sc_news(t, lt, dur):
    img = to_img(grad((12, 18, 36), (6, 8, 16)))
    c = C(img)
    cx, cy = 560, 920
    wave = lt * 360
    for x, y in country_dots():
        dd = math.hypot(x - cx, y - cy)
        k = clamp((wave - dd) / 120)
        col = mix((40, 50, 80), GOLD_L, k)
        c.ell(x, y, 4, 4, col)
    for j in range(3):
        rr = ((lt * 300 + j * 200) % 600)
        c.ell(cx, cy, rr, rr, None, outline=GOLD + (int(120 * (1 - rr / 600)),), w=3)
    frame = cam(img, 1.0 + lt * 0.02, 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.rectangle([60, 1380, 330, 1450], fill=RED)
        d.text((195, 1416), "СРОЧНО", font=MS(40, 800), fill=(255, 255, 255), anchor="mm")
        d.rectangle([60, 1450, 1020, 1600], fill=(245, 242, 236))
        d.text((90, 1495), "ПРЕЗИДЕНТ ПОДПИСАЛ", font=MS(46, 800), fill=(20, 20, 30), anchor="lm")
        d.text((90, 1555), "ТРИ НОВЫХ УКАЗА", font=MS(46, 800), fill=(20, 20, 30), anchor="lm")
        d.rectangle([60, 1600, 1020, 1660], fill=(20, 26, 48))
        tick = "  •  ШКОЛЬНЫЙ ДЕНЬ ДО 12:00  •  +1 ВЫХОДНОЙ  •  БЕСПЛАТНЫЙ ИНТЕРНЕТ ДЛЯ ВСЕХ"
        f = MS(30, 600)
        tw = d.textlength(tick, font=f)
        x = 1020 - (lt * 240) % tw
        for k in range(3):
            d.text((x + k * tw, 1630), tick, font=f, fill=CREAM, anchor="lm")
        d.rectangle([0, 1600, 60, 1660], fill=(20, 26, 48))
        d.rectangle([1020, 1600, W, 1660], fill=(0, 0, 0))
        d.ellipse([920, 1395, 950, 1425], fill=(255, 60, 60) if int(t * 2) % 2 else (120, 20, 20))
        d.text((960, 1410), "LIVE", font=MS(28, 800), fill=CREAM, anchor="lm")
    return frame, txt


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


def sc_crowd(t, lt, dur):
    img = city_bg().copy()
    c = C(img)
    r = np.random.default_rng(31)
    for row in range(4):
        y = 1450 + row * 130
        s = 0.8 + row * 0.25
        for i in range(14):
            x = (i + r.uniform(0, 0.8)) * 82 - 20
            bob = abs(math.sin(t * 4 + i + row)) * 14 * s
            c.ell(x, y - bob, 36 * s, 44 * s, INK)
            c.ell(x, y + 80 * s - bob, 70 * s, 60 * s, INK)
            if (i + row) % 3 == 0:
                px, py = x + 30 * s, y - 120 * s - bob
                c.line([(x + 20 * s, y + 20 * s - bob), (px, py + 40 * s)], INK, 14 * s)
                c.rect(px - 18 * s, py - 30 * s, px + 18 * s, py + 30 * s, (240, 240, 255), r=4)
    for i in range(80):
        x = (i * 97 + t * 40 * (1 + i % 3)) % W
        y = (i * 211 + t * 160 * (1 + i % 2)) % 1500
        c.rect(x, y, x + 10, y + 16, [GOLD, RED, CREAM][i % 3] + (200,))
    img = light(img, lambda d, k: [d.ellipse([((i * 83) % W - 40) * k, (1200 + (i * 37) % 300) * k, ((i * 83) % W + 40) * k,
                                              (1280 + (i * 37) % 300) * k], fill=(120, 120, 160)) for i in range(14)], 30)
    flash = (int(t * 3) % 5 == 0) and (t * 3) % 1 < 0.3
    frame = cam(img, lerp(1.0, 1.1, lt / dur), 540, 1100)
    if flash:
        frame = to_img(np.asarray(frame, np.float32) * 1.25)
    return frame, None


def sc_trend(t, lt, dur):
    img = to_img(grad((16, 14, 30), (6, 6, 12)) + radial(540, 900, 700)[:, :, None] * np.array([60, 40, 70], np.float32))
    c = C(img)
    c.rect(250, 300, 830, 1560, (14, 14, 18), r=70)
    c.rect(270, 320, 810, 1540, (24, 26, 34), r=56)
    c.rect(470, 340, 610, 370, (14, 14, 18), r=15)
    frame = cam(img, lerp(1.0, 1.05, lt / dur), 540, 930)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        d.text((540, 440), "В ТРЕНДАХ", font=MS(34, 700), fill=GOLD, anchor="mm")
        items = [("#президент", 12.4), ("#24часа", 8.1), ("#указ", 5.6), ("#выходной", 3.9)]
        for i, (tag, mln) in enumerate(items):
            k = eout((lt - i * 0.25) / 0.5)
            y = 560 + i * 200
            x0 = 300 + (1 - k) * 200
            d.rounded_rectangle([x0, y, x0 + 480, y + 160], 24, fill=(36, 40, 54, int(255 * k)))
            d.text((x0 + 30, y + 52), f"{i + 1}", font=PF(56, 800), fill=GOLD_L if i == 0 else CREAM, anchor="lm")
            d.text((x0 + 90, y + 52), tag, font=MS(40, 700), fill=CREAM, anchor="lm")
            cnt = mln * ease(lt / 2.5)
            d.text((x0 + 90, y + 112), f"{cnt:.1f} млн просмотров", font=MS(28, 500), fill=(160, 166, 190), anchor="lm")
        for i in range(10):
            hk = (lt * 0.8 + i * 0.13) % 1
            hx, hy = 860 + math.sin(i * 3 + lt * 2) * 40, 1500 - hk * 900
            a_ = int(255 * (1 - hk))
            d.ellipse([hx - 20, hy - 16, hx, hy + 4], fill=(240, 70, 90, a_))
            d.ellipse([hx, hy - 16, hx + 20, hy + 4], fill=(240, 70, 90, a_))
            d.polygon([(hx - 19, hy - 2), (hx + 19, hy - 2), (hx, hy + 22)], fill=(240, 70, 90, a_))
    return frame, txt


def sc_millions(t, lt, dur):
    img = to_img(grad((10, 12, 24), (4, 4, 10)))
    c = C(img)
    cols, rows = 30, 46
    wave = lt / (dur * 0.8)
    for j in range(rows):
        for i in range(cols):
            x, y = 60 + i * 33, 200 + j * 33
            dd = math.hypot(i - cols / 2, j - rows / 2) / 30
            k = clamp((wave - dd) * 3)
            c.ell(x, y, 6, 6, mix((40, 44, 70), GOLD_L, k))
    frame = cam(img, lerp(1.25, 1.0, ease(lt / dur)), 540, 960)

    def txt(im):
        kk = eout((t - word_t("миллионов", 64)) / 0.5)
        if kk > 0:
            d = ImageDraw.Draw(im)
            d.rectangle([0, 820, W, 1100], fill=(4, 4, 10))
            d.text((540, 900), "изменить жизнь", font=PFI(70, 500), fill=mix(NAVY, CREAM, kk), anchor="mm")
            d.text((540, 1010), "МИЛЛИОНОВ", font=PF(120, 900), fill=mix(NAVY, GOLD_L, kk), anchor="mm")
    return frame, txt


def sc_ominous(t, lt, dur):
    img = bg_palace().copy()
    img = to_img(np.asarray(img, np.float32) * np.array([0.45, 0.45, 0.6], np.float32))
    flash = 0 <= lt - 1.2 < 0.12
    if flash:
        img = to_img(np.asarray(img, np.float32) * 2.2)
    return cam(img, lerp(1.0, 1.15, ease(lt / dur)), 540, 900), None


@lru_cache(None)
def office_bg():
    img = bg_room("office").copy()
    c = C(img)
    c.rect(400, 520, 680, 1240, (14, 12, 16))
    return img


def sc_minister(t, lt, dur):
    t_burst = v(73.6)
    t_slam = word_t("кладёт", 73) + 0.6
    if t < t_slam - 0.3:
        img = office_bg().copy()
        c = C(img)
        k = eout((t - t_burst) / 0.25)
        img = light(img, lambda d, kk: d.polygon([(400 * kk, 520 * kk), ((400 + 280 * k) * kk, 520 * kk), ((700 + 500 * k) * kk, 1920 * kk),
                                                  (200 * kk, 1920 * kk)], fill=(150, 130, 100)), 30)
        c = C(img)
        c.rect(400, 520, 400 + 280 * k, 1240, (230, 210, 170))
        fk = eout((t - t_burst - 0.3) / 1.6)
        figure(c, lerp(540, 600, fk), lerp(1240, 1800, fk), lerp(560, 1250, fk), INK, rim=(230, 200, 150), folder=True)
        c.rect(0, 1600, W, H, (24, 18, 16))
        c.rect(0, 1580, W, 1610, (40, 30, 24))
        return cam(img, 1.0, 540, 960), None
    img = desk_bg().copy()
    c = C(img)
    k = eout((t - t_slam + 0.3) / 0.3)
    s = lerp(1.6, 1.0, k)
    fx, fy = 540, 960
    c.rect(fx - 330 * s + 20, fy - 440 * s + 24, fx + 330 * s + 20, fy + 440 * s + 24, (0, 0, 0, 100))
    c.rect(fx - 330 * s, fy - 440 * s, fx + 330 * s, fy + 440 * s, (120, 40, 36))
    c.rect(fx - 300 * s, fy - 410 * s, fx + 300 * s, fy + 410 * s, None, outline=(160, 70, 60), w=4)
    c.rect(fx - 200 * s, fy - 300 * s, fx + 200 * s, fy - 180 * s, (232, 222, 200))
    c.text(fx, fy - 240 * s, "БЮДЖЕТ", F("pf", int(60 * s), 800), (40, 30, 30))
    c.rect(fx + 330 * s, fy - 400 * s, fx + 350 * s, fy + 420 * s, (200, 190, 170))
    for i in range(6):
        c.rect(fx + 330 * s, fy - 380 * s + i * 140 * s, fx + 345 * s, fy - 300 * s + i * 140 * s, (230, 222, 200))
    sk = clamp((t - t_slam - 0.4) / 0.2)
    if sk > 0:
        ss = lerp(1.5, 1.0, sk)
        c.rect(fx - 190 * ss, fy + 40 * ss, fx + 190 * ss, fy + 160 * ss, None, outline=(230, 60, 60, 230), w=8)
        c.text(fx, fy + 100 * ss, "СРОЧНО", F("ms", int(60 * ss), 800), (230, 60, 60, 230))
    return cam(img, lerp(1.0, 1.05, lt / dur), 540, 960, rot=2), None


def sc_budget(t, lt, dur):
    img = to_img(grad((14, 16, 26), (6, 6, 10)))
    c = C(img)
    for i in range(6):
        y = 500 + i * 150
        c.line([(120, y), (960, y)], (40, 44, 60), 2)
    pts = []
    k = clamp(lt / 3.5)
    n = 40
    for i in range(int(n * k) + 1):
        x = 120 + i * 840 / n
        y = 600 + (i / n) ** 2.2 * 650 + math.sin(i * 1.3) * 25
        pts.append((x, y))
    if len(pts) > 1:
        c.line(pts, RED, 8)
        c.ell(pts[-1][0], pts[-1][1], 14, 14, RED)
    frame = cam(img, lerp(1.0, 1.06, lt / dur), 540, 900)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.text((120, 400), "БЮДЖЕТ СТРАНЫ", font=MS(38, 700), fill=CREAM, anchor="lm")
        val = -int(1000 * ease(lt / 3.5) ** 1.5)
        d.text((960, 400), f"{val:+d} млрд".replace("-", "−"), font=MS(44, 800), fill=RED, anchor="rm")
    return frame, txt


def sc_answer(t, lt, dur):
    img = to_img(grad((10, 14, 30), (4, 4, 10)))
    c = C(img)
    for i in range(3):
        x0 = 120 + i * 300
        c.rect(x0, 200, x0 + 260, 1500, (24, 36, 70))
        for wy in range(260, 1460, 60):
            for wx in range(int(x0 + 20), int(x0 + 240), 50):
                if (wx * 7 + wy * 3) % 5:
                    c.rect(wx, wy, wx + 14, wy + 8, (200, 170, 110))
        c.rect(x0 + 260, 200, x0 + 300, 1500, (8, 8, 14))
    c.rect(0, 1500, W, H, (8, 8, 12))
    c.rect(330, 900, 750, 1700, (6, 6, 10), r=60)
    c.rect(300, 1300, 780, 1500, (6, 6, 10), r=40)
    figure(c, 540, 1700, 1000, (6, 6, 10), rim=(120, 140, 200), rim_dx=6, seated=True)
    return cam(img, lerp(1.08, 1.0, ease(lt / dur)), 540, 1050), None


def sc_leave(t, lt, dur):
    img = office_bg().copy()
    c = C(img)
    k = 1 - eout(lt / 1.2)
    c.rect(400, 520, 400 + 280 * k, 1240, (230, 210, 170))
    if k > 0.3:
        figure(c, 400 + 140 * k, 1240, 560, INK)
    c.rect(0, 1600, W, H, (24, 18, 16))
    return cam(img, 1.0, 540, 960), None


def sc_phone(t, lt, dur):
    img = desk_bg().copy()
    c = C(img)
    ring = int(t * 6) % 2 == 0
    off = math.sin(t * 90) * 4 if ring else 0
    c.rect(220 + off, 700, 860 + off, 1300, (14, 14, 16), r=60)
    c.rect(180 + off, 560, 900 + off, 720, (20, 20, 22), r=60)
    c.ell(300 + off, 640, 90, 80, (14, 14, 16))
    c.ell(780 + off, 640, 90, 80, (14, 14, 16))
    for i in range(3):
        for j in range(4):
            c.ell(400 + i * 140 + off, 820 + j * 110, 40, 34, (40, 40, 44))
    c.rect(320 + off, 1180, 760 + off, 1260, (40, 30, 30), r=10)
    c.text(540 + off, 1220, "СЕКРЕТНАЯ ЛИНИЯ", F("ms", 34, 700), (230, 210, 180))
    if ring:
        img = light(img, lambda d, k: d.ellipse([(810) * k, (720) * k, 880 * k, 790 * k], fill=(255, 40, 40)), 30, 2)
        c = C(img)
    c.ell(845 + off, 755, 18, 18, (255, 60, 60) if ring else (90, 20, 20))
    return cam(img, lerp(1.0, 1.08, lt / dur), 540, 960, rot=-3), None


def sc_corridor(t, lt, dur):
    img = corridor_bg().copy()
    img = to_img(np.asarray(img, np.float32) * np.array([0.55, 0.6, 0.75], np.float32))
    c = C(img)
    k = lt / dur
    figure(c, 520, lerp(1700, 1250, k), lerp(900, 480, k), INK, rim=(150, 150, 190))
    c.rect(870, 860, 900, 920, (20, 20, 24), r=4)
    c.ell(885, 875, 6, 6, (60, 255, 120) if lt > dur - 0.8 else (255, 60, 60))
    return cam(img, lerp(1.0, 1.15, ease(k)), 540, 900), None


def sc_table(t, lt, dur):
    img = to_img(grad((10, 10, 14), (4, 4, 6)))
    img = light(img, lambda d, k: d.ellipse([140 * k, 380 * k, 940 * k, 1540 * k], fill=(150, 120, 80)), 80)
    c = C(img)
    c.rect(330, 500, 750, 1420, (60, 38, 26), r=50)
    c.rect(350, 520, 730, 1400, (74, 48, 32), r=40)
    look = ease((t - word_t("смотрят", 101)) / 0.8)
    seats = [(250, 600 + i * 200, 1) for i in range(5)] + [(830, 600 + i * 200, -1) for i in range(5)]
    for i, (x, y, side) in enumerate(seats):
        ang = lerp(0, -side * 0.9 * (1 if y < 1300 else 0.4), look)
        c.ell(x, y, 80, 54, (16, 16, 20))
        hx, hy = x + math.sin(ang) * 10 + side * 14, y + math.cos(ang) * 4
        c.ell(hx, hy, 40, 40, (26, 24, 26))
        c.ell(hx + side * 10 * math.cos(ang) + math.sin(ang) * 30, hy + 30 * (1 - abs(math.sin(ang))) * 0 + math.cos(ang) * 0, 8, 8, (40, 36, 36))
    c.ell(540, 1450 + 100 * (1 - eout(lt / 1.5)), 100, 66, (12, 12, 16))
    c.ell(540, 1460 + 100 * (1 - eout(lt / 1.5)), 46, 46, (30, 28, 30))
    return cam(img, lerp(1.1, 1.0, ease(lt / dur)), 540, 1000), None


def sc_reveal(t, lt, dur):
    img = to_img(grad((8, 8, 12), (3, 3, 5)))
    img = light(img, lambda d, k: d.polygon([(470 * k, 0), (610 * k, 0), (900 * k, 1920 * k), (180 * k, 1920 * k)], fill=(90, 80, 60)), 60)
    c = C(img)
    c.rect(0, 1400, W, H, (40, 26, 18))
    figure(c, 540, 1700, 1300, INK, rim=(140, 120, 90), rim_dx=4, seated=True)
    c.ell(505, 760, 6, 3, (230, 220, 200))
    c.ell(575, 760, 6, 3, (230, 220, 200))
    return cam(img, lerp(1.0, 1.4, ease(lt / dur)), 540, 900), None


def sc_half(t, lt, dur):
    img = to_img(grad((8, 10, 18), (2, 2, 4)))
    c = C(img)
    r = 360
    c.ell(540, 900, r, r, None, outline=(40, 40, 52), w=6)
    used = (24 - remaining(t)) / 24
    c.arc(540, 900, r, -90, -90 + 360 * used * ease(lt / 1.2), GOLD, 10)
    c.line([(540, 900 - r - 40), (540, 900 - r + 40)], (90, 90, 100), 3)
    c.line([(540, 900 + r - 40), (540, 900 + r + 40)], RED, 4)
    frame = cam(img, lerp(1.0, 1.05, lt / dur), 540, 920)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.text((540, 860), clock_text(remaining(t)), font=MS(120, 600), fill=CREAM, anchor="mm")
        d.text((540, 960), "ОСТАЛОСЬ", font=MS(34, 600), fill=GOLD, anchor="mm")
        kk = eout((lt - 1.4) / 0.5)
        d.text((540, 1420), "ещё даже не", font=PFI(56, 500), fill=mix(NAVY, CREAM, kk), anchor="mm")
        d.text((540, 1510), "ПОЛОВИНА", font=PF(100, 900), fill=mix(NAVY, GOLD_L, kk), anchor="mm")
    return frame, txt


def sc_debt(t, lt, dur):
    img = to_img(grad((12, 10, 16), (4, 3, 6)))
    c = C(img)
    dim = ease((t - word_t("последствия", 122)) / 1.5)
    for x, y in country_dots():
        c.ell(x, y - 120, 4, 4, mix((60, 70, 100), (160, 30, 30), dim))
    c.rect(170, 1150, 910, 1700, (238, 232, 218))
    c.text(540, 1230, "ГОСУДАРСТВЕННЫЙ ДОЛГ", F("ms", 40, 800), (30, 26, 30))
    c.line([(260, 1280), (820, 1280)], (180, 150, 100), 2)
    c.text(540, 1360, "Срок выплаты:", F("pfi", 40, 500), (90, 80, 76))
    c.text(540, 1440, "ЗАВТРА, 09:00", F("pf", 76, 800), (30, 26, 30))
    sk = clamp((t - word_t("долг", 122) - 0.2) / 0.25)
    if sk > 0:
        ss = lerp(1.6, 1.0, sk)
        c.rect(540 - 260 * ss, 1580 - 60 * ss, 540 + 260 * ss, 1580 + 60 * ss, None, outline=(190, 30, 40), w=8)
        c.text(540, 1580, "НЕ ОПЛАЧЕНО", F("ms", int(54 * ss), 800), (190, 30, 40))
    return cam(img, lerp(1.0, 1.06, lt / dur), 540, 1000, rot=-1), None


def sc_eyes(t, lt, dur):
    img = to_img(grad((8, 8, 14), (2, 2, 4)))
    img = light(img, lambda d, k: d.ellipse([200 * k, 500 * k, 900 * k, 1100 * k], fill=(70, 60, 50)), 120)
    c = C(img)
    lift = eout((lt - 0.4) / 1.2)
    hy = lerp(1060, 900, lift)
    c.ell(540, hy, 330, 410, INK)
    c.rect(380, hy + 300, 700, H, INK)
    c.poly([(0, H), (120, hy + 560), (960, hy + 560), (W, H)], INK)
    if lift > 0.5:
        a = int(255 * (lift - 0.5) * 2)
        c.ell(440, hy - 20, 22, 8, (230, 220, 200, a))
        c.ell(640, hy - 20, 22, 8, (230, 220, 200, a))
    return cam(img, lerp(1.0, 1.1, lt / dur), 540, 960), None


def sc_sum(t, lt, dur):
    img = to_img(grad((10, 8, 14), (2, 2, 4)) + radial(540, 900, 600)[:, :, None] * np.array([60, 20, 20], np.float32))
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im)
        d.text((540, 700), "СУММА", font=MS(40, 600), fill=GOLD, anchor="mm")
        r = np.random.default_rng(int(t * 20))
        if lt < 1.4:
            num = " ".join(f"{r.integers(100, 999)}" for _ in range(4))
            d.text((540, 900), num, font=MS(96, 700), fill=CREAM, anchor="mm")
        else:
            for i in range(4):
                x = 140 + i * 205
                d.rectangle([x, 850, x + 180, 950], fill=(10, 10, 12))
                d.rectangle([x, 850, x + 180, 950], outline=(60, 60, 70), width=2)
            d.text((540, 1040), "ЗАСЕКРЕЧЕНО", font=MS(36, 700), fill=RED, anchor="mm")
    return frame, txt


def sc_chaos(t, lt, dur):
    img = bg_palace().copy()
    a = np.asarray(img, np.float32) * np.array([1.2, 0.4, 0.4], np.float32)
    img = to_img(a)
    frame = cam(img, lerp(1.0, 1.3, ease(lt / dur)), 540, 900)
    arr = np.asarray(frame).copy()
    g = int(6 + 30 * ease(lt / dur))
    if int(t * 12) % 3 == 0:
        arr[:, :, 0] = np.roll(arr[:, :, 0], g, axis=1)
        arr[:, :, 2] = np.roll(arr[:, :, 2], -g, axis=1)
        r = np.random.default_rng(int(t * 24))
        for _ in range(6):
            y = r.integers(0, H - 60)
            arr[y:y + 40] = np.roll(arr[y:y + 40], r.integers(-80, 80), axis=1)
    frame = Image.fromarray(arr)

    def txt(im):
        kk = eout((t - word_t("хаос", 137)) / 0.3)
        if kk > 0:
            d = ImageDraw.Draw(im)
            d.text((540 + (r_ := math.sin(t * 50) * 6), 960), "ХАОС", font=PF(230, 900), fill=mix(NAVY, (240, 60, 60), kk), anchor="mm")
            d.text((540, 820), "завтра может наступить настоящий", font=PFI(44, 500), fill=mix(NAVY, CREAM, kk), anchor="mm")
    return frame, txt


def sc_end(t, lt, dur):
    img = Image.new("RGB", (W, H), NAVY)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = eout((lt - 0.3) / 0.8) * (1 - ease((lt - dur + 0.6) / 0.6))
        d.text((540, 880), TITLE, font=PF(150, 900), fill=mix(NAVY, GOLD_L, k), anchor="mm")
        d.text((540, 1010), SUBTITLE, font=MS(44, 600), fill=mix(NAVY, CREAM, k), anchor="mm")
        d.line([(540 - 160 * k, 1070), (540 + 160 * k, 1070)], fill=mix(NAVY, GOLD, k), width=2)
    return img, txt


# ================================================================ монтаж
SC = {name: globals()["sc_" + name] for _, name in SCENES}
SHOTS = []
for i, (a, name) in enumerate(SCENES):
    b = SCENES[i + 1][0] if i + 1 < len(SCENES) else DURATION - VOICE_OFFSET
    SHOTS.append((max(0.0, v(a)), v(b) if i + 1 < len(SCENES) else DURATION, name))

SHAKES += [(word_t("кладёт", 73) + 0.3, 30, 0.3), (word_t("нюанс", 12), 10, 0.4), (word_t("хаос", 137), 30, 0.8)]
for i in range(3):
    SHAKES.append(([word_t("первый", 33.5) + 0.3, word_t("второй", 37), word_t("третий", 40)][i] + 1.6, 22, 0.25))
SHAKES.append((word_t("долг", 122) + 0.2, 18, 0.3))

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
    if t < v(CLOCK_FROM) or name in ("timer", "half", "end", "sum") or t >= SHOTS[-1][0]:
        return
    k = eout((t - v(CLOCK_FROM)) / 0.6)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([340, 200, 740, 300], 50, fill=(6, 8, 14, int(170 * k)), outline=GOLD + (int(160 * k),), width=2)
    d.text((540, 230), "ОСТАЛОСЬ", font=MS(22, 600), fill=GOLD + (int(255 * k),), anchor="mm")
    late = name in ("debt", "eyes", "chaos")
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

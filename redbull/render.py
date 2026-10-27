"""Renderer for the Short "He Sued Red Bull for $13 Million!" — realistic-leaning motion design
(engine of the «24 часа президентом» videos; fonts from ../president/fonts).

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

from timeline import CAPTIONS, DURATION, FPS, H, SCENES, VOICE_OFFSET, W, v

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




# ================================================================ words / captions
@lru_cache(None)
def words():
    out = []
    for line in open("voice/captions.tsv", encoding="utf-8"):
        s, e, w = line.rstrip("\n").split("\t")
        out.append((v(float(s)), v(float(e)), w))
    return out


def word_t(word, after=0.0):
    """Video time of the first script word starting with `word` after voice second `after`."""
    for s, e, w in words():
        if s >= v(after) and w.lower().strip(".,!?“”‘’\"…:").startswith(word):
            return s
    return v(after)


YEL = (255, 210, 40)
WHITE = (250, 250, 250)
GREEN = (60, 170, 90)
BLUE = (30, 70, 160)


# ================================================================ realistic props
@lru_cache(None)
def bokeh_bg(seed, tint):
    """Dark background with soft out-of-focus lights (depth of field)."""
    a = grad((18, 20, 30), (4, 4, 8))
    img = to_img(a)
    r = np.random.default_rng(seed)
    lay = Image.new("RGB", (W // 4, H // 4))
    d = ImageDraw.Draw(lay)
    for _ in range(40):
        x, y, rr = r.uniform(0, W / 4), r.uniform(0, H / 4), r.uniform(6, 26)
        k = r.uniform(0.3, 1.0)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=tuple(int(c * k) for c in tint))
    lay = lay.filter(ImageFilter.GaussianBlur(6)).resize(img.size, Image.BILINEAR)
    return ImageChops.add(img, to_img(np.asarray(lay, np.float32) * 0.55))


@lru_cache(None)
def can_img():
    """Generic energy-drink can with cylindrical shading (no real brand logo)."""
    cw, ch = 300 * S, 560 * S
    xs = np.linspace(-1, 1, cw)
    shade = 0.35 + 0.65 * np.sqrt(np.clip(1 - xs ** 2, 0, 1))
    spec = np.exp(-((xs + 0.45) / 0.09) ** 2) * 0.9 + np.exp(-((xs - 0.55) / 0.05) ** 2) * 0.35
    body = np.zeros((ch, cw, 4), np.float32)
    ys = np.linspace(0, 1, ch)[:, None]
    diag = (ys * 1.0 + xs[None, :] * 0.18) < 0.52
    col = np.where(diag[:, :, None], np.array([28, 60, 150], np.float32), np.array([196, 202, 212], np.float32))
    body[:, :, :3] = col * shade[None, :, None] + spec[None, :, None] * 255
    body[:, :, 3] = 255
    img = to_img(body[:, :, :3]).convert("RGBA")
    m = Image.new("L", (cw, ch), 0)
    md = ImageDraw.Draw(m)
    md.rounded_rectangle([0, 18 * S, cw, ch - 10 * S], 26 * S, fill=255)
    md.ellipse([0, 0, cw, 40 * S], fill=255)
    img.putalpha(m)
    d = ImageDraw.Draw(img)
    d.ellipse([6 * S, 2 * S, cw - 6 * S, 36 * S], fill=(205, 210, 218))
    d.ellipse([30 * S, 8 * S, cw - 30 * S, 30 * S], fill=(150, 156, 166))
    d.rounded_rectangle([120 * S, 12 * S, 180 * S, 24 * S], 6 * S, fill=(220, 224, 230))
    # yellow bolt + vertical "ENERGY"
    bolt = [(170, 150), (110, 290), (155, 290), (125, 420), (200, 250), (152, 250), (185, 150)]
    d.polygon([(x * S, y * S) for x, y in bolt], fill=(250, 200, 40))
    f = _font(FONT_DIR + "/Montserrat[wght].ttf", 46 * S, 800)
    txt = Image.new("RGBA", (420 * S, 70 * S), (0, 0, 0, 0))
    ImageDraw.Draw(txt).text((210 * S, 35 * S), "ENERGY", font=f, fill=(240, 50, 60), anchor="mm")
    txt = txt.rotate(90, expand=True)
    img.alpha_composite(txt, (40 * S, 90 * S))
    return img


def put_can(img, x, y, scale=1.0, rot=0.0):
    c = can_img()
    if scale != 1.0:
        c = c.resize((int(c.width * scale), int(c.height * scale)), Image.LANCZOS)
    if rot:
        c = c.rotate(rot, Image.BICUBIC, expand=True)
    sh = Image.new("RGBA", c.size, (0, 0, 0, 0))
    sh.putalpha(c.split()[3].point(lambda p: int(p * 0.45)))
    sh = sh.filter(ImageFilter.GaussianBlur(16))
    img.paste(sh, (int(x * S - c.width / 2 + 20), int(y * S - c.height / 2 + 30)), sh)
    img.paste(c, (int(x * S - c.width / 2), int(y * S - c.height / 2)), c)


def wing(c, x, y, s, side, col=(250, 248, 240), alpha=255):
    for k in range(7):
        ln = (180 - k * 16) * s
        ang = math.radians(-35 + k * 15)
        bx, by = x + side * 20 * s, y + k * 22 * s
        tip = (bx + side * math.cos(ang) * ln, by - math.sin(ang) * ln * 0.6)
        c.poly([(bx, by - 16 * s), tip, (bx, by + 18 * s)], col + (alpha,))


def gavel(c, cx, cy, ang, s=1.0):
    """Wooden gavel rotated by ang (radians) around the handle end (cx, cy)."""
    def pt(px, py):
        return (cx + (px * math.cos(ang) - py * math.sin(ang)) * s, cy + (px * math.sin(ang) + py * math.cos(ang)) * s)
    c.poly([pt(0, -14), pt(330, -14), pt(330, 14), pt(0, 14)], (120, 74, 40))
    c.poly([pt(0, -8), pt(330, -8), pt(330, -2), pt(0, -2)], (160, 104, 62))
    c.poly([pt(300, -70), pt(420, -70), pt(420, 70), pt(300, 70)], (96, 56, 30))
    c.poly([pt(300, -60), pt(420, -60), pt(420, -38), pt(300, -38)], (150, 98, 58))
    for q in (318, 402):
        c.poly([pt(q - 6, -72), pt(q + 6, -72), pt(q + 6, 72), pt(q - 6, 72)], (200, 160, 80))


def money_stack(c, x, y, n, s=1.0):
    for i in range(n):
        yy = y - i * 9 * s
        c.rect(x - 120 * s, yy - 50 * s, x + 120 * s, yy, (70, 120, 70), r=4 * s)
        c.rect(x - 112 * s, yy - 44 * s, x + 112 * s, yy - 6 * s, (110, 160, 105), r=3 * s)
        c.rect(x - 20 * s, yy - 52 * s, x + 20 * s, yy + 2 * s, (230, 222, 196))
    c.text(x, y - n * 9 * s - 24 * s, "$100", F("ms", int(30 * s), 800), (40, 80, 45))


def big_text(d, x, y, txt, size, col=WHITE, stroke=10, font=None):
    d.text((x, y), txt, font=font or MS(size, 800), fill=col, anchor="mm", stroke_width=stroke, stroke_fill=(0, 0, 0))


def pop(t, t0, dur=0.35):
    """Overshoot scale-in for punchy text."""
    k = clamp((t - t0) / dur)
    return 0 if k <= 0 else 1 + 0.25 * math.sin(k * math.pi) * (1 - k) + (1 - k) * -0.3


# ================================================================ scenes
def sc_hook(t, lt, dur):
    img = bokeh_bg(1, (180, 40, 40)).copy()
    c = C(img)
    slam = 0.35
    ang = lerp(-1.2, 0.05, eout(lt / slam)) if lt < slam + 0.05 else 0.05 - 0.04 * math.exp(-(lt - slam) * 6)
    c.rect(200, 1290, 880, 1340, (60, 36, 22), r=10)
    c.ell(540, 1290, 210, 36, (80, 50, 30))
    gavel(c, 160, 1060, ang, 1.6)
    frame = cam(img, lerp(1.08, 1.0, ease(lt / dur)), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = pop(t, v(0.0) + 0.05)
        if k:
            big_text(d, 540, 420, "HE SUED", int(120 * max(k, 0.7)), WHITE, 10)
            big_text(d, 540, 560, "RED BULL", int(150 * max(k, 0.7)), (240, 60, 60), 12)
        amt = int(13_000_000 * eout((lt - 0.4) / 1.8))
        if lt > 0.4:
            big_text(d, 540, 760, f"${amt:,}", 104, YEL, 10)
    return frame, txt


def sc_fridge(t, lt, dur):
    img = to_img(grad((16, 20, 30), (6, 6, 10)))
    c = C(img)
    c.rect(120, 220, 960, 1700, (30, 34, 44), r=24)
    c.rect(150, 250, 930, 1670, (190, 220, 240), r=10)
    img = light(img, lambda d, k: d.rectangle([150 * k, 250 * k, 930 * k, 1670 * k], fill=(60, 80, 100)), 60)
    c = C(img)
    for row in range(3):
        y = 560 + row * 430
        c.rect(150, y + 150, 930, y + 170, (150, 170, 190))
        for i in range(5):
            if row == 1 and i == 2 and lt > 1.6:
                continue
            put_can(img, 230 + i * 155, y, 0.42)
    c = C(img)
    c.rect(150, 250, 930, 1670, (200, 230, 255, 30))
    if lt > 1.0:
        k = eout((lt - 1.0) / 0.7)
        hx, hy = lerp(1200, 560, k), lerp(1100, 990, k)
        if lt > 1.6:
            put_can(img, hx - 20, hy - 40, 0.42)
            c = C(img)
        c.ell(hx + 60, hy, 70, 52, (225, 180, 150))
        c.rect(hx + 60, hy - 40, hx + 600, hy + 40, (225, 180, 150))
    frame = cam(img, lerp(1.0, 1.12, ease(lt / dur)), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = pop(t, word_t("advertising", 3))
        if k:
            big_text(d, 540, 300, "YOU BELIEVE THE AD", int(72 * max(k, 0.7)), YEL, 8)
    return frame, txt


def sc_slogan(t, lt, dur):
    img = bokeh_bg(2, (40, 80, 200)).copy()
    img = light(img, lambda d, k: d.ellipse([240 * k, 520 * k, 840 * k, 1300 * k], fill=(120, 140, 220)), 90)
    c = C(img)
    wk = eout((t - word_t("wings", 7)) / 0.6)
    if wk > 0:
        for side in (-1, 1):
            wing(c, 540 + side * 60, 760, lerp(0.4, 1.6, wk), side, (255, 250, 235), int(255 * wk))
    dust(c, t, 60, seed=4, col=(255, 240, 200))
    put_can(img, 540, 960, 1.15, math.sin(t * 1.5) * 3)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = eout((t - word_t("gives", 7) + 0.3) / 0.4)
        if k > 0:
            big_text(d, 540, 340, "“GIVES YOU", 86, WHITE, 8, PF(96, 900))
            big_text(d, 540, 470, "WINGS!”", 130, YEL, 10, PF(140, 900))
    return frame, txt


def question_marks(c, t, cx, cy):
    for i in range(5):
        ph = t * 1.2 + i * 1.3
        x = cx + math.cos(ph) * 260
        y = cy - 200 + math.sin(ph * 0.8) * 120 - i * 40
        c.text(x, y, "?", F("pf", 110, 900), (255, 210, 40, 200))


def sc_man(t, lt, dur):
    img = bokeh_bg(3, (200, 140, 60)).copy()
    img = light(img, lambda d, k: d.polygon([(400 * k, 0), (680 * k, 0), (860 * k, 1920 * k), (220 * k, 1920 * k)], fill=(90, 80, 60)), 60)
    c = C(img)
    c.rect(0, 1600, W, H, (14, 12, 14))
    turn = math.sin(lt * 2.4) if t > word_t("where", 11) else 0.0
    figure(c, 540, 1640, 1150, INK, rim=(230, 190, 120), rim_dx=6, turn=turn)
    put_can(img, 690, 1180, 0.35, -10)
    c = C(img)
    if t > word_t("where", 11):
        question_marks(c, t, 540, 700)
    frame = cam(img, lerp(1.0, 1.12, ease(lt / dur)), 540, 1050)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = pop(t, word_t("where", 11))
        if k:
            big_text(d, 540, 330, "“WHERE ARE", int(96 * max(k, 0.7)), WHITE, 9)
            big_text(d, 540, 450, "MY WINGS?”", int(120 * max(k, 0.7)), YEL, 10)
        elif lt > 0.2:
            big_text(d, 540, 360, "ONE MAN ASKED…", 70, WHITE, 8)
    return frame, txt


def sc_joke(t, lt, dur):
    img = bokeh_bg(4, (220, 180, 60)).copy()
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im)
        serious = t >= word_t("listen", 17)
        if not serious:
            k = pop(t, v(16.9) + 0.1)
            big_text(d, 540, 820, "A JOKE?", int(150 * max(k, 0.7)), YEL, 12)
            for i in range(6):
                a = t * 3 + i
                d.text((540 + math.cos(a) * 330, 820 + math.sin(a * 1.3) * 230), "HA", font=MS(60, 800),
                       fill=(255, 255, 255), anchor="mm")
        else:
            k = pop(t, word_t("listen", 17))
            big_text(d, 540, 820, "LISTEN", int(150 * max(k, 0.7)), (240, 60, 60), 12)
            big_text(d, 540, 960, "TO THIS", 90, WHITE, 9)
    return frame, txt


def legal_doc(c, x, y, s, t, hl, filed):
    w, h = 720 * s, 980 * s
    c.rect(x - w / 2 + 16, y - h / 2 + 20, x + w / 2 + 16, y + h / 2 + 20, (0, 0, 0, 110))
    c.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, (246, 244, 236))
    c.text(x, y - h / 2 + 70 * s, "UNITED STATES DISTRICT COURT", F("ms", int(30 * s), 800), (30, 30, 36))
    c.line([(x - 280 * s, y - h / 2 + 110 * s), (x + 280 * s, y - h / 2 + 110 * s)], (60, 60, 70), 2)
    c.text(x - 150 * s, y - h / 2 + 170 * s, "Plaintiff,", F("pfi", int(34 * s), 500), (40, 40, 46))
    c.text(x, y - h / 2 + 230 * s, "v.", F("pf", int(40 * s), 800), (40, 40, 46))
    c.text(x + 70 * s, y - h / 2 + 290 * s, "RED BULL", F("pf", int(54 * s), 900), (30, 30, 36))
    c.text(x, y - h / 2 + 380 * s, "CLASS ACTION COMPLAINT", F("ms", int(36 * s), 800), (170, 30, 30))
    for i in range(10):
        yy = y - h / 2 + 470 * s + i * 44 * s
        ln = 560 * s if i % 4 != 3 else 300 * s
        if i in (3, 4, 5) and hl > 0:
            c.rect(x - 290 * s, yy - 16 * s, x - 290 * s + 580 * s * hl, yy + 16 * s, (255, 230, 60, 170))
        c.line([(x - 280 * s, yy), (x - 280 * s + ln, yy)], (150, 150, 160), 6 * s)
    if filed > 0:
        sc = lerp(1.6, 1.0, eout(filed))
        a = int(230 * min(1, filed * 3))
        c.rect(x + 40 * sc - 160 * sc, y + 330 * s - 60 * sc, x + 40 * sc + 160 * sc, y + 330 * s + 60 * sc, None,
               outline=(190, 30, 40, a), w=8)
        c.text(x + 40 * sc, y + 330 * s, "FILED", F("ms", int(70 * sc), 800), (190, 30, 40, a))


def sc_lawsuit(t, lt, dur):
    img = bokeh_bg(5, (200, 170, 120)).copy()
    c = C(img)
    k = eout(lt / 0.6)
    hl = ease((t - word_t("suggested", 19)) / 2.0)
    filed = clamp((t - word_t("filed", 19) - 0.2) / 0.3)
    legal_doc(c, 540, lerp(1700, 1000, k), 1.0, t, hl, filed)
    frame = cam(img, lerp(1.0, 1.15, ease(lt / dur)), 540, lerp(980, 1080, ease(lt / dur)))

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([40, 230, 420, 300], 35, fill=(200, 30, 40, 230))
        d.text((230, 266), "REAL CASE · USA", font=MS(32, 800), fill=WHITE, anchor="mm")
        if t > word_t("weren’t", 19) - 0.3:
            big_text(d, 540, 1720, "“NOT SUPPORTED”", 84, YEL, 9)
    return frame, txt


def sc_classaction(t, lt, dur):
    img = bokeh_bg(6, (120, 120, 200)).copy()
    c = C(img)
    grow = ease((t - word_t("class-action", 29)) / 1.5)
    n = 1 + int(grow * 40)
    r = np.random.default_rng(3)
    spots = [(540, 1500, 700)] + [(r.uniform(60, 1020), r.uniform(1150, 1750), r.uniform(260, 520)) for _ in range(40)]
    spots = sorted(spots[:n], key=lambda p: p[1])
    for x, y, h in spots:
        figure(c, x, y, h * (0.6 + 0.4 * (y - 1100) / 700), INK, rim=(140, 150, 200), rim_dx=3)
    frame = cam(img, lerp(1.0, 1.05, lt / dur), 540, 1000)

    def txt(im):
        d = ImageDraw.Draw(im)
        if t < word_t("case", 29):
            k = pop(t, v(29.8) + 0.1)
            big_text(d, 540, 520, "THINGS GET", 80, WHITE, 8)
            big_text(d, 540, 640, "INTERESTING", int(110 * max(k, 0.7)), YEL, 10)
        else:
            k = pop(t, word_t("class-action", 29))
            if k:
                big_text(d, 540, 520, "CLASS ACTION", int(120 * max(k, 0.7)), (240, 60, 60), 11)
                big_text(d, 540, 650, "ONE CASE FOR MANY BUYERS", 54, WHITE, 7)
    return frame, txt


def sc_millions(t, lt, dur):
    img = bokeh_bg(7, (60, 200, 90)).copy()
    c = C(img)
    c.rect(0, 1450, W, H, (24, 18, 14))
    stacks = [(240, 1460), (420, 1480), (600, 1470), (780, 1490), (330, 1600), (690, 1610), (510, 1640)]
    grow = eout(lt / 2.2)
    for i, (x, y) in enumerate(stacks):
        n = int(lerp(0, 22 + (i % 3) * 6, clamp(grow * 1.3 - i * 0.05)))
        if n:
            money_stack(c, x, y, n, 0.9)
    can_k = eout((t - word_t("energy", 37)) / 0.5)
    if can_k > 0:
        put_can(img, 900, lerp(2100, 1300, can_k), 0.55, 12)
        c = C(img)
    slam = word_t("settlement", 35)
    ang = lerp(-1.1, 0.05, eout((t - slam + 0.3) / 0.3)) if t > slam - 0.3 else -1.1
    if t > slam - 0.3:
        gavel(c, 80, 820, ang, 0.9)
    frame = cam(img, lerp(1.05, 1.0, ease(lt / dur)), 540, 1100)

    def txt(im):
        d = ImageDraw.Draw(im)
        amt = int(13_000_000 * eout((lt - 0.2) / 2.0))
        big_text(d, 540, 420, f"${amt:,}", 118, YEL, 11)
        k = pop(t, slam)
        if k:
            big_text(d, 540, 560, "SETTLEMENT", int(86 * max(k, 0.7)), WHITE, 8)
        if t > word_t("over", 38):
            big_text(d, 540, 700, "…OVER AN AD!", 72, (240, 60, 60), 8)
    return frame, txt


def sc_catch(t, lt, dur):
    img = bokeh_bg(8, (200, 60, 60)).copy()
    c = C(img)
    split = ease((t - word_t("eligible", 42)) / 1.6)
    cx, cy = 540, 980
    if split < 1:
        c.rect(cx - 220 * (1 - split), cy - 120 * (1 - split), cx + 220 * (1 - split), cy + 120 * (1 - split), (70, 120, 70), r=20)
    r = np.random.default_rng(9)
    for i in range(60):
        a = r.uniform(0, 2 * math.pi)
        dist = r.uniform(150, 520) * split
        x, y = cx + math.cos(a) * dist, cy + math.sin(a) * dist * 1.3
        c.ell(x, y, 22, 22, (230, 190, 70))
        c.ell(x, y, 15, 15, (250, 214, 100))
    man = clamp((t - word_t("one", 46)) / 0.5)
    if man > 0:
        figure(c, 540, 1750, 520, INK, rim=(240, 120, 120), rim_dx=4)
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = pop(t, v(42.4) + 0.1)
        if k:
            big_text(d, 540, 330, "THE CATCH", int(130 * max(k, 0.7)), (240, 60, 60), 12)
        if split < 0.3:
            big_text(d, 540, 980, "$13M", 120, WHITE, 10)
        if split > 0.6:
            big_text(d, 540, 520, "SPLIT AMONG CLAIMANTS", 56, WHITE, 7)
            big_text(d, 540, 600, "≈ $10 cash or $15 in product", 46, YEL, 6)
        if man > 0:
            big_text(d, 540, 1150, "NOT $13M FOR ONE MAN", 60, YEL, 7)
    return frame, txt


def sc_nowrong(t, lt, dur):
    img = bokeh_bg(5, (200, 170, 120)).copy()
    c = C(img)
    legal_doc(c, 540, 1000, 1.0, t, 1.0, 0)
    s = clamp((t - word_t("wrongdoing", 49) - 0.1) / 0.25)
    if s > 0:
        sc = lerp(1.8, 1.0, eout(s))
        c.rect(540 - 330 * sc, 1060 - 110 * sc, 540 + 330 * sc, 1060 + 110 * sc, (255, 255, 255, 60))
        c.rect(540 - 330 * sc, 1060 - 110 * sc, 540 + 330 * sc, 1060 + 110 * sc, None, outline=(190, 30, 40), w=12)
        c.text(540, 1020, "NO WRONGDOING", F("ms", int(62 * sc), 800), (190, 30, 40))
        c.text(540, 1100, "ADMITTED", F("ms", int(62 * sc), 800), (190, 30, 40))
    return cam(img, lerp(1.05, 1.15, lt / dur), 540, 1000, rot=-3), None


HEADLINES = [("MAN SUES OVER", "MISSING “WINGS”"), ("$13 MILLION", "ENERGY DRINK DEAL"), ("THE LAWSUIT", "THAT WENT VIRAL")]


def newspaper(c, x, y, rot, s, h1, h2, show_text=True):
    pts = [(-330, -420), (330, -420), (330, 420), (-330, 420)]
    a = math.radians(rot)

    def P(px, py):
        return (x + (px * math.cos(a) - py * math.sin(a)) * s, y + (px * math.sin(a) + py * math.cos(a)) * s)
    c.poly([P(px + 14, py + 18) for px, py in pts], (0, 0, 0, 110))
    c.poly([P(px, py) for px, py in pts], (236, 232, 220))
    if not show_text:
        return
    c.text(*P(0, -350), "THE DAILY TIMES", F("pf", int(54 * s), 900), (30, 30, 34))
    c.line([P(-300, -300), P(300, -300)], (60, 60, 66), 3)
    c.text(*P(0, -220), h1, F("ms", int(52 * s), 800), (30, 30, 34))
    c.text(*P(0, -150), h2, F("ms", int(52 * s), 800), (190, 30, 40))
    c.poly([P(-300, -90), P(0, -90), P(0, 160), P(-300, 160)], (180, 176, 168))
    for i in range(9):
        c.line([P(20, -80 + i * 26), P(300, -80 + i * 26)], (170, 166, 158), 5 * s)
    for i in range(7):
        c.line([P(-300, 190 + i * 30), P(300, 190 + i * 30)], (170, 166, 158), 5 * s)


def sc_legend(t, lt, dur):
    img = bokeh_bg(9, (200, 180, 120)).copy()
    c = C(img)
    for i, (h1, h2) in enumerate(HEADLINES):
        k = eout((lt - i * 0.7) / 0.6)
        if k <= 0:
            continue
        rot = lerp(180 + i * 40, 0, k)
        newspaper(c, [400, 680, 540][i], [640, 960, 1290][i], rot, lerp(2.0, 0.72, k), h1, h2, k > 0.97)
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im)
        k = pop(t, word_t("legendary", 54))
        if k:
            big_text(d, 540, 330, "LEGENDARY", int(130 * max(k, 0.7)), YEL, 11)
    return frame, txt


def sc_david(t, lt, dur):
    img = to_img(grad((20, 24, 40), (6, 6, 10)) + radial(540, 500, 700)[:, :, None] * np.array([40, 40, 70], np.float32))
    c = C(img)
    x0, x1 = 300, 860
    c.rect(x0, 200, x1, 1600, (28, 34, 54))
    for y in range(240, 1580, 60):
        for x in range(x0 + 30, x1 - 30, 56):
            c.rect(x, y, x + 34, y + 36, (60, 80, 120) if (x + y) % 7 else (200, 180, 120))
    c.rect(x0 - 40, 1600, x1 + 40, 1660, (20, 22, 30))
    c.text(580, 640, "CORP.", F("ms", 80, 800), (200, 200, 220))
    c.rect(0, 1660, W, H, (12, 12, 16))
    figure(c, 150, 1700, 260, INK, rim=(250, 210, 120), rim_dx=3)
    img = light(img, lambda d, k: d.ellipse([60 * k, 1380 * k, 240 * k, 1720 * k], fill=(120, 100, 60)), 40)
    frame = cam(img, lerp(1.25, 1.0, ease(lt / dur)), lerp(260, 540, ease(lt / dur)), lerp(1500, 1000, ease(lt / dur)))

    def txt(im):
        d = ImageDraw.Draw(im)
        k = pop(t, word_t("one", 58))
        if k:
            big_text(d, 540, 330, "ONE PERSON", int(110 * max(k, 0.7)), YEL, 10)
        k2 = pop(t, word_t("massive", 58))
        if k2:
            big_text(d, 540, 450, "VS A GIANT", int(100 * max(k2, 0.7)), WHITE, 10)
    return frame, txt


CLAIMS = [("100% NATURAL*", (200, 80, 60)), ("CLINICALLY PROVEN*", (60, 110, 190)), ("LOSE 10 LBS FAST*", (200, 150, 40)),
          ("#1 DOCTOR RECOMMENDED*", (90, 150, 90))]


def sc_promises(t, lt, dur):
    img = bokeh_bg(10, (120, 120, 200)).copy()
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        for i, (claim, col) in enumerate(CLAIMS):
            k = eout((lt - i * 0.45) / 0.4)
            if k <= 0:
                continue
            y = 420 + i * 260
            x = lerp(1400, 540, k) + (i % 2) * 40 - 20
            d.rounded_rectangle([x - 430, y - 90, x + 430, y + 90], 24, fill=col + (230,))
            d.text((x, y), claim, font=MS(60, 800), fill=WHITE, anchor="mm")
        q = pop(t, word_t("questioning", 65))
        if q:
            big_text(d, 540, 1680, "DO YOU", 80, WHITE, 8)
            big_text(d, 540, 1800, "QUESTION THEM?", int(100 * max(q, 0.7)), YEL, 10)
    return frame, txt


def sc_cta(t, lt, dur):
    img = bokeh_bg(11, (220, 50, 50)).copy()
    frame = cam(img, 1.0)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = pop(t, v(69.8) + 0.1)
        if k:
            big_text(d, 540, 520, "WANT THE", 80, WHITE, 8)
            big_text(d, 540, 640, "FULL STORY?", int(120 * max(k, 0.7)), YEL, 11)
        k2 = eout((t - word_t("watch", 69)) / 0.4)
        if k2 > 0:
            d.text((540, 820), "FULL VIDEO ON MY CHANNEL", font=MS(48, 800), fill=WHITE + (int(255 * k2),), anchor="mm")
        s_t = word_t("subscribe", 73)
        k3 = eout((t - s_t + 0.6) / 0.4)
        if k3 > 0:
            pressed = t > s_t + 0.4
            bw = 560 * (0.92 if pressed and t < s_t + 0.6 else 1.0)
            col = (90, 90, 96) if pressed else (220, 30, 30)
            y = 1180
            d.rounded_rectangle([540 - bw / 2, y - 80, 540 + bw / 2, y + 80], 80, fill=col + (int(255 * k3),))
            d.text((540, y), "SUBSCRIBED" if pressed else "SUBSCRIBE", font=MS(64, 800), fill=WHITE, anchor="mm")
            # bell
            bx, by = 540 + bw / 2 + 90, y
            d.ellipse([bx - 50, by - 50, bx + 50, by + 50], fill=(250, 250, 250, int(255 * k3)))
            sw = math.sin(t * 20) * 10 if pressed else 0
            d.polygon([(bx - 24 + sw, by + 18), (bx + 24 + sw, by + 18), (bx + 16 + sw, by - 14), (bx + sw, by - 26),
                       (bx - 16 + sw, by - 14)], fill=(30, 30, 30))
            d.ellipse([bx - 6 + sw, by + 18, bx + 6 + sw, by + 30], fill=(30, 30, 30))
            # pointer
            if not pressed:
                px, py = lerp(900, 600, ease((t - s_t + 0.2) / 0.5)), lerp(1500, 1210, ease((t - s_t + 0.2) / 0.5))
                d.polygon([(px, py), (px + 22, py + 70), (px + 36, py + 46), (px + 70, py + 60)], fill=WHITE)
        k4 = eout((t - word_t("crazy", 73)) / 0.4)
        if k4 > 0:
            big_text(d, 540, 1480, "MORE CRAZY", 70, WHITE, 8)
            big_text(d, 540, 1580, "REAL-LIFE STORIES", 70, YEL, 8)
    return frame, txt


# ================================================================ edit
SC = {name: globals()["sc_" + name] for _, name in SCENES}
SHOTS = []
for i, (a, name) in enumerate(SCENES):
    b = SCENES[i + 1][0] if i + 1 < len(SCENES) else None
    SHOTS.append((max(0.0, v(a)), v(b) if b is not None else DURATION, name))

SHAKES += [(v(0.0) + 0.4, 34, 0.35), (word_t("settlement", 35), 26, 0.35), (word_t("wrongdoing", 49) + 0.1, 22, 0.3),
           (word_t("where", 11), 10, 0.4), (word_t("listen", 17), 14, 0.3), (word_t("filed", 19) + 0.2, 14, 0.3)]

# scenes whose own big text already says the line — captions hidden there
NO_CAPTIONS = {"hook", "cta"}


# ================================================================ post & captions
@lru_cache(None)
def vignette():
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.7)) ** 2 + ((y - H / 2) / (H * 0.7)) ** 2)
    return np.clip(1.1 - d ** 2.2, 0.35, 1).astype(np.float32)[:, :, None]


@lru_cache(None)
def grain_bank():
    r = np.random.default_rng(9)
    return [r.normal(0, 1, (H // 2, W // 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:, :, None] for _ in range(6)]


def post(img, t):
    a = np.asarray(img, np.float32) * vignette()
    lum = a.mean(axis=2, keepdims=True) / 255
    a = a + (1 - lum) * np.array([-3, 0, 6], np.float32) + lum * np.array([6, 3, -4], np.float32)
    a = a + grain_bank()[int(t * FPS) % 6] * 4
    if t < 0.12:
        a *= t / 0.12
    return to_img(a)


@lru_cache(None)
def chunks():
    """Script words grouped into 1–3-word caption chunks, split at punctuation."""
    out, cur = [], []
    for s, e, w in words():
        cur.append((s, e, w))
        if w[-1] in ".,!?…:;”" or len(cur) == 3:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def draw_captions(im, t):
    cs = chunks()
    for i, ch in enumerate(cs):
        start = ch[0][0]
        end = cs[i + 1][0][0] if i + 1 < len(cs) else ch[-1][1] + 0.4
        end = min(end, ch[-1][1] + 0.6)
        if not (start <= t < end):
            continue
        d = ImageDraw.Draw(im)
        f = MS(76, 800)
        toks = [w.upper().strip("“”\"") for _, _, w in ch]
        widths = [d.textlength(tk, font=f) for tk in toks]
        gap = 22
        total = sum(widths) + gap * (len(toks) - 1)
        scale = min(1.0, 960 / total)
        if scale < 1.0:
            f = MS(int(76 * scale), 800)
            widths = [d.textlength(tk, font=f) for tk in toks]
            total = sum(widths) + gap * (len(toks) - 1)
        x = (W - total) / 2
        y = 1500 - 10 * max(0.0, 1 - (t - start) / 0.12)
        for (s, e, _), tk, wd in zip(ch, toks, widths):
            active = s <= t < max(e, s + 0.2)
            d.text((x, y), tk, font=f, fill=YEL if active else WHITE, anchor="lm", stroke_width=9, stroke_fill=(0, 0, 0))
            x += wd + gap
        return


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
    if CAPTIONS and name not in NO_CAPTIONS:
        draw_captions(img, t)
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
                print(f"  frame {n}/{total}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

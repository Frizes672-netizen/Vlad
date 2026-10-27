"""Рендер серии 4 (финал) «Апокалипсис только начался» в плоском мультяшном стиле (движок серий 2–3).

Мир рисуется в двукратном разрешении (сглаживание), камера вырезает кадр,
поверх кладутся плашки HUD: день/время, глубина, зомби рядом, локация,
бейджи с именами персонажей.
Запуск: python3 render.py build/video.mp4 [--preview СЕКУНДЫ ...]
"""
import math
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from timeline import (CAPTIONS, CLOCK_START, CLOCK_STEP, COUNTDOWN_FROM, COUNTDOWN_SECONDS, COUNTDOWN_TO,
                      DAY_START, DEPTH, DURATION, EV, FINAL_TEXT, FPS, H, LOCATIONS,
                      VOICE_OFFSET, W, ZOMBIES_NEAR, v)

S = 2                       # масштаб мира относительно кадра
WW, WH = W * S, H * S
FB = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FX = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# реплики (для анимации рта)
TALK = {
    "vlad": [(v(9.52), v(10.6)), (v(42.56), v(44.34)), (v(76.3), v(78.16)), (v(102.66), v(103.9)),
             (v(145.84), v(146.9)), (v(154.56), v(156.8))],
    "double": [(v(4.74), v(6.88)), (v(12.04), v(14.3)), (v(45.84), v(47.68)), (v(98.48), v(100.78)),
               (v(105.56), v(107.98)), (v(127.62), v(130.92))],
    "vika": [(v(66.48), v(68.46)), (v(71.1), v(74.8)), (v(79.76), v(85.08)), (v(110.06), v(113.52))],
    "sci": [(v(20.0), v(26.42)), (v(35.02), v(40.32))],
}

SKIN = (247, 214, 180)
VLAD = dict(skin=SKIN, hair="spiky", hair_col=(30, 28, 34), top=(24, 170, 150), pants=(28, 34, 58),
            shoes=(245, 245, 245), extra={"hood"})
VIKA = dict(skin=SKIN, hair="long", hair_col=(52, 30, 26), top=(232, 64, 52), pants=(30, 36, 64),
            shoes=(245, 245, 245), extra=set())
RESCUER = dict(skin=(196, 204, 198), hair="helmet", hair_col=(236, 112, 30), top=(240, 120, 36),
               pants=(40, 44, 54), shoes=(30, 30, 30), extra={"stripes"})
SCI = dict(skin=SKIN, hair="bald", hair_col=(225, 225, 225), top=(245, 245, 245), pants=(70, 74, 90),
           shoes=(40, 40, 40), extra={"coat", "glasses"})


# ================================================================ утилиты
@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(x, y, k)) for x, y in zip(c1, c2))


def interp(points, t):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.interp(t, xs, ys))


def talking(who, t):
    for a, b in TALK.get(who, ()):
        if a <= t < b:
            return 0.35 + 0.65 * abs(math.sin(t * 13.0 + a))
    return 0.0


OY = [0]  # вертикальный сдвиг сцены вниз (место под плашки HUD)


class C:
    """Холст в «дизайнерских» координатах 1080×1920, рисует в мире ×S
    со сдвигом сцены вниз на OY."""

    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img, "RGBA")

    def rect(self, x0, y0, x1, y1, fill, r=0):
        y0, y1 = min(y0, y1), max(y0, y1)
        y0 = y0 - OY[0] if y0 <= 0 else y0  # фон, начинающийся у верхнего края, тянется до края
        b = [min(x0, x1) * S, (y0 + OY[0]) * S, max(x0, x1) * S, (y1 + OY[0]) * S]
        if r:
            self.d.rounded_rectangle(b, r * S, fill=fill)
        else:
            self.d.rectangle(b, fill=fill)

    def ell(self, cx, cy, rx, ry, fill, outline=None, w=0):
        cy += OY[0]
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=fill,
                       outline=outline, width=int(w * S))

    def poly(self, pts, fill):
        self.d.polygon([(x * S, (y + OY[0]) * S) for x, y in pts], fill=fill)

    def line(self, pts, fill, w, round_=True):
        pts = [(x, y - OY[0] if y <= 0 else y) for x, y in pts]
        self.d.line([(x * S, (y + OY[0]) * S) for x, y in pts], fill=fill, width=max(1, int(w * S)), joint="curve")
        if round_:
            for x, y in (pts[0], pts[-1]):
                self.ell(x, y, w / 2, w / 2, fill)

    def chord(self, cx, cy, r, a0, a1, fill):
        cy += OY[0]
        self.d.chord([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], a0, a1, fill=fill)

    def arc(self, cx, cy, rx, ry, a0, a1, fill, w):
        cy += OY[0]
        self.d.arc([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], a0, a1, fill=fill,
                   width=max(1, int(w * S)))

    def text(self, x, y, txt, path, size, fill, anchor="mm", stroke=0, stroke_fill=(0, 0, 0)):
        self.d.text((x * S, (y + OY[0]) * S), txt, font=font(path, int(size * S)), fill=fill, anchor=anchor,
                    stroke_width=int(stroke * S), stroke_fill=stroke_fill)


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ================================================================ камера
SHAKES = [
    (EV["grab"], 8, 0.3), (EV["fall"], 44, 0.6), (EV["fall"] + 1.2, 20, 0.5), (EV["fall"] + 2.6, 16, 0.5),
    (EV["pipe"] + 0.05, 30, 0.35), (EV["shot"] + 0.4, 20, 0.3), (EV["alarm"], 14, 0.5), (EV["press"], 18, 0.4),
    (EV["doors"], 12, 0.6), (EV["doors"] + 8.0, 34, 0.5), (EV["boom"], 70, 1.4), (v(152.0), 10, 0.6),
    (v(170.6), 12, 0.6), (EV["final"], 14, 0.6),
]
CUR_T = [0.0]


def shake(t):
    dx = dy = 0.0
    for i, (te, a, dec) in enumerate(SHAKES):
        if t >= te:
            k = a * math.exp(-(t - te) / dec)
            dx += k * math.sin(t * 61 + i * 1.7)
            dy += k * math.cos(t * 53 + i * 2.3)
    return dx, dy


def cam(world, zoom=1.0, cx=540, cy=960):
    """Вырезает кадр; cx, cy — центр в дизайнерских координатах.
    Возвращает кадр W×H и функцию перевода координат мира в кадр."""
    dx, dy = shake(CUR_T[0])
    if abs(dx) + abs(dy) > 0.5:
        zoom = max(zoom, 1.0 + (abs(dx) + abs(dy)) * 2.2 / W + 0.02)
    ww, hh = W / zoom, H / zoom
    oy = OY[0]
    cy += oy
    x0 = clamp(cx - ww / 2 + dx / zoom, 0, W - ww)
    y0 = clamp(cy - hh / 2 + dy / zoom, 0, H - hh)
    box = (x0 * S, y0 * S, (x0 + ww) * S, (y0 + hh) * S)
    frame = world.resize((W, H), Image.BILINEAR, box=box, reducing_gap=1.5)

    def f(x, y):
        return (x - x0) * zoom, (y + oy - y0) * zoom

    return frame, f


# ================================================================ персонаж
def char(c, x, y, h, sp, t=0.0, run=0.0, talk=0.0, expr="neutral", look=(0.0, 0.0),
         arms=(8, 8), kneel=0.0, bow=0.0, gun=None, phone=False, zombie=False, mouth=None):
    """Персонаж «как в референсе»: круглая голова, прямоугольное тело.
    arms — углы рук в градусах (0 вниз, 90 в сторону, 180 вверх), левая/правая.
    Возвращает точку над головой (для бейджа)."""
    u = h / 100
    ph = t * 15
    bob = abs(math.sin(ph)) * 2.5 * u * run
    drop = kneel * 20 * u
    by = y - bob + drop
    skin, top = sp["skin"], sp["top"]
    ex = sp["extra"]
    # тень
    c.ell(x, y + 1 * u, 22 * u, 3.2 * u, (0, 0, 0, 60))
    # ноги
    for s in (-1, 1):
        lx = x + s * 6.5 * u
        if kneel > 0.05:
            c.rect(lx - 5 * u, by - 32 * u, lx + 5 * u, y - 3 * u, sp["pants"], r=2 * u)
            c.rect(lx - 6 * u, y - 6 * u, lx + 6 * u + s * 4 * u, y, sp["pants"], r=3 * u)
            continue
        lift = max(0.0, math.sin(ph + (0 if s < 0 else math.pi))) * 5 * u * run
        c.rect(lx - 5 * u, by - 32 * u, lx + 5 * u, y - 4 * u - lift, sp["pants"], r=1.5 * u)
        c.rect(lx - 6.5 * u + min(0, s) * 1 * u, y - 7 * u - lift, lx + 6.5 * u + max(0, s) * 1 * u, y - lift,
               sp["shoes"], r=3 * u)
    # тело
    bt, bb = by - 63 * u, by - 29 * u
    if "coat" in ex:
        c.rect(x - 16 * u, bt, x + 16 * u, bb + 9 * u, top, r=5 * u)
        c.poly([(x - 4 * u, bt), (x + 4 * u, bt), (x + 2 * u, bb + 2 * u), (x - 2 * u, bb + 2 * u)], (150, 190, 220))
    else:
        c.rect(x - 15 * u, bt, x + 15 * u, bb, top, r=5 * u)
    if "stripes" in ex:
        c.rect(x - 15 * u, by - 43 * u, x + 15 * u, by - 39 * u, (225, 230, 225))
    if "hood" in ex:
        c.rect(x - 11 * u, bt - 3 * u, x + 11 * u, bt + 5 * u, tuple(int(q * 0.85) for q in top), r=4 * u)
        for s in (-1, 1):
            c.line([(x + s * 4 * u, bt + 3 * u), (x + s * 4.5 * u, bt + 13 * u)], (245, 245, 245), 1.6 * u)
        c.rect(x - 8 * u, by - 46 * u, x + 8 * u, by - 37 * u, tuple(int(q * 0.85) for q in top), r=3 * u)
    if zombie:
        for k in range(4):
            px = x - 13 * u + k * 8 * u
            c.poly([(px, bb), (px + 4 * u, bb - 6 * u), (px + 8 * u, bb)], (0, 0, 0, 90))
        c.ell(x - 6 * u, by - 50 * u, 3 * u, 2.4 * u, (140, 20, 20))
        c.ell(x + 7 * u, by - 41 * u, 2.2 * u, 3 * u, (140, 20, 20))
    # руки
    hands = []
    for s, a in zip((-1, 1), arms):
        sw = math.sin(ph + (math.pi if s < 0 else 0)) * 25 * run
        a = math.radians(a + sw)
        sx, sy = x + s * 14 * u, by - 59 * u
        dxv, dyv = s * math.sin(a), math.cos(a)
        exx, eyy = sx + dxv * 23 * u, sy + dyv * 23 * u
        c.line([(sx, sy), (exx, eyy)], top, 8 * u)
        hx, hy = exx + dxv * 1.5 * u, eyy + dyv * 1.5 * u
        if "stripes" in ex:
            mx, my = sx + dxv * 15 * u, sy + dyv * 15 * u
            c.line([(mx - dyv * 0.1, my), (mx + dxv * 2.5 * u, my + dyv * 2.5 * u)], (225, 230, 225), 8.2 * u, False)
        if gun is not None and s == gun:
            gx, gy = hx + dxv * 3 * u, hy + dyv * 3 * u
            nx, ny = -dyv, dxv
            c.poly([(hx - nx * 2 * u, hy - ny * 2 * u), (gx + dxv * 9 * u - nx * 2 * u, gy + dyv * 9 * u - ny * 2 * u),
                    (gx + dxv * 9 * u + nx * 2.2 * u, gy + dyv * 9 * u + ny * 2.2 * u), (hx + nx * 2.2 * u, hy + ny * 2.2 * u)],
                   (35, 35, 40))
            c.poly([(hx, hy), (hx + nx * 6 * u * s, hy + ny * 6 * u * s), (hx + dxv * 3 * u + nx * 6 * u * s,
                    hy + dyv * 3 * u + ny * 6 * u * s), (hx + dxv * 3 * u, hy + dyv * 3 * u)], (35, 35, 40))
        if phone and s == 1:
            c.rect(hx - 2.5 * u, hy - 7 * u, hx + 2.5 * u, hy + 1 * u, (30, 30, 36), r=1 * u)
            c.rect(hx - 1.8 * u, hy - 6 * u, hx + 1.8 * u, hy - 3 * u, (250, 250, 230))
        c.ell(hx, hy, 4.6 * u, 4.6 * u, skin)
        hands.append((hx, hy))
    # голова
    hx, hy = x, by - 82 * u
    r = 17 * u
    hc = sp["hair_col"]
    if sp["hair"] == "long":
        c.rect(hx - 19 * u, hy - 10 * u, hx + 19 * u, hy + 15 * u, hc, r=7 * u)
        c.ell(hx + 15 * u, hy - 12 * u, 6 * u, 7 * u, hc)
    c.ell(hx, hy, r, r, skin)
    if sp["hair"] == "spiky":
        pts = [(hx - r * 1.05, hy - 1 * u)]
        for k in range(9):
            a = math.pi + k * math.pi / 8
            rr = r * (1.28 if k % 2 else 1.05)
            pts.append((hx + math.cos(a) * rr, hy + math.sin(a) * rr - 2 * u))
        pts += [(hx + r * 1.05, hy - 1 * u), (hx + r * 0.8, hy - 7 * u), (hx - r * 0.8, hy - 7 * u)]
        c.poly(pts, hc)
    elif sp["hair"] == "long":
        c.chord(hx, hy - 1 * u, r * 1.08, 180, 360, hc)
        c.poly([(hx - r * 1.05, hy - 1 * u), (hx - r * 0.2, hy - r * 0.95), (hx + r * 0.4, hy - 7 * u),
                (hx - r * 0.6, hy - 6 * u)], hc)
        c.poly([(hx + r * 1.05, hy - 2 * u), (hx + r * 0.3, hy - 9 * u), (hx + r * 0.9, hy - r * 0.8)], hc)
    elif sp["hair"] == "helmet":
        c.chord(hx, hy - 2 * u, r * 1.14, 180, 360, hc)
        c.rect(hx - r * 1.2, hy - 4 * u, hx + r * 1.2, hy - 1.5 * u, tuple(int(q * 0.8) for q in hc), r=1 * u)
        c.rect(hx - 2 * u, hy - r * 1.12, hx + 2 * u, hy - 4 * u, (240, 240, 240))
    elif sp["hair"] == "bald":
        for s in (-1, 1):
            c.ell(hx + s * 15 * u, hy - 1 * u, 5 * u, 6.5 * u, hc)
    elif sp["hair"] == "zombie":
        for k in range(5):
            a = math.pi + 0.4 + k * 0.55
            c.ell(hx + math.cos(a) * r * 0.9, hy + math.sin(a) * r * 0.9, 4 * u, 3 * u, hc)
    # лицо
    fy = hy + bow * 5 * u
    lx, ly = look[0] * 2.4 * u, look[1] * 2.2 * u
    if zombie:
        for s in (-1, 1):
            c.ell(hx + s * 6.5 * u, fy, 3.2 * u, 3.2 * u, (250, 250, 235))
            c.ell(hx + s * 6.5 * u + lx * 0.4, fy + ly * 0.4, 1 * u, 1 * u, (190, 20, 20))
        if bow < 0.5:
            c.poly([(hx - 6 * u, fy + 8 * u), (hx - 3 * u, fy + 10.5 * u), (hx, fy + 8.5 * u), (hx + 3 * u, fy + 11 * u),
                    (hx + 6 * u, fy + 8 * u), (hx + 6 * u, fy + 12 * u), (hx - 6 * u, fy + 12 * u)], (40, 10, 10))
        return (hx, hy - r * 1.4), hands
    big = expr in ("scared", "shock")
    for s in (-1, 1):
        ex_, ey_ = hx + s * 6.5 * u, fy
        if big:
            c.ell(ex_, ey_, 3.4 * u, 3.4 * u, (255, 255, 255))
            c.ell(ex_ + lx * 0.6, ey_ + ly * 0.6, 1.7 * u, 1.7 * u, (20, 20, 24))
        else:
            c.ell(ex_ + lx, ey_ + ly, 2.1 * u, 2.1 * u, (20, 20, 24))
        if "glasses" in ex:
            c.ell(ex_, ey_, 4.8 * u, 4.8 * u, None, outline=(40, 40, 50), w=1.1 * u)
        inner, outer = {"angry": (2.2, -1.2), "sad": (-2.4, 1.2), "scared": (-2.8, 0.6), "shock": (-3.2, -0.5),
                        "creepy": (2.6, -2.0), "neutral": (0.6, 0.0)}.get(expr, (0, 0))
        c.line([(hx + s * 3 * u, fy - 6.5 * u + inner * u), (hx + s * 11 * u, fy - 6.5 * u + outer * u)],
               (35, 28, 30), 2.5 * u)
    if "glasses" in ex:
        c.line([(hx - 1.7 * u, fy), (hx + 1.7 * u, fy)], (40, 40, 50), 1 * u)
    my = fy + 8.5 * u
    m = mouth or expr
    if bow > 0.5:
        return (hx, hy - r * 1.4), hands
    if talk > 0.05:
        c.ell(hx, my + 1 * u, 3.6 * u, 0.8 * u + talk * 2.6 * u, (120, 30, 40))
    elif m == "creepy":
        my += OY[0]
        c.d.chord([(hx - 10 * u) * S, (my - 6 * u) * S, (hx + 10 * u) * S, (my + 6 * u) * S], 0, 180, fill=(40, 12, 14))
        c.d.chord([(hx - 9 * u) * S, (my - 4.5 * u) * S, (hx + 9 * u) * S, (my + 2.5 * u) * S], 0, 180, fill=(250, 250, 240))
    elif m in ("shock", "scared"):
        c.ell(hx, my + 0.5 * u, 2.4 * u, 2.8 * u, (90, 25, 35))
    elif m == "sad":
        c.arc(hx, my + 3 * u, 4 * u, 2.5 * u, 200, 340, (60, 30, 30), 1.4 * u)
    else:
        c.line([(hx - 3.6 * u, my), (hx + 3.6 * u, my)], (60, 30, 30), 1.5 * u)
    return (hx, hy - r * 1.4), hands


ZOMBIE_TOPS = [(110, 120, 100), (120, 90, 110), (80, 100, 130), (140, 120, 90), (90, 110, 90), (130, 80, 80)]


def zombie_spec(i):
    return dict(skin=(150, 182, 128), hair="zombie", hair_col=(50, 50, 40), top=ZOMBIE_TOPS[i % 6],
                pants=(50, 52, 60), shoes=(40, 40, 40), extra=set())


# ================================================================ HUD
def hud_number(s):
    return s.replace("-", "−")


def draw_hud(img, t):
    d = ImageDraw.Draw(img, "RGBA")
    # день и время — карточка слева сверху
    card = Image.new("RGBA", (340, 150), (0, 0, 0, 0))
    cd = ImageDraw.Draw(card)
    cd.rectangle([10, 14, 330, 140], fill=(0, 0, 0, 70))
    cd.rectangle([0, 4, 322, 130], fill=(250, 246, 236))
    mins = CLOCK_START[0] * 60 + CLOCK_START[1] + int(t / CLOCK_STEP)
    cd.text((176, 52), f"ДЕНЬ {DAY_START + mins // 1440}", font=font(FB, 58), fill=(25, 25, 30), anchor="mm")
    cd.text((176, 104), f"{mins // 60 % 24:02}:{mins % 60:02}", font=font(FB, 38), fill=(205, 40, 34), anchor="mm")
    card = card.rotate(2.5, Image.BICUBIC, expand=True)
    img.paste(card, (-30, 214), card)
    # плашки справа
    dep = int(round(interp(DEPTH, t)))
    zn = int(round(interp(ZOMBIES_NEAR, t)))
    for i, (label, val) in enumerate((("ГЛУБИНА", hud_number(f"{dep} М")), ("ЗОМБИ РЯДОМ", str(zn)))):
        y0 = 236 + i * 92
        x1, x0 = 1040, 600
        d.rounded_rectangle([x0, y0, x1, y0 + 76], 38, fill=(22, 25, 36, 235))
        d.text((x0 + 36, y0 + 39), label, font=font(FX, 28), fill=(225, 228, 240), anchor="lm")
        col = (255, 196, 36) if i == 0 else ((255, 80, 70) if zn > 2 else (255, 196, 36))
        d.text((x1 - 34, y0 + 39), val, font=font(FB, 44), fill=col, anchor="rm")
    # обратный отсчёт до взрыва — красная плашка
    if COUNTDOWN_FROM <= t < COUNTDOWN_TO + 1.0:
        left = max(0, int(COUNTDOWN_SECONDS * (1 - (t - COUNTDOWN_FROM) / (COUNTDOWN_TO - COUNTDOWN_FROM))))
        k = ease((t - COUNTDOWN_FROM) / 0.4)
        y0, x1, x0 = 236 + 2 * 92, 1040, 640
        x0s = x1 - (x1 - x0) * k
        pulse = 0.5 + 0.5 * math.sin(t * 6)
        d.rounded_rectangle([x0s, y0, x1, y0 + 76], 38, fill=(int(150 + 60 * pulse), 24, 28, 240))
        if k > 0.9:
            d.text((x0 + 34, y0 + 39), "ДО ВЗРЫВА", font=font(FX, 28), fill=(255, 235, 235), anchor="lm")
            d.text((x1 - 34, y0 + 39), f"{left // 60:02}:{left % 60:02}", font=font(FB, 44), fill=(255, 255, 255), anchor="rm")
    # локация — красная плашка слева
    loc, t0 = LOCATIONS[0][1], 0.0
    for a, name in LOCATIONS:
        if t >= a:
            loc, t0 = name, a
    f = font(FB, 44)
    tw = d.textlength(loc, font=f)
    slide = ease((t - t0) / 0.35) if t0 > 0 else 1.0
    x = -tw - 60 + (tw + 60) * slide
    d.rectangle([x - 10, 426, x + tw + 44, 500], fill=(226, 46, 46))
    d.text((x + 18, 464), loc, font=f, fill=(255, 255, 255), anchor="lm")


def name_tag(img, pos, name, sub, sub_col=(70, 70, 76)):
    d = ImageDraw.Draw(img, "RGBA")
    x, y = pos
    fn, fs = font(FB, 52), font(FX, 24)
    w = max(d.textlength(name, font=fn), d.textlength(sub, font=fs)) + 44
    x0, y1 = x - w / 2, y - 18
    y0 = y1 - 112
    if y0 < 515 or y0 > H - 150:
        return
    d.rectangle([x0 + 6, y0 + 6, x0 + w + 6, y1 + 6], fill=(0, 0, 0, 70))
    d.rectangle([x0, y0, x0 + w, y1], fill=(250, 250, 248))
    d.text((x, y0 + 40), name, font=fn, fill=(20, 20, 24), anchor="mm")
    d.text((x, y0 + 86), sub, font=fs, fill=sub_col, anchor="mm")


def vlad_sub(t):
    if t >= EV["smile"]:
        return "ОРИГИНАЛ", (205, 30, 30)
    if t >= v(152.0):
        return "ЗАРАЖЁН", (205, 30, 30)
    if t >= v(37.3):
        return "СОХРАНИЛ СОЗНАНИЕ", (70, 70, 76)
    return "КТО ИЗ НАС НАСТОЯЩИЙ?", (70, 70, 76)


def vika_sub(t):
    if t >= EV["shore"]:
        return "ВЫЖИЛА", (70, 70, 76)
    if t >= EV["couldnt"]:
        return "НЕ СМОГЛА", (70, 70, 76)
    if t >= EV["worked"]:
        return "РАБОТАЛА В ЛАБОРАТОРИИ", (205, 30, 30)
    return "АГЕНТ", (205, 30, 30)


def double_sub(t):
    if t >= v(40.0):
        return "НОСИТЕЛЬ ВИРУСА", (205, 30, 30)
    return "ОБЪЕКТ № 1", (205, 30, 30)


TAGS = {
    "vlad": lambda t: ("ВЛАД",) + vlad_sub(t),
    "vika": lambda t: ("ВИКА",) + vika_sub(t),
    "double": lambda t: ("ВЛАД?",) + double_sub(t),
    "sci": lambda t: ("УЧЁНЫЙ", "ЗАПИСЬ", (70, 70, 76)),
}




def lamp_glow(c, x, y, r=170, col=(255, 222, 140)):
    for k in range(6):
        rr = r * (1 - k / 7)
        c.ell(x, y, rr, rr, col + (14,))
    c.ell(x, y, 22, 22, (255, 240, 190))


@lru_cache(None)
def bg_lab(oy=0):
    img = Image.new("RGB", (WW, WH), (32, 60, 70))
    c = C(img)
    c.rect(0, 0, 1080, 90, (24, 44, 52))
    c.rect(60, 280, 1020, 1000, (60, 84, 92))
    c.rect(80, 300, 1000, 980, (16, 26, 32))
    c.rect(80, 820, 1000, 980, (24, 36, 42))
    for x in (380, 700):
        c.rect(x - 8, 300, x + 8, 980, (60, 84, 92))
    c.rect(0, 1220, 1080, 1920, (40, 52, 58))
    c.rect(0, 1220, 1080, 1238, (28, 38, 44))
    for k in range(10):
        c.rect(0, 1300 + k * 70, 1080, 1304 + k * 70, (34, 46, 52))
    # стол и компьютер справа
    c.rect(660, 1020, 1060, 1060, (90, 98, 104))
    c.rect(690, 1060, 714, 1240, (70, 76, 80))
    c.rect(1006, 1060, 1030, 1240, (70, 76, 80))
    c.rect(740, 820, 980, 1020, (214, 206, 180), r=16)
    c.rect(870, 1000, 960, 1022, (190, 182, 160))
    # сирена на потолке
    c.rect(510, 90, 570, 120, (60, 60, 60))
    return img


def glass_overlay(c):
    for x0 in (120, 470, 780):
        c.poly([(x0, 980), (x0 + 120, 300), (x0 + 170, 300), (x0 + 50, 980)], (255, 255, 255, 18))


def lab_world(t, screen="off", siren=0.0, zombies=0.0, kneel=0.0, bow_override=None):
    img = bg_lab(OY[0]).copy()
    c = C(img)
    # зомби за стеклом
    if zombies > 0:
        rows = [(780, 0.62, 13), (880, 0.78, 11), (990, 0.95, 9)]
        for ri, (fy, sc, n) in enumerate(rows):
            for i in range(n):
                x = 120 + (i + 0.5) * (880 / n) + (ri % 2) * 20
                rise = (1 - ease((zombies - (i % 4) * 0.08 - ri * 0.1) / 0.6)) * 400
                kd = ease((kneel - i * 0.02 - ri * 0.05) / 0.5)
                char(c, x, fy + 120 * sc + rise, 330 * sc, zombie_spec(i + ri), t, zombie=True, kneel=kd,
                     bow=kd if bow_override is None else bow_override,
                     arms=(160 - 110 * kd, 160 - 110 * kd), look=(0, 0))
        c.rect(80, 980, 1000, 1000, (60, 84, 92))
    glass_overlay(c)
    # экран компьютера
    if screen != "off":
        col = {"on": (60, 160, 120), "found": (40, 120, 70), "red": (170, 30, 30)}[screen]
        c.rect(760, 840, 960, 990, col, r=10)
    else:
        c.rect(760, 840, 960, 990, (30, 34, 34), r=10)
    # сирена
    if siren > 0:
        on = 0.5 + 0.5 * math.sin(t * 9)
        c.ell(540, 132, 34, 22, (255, 40, 40))
        for k in range(5):
            c.ell(540, 132, 120 * (1 - k / 6), 90 * (1 - k / 6), (255, 30, 30, int(30 * on)))
    return img, c



# ================================================================ новые персонажи
VLAD_COAT = dict(VLAD, top=(245, 245, 245), extra={"coat"})
OLD = dict(VLAD, hair_col=(205, 205, 205), top=(240, 240, 240), extra={"coat", "glasses"})
MASKED = dict(skin=(28, 28, 32), hair="none", hair_col=(28, 28, 32), top=(36, 36, 42), pants=(24, 24, 28),
              shoes=(20, 20, 20), extra=set())
DOUBLE = dict(VLAD, top=(34, 34, 40), pants=(20, 20, 24), shoes=(30, 30, 30), extra={"hood"})
VIKA_TAPE = dict(VIKA, top=(200, 200, 205))


def human_spec(i):
    tops = [(120, 140, 170), (170, 120, 100), (110, 150, 120), (160, 150, 110), (140, 110, 150)]
    hair = [(40, 30, 20), (90, 60, 30), (20, 20, 20), (150, 110, 60), (60, 50, 40)]
    return dict(skin=SKIN, hair="spiky" if i % 2 else "long", hair_col=hair[i % 5], top=tops[i % 5],
                pants=(50, 52, 60), shoes=(240, 240, 240), extra=set())


def head_center(top, h):
    """Центр головы по точке над головой, которую возвращает char()."""
    return top[0], top[1] + 17 * h / 100 * 1.4


def tear(c, top, h, k, side=-1):
    hx, hy = head_center(top, h)
    u = h / 100
    x = hx + side * 6.5 * u
    y = hy + 3 * u + k * 9 * u
    c.ell(x, y, 1.6 * u, 2.4 * u, (120, 190, 255))


def ripples(c, x, y, t, col=(170, 90, 255)):
    for k in range(5):
        r = ((t * 260 + k * 120) % 600)
        a = int(170 * (1 - r / 600))
        c.ell(x, y, r, r, None, outline=col + (a,), w=6)


# ================================================================ шоты
def tags_from(f, items, t):
    out = []
    for who, pos in items:
        if pos is None:
            continue
        name, sub, col = TAGS[who](t)
        out.append((f(*pos), name, sub, col))
    return out


def char_rot(img, x, y, h, spec, angle, t=0.0, alpha=1.0, **kw):
    """Персонаж, повёрнутый вокруг ступней (падение, лежит на полу)."""
    # холст с запасом: ступни в центре, чтобы при любом повороте голова не обрезалась
    size = h * 2.4
    lay = Image.new("RGBA", (int(size * S), int(size * S)), (0, 0, 0, 0))
    saved, OY[0] = OY[0], 0
    top, _ = char(C(lay), size / 2, size / 2, h, spec, t, **kw)
    OY[0] = saved
    if angle:
        lay = lay.rotate(angle, Image.BICUBIC, center=(size / 2 * S, size / 2 * S))
    if alpha < 1:
        lay.putalpha(lay.split()[3].point(lambda p: int(p * alpha)))
    img.paste(lay, (int((x - size / 2) * S), int((y + OY[0] - size / 2) * S)), lay)
    return (x, y - (size / 2 - top[1])) if abs(angle) < 20 else None


def lab_two(t, vlad_x=420, dbl_x=780, vika_x=None, vlad=None, dbl=None, vika=None, screen="on", zoom=1.0, cx=560, cy=940,
            siren=0.0):
    img, c = lab_world(t, screen=screen, siren=siren, zombies=1.0, kneel=0.0, bow_override=0)
    items = []
    dbl = dbl or {}
    if dbl_x is not None:
        p, _ = char(c, dbl_x, dbl.pop("y", 1250), dbl.pop("h", 580), DOUBLE, t, talk=dbl.pop("talk", talking("double", t)),
                    mouth=dbl.pop("mouth", "creepy"), **dbl)
        items.append(("double", p))
    vlad = vlad or {}
    p1, _ = char(c, vlad_x, vlad.pop("y", 1250), vlad.pop("h", 580), VLAD, t, talk=talking("vlad", t), **vlad)
    items.append(("vlad", p1))
    if vika_x is not None:
        vika = vika or {}
        p2, _ = char(c, vika_x, vika.pop("y", 1235), vika.pop("h", 520), VIKA, t, talk=talking("vika", t), **vika)
        items.append(("vika", p2))
    frame, f = cam(img, zoom, cx, cy)
    return frame, tags_from(f, items, t)


def shot_face(t, lt):
    return lab_two(t, vlad_x=330, dbl_x=760, vika_x=520,
                   vlad={"expr": "shock", "look": (1.3, 0)}, dbl={"expr": "creepy", "look": (-1.3, 0)},
                   vika={"expr": "scared", "look": (1.3, 0)}, zoom=lerp(1.0, 1.1, ease(lt / 7)), cx=560)


def shot_backoff(t, lt):
    g = ease((t - EV["grab"]) / 0.4)
    back = ease((t - EV["grab"]) / 1.5)
    laugh = v(10.6) <= t < v(11.6)
    return lab_two(t, vlad_x=lerp(330, 260, back), dbl_x=780, vika_x=lerp(520, 420, back),
                   vlad={"expr": "angry", "look": (1.3, 0), "arms": (8, lerp(8, 60, g))},
                   dbl={"expr": "creepy", "look": (-1.3, 0), "talk": 0.8 if laugh else talking("double", t)},
                   vika={"expr": "scared", "look": (1.3, 0), "arms": (lerp(8, 60, g), 8)}, zoom=1.08, cx=540)


def shot_topc(t, lt):
    k = ease(lt / 2.6)
    return lab_two(t, vlad_x=260, dbl_x=lerp(780, 860, k), vika_x=420,
                   vlad={"expr": "angry", "look": (1.3, 0), "arms": (8, 60)},
                   dbl={"expr": "creepy", "look": (1.2, 0), "run": 1.0 if 0.05 < k < 0.95 else 0.0},
                   vika={"expr": "scared", "look": (1.3, 0), "arms": (60, 8)}, zoom=1.08, cx=600)


def screen_frame(t, mode):
    """Крупный план экрана компьютера."""
    img = Image.new("RGB", (WW, WH), (30, 36, 40))
    c = C(img)
    c.rect(30, 420, 1050, 1500, (214, 206, 180), r=60)
    c.rect(90, 480, 990, 1260, (18, 22, 20), r=40)
    tags = []
    if mode == "sci":
        c.rect(110, 500, 970, 1240, (150, 170, 175), r=30)
        c.rect(110, 1000, 970, 1240, (120, 136, 140))
        p, _ = char(c, 540, 1500, 900, SCI, t, talk=talking("sci", t), expr="sad", arms=(10, 10))
        tags.append(("sci", p))
        c.text(170, 540, "● REC", FB, 30, (220, 40, 40), anchor="lm")
        c.text(930, 540, "ЭКСПЕРИМЕНТ 17", FM, 26, (240, 240, 240), anchor="rm")
    elif mode == "footage":
        c.rect(110, 500, 970, 1240, (170, 176, 180), r=30)
        c.rect(110, 1000, 970, 1240, (140, 146, 150))
        c.rect(180, 960, 760, 1000, (120, 126, 132))
        c.rect(200, 1000, 230, 1180, (100, 104, 110))
        c.rect(710, 1000, 740, 1180, (100, 104, 110))
        char_rot(img, 220, 955, 470, VLAD_COAT, -90, t, expr="neutral", arms=(8, 8))
        c = C(img)
        char(c, 860, 1210, 560, DOUBLE, t, expr="neutral", look=(-1.3, 0.5))
        c.text(170, 540, "● REC", FB, 30, (220, 40, 40), anchor="lm")
        c.text(930, 540, "АРХИВ · ИСПЫТАНИЯ", FM, 26, (40, 40, 40), anchor="rm")
    elif mode == "sixty":
        c.rect(110, 500, 970, 1240, (60, 8, 10), r=30)
        blink = int(t * 4) % 2 == 0
        c.text(540, 700, "ДО УНИЧТОЖЕНИЯ", FB, 64, (255, 230, 230))
        c.text(540, 790, "ОСТРОВА", FB, 64, (255, 230, 230))
        c.text(540, 990, "60", FB, 260, (255, 70, 70) if blink else (200, 40, 40))
        c.text(540, 1150, "СЕКУНД", FB, 54, (255, 230, 230))
    for y in range(500, 1240, 10):
        c.rect(110, y, 970, y + 3, (0, 0, 0, 40))
    if mode == "footage":
        a = np.asarray(img, np.float32)
        g = a.mean(axis=2, keepdims=True)
        img = to_img(a * 0.4 + g * 0.6)
    return img, tags


def shot_screen(t, lt):
    mode = "footage" if EV["footage"] <= t < EV["sci2"] else "sci"
    img, tags = screen_frame(t, mode)
    frame, f = cam(img, lerp(1.0, 1.12, ease(lt / 8)), 540, 880)
    return frame, tags_from(f, tags, t)


def shot_who(t, lt):
    smile = t >= v(44.72)
    return lab_two(t, vlad_x=380, dbl_x=700, vika_x=None,
                   vlad={"expr": "angry", "look": (1.3, 0)},
                   dbl={"expr": "creepy", "look": (-1.3, 0), "mouth": "creepy" if smile else "neutral"},
                   zoom=lerp(1.1, 1.25, ease(lt / 7)), cx=540, cy=900)


def shot_fight(t, lt):
    img, c = lab_world(t, screen="on", siren=0.0, zombies=1.0, kneel=0.0, bow_override=0)
    items = []
    if t < EV["fall"]:
        k = ease((t - EV["fight"]) / (EV["fall"] - EV["fight"]))
        p = char_rot(img, lerp(780, 520, k), 1250, 580, DOUBLE, lerp(0, 25, k), t, run=1.0, expr="creepy",
                     mouth="creepy", arms=(150, 150))
        c = C(img)
        p1 = char_rot(img, 380, 1250, 580, VLAD, lerp(0, 15, k), t, expr="shock", look=(1.3, 0))
        items = [("double", p), ("vlad", p1)]
    elif t < EV["pipe"] - 0.3:
        char_rot(img, 260, 1290, 580, VLAD, -86, t, expr="shock", arms=(160, 160))
        char_rot(img, 330, 1200, 580, DOUBLE, -72, t, expr="creepy", mouth="creepy", arms=(170, 170))
    else:
        swing = ease((t - EV["pipe"] + 0.3) / 0.35)
        p = char_rot(img, 720, 1250, 580, DOUBLE, 0, t, expr="creepy", mouth="creepy", look=(-1.3, 0))
        c = C(img)
        p1, hands = char(c, 420, 1250, 580, VLAD, t, expr="angry", look=(1.3, 0), arms=(8, lerp(160, 75, swing)))
        hx, hy = hands[1]
        a = math.radians(lerp(160, 75, swing))
        c.line([(hx, hy), (hx + math.sin(a) * 240, hy + math.cos(a) * 240)], (150, 156, 164), 18)
        items = [("double", p), ("vlad", p1)]
    frame, f = cam(img, 1.1, 560, 940)
    return frame, tags_from(f, items, t)


def shot_shot(t, lt):
    img, c = lab_world(t, screen="on", zombies=1.0, kneel=0.0, bow_override=0)
    up = ease((t - EV["shot"]) / 0.3)
    p2, hands = char(c, 300, 1235, 520, VIKA, t, expr="angry", look=(0, -1.5), arms=(8, lerp(8, 172, up)))
    hx, hy = hands[1]
    c.rect(hx - 10, hy - 60, hx + 12, hy, (35, 35, 40))
    if 0 <= t - (EV["shot"] + 0.4) < 0.12:
        lamp_glow(c, hx, hy - 80, 160, (255, 220, 120))
    r = np.random.default_rng(3)
    for i in range(80):
        x = r.uniform(80, 1000)
        y = r.uniform(-100, 200) + max(0, t - EV["shot"] - 0.4) * r.uniform(400, 900)
        c.ell(x, y, 3, 3, (180, 180, 170))
    p = char_rot(img, 760, 1250, 580, DOUBLE, 0, t, expr="shock", look=(0, -1.5))
    c = C(img)
    p1, _ = char(c, 520, 1250, 580, VLAD, t, expr="shock", look=(0, -1.0))
    frame, f = cam(img, 1.05, 540, 960)
    return frame, tags_from(f, [("vika", p2), ("vlad", p1), ("double", p)], t)


@lru_cache(None)
def bg_corridor(oy=0):
    img = Image.new("RGB", (WW, WH), (52, 58, 64))
    c = C(img)
    vx, vy = 540, 860
    c.poly([(0, 0), (1080, 0), (vx + 140, vy - 160), (vx - 140, vy - 160)], (38, 42, 48))
    c.poly([(0, 1920), (1080, 1920), (vx + 140, vy + 160), (vx - 140, vy + 160)], (40, 44, 48))
    c.poly([(0, 0), (vx - 140, vy - 160), (vx - 140, vy + 160), (0, 1920)], (60, 66, 72))
    c.poly([(1080, 0), (vx + 140, vy - 160), (vx + 140, vy + 160), (1080, 1920)], (56, 62, 68))
    c.rect(vx - 140, vy - 160, vx + 140, vy + 160, (16, 18, 22))
    for k in range(1, 6):
        f = k / 6
        x0 = lerp(0, vx - 140, f)
        y0, y1 = lerp(0, vy - 160, f), lerp(1920, vy + 160, f)
        c.line([(x0, y0), (x0, y1)], (44, 48, 54), 4)
        c.line([(1080 - x0, y0), (1080 - x0, y1)], (44, 48, 54), 4)
        c.rect(vx - 60 * (1 - f) - 20, lerp(40, vy - 150, f), vx + 60 * (1 - f) + 20, lerp(60, vy - 140, f), (230, 230, 210))
    # аварийный выход справа
    c.rect(900, 700, 1060, 1500, (30, 36, 34))
    c.rect(890, 600, 1070, 670, (30, 160, 80), r=8)
    c.text(980, 635, "ВЫХОД", FB, 40, (255, 255, 255))
    return img


def corridor_world(t, red=False):
    img = bg_corridor(OY[0]).copy()
    c = C(img)
    if red:
        on = 0.5 + 0.5 * math.sin(t * 8)
        for x in (200, 880):
            lamp_glow(c, x, 140, 220, (255, 40, 40))
        c.rect(0, 0, 1080, 1920, (255, 0, 0, int(40 * on)))
    return img, c


def shot_run(t, lt):
    img, c = corridor_world(t)
    k = ease((t - EV["run"]) / 2.6)
    stop = t >= v(65.06)
    p1, _ = char(c, lerp(260, 640, k), 1700, 640, VLAD, t, run=0.0 if stop else 1.0, expr="scared", look=(1.2, 0) if not stop else (-1.2, 0),
                 arms=(8, 60))
    p2, _ = char(c, lerp(160, 470, k), 1690, 600, VIKA, t + 0.1, run=0.0 if stop else 1.0, expr="sad", look=(1.2, 0), arms=(60, 8))
    frame, f = cam(img, 1.0, 540, 1200)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def remote(c, x, y, s=1.0, pressed=False, led=True):
    c.rect(x - 40 * s, y - 70 * s, x + 40 * s, y + 70 * s, (40, 42, 48), r=14 * s)
    c.ell(x, y, 26 * s, 26 * s, (150, 20, 20) if pressed else (230, 40, 40))
    if led:
        c.ell(x, y - 50 * s, 6 * s, 6 * s, (255, 80, 80))


def shot_confess(t, lt):
    red = t >= EV["alarm"]
    img, c = corridor_world(t, red=red)
    p1, _ = char(c, 380, 1700, 640, VLAD, t, talk=talking("vlad", t), expr="shock" if t > EV["worked"] else "scared",
                 look=(1.2, 0))
    nod = 0.6 if EV["nod"] <= t < EV["nod"] + 0.6 else 0.0
    down = EV["nod"] <= t < EV["couldnt"]
    p2, hands = char(c, 700, 1690, 600, VIKA, t, talk=talking("vika", t), expr="sad",
                     look=(-1.2, 0.8 if down else 0) if not nod else (-1.2, 1.4), arms=(8, 40 if t > EV["remote"] else 8))
    if t > EV["remote"]:
        hx, hy = hands[1]
        remote(c, hx + 10, hy - 50, 0.7)
    zoom = 1.15 if t < EV["alarm"] else 1.0
    frame, f = cam(img, zoom, 540, 1150)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_sixty(t, lt):
    img, _ = screen_frame(t, "sixty")
    return cam(img, lerp(1.0, 1.1, ease(lt / 3)), 540, 880)[0], []


def corridor_crowd(c, t, k_dbl, cover=0.0):
    """Двойник в конце коридора и сотни заражённых за ним."""
    r = np.random.default_rng(14)
    for row in range(5):
        for i in range(16):
            x = 400 + r.uniform(0, 280) + (row - 2) * 6
            y = 980 - row * 22
            hh = 150 - row * 14
            char(c, x, y, hh, zombie_spec(i + row), t, zombie=True, arms=(150, 150))
    pd, _ = char(c, 540, 1160, 380, DOUBLE, t, talk=talking("double", t),
                 expr="creepy" if t < EV["no_smile"] else "angry",
                 mouth="creepy" if t < EV["no_smile"] else None, arms=(8, 8) if cover <= 0 else (170, 170))
    if cover > 0:
        r2 = np.random.default_rng(15)
        for i in range(int(20 * cover)):
            x = 540 + r2.uniform(-120, 120) * (1 - cover * 0.5)
            char(c, x, 1170 + r2.uniform(-20, 20), 330, zombie_spec(i), t, zombie=True, arms=(160, 160))
    return pd


def shot_corridor(t, lt):
    img, c = corridor_world(t, red=True)
    pd = corridor_crowd(c, t, 1.0)
    p1, _ = char(c, 230, 2050, 1100, VLAD, t, talk=talking("vlad", t), expr="angry", look=(1.0, -0.6))
    p2, hands = char(c, 880, 2040, 1000, VIKA, t, expr="scared", look=(-1.0, -0.6), arms=(8, 40))
    hx, hy = hands[1]
    remote(c, hx, hy - 70, 1.0)
    frame, f = cam(img, lerp(1.0, 1.12, ease(lt / 14)), 540, 1080)
    return frame, tags_from(f, [("double", pd)], t)


def shot_warn(t, lt):
    img, c = corridor_world(t, red=True)
    g = ease((t - EV["warn"]) / 0.4)
    p1, _ = char(c, 400, 1700, 640, VLAD, t, expr="angry", look=(1.2, 0), arms=(8, lerp(8, 55, g)))
    p2, _ = char(c, 680, 1690, 600, VIKA, t, talk=talking("vika", t), expr="scared", look=(-1.2, 0), arms=(lerp(8, 55, g), 8))
    frame, f = cam(img, 1.1, 540, 1150)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_button(t, lt):
    img = Image.new("RGB", (WW, WH), (30, 8, 10))
    c = C(img)
    lamp_glow(c, 540, 960, 600, (160, 20, 20))
    pressed = t >= EV["press"]
    c.ell(540, 1320, 330, 240, SKIN)
    c.rect(300, 1320, 780, 1920, SKIN)
    remote(c, 540, 960, 4.0, pressed=pressed)
    if not pressed:
        fy = lerp(1700, 1080, ease((t - EV["press"] + 1.2) / 1.0))
        c.rect(500, fy, 580, fy + 600, SKIN, r=40)
    else:
        c.rect(500, 980, 580, 1580, SKIN, r=40)
        if t - EV["press"] < 0.3:
            lamp_glow(c, 540, 960, 400, (255, 80, 80))
    frame, f = cam(img, lerp(1.0, 1.15, ease(lt / 8)), 540, 980)
    return frame, []


def shot_doors(t, lt):
    img, c = corridor_world(t, red=True)
    cover = ease((t - v(123.38)) / 2.5)
    pd = corridor_crowd(c, t, 1.0, cover)
    shut = ease((t - EV["doors"]) / 8.5)
    y = lerp(-200, 1150, shut)
    c.rect(300, -400 + 0, 780, y, (70, 74, 80))
    for k in range(int((y + 400) // 60)):
        yy = y - k * 60
        c.rect(300, yy - 4, 780, yy, (50, 54, 60))
    c.rect(300, y - 40, 780, y, (230, 190, 30))
    for i in range(8):
        c.poly([(300 + i * 60, y), (330 + i * 60, y - 40), (360 + i * 60, y - 40), (330 + i * 60, y)], (30, 30, 30))
    frame, f = cam(img, lerp(1.1, 1.3, ease(lt / 9)), 540, 1050)
    return frame, tags_from(f, [("double", pd)] if shut < 0.8 else [], t)


def shot_boom(t, lt):
    img = Image.new("RGB", (WW, WH), (0, 0, 0))
    c = C(img)
    if t < EV["silence"]:
        k = ease(lt / 2.5)
        for i, col in enumerate(((255, 250, 230), (255, 200, 90), (240, 110, 30), (150, 40, 20))):
            rr = 200 + k * 900 * (1 - i * 0.18)
            c.ell(540, 960, rr, rr, col + (int(220 - i * 40),))
    return cam(img, 1.0)[0], []


@lru_cache(None)
def bg_shore(oy=0):
    img = Image.new("RGB", (WW, WH), (30, 16, 20))
    c = C(img)
    for i in range(40):
        y = i * 26
        c.rect(0, y, 1080, y + 26, mix((24, 12, 20), (200, 80, 40), (i / 40) ** 1.5))
    c.rect(0, 1040, 1080, 1360, (40, 30, 40))
    for k in range(14):
        y = 1060 + k * 22
        c.line([(0, y), (1080, y + 4)], (90, 50, 50), 2)
    c.rect(0, 1360, 1080, 1920, (110, 86, 60))
    c.poly([(0, 1360), (1080, 1330), (1080, 1380), (0, 1400)], (150, 140, 130))
    # горящий остров на горизонте
    c.poly([(160, 1040), (300, 900), (420, 860), (560, 880), (700, 940), (820, 1040)], (24, 14, 14))
    return img


def shore_world(t):
    img = bg_shore(OY[0]).copy()
    c = C(img)
    r = np.random.default_rng(int(t * 10))
    for i in range(24):
        x = r.uniform(220, 780)
        hgt = r.uniform(60, 220)
        base = 1040 - max(0, 160 - abs(x - 460) * 0.6)
        c.poly([(x - 30, base), (x + 30, base), (x + r.uniform(-15, 15), base - hgt)], (255, int(r.uniform(90, 190)), 30, 230))
    lamp_glow(c, 480, 900, 400, (255, 120, 40))
    sink = ease((t - EV["shore"]) / 10)
    sy = lerp(1040, 1090, sink)
    c.poly([(880, sy), (1020, sy - 20), (1000, sy + 10), (900, sy + 14)], (14, 10, 12))
    c.line([(940, sy - 10), (950 - sink * 20, sy - 120)], (14, 10, 12), 6)
    c.rect(0, 1040, 1080, 1100, (40, 30, 40, int(220 * sink)))
    return img, c


def shot_shore(t, lt):
    img, c = shore_world(t)
    up = ease((t - EV["shore"] - 1.2) / 1.2)
    help_k = ease((t - EV["help_up"]) / 1.2)
    items = []
    if up < 0.5:
        char_rot(img, -10, 1560, 560, VLAD, -85, t, expr="sad", look=(0, 0))
        c = C(img)
    else:
        p1, _ = char(c, 420, 1640, 600, VLAD, t, talk=talking("vlad", t), expr="sad", look=(1.2, 0.4 if help_k < 1 else 0),
                     arms=(8, lerp(8, 70, help_k)))
        items.append(("vlad", p1))
    if help_k < 0.5:
        char_rot(img, 1100, 1820, 500, VIKA, 85, t, expr="sad")
    else:
        p2, _ = char(c, 680, 1640, 540, VIKA, t, expr="sad", look=(-1.2, 0), arms=(70, 8))
        items.append(("vika", p2))
    frame, f = cam(img, lerp(1.1, 1.0, ease(lt / 10)), 540, 1150)
    return frame, tags_from(f, items, t)


def shot_veins(t, lt):
    img = Image.new("RGB", (WW, WH), (20, 14, 16))
    c = C(img)
    lamp_glow(c, 540, 700, 600, (160, 70, 30))
    c.rect(360, 1100, 720, 1920, SKIN, r=60)
    c.ell(540, 950, 260, 230, SKIN)
    for i, (dx, ln) in enumerate(((-190, 260), (-90, 320), (10, 340), (110, 300))):
        c.rect(540 + dx - 40, 950 - ln, 540 + dx + 40, 900, SKIN, r=40)
    c.rect(720, 950, 800, 1150, SKIN, r=40)
    grow = ease((t - v(150.1)) / 4.0)
    r = np.random.default_rng(6)
    for b in range(9):
        x, y = 540 + r.uniform(-120, 120), 1900
        pts = [(x, y)]
        steps = int(18 * grow)
        for _ in range(steps):
            x += r.uniform(-35, 35)
            y -= r.uniform(40, 70)
            pts.append((x, y))
        if len(pts) > 1:
            c.line(pts, (25, 15, 30), 9)
    frame, f = cam(img, lerp(1.0, 1.12, ease(lt / 10)), 540, 1100)
    return frame, []


def shot_inner(t, lt):
    img, c = shore_world(t)
    img = to_img(np.asarray(img, np.float32) * 0.6)
    c = C(img)
    ripples(c, 540, 900, t, (200, 40, 40))
    char_rot(img, 540, 2100, 1400, DOUBLE, 0, t, alpha=0.25 + 0.15 * math.sin(t * 3), expr="creepy", mouth="creepy")
    c = C(img)
    p1, _ = char(c, 540, 2350, 1500, VLAD, t, expr="scared", look=(0, -0.3), arms=(10, 10))
    frame, f = cam(img, lerp(1.0, 1.1, lt / 10), 540, 960)
    return frame, tags_from(f, [("vlad", p1)], t)


def shot_smile(t, lt):
    img, c = shore_world(t)
    img = to_img(np.asarray(img, np.float32) * 0.5)
    c = C(img)
    smile = t >= v(170.6)
    p1, _ = char(c, 540, 2350, 1500, VLAD, t, expr="creepy" if smile else "neutral", mouth="creepy" if smile else None,
                 look=(0, 0), arms=(10, 10))
    hx, hy = head_center(p1, 1500)
    grow = ease((t - EV["smile"]) / 3.0)
    r = np.random.default_rng(8)
    for b in range(5):
        x, y = hx + r.uniform(-200, 200), hy + 260
        pts = [(x, y)]
        for _ in range(int(8 * grow)):
            x += r.uniform(-25, 25)
            y -= r.uniform(30, 50)
            pts.append((x, y))
        if len(pts) > 1:
            c.line(pts, (30, 15, 35, 200), 7)
    if smile:
        for s in (-1, 1):
            lamp_glow(c, hx + s * 6.5 * 15, hy, 60, (255, 40, 40))
    frame, f = cam(img, lerp(1.0, 1.25, ease(lt / 5)), 540, 900)
    return frame, tags_from(f, [("vlad", p1)], t)


def shot_final(t, lt):
    img = Image.new("RGB", (W, H), (10, 6, 8))
    d = ImageDraw.Draw(img)
    fade_out = 1 - ease((t - EV["black"] - 1.0) / 2.0)
    for i, line in enumerate(FINAL_TEXT):
        k = ease((lt - i * 1.4) / 0.6) * fade_out
        f = font(FB, 50 if i == 0 else 78)
        col = (int(255 * k), int(255 * k), int(255 * k)) if i == 0 else (int(230 * k), int(50 * k), int(50 * k))
        d.text((540, 880 + i * 130), line, font=f, fill=col, anchor="mm")
    return img, []


SHOTS = [
    (0.0, EV["backoff"], shot_face), (EV["backoff"], EV["to_pc"], shot_backoff), (EV["to_pc"], EV["screen"], shot_topc),
    (EV["screen"], EV["who"], shot_screen), (EV["who"], EV["fight"], shot_who), (EV["fight"], EV["shot"], shot_fight),
    (EV["shot"], EV["run"], shot_shot), (EV["run"], EV["confess"], shot_run), (EV["confess"], EV["sixty"], shot_confess),
    (EV["sixty"], EV["corridor"], shot_sixty), (EV["corridor"], EV["warn"], shot_corridor), (EV["warn"], EV["button"], shot_warn),
    (EV["button"], EV["doors"], shot_button), (EV["doors"], EV["boom"], shot_doors), (EV["boom"], EV["shore"], shot_boom),
    (EV["shore"], EV["veins"], shot_shore), (EV["veins"], EV["inner"], shot_veins), (EV["inner"], EV["smile"], shot_inner),
    (EV["smile"], EV["final"], shot_smile), (EV["final"], DURATION, shot_final),
]

FLASHBACK = ()
SHIFTED = {fn: 250 for _, _, fn in SHOTS if fn in (shot_face, shot_backoff, shot_topc, shot_who, shot_fight, shot_shot)}

FLASHES = [(EV["pipe"] + 0.05, (255, 255, 255), 0.15), (EV["shot"] + 0.4, (255, 240, 200), 0.12),
           (EV["alarm"], (255, 30, 30), 0.3), (EV["press"], (255, 40, 40), 0.25), (EV["boom"], (255, 255, 255), 0.6),
           (EV["shore"], (0, 0, 0), 0.5), (v(152.0), (60, 0, 20), 0.4), (v(170.6), (255, 30, 30), 0.25)]


# ================================================================ постобработка и надписи
@lru_cache(None)
def vignette():
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    return np.clip(1.08 - d ** 2.4, 0.45, 1).astype(np.float32)[:, :, None]


def post(img, t, fn):
    a = np.asarray(img, np.float32)
    if fn not in (shot_final, shot_boom):
        a = a * vignette()
    if fn in FLASHBACK:
        # воспоминание: сепия и выцветание
        g = a.mean(axis=2, keepdims=True)
        a = g * np.array([1.08, 0.94, 0.74], np.float32) * 0.9 + 18
    elif EV["alarm"] <= t < EV["boom"] and fn not in (shot_sixty, shot_button):
        p = 0.5 + 0.5 * math.sin(t * 9)
        a = a * (1 - 0.2 * p) + np.array([255, 20, 20], np.float32) * 0.2 * p
    for ft, col, dur in FLASHES:
        if 0 <= t - ft < dur:
            k = 1 - (t - ft) / dur
            a = a * (1 - k * 0.6) + np.array(col, np.float32) * k * 0.6
    if t < 0.3:
        a *= t / 0.3
    return to_img(a)


@lru_cache(None)
def words():
    out = []
    try:
        for line in open("voice/words.tsv", encoding="utf-8"):
            s, e, wd = line.rstrip("\n").split("\t")
            out.append((float(s) + VOICE_OFFSET, float(e) + VOICE_OFFSET, wd))
    except FileNotFoundError:
        pass
    return out


def caption(img, t):
    for s, e, wd in words():
        if s <= t < max(e, s + 0.25):
            d = ImageDraw.Draw(img)
            d.text((540, 1500), wd.upper(), font=font(FB, 120), fill=(255, 255, 255), anchor="mm",
                   stroke_width=10, stroke_fill=(10, 10, 10))
            return


def shot_at(t):
    for a, b, fn in SHOTS:
        if a <= t < b:
            return fn, t - a
    return SHOTS[-1][2], t - SHOTS[-1][0]


def frame(i):
    t = i / FPS
    CUR_T[0] = t
    fn, lt = shot_at(t)
    OY[0] = SHIFTED.get(fn, 0)
    img, tags = fn(t, lt)
    img = post(img.convert("RGB"), t, fn)
    if fn not in (shot_boom, shot_final):
        draw_hud(img, t)
        for pos, name, sub, col in tags:
            name_tag(img, pos, name, sub, col)
        if CAPTIONS:
            caption(img, t)
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
                           "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool() as pool:
        for n, buf in enumerate(pool.imap(frame, range(total), chunksize=6)):
            ff.stdin.write(buf)
            if n % (FPS * 10) == 0:
                print(f"  кадр {n}/{total}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

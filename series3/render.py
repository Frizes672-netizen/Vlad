"""Рендер серии 3 «Кто из нас настоящий» в плоском мультяшном стиле (движок серии 2).

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

from timeline import (CAPTIONS, CLOCK_START, CLOCK_STEP, COUNTDOWN_FROM, COUNTDOWN_SECONDS,
                      DAY_START, DEPTH, DURATION, EV, FINAL_TEXT, FPS, H, LOCATIONS,
                      VOICE_OFFSET, W, ZOMBIES_NEAR, v)

S = 2                       # масштаб мира относительно кадра
WW, WH = W * S, H * S
FB = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FX = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# реплики (для анимации рта)
TALK = {
    "vlad": [(v(70.32), v(71.6)), (v(74.24), v(75.78)), (v(84.1), v(86.46)), (v(131.94), v(133.66))],
    "vika": [(v(7.64), v(9.62)), (v(50.48), v(53.2)), (v(77.38), v(82.66)), (v(86.46), v(90.46)),
             (v(105.52), v(109.22)), (v(137.0), v(138.02)), (v(150.08), v(154.74))],
    "old": [(v(64.34), v(68.42))],
    "tapevika": [(v(119.08), v(121.52))],
    "double": [(v(163.46), v(167.28))],
    "speaker": [(v(42.48), v(48.82)), (v(145.52), v(147.46))],
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
    (EV["heads_up"], 10, 0.6), (EV["pain"], 30, 1.2), (EV["pain"] + 1.5, 20, 1.0), (EV["press"], 26, 0.5),
    (EV["island"] + 0.3, 14, 0.8), (EV["virus"], 12, 0.5), (EV["siren"], 14, 0.5), (EV["shout"], 18, 0.4),
    (EV["beep"], 8, 0.3), (EV["auth"], 10, 0.3), (EV["press2"], 22, 0.4), (EV["unmask"], 12, 0.4),
    (EV["tape_cut"], 16, 0.3), (EV["boom"], 46, 0.7), (EV["stand_up"], 16, 0.8), (EV["awake"], 10, 0.5),
    (EV["double"], 12, 0.6), (EV["final"], 14, 0.6),
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
    if t >= COUNTDOWN_FROM:
        left = max(0, COUNTDOWN_SECONDS - int(t - COUNTDOWN_FROM))
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
    if t >= EV["double"] + 1.0:
        return "НАСТОЯЩИЙ?", (205, 30, 30)
    if t >= EV["virus"]:
        return "ВЫПУСТИЛ ВИРУС", (205, 30, 30)
    return "СОЗДАТЕЛЬ", (205, 30, 30)


def vika_sub(t):
    if t >= EV["tear"]:
        return "БЫЛА В ПЛЕНУ", (70, 70, 76)
    if t >= v(77.4):
        return "АГЕНТ", (205, 30, 30)
    return "В ШОКЕ", (70, 70, 76)


TAGS = {
    "vlad": lambda t: ("ВЛАД",) + vlad_sub(t),
    "vika": lambda t: ("ВИКА",) + vika_sub(t),
    "vlad_past": lambda t: ("ВЛАД", "В ПРОШЛОМ", (70, 70, 76)),
    "sci": lambda t: ("УЧЁНЫЕ", "ВОСПОМИНАНИЕ", (70, 70, 76)),
    "old": lambda t: ("ВЛАД?", "НАМНОГО СТАРШЕ", (205, 30, 30)),
    "tapevika": lambda t: ("ВИКА", "НА ЗАПИСИ", (70, 70, 76)),
    "masked": lambda t: ("ВЛАД", "НА ЗАПИСИ", (205, 30, 30)) if t >= EV["unmask"] else ("???", "В МАСКЕ", (70, 70, 76)),
    "double": lambda t: ("ВЛАД?", "ОБЪЕКТ № 1", (205, 30, 30)),
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
        name, sub, col = TAGS[who](t)
        out.append((f(*pos), name, sub, col))
    return out


def zombies_kneel(t):
    """0 — стоят, 1 — на коленях (стоят на коленях до удара за стеной)."""
    return 1 - ease((t - EV["stand_up"]) / 1.4)


def zombie_heads_up(t):
    return ease((t - EV["heads_up"]) / 0.8)


def lab_scene(t, vlad_x=470, vika_x=800, vlad=None, vika=None, siren=0.0, screen="on", zoom=1.0, cx=540, cy=980):
    kn = zombies_kneel(t)
    img, c = lab_world(t, screen=screen, siren=siren, zombies=1.0, kneel=kn, bow_override=kn * (1 - zombie_heads_up(t)))
    vlad = vlad or {}
    vika = vika or {}
    p1, _ = char(c, vlad_x, vlad.pop("y", 1250), vlad.pop("h", 580), vlad.pop("spec", VLAD), t,
                 talk=talking("vlad", t), **vlad)
    p2, _ = char(c, vika_x, vika.pop("y", 1230), vika.pop("h", 520), VIKA, t, talk=talking("vika", t), **vika)
    frame, f = cam(img, zoom, cx, cy)
    return frame, f, c, p1, p2


def shot_kneel(t, lt):
    back = ease(lt / 8)
    frame, f, c, p1, p2 = lab_scene(t, vika_x=lerp(780, 880, back),
                                    vlad={"expr": "shock", "look": (0, -0.5), "arms": (20, 20)},
                                    vika={"expr": "scared", "look": (-1.3, 0)},
                                    zoom=lerp(1.0, 1.12, ease(lt / 9)), cy=lerp(980, 940, lt / 9))
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_step(t, lt):
    k = ease(lt / 0.8)
    frame, f, c, p1, p2 = lab_scene(t, vlad_x=lerp(470, 540, k), vika_x=880,
                                    vlad={"expr": "angry", "run": 1.0 if 0.05 < k < 0.95 else 0.0, "look": (0, -0.5)},
                                    vika={"expr": "scared", "look": (-1.3, 0), "arms": (60, 8)},
                                    zoom=lerp(1.12, 1.0, ease(lt / 2)), cy=940)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_voice(t, lt):
    img = Image.new("RGB", (WW, WH), (16, 10, 24))
    c = C(img)
    for i in range(7):
        x = 90 + i * 150
        char(c, x, 1300 + (i % 2) * 60, 520, zombie_spec(i), t, zombie=True, arms=(10, 10), look=(0, 0))
        c.ell(x - 22, 1300 + (i % 2) * 60 - 460, 10, 10, (190, 90, 255, 160))
        c.ell(x + 22, 1300 + (i % 2) * 60 - 460, 10, 10, (190, 90, 255, 160))
    c.rect(0, 0, 1080, 1920, (10, 6, 18, 150))
    ripples(c, 540, 900, t)
    p1, _ = char(c, 540, 2350, 1500, VLAD, t, expr="scared", look=(0, -0.3), arms=(10, 10))
    frame, f = cam(img, lerp(1.0, 1.1, lt / 5), 540, 960)
    return frame, tags_from(f, [("vlad", p1)], t)


def shot_pain(t, lt):
    img = Image.new("RGB", (WW, WH), (40, 8, 12))
    c = C(img)
    ripples(c, 540, 1000, t * 1.6, (255, 60, 60))
    p1, _ = char(c, 540, 2200, 1350, VLAD, t, expr="shock", look=(0, 0.4), arms=(165, 165))
    frame, f = cam(img, lerp(1.0, 1.15, lt / 4.5), 540, 960)
    return frame, tags_from(f, [("vlad", p1)], t)


def shot_fb_lab(t, lt):
    img, c = lab_world(t, screen="on", zombies=0.0)
    frame, f = cam(img, lerp(1.25, 1.05, ease(lt / 2)), 540, 940)
    return frame, []


def shot_fb_sci(t, lt):
    img, c = lab_world(t, screen="on", zombies=0.0)
    pts = []
    for i, x in enumerate((300, 540, 780)):
        p, _ = char(c, x, 1240, 540, SCI, t, expr="neutral", look=(math.sin(t + i), 0), arms=(8, 30 if i == 1 else 8))
        pts.append(p)
    frame, f = cam(img, 1.05, 540, 960)
    return frame, tags_from(f, [("sci", pts[1])], t)


def shot_fb_glass(t, lt):
    img, c = lab_world(t, screen="on", zombies=0.0)
    for i in range(6):
        x = 170 + i * 150
        char(c, x, 990, 420, human_spec(i), t, expr="scared", look=(0, 0), arms=(150, 150) if i % 2 else (20, 20))
    c.rect(80, 980, 1000, 1000, (60, 84, 92))
    glass_overlay(c)
    frame, f = cam(img, lerp(1.0, 1.3, ease(lt / 1.6)), 540, 760)
    return frame, []


def shot_fb_self(t, lt):
    img, c = lab_world(t, screen="on", zombies=0.0)
    p1, _ = char(c, 540, 1250, 620, VLAD_COAT, t, expr="angry", look=(0, 0), arms=(8, 8))
    frame, f = cam(img, lerp(1.0, 1.25, ease(lt / 1.4)), 540, 900)
    return frame, tags_from(f, [("vlad_past", p1)], t)


@lru_cache(None)
def bg_vault(oy=0):
    img = Image.new("RGB", (WW, WH), (36, 44, 52))
    c = C(img)
    c.rect(0, 1220, 1080, 1920, (44, 50, 56))
    c.rect(0, 1220, 1080, 1238, (30, 34, 38))
    c.ell(420, 760, 400, 400, (90, 98, 108))
    c.ell(420, 760, 350, 350, (120, 128, 138))
    for k in range(12):
        a = k * math.pi / 6
        c.ell(420 + math.cos(a) * 375, 760 + math.sin(a) * 375, 16, 16, (70, 76, 84))
    c.ell(420, 760, 120, 120, None, outline=(70, 76, 84), w=26)
    for k in range(3):
        a = k * math.pi / 3
        c.line([(420 - math.cos(a) * 160, 760 - math.sin(a) * 160), (420 + math.cos(a) * 160, 760 + math.sin(a) * 160)],
               (70, 76, 84), 24)
    c.rect(860, 760, 1040, 1020, (70, 76, 84), r=16)
    c.rect(880, 780, 1020, 1000, (50, 54, 60), r=12)
    c.rect(860, 1020, 900, 1230, (60, 64, 70))
    return img


def shot_fb_door(t, lt):
    img = bg_vault(OY[0]).copy()
    c = C(img)
    pressed = t >= EV["press"]
    c.ell(950, 890, 50, 50, (150, 20, 20) if pressed else (230, 40, 40))
    if pressed and int(t * 6) % 2 == 0:
        lamp_glow(c, 950, 890, 220, (255, 60, 60))
    k = ease((t - EV["press"] + 0.6) / 0.6)
    p1, _ = char(c, 760, 1250, 600, VLAD_COAT, t, expr="angry", look=(1.3, -0.2), arms=(8, lerp(10, 82, k)))
    frame, f = cam(img, lerp(1.0, 1.15, ease(lt / 3)), 620, 960)
    return frame, tags_from(f, [("vlad_past", p1)], t)


def shot_island(t, lt):
    img = Image.new("RGB", (WW, WH), (30, 90, 140))
    c = C(img)
    for k in range(12):
        y = 200 + k * 140 + math.sin(t + k) * 10
        c.line([(60 + (k * 70) % 300, y), (240 + (k * 70) % 300, y)], (60, 120, 170), 6)
    pts = []
    for k in range(40):
        a = k / 40 * 2 * math.pi
        r = 380 + 50 * math.sin(a * 3) + 30 * math.cos(a * 5)
        pts.append((540 + math.cos(a) * r, 1000 + math.sin(a) * r * 1.2))
    c.poly(pts, (230, 214, 160))
    c.poly([(540 + (x - 540) * 0.9, 1000 + (y - 1000) * 0.9) for x, y in pts], (90, 160, 90))
    for k in range(9):
        a = k * 0.7
        c.ell(540 + math.cos(a) * 220, 1000 + math.sin(a) * 260, 40, 40, (60, 130, 70))
    c.rect(500, 960, 600, 1040, (200, 200, 205))
    spread = ease(lt / 2.0)
    for k in range(3):
        r = spread * (180 + k * 160)
        c.ell(550, 1000, r, r * 1.15, (220, 30, 30, 70))
    frame, f = cam(img, lerp(1.0, 1.15, lt / 2.6), 540, 1000)
    return frame, []


def shot_realize(t, lt):
    frame, f, c, p1, p2 = lab_scene(t, vlad_x=540, vika_x=880,
                                    vlad={"expr": "shock", "look": (0, 1.4) if t < EV["virus"] else (0, 0), "arms": (55, 55)},
                                    vika={"expr": "scared", "look": (-1.3, 0)},
                                    zoom=lerp(1.0, 1.3, ease(lt / 3.6)), cx=600, cy=lerp(940, 860, ease(lt / 3.6)))
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def speaker(c, x, y, t, on):
    c.rect(x - 90, y - 60, x + 90, y + 60, (40, 44, 50), r=14)
    for gx in range(-60, 70, 24):
        for gy in range(-36, 40, 24):
            c.ell(x + gx, y + gy, 6, 6, (16, 18, 22))
    if on:
        for k in range(3):
            r = 110 + ((t * 200 + k * 60) % 180)
            c.arc(x, y, r, r, -40, 40, (255, 230, 120, int(200 * (1 - (r - 110) / 180))), 6)
            c.arc(x, y, r, r, 140, 220, (255, 230, 120, int(200 * (1 - (r - 110) / 180))), 6)


def shot_siren(t, lt):
    kn = zombies_kneel(t)
    img, c = lab_world(t, screen="red", siren=1.0, zombies=1.0, kneel=kn, bow_override=0)
    speaker(c, 200, 180, t, talking("speaker", t) > 0)
    p1, _ = char(c, 540, 1250, 580, VLAD, t, expr="shock", look=(-1.0, -1.0), arms=(30, 30))
    p2, _ = char(c, 880, 1230, 520, VIKA, t, expr="scared", look=(-1.2, -1.0))
    frame, f = cam(img, lerp(1.0, 1.08, lt / 9), 520, 940)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_run(t, lt):
    kn = zombies_kneel(t)
    img, c = lab_world(t, screen="on", siren=1.0, zombies=1.0, kneel=kn, bow_override=0)
    g = ease((t - EV["grab"]) / 0.3)
    rk = ease((t - EV["run"]) / 1.6)
    running = EV["run"] < t < EV["pc"]
    vx = lerp(540, 300, rk)
    kx = lerp(880, 420, rk)
    p1, _ = char(c, vx, 1250, 580, VLAD, t, run=1.0 if running else 0.0, expr="scared",
                 look=(1.3, 0) if t >= EV["pc"] - 0.3 else (-1, 0), arms=(8, lerp(8, 60, g)))
    p2, _ = char(c, kx, 1230, 520, VIKA, t + 0.1, run=1.0 if running else 0.0, talk=talking("vika", t), expr="scared",
                 look=(-1.2, 0), arms=(lerp(8, 60, g), 8))
    frame, f = cam(img, lerp(1.05, 1.15, ease(lt / 7)), lerp(620, 560, ease(lt / 7)), 960)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def screen_frame(t, mode):
    """Крупный план экрана компьютера."""
    img = Image.new("RGB", (WW, WH), (30, 36, 40))
    c = C(img)
    c.rect(30, 420, 1050, 1500, (214, 206, 180), r=60)
    c.rect(90, 480, 990, 1260, (18, 22, 20), r=40)
    blink = int(t * 4) % 2 == 0
    tags = []
    if mode == "old":
        c.rect(110, 500, 970, 1240, (150, 160, 170), r=30)
        c.rect(110, 1000, 970, 1240, (120, 130, 136))
        p, _ = char(c, 540, 1500, 900, OLD, t, talk=talking("old", t), expr="sad", arms=(10, 10))
        tags.append(("old", p))
        c.text(170, 540, "● REC", FB, 30, (220, 40, 40), anchor="lm")
        c.text(930, 540, "ЗАПИСЬ 00", FM, 26, (240, 240, 240), anchor="rm")
    elif mode == "tape":
        c.rect(110, 500, 970, 1240, (70, 74, 80), r=30)
        c.rect(110, 1050, 970, 1240, (52, 56, 60))
        c.rect(420, 700, 660, 1150, (140, 146, 152), r=10)
        c.rect(400, 980, 680, 1020, (120, 126, 132))
        vt = char(c, 540, 1180, 560, VIKA_TAPE, t, talk=talking("tapevika", t), expr="scared", arms=(18, 18))[0]
        for yy in (900, 960, 1040):
            c.line([(440, yy), (640, yy + 6)], (150, 110, 70), 7)
        tear(c, vt, 560, (t * 0.8) % 1)
        tear(c, vt, 560, (t * 0.8 + 0.5) % 1, side=1)
        tags.append(("tapevika", vt))
        if t >= EV["mask_in"]:
            k = ease((t - EV["mask_in"]) / 1.0)
            mx = lerp(1050, 800, k)
            un = ease((t - EV["unmask"]) / 0.4)
            spec = MASKED if un < 0.5 else dict(VLAD, top=(36, 36, 42), pants=(24, 24, 28))
            mp, _ = char(c, mx, 1200, 600, spec, t, expr="angry", look=(-1.2, 0),
                         arms=(8, 150 * (1 - un) + 8 if t > EV["unmask"] - 0.5 else 8))
            if un < 0.5:
                hx, hy = head_center(mp, 600)
                for s in (-1, 1):
                    c.ell(hx + s * 39 - 8, hy, 15, 10, (240, 240, 240))
            tags.append(("masked", mp))
        c.text(170, 540, "● REC", FB, 30, (220, 40, 40), anchor="lm")
        c.text(930, 540, "ЗАПИСЬ 17", FM, 26, (240, 240, 240), anchor="rm")
    elif mode in ("protocol", "auth", "beep"):
        c.rect(110, 500, 970, 1240, (10, 24, 14), r=30)
        if mode == "beep":
            if blink:
                c.text(540, 860, "!", FB, 200, (240, 200, 60))
        if mode in ("protocol", "auth"):
            c.rect(150, 560, 930, 780, (150, 20, 20) if blink else (100, 10, 10), r=10)
            c.text(540, 620, "ЗАПУСК ПРОТОКОЛА", FB, 50, (255, 240, 240))
            c.text(540, 710, "УНИЧТОЖЕНИЯ", FB, 64, (255, 240, 240))
        if mode == "auth":
            c.text(540, 860, "АВТОРИЗАЦИЯ ТРЕБУЕТСЯ:", FB, 44, (90, 240, 120))
            c.text(540, 940, "ВЛАД", FB, 90, (90, 240, 120))
            c.rect(330, 1050, 750, 1170, (230, 40, 40) if blink else (180, 30, 30), r=20)
            c.text(540, 1110, "ПОДТВЕРДИТЬ", FB, 46, (255, 255, 255))
    elif mode == "static":
        r = np.random.default_rng(int(t * 24))
        for _ in range(500):
            x, y = r.uniform(110, 970), r.uniform(500, 1240)
            g = int(r.uniform(40, 220))
            c.rect(x, y, x + r.uniform(4, 40), y + 4, (g, g, g))
    for y in range(500, 1240, 10):
        c.rect(110, y, 970, y + 3, (0, 0, 0, 40))
    c.ell(930, 1380, 20, 20, (90, 220, 110))
    return img, tags


def shot_old(t, lt):
    if t < EV["old_video"] + 0.8:
        kn = zombies_kneel(t)
        img, c = lab_world(t, screen="on", siren=1.0, zombies=1.0, kneel=kn, bow_override=0)
        p1, _ = char(c, 600, 1250, 580, VLAD, t, expr="angry", look=(1.4, 0))
        frame, f = cam(img, lerp(1.1, 2.0, ease(lt / 0.8)), lerp(600, 860, ease(lt / 0.8)), lerp(960, 920, ease(lt / 0.8)))
        return frame, []
    img, tags = screen_frame(t, "old")
    frame, f = cam(img, lerp(1.0, 1.15, ease((lt - 0.8) / 10)), 540, 880)
    return frame, tags_from(f, tags, t)


def shot_talk(t, lt):
    closer = ease((t - EV["closer"]) / 0.6) - ease((t - EV["step_back"]) / 0.6) * 0.8
    shout = EV["shout"] <= t < EV["look_down"]
    down = t >= EV["look_down"]
    turn = ease((t - EV["confront"]) / 0.5)
    vlad_x = lerp(460, 600, closer)
    frame, f, c, p1, p2 = lab_scene(
        t, vlad_x=vlad_x, vika_x=820,
        vlad={"expr": "angry" if t > EV["closer"] else "shock", "look": (lerp(-0.5, 1.3, turn), 0),
              "arms": (30, 30) if shout else (8, 8)},
        vika={"expr": "sad", "look": (0.6, 1.4) if down and not talking("vika", t) else (-1.3, 0.3)},
        siren=1.0, zoom=1.2 if shout else 1.12, cx=lerp(560, 640, closer), cy=900)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_protocol(t, lt):
    mode = "beep" if t < EV["protocol"] else ("protocol" if t < EV["auth"] else "auth")
    img, _ = screen_frame(t, mode)
    frame, f = cam(img, lerp(1.0, 1.15, ease(lt / 12)), 540, 880)
    return frame, []


def shot_pale(t, lt):
    pressk = ease((t - EV["press2"] + 0.5) / 0.4)
    frame, f, c, p1, p2 = lab_scene(
        t, vlad_x=520, vika_x=800,
        vlad={"expr": "angry", "look": (1.3, 0) if t < EV["press2"] - 1 else (1.5, -0.3), "arms": (8, lerp(8, 80, pressk))},
        vika={"expr": "scared", "look": (-1.3, 0), "arms": (lerp(10, 95, ease((t - EV["pale"] - 2.0) / 0.4)), 10)},
        siren=1.0, zoom=1.15, cx=640, cy=900)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_tape(t, lt):
    mode = "static" if t >= EV["tape_cut"] else "tape"
    img, tags = screen_frame(t, mode)
    frame, f = cam(img, lerp(1.0, 1.12, ease(lt / 17)), 540, 880)
    return frame, tags_from(f, tags, t) if mode == "tape" else []


def shot_truth(t, lt):
    if t < EV["tear"]:
        frame, f, c, p1, p2 = lab_scene(t, vlad_x=500, vika_x=780,
                                        vlad={"expr": "shock", "look": (1.3, 0)},
                                        vika={"expr": "sad", "look": (-1.3, 0)},
                                        siren=1.0, zoom=1.12, cx=620, cy=900)
        return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)
    img, c = lab_world(t, screen="on", siren=1.0, zombies=1.0, kneel=1.0, bow_override=0)
    p2, _ = char(c, 560, 1900, 1200, VIKA, t, talk=talking("vika", t), expr="sad", look=(-0.6, 0.2), arms=(8, 8))
    tear(c, p2, 1200, ease((t - EV["tear"] - 0.5) / 2.0))
    frame, f = cam(img, lerp(1.0, 1.1, ease((t - EV["tear"]) / 4)), 560, 1000)
    return frame, tags_from(f, [("vika", p2)], t)


def shot_awake(t, lt):
    kn = zombies_kneel(t)
    hold = ease((t - EV["leave"]) / 0.3)
    frame, f, c, p1, p2 = lab_scene(
        t, vlad_x=500, vika_x=lerp(820, 700, hold),
        vlad={"expr": "shock", "look": (-0.5, -1.0) if t < EV["leave"] else (1.2, 0), "arms": (8, lerp(8, 55, hold))},
        vika={"expr": "scared", "look": (-1.2, -0.6), "arms": (lerp(8, 55, hold), 8)},
        siren=1.0, zoom=lerp(1.0, 1.06, lt / 16), cy=960)
    speaker_on = talking("speaker", t) > 0
    if speaker_on:
        d = ImageDraw.Draw(frame, "RGBA")
        for k in range(3):
            r = 60 + ((t * 200 + k * 50) % 150)
            d.arc([120 - r, 300 - r, 120 + r, 300 + r], -40, 40, fill=(255, 230, 120, 180), width=6)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_double(t, lt):
    img, c = lab_world(t, screen="on", siren=1.0, zombies=1.0, kneel=0.0, bow_override=0)
    # дверь лаборатории слева
    open_k = ease((t - EV["door_open"]) / 2.5)
    c.rect(30, 560, 290, 1240, (20, 22, 26))
    c.rect(44, 574, 276, 1240, (6, 6, 8))
    tags = []
    if t >= EV["double"] - 0.6:
        k = ease((t - EV["double"] + 0.6) / 1.6)
        dx = lerp(160, 300, k)
        dp, _ = char(c, dx, 1240, 580, DOUBLE, t, talk=talking("double", t), run=0.0,
                     expr="creepy" if t >= EV["smile"] else "neutral",
                     mouth="creepy" if t >= EV["smile"] and not talking("double", t) else None, look=(1.2, 0))
        tags.append(("double", dp))
    c.rect(44 + 232 * open_k, 574, 276, 1240, (70, 76, 84))
    p1, _ = char(c, 640, 1250, 580, VLAD, t, expr="shock", look=(-1.4, 0))
    p2, _ = char(c, 860, 1230, 520, VIKA, t, expr="scared", look=(-1.4, 0), arms=(55, 8))
    tags += [("vlad", p1), ("vika", p2)]
    if t < EV["smile"]:
        frame, f = cam(img, lerp(1.0, 1.15, ease(lt / 5)), lerp(540, 430, ease(lt / 5)), lerp(940, 860, ease(lt / 5)))
    else:
        k = ease((t - EV["smile"]) / 1.2)
        frame, f = cam(img, lerp(1.15, 1.25, k), lerp(430, 380, k), lerp(860, 800, k))
    return frame, tags_from(f, tags, t)


def shot_black(t, lt):
    return Image.new("RGB", (W, H)), []


def shot_final(t, lt):
    img = Image.new("RGB", (W, H), (10, 10, 14))
    d = ImageDraw.Draw(img)
    a = ease(lt / 0.5) * (1 - ease((lt - 3.3) / 0.6))
    f = font(FB, 92)
    for i, wd in enumerate(FINAL_TEXT.split()):
        k = ease((lt - i * 0.25) / 0.35)
        col = tuple(int(255 * a * k) for _ in range(3))
        d.text((540, 840 + i * 120 - (1 - k) * 30), wd, font=f, fill=col, anchor="mm")
    d.rectangle([290, 1110, 290 + 500 * ease((lt - 0.8) / 0.8), 1122], fill=(int(226 * a), int(46 * a), int(46 * a)))
    return img, []


SHOTS = [
    (0.0, EV["step"], shot_kneel), (EV["step"], EV["voice"], shot_step), (EV["voice"], EV["pain"], shot_voice),
    (EV["pain"], EV["flash"], shot_pain), (EV["flash"], EV["scientists"], shot_fb_lab),
    (EV["scientists"], EV["glass"], shot_fb_sci), (EV["glass"], EV["self"], shot_fb_glass),
    (EV["self"], EV["door"], shot_fb_self), (EV["door"], EV["island"], shot_fb_door),
    (EV["island"], EV["back"], shot_island), (EV["back"], EV["siren"], shot_realize),
    (EV["siren"], EV["grab"], shot_siren), (EV["grab"], EV["old_video"], shot_run),
    (EV["old_video"], EV["confront"], shot_old), (EV["confront"], EV["protocol_shot"], shot_talk),
    (EV["protocol_shot"], EV["pale"], shot_protocol), (EV["pale"], EV["tape"], shot_pale),
    (EV["tape"], EV["truth"], shot_tape), (EV["truth"], EV["boom"], shot_truth),
    (EV["boom"], EV["door_open"], shot_awake), (EV["door_open"], EV["black"], shot_double),
    (EV["black"], EV["final"], shot_black), (EV["final"], DURATION, shot_final),
]

FLASHBACK = (shot_fb_lab, shot_fb_sci, shot_fb_glass, shot_fb_self, shot_fb_door, shot_island)
SHIFTED = {fn: 250 for _, _, fn in SHOTS if fn not in (shot_voice, shot_pain, shot_black, shot_final, shot_island)}
for fn in (shot_old, shot_protocol, shot_tape):
    SHIFTED[fn] = 0

FLASHES = [(EV["pain"], (255, 255, 255), 0.2), (EV["flash"], (255, 255, 255), 0.35), (EV["press"], (255, 40, 30), 0.3),
           (EV["back"], (255, 255, 255), 0.35), (EV["virus"], (220, 30, 30), 0.25), (EV["unmask"], (255, 255, 255), 0.15),
           (EV["boom"], (255, 255, 255), 0.2), (EV["double"], (0, 0, 0), 0.3), (EV["final"], (120, 0, 0), 0.4)]


# ================================================================ постобработка и надписи
@lru_cache(None)
def vignette():
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    return np.clip(1.08 - d ** 2.4, 0.45, 1).astype(np.float32)[:, :, None]


def post(img, t, fn):
    a = np.asarray(img, np.float32)
    if fn not in (shot_final, shot_black):
        a = a * vignette()
    if fn in FLASHBACK:
        # воспоминание: сепия и выцветание
        g = a.mean(axis=2, keepdims=True)
        a = g * np.array([1.08, 0.94, 0.74], np.float32) * 0.9 + 18
    elif EV["siren"] <= t < EV["black"] and fn not in (shot_old, shot_protocol, shot_tape):
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
    if fn not in (shot_black, shot_final):
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

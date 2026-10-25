"""Рендер серии 2 «Он знает моё имя» в плоском мультяшном стиле.

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

from timeline import (CAPTIONS, CLOCK_START, CLOCK_STEP, DAY, DEPTH, DURATION,
                      EV, FINAL_TEXT, FPS, H, LOCATIONS, VOICE_OFFSET, W,
                      ZOMBIES_NEAR, v)

S = 2                       # масштаб мира относительно кадра
WW, WH = W * S, H * S
FB = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FX = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraBold.otf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# реплики (для анимации рта)
TALK = {
    "vlad": [(v(4.9), v(6.06)), (v(50.22), v(52.26))],
    "vika": [(v(7.06), v(9.3)), (v(53.82), v(57.28)), (v(71.48), v(72.64)), (v(93.9), v(97.8))],
    "rescuer": [(v(20.34), v(22.46))],
    "sci": [(v(61.4), v(67.2))],
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
SHAKES = [(t_, 10 + 4 * i, 0.25) for i, t_ in enumerate(EV["door_hits"])] + [
    (EV["door_off"], 40, 0.5), (EV["jump"] + 0.5, 22, 0.3), (EV["lid_close"], 28, 0.35),
    (EV["roar"], 26, 1.0), (EV["drop"], 20, 0.4), (EV["slam"], 34, 0.4), (EV["label"], 10, 0.4),
    (EV["gun"], 8, 0.3), (EV["siren"], 14, 0.5), (EV["zombies"], 12, 0.6), (EV["kneel"] + 0.8, 18, 0.5),
    (EV["drop_gun"] + 0.35, 8, 0.25), (EV["creator"], 16, 0.7), (EV["final"], 14, 0.6),
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
    cd.text((176, 52), DAY, font=font(FB, 58), fill=(25, 25, 30), anchor="mm")
    mins = CLOCK_START[0] * 60 + CLOCK_START[1] + int(t / CLOCK_STEP)
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
    if t >= EV["creator"]:
        return "СОЗДАТЕЛЬ", (205, 30, 30)
    if t >= EV["label"]:
        return "ОБЪЕКТ № 17", (205, 30, 30)
    return "НИЧЕГО НЕ ПОМНИТ", (70, 70, 76)


def vika_sub(t):
    if t >= EV["drop_gun"]:
        return "В ШОКЕ", (70, 70, 76)
    if t >= EV["gun"]:
        return "ДЕРЖИТ НА МУШКЕ", (205, 30, 30)
    return "ЧТО-ТО СКРЫВАЕТ", (70, 70, 76)


TAGS = {
    "vlad": lambda t: ("ВЛАД",) + vlad_sub(t),
    "vika": lambda t: ("ВИКА",) + vika_sub(t),
    "rescuer": lambda t: ("СПАСАТЕЛЬ", "БЫЛ МЁРТВ 3 ДНЯ", (205, 30, 30)),
    "sci": lambda t: ("УЧЁНЫЙ", "ВИДЕОЗАПИСЬ", (70, 70, 76)),
}


# ================================================================ локации
@lru_cache(None)
def bg_hut(oy=0):
    img = Image.new("RGB", (WW, WH), (128, 86, 54))
    c = C(img)
    for i in range(14):
        x = i * 80
        c.rect(x, 0, x + 78, 1250, (136 - (i % 3) * 8, 92 - (i % 3) * 5, 58 - (i % 3) * 4))
        c.rect(x + 76, 0, x + 80, 1250, (92, 60, 38))
    c.rect(0, 1250, 1080, 1920, (98, 66, 44))
    for k in range(9):
        y = 1250 + k * 80
        c.rect(0, y, 1080, y + 5, (78, 52, 34))
    c.rect(0, 1236, 1080, 1256, (78, 52, 34))
    # окно с луной
    c.rect(90, 380, 360, 720, (82, 56, 36))
    c.rect(106, 396, 344, 704, (40, 56, 96))
    c.ell(270, 470, 34, 34, (236, 236, 214))
    c.rect(222, 396, 230, 704, (82, 56, 36))
    c.rect(106, 546, 344, 554, (82, 56, 36))
    # лампа
    c.line([(540, 0), (540, 210)], (40, 30, 24), 4)
    c.poly([(500, 210), (580, 210), (600, 250), (480, 250)], (60, 60, 64))
    # дверной проём
    c.rect(748, 690, 1022, 1250, (70, 46, 30))
    c.rect(762, 704, 1008, 1250, (24, 30, 52))
    # люк в полу
    c.poly([(390, 1350), (690, 1350), (720, 1470), (360, 1470)], (84, 56, 36))
    for k in range(4):
        x = 400 + k * 75
        c.line([(x, 1352), (x - 8 + k * 5, 1468)], (66, 44, 28), 3)
    c.ell(540, 1412, 20, 9, None, outline=(150, 150, 150), w=4)
    return img


def lamp_glow(c, x, y, r=170, col=(255, 222, 140)):
    for k in range(6):
        rr = r * (1 - k / 7)
        c.ell(x, y, rr, rr, col + (14,))
    c.ell(x, y, 22, 22, (255, 240, 190))


def hut_world(t, door_k=0.0, hatch_open=0.0, chars=()):
    img = bg_hut(OY[0]).copy()
    c = C(img)
    lamp_glow(c, 540, 262)
    # дверь: дрожит от ударов, потом вылетает
    hits = sum(t >= q for q in EV["door_hits"])
    jit = sum(math.exp(-(t - q) / 0.12) for q in EV["door_hits"] if t >= q) * 6
    if door_k <= 0:
        dx = math.sin(t * 70) * jit
        c.rect(764 + dx, 706, 1006 + dx, 1250, (150, 98, 60))
        for k in range(4):
            c.rect(764 + dx + k * 61, 706, 768 + dx + k * 61, 1250, (112, 72, 44))
        c.rect(764 + dx, 800, 1006 + dx, 830, (112, 72, 44))
        c.rect(764 + dx, 1110, 1006 + dx, 1140, (112, 72, 44))
        c.ell(982 + dx, 990, 9, 9, (200, 180, 120))
        r = np.random.default_rng(5)
        for k in range(hits * 2):
            x, y = r.uniform(790, 980), r.uniform(740, 1200)
            pts = [(x + dx, y)]
            for _ in range(4):
                x += r.uniform(-25, 25)
                y += r.uniform(15, 45)
                pts.append((x + dx, y))
            c.line(pts, (40, 24, 14), 4)
    else:
        # спасатель в проёме
        char(c, 885, 1250, 520, RESCUER, t, expr="creepy", mouth="creepy", arms=(30, 30))
        k = ease(door_k)
        cx, cy = lerp(885, 520, k), lerp(978, 1500, k)
        wdt, hgt = 242 * (1 - 0.3 * k), 544 * (1 + 0.25 * k)
        a = math.radians(-70 * k)
        pts = [(-wdt / 2, -hgt / 2), (wdt / 2, -hgt / 2), (wdt / 2, hgt / 2), (-wdt / 2, hgt / 2)]
        pts = [(cx + px * math.cos(a) - py * math.sin(a), cy + px * math.sin(a) + py * math.cos(a)) for px, py in pts]
        c.poly(pts, (150, 98, 60))
    # люк
    if hatch_open > 0:
        k = ease(hatch_open)
        c.poly([(390, 1350), (690, 1350), (720, 1470), (360, 1470)], (10, 8, 8))
        top = lerp(1350, 1080, k)
        c.poly([(390, 1350), (690, 1350), (700, top), (380, top)], (110, 74, 48))
        c.ell(540, lerp(1412, top + 60, k), 20, 9 * (1 - k) + 3, None, outline=(150, 150, 150), w=4)
    tags = []
    for spec in chars:
        tags.append(spec)
    return img, c


@lru_cache(None)
def floor_front_mask(oy):
    """Маска пола перед люком: скрывает персонажей, провалившихся в люк."""
    m = Image.new("L", (WW, WH), 0)
    d = ImageDraw.Draw(m)
    m0 = m
    d.rectangle([0, 1470 * S, WW, WH], fill=255)
    d.polygon([(0, 1350 * S), (360 * S, 1470 * S), (0, 1470 * S)], fill=255)
    d.polygon([(WW, 1350 * S), (720 * S, 1470 * S), (WW, 1470 * S)], fill=255)
    d.polygon([(0, 1350 * S), (390 * S, 1350 * S), (360 * S, 1470 * S), (0, 1470 * S)], fill=255)
    d.polygon([(690 * S, 1350 * S), (WW, 1350 * S), (WW, 1470 * S), (720 * S, 1470 * S)], fill=255)
    out = Image.new("L", (WW, WH), 0)
    out.paste(m0, (0, oy * S))
    return out


def tunnel_world(t, scroll, door_x=None):
    img = Image.new("RGB", (WW, WH), (58, 62, 76))
    c = C(img)
    c.rect(0, 0, 1080, 260, (44, 47, 60))
    for k in range(-1, 5):
        x = k * 360 - (scroll % 360)
        c.rect(x + 40, 330, x + 300, 1190, (34, 39, 58))
        c.ell(x + 170, 330, 130, 120, (34, 39, 58))
        c.rect(x + 300, 260, x + 400, 1190, (70, 72, 86))
        on = (int((scroll + k * 360) / 360) * 7 + int(t * 3)) % 9 != 0
        c.ell(x + 350, 300, 26, 26, (230, 210, 140) if on else (80, 80, 70))
    for k in range(3):
        y = 120 + k * 40
        c.rect(0, y, 1080, y + 18, (74, 78, 92))
    c.rect(0, 1190, 1080, 1920, (46, 46, 54))
    c.rect(0, 1190, 1080, 1206, (210, 180, 40))
    for k in range(-1, 12):
        x = k * 120 - (scroll * 1.25 % 120)
        c.rect(x, 1500, x + 60, 1520, (70, 60, 50))
    c.rect(0, 1470, 1080, 1480, (110, 110, 120))
    c.rect(0, 1560, 1080, 1570, (110, 110, 120))
    for k in range(4):
        x = (k * 410 + 150 - scroll * 1.1) % 1400 - 200
        c.ell(x, 1720, 110, 18, (60, 68, 90))
    if door_x is not None:
        x = door_x
        c.rect(x - 170, 520, x + 170, 1192, (96, 102, 112))
        c.rect(x - 150, 540, x + 150, 1192, (124, 130, 140))
        for yy in (560, 1170):
            for xx in range(-130, 150, 52):
                c.ell(x + xx, yy, 7, 7, (90, 96, 104))
        c.rect(x - 150, 1110, x + 150, 1150, (230, 190, 30))
        for k in range(8):
            c.poly([(x - 150 + k * 40, 1150), (x - 130 + k * 40, 1110), (x - 110 + k * 40, 1110), (x - 130 + k * 40, 1150)],
                   (30, 30, 30))
        c.rect(x - 130, 640, x + 130, 800, (236, 196, 40))
        c.rect(x - 122, 648, x + 122, 792, (250, 214, 60))
        c.text(x, 690, "ПОДЗЕМНЫЙ", FB, 34, (20, 20, 20))
        c.text(x, 748, "БУНКЕР № 7", FB, 44, (20, 20, 20))
        c.rect(x + 100, 900, x + 124, 980, (70, 74, 80), r=6)
    return img, c


@lru_cache(None)
def bg_bunker(oy=0):
    img = Image.new("RGB", (WW, WH), (112, 124, 118))
    c = C(img)
    for k in range(7):
        y = 120 + k * 160
        c.rect(0, y, 1080, y + 6, (94, 104, 98))
    c.rect(0, 1220, 1080, 1920, (70, 74, 76))
    c.rect(0, 1220, 1080, 1236, (52, 56, 58))
    for k in range(10):
        c.rect(0, 1300 + k * 70, 1080, 1304 + k * 70, (60, 64, 66))
    # стальная дверь слева
    c.rect(20, 560, 250, 1222, (78, 84, 92))
    c.rect(34, 574, 236, 1222, (104, 110, 118))
    c.rect(200, 860, 222, 940, (60, 64, 70), r=6)
    # пробковая доска
    c.rect(400, 330, 1050, 980, (120, 84, 50))
    c.rect(414, 344, 1036, 966, (186, 140, 92))
    r = np.random.default_rng(3)
    for k in range(60):
        c.ell(r.uniform(420, 1030), r.uniform(350, 960), 3, 3, (160, 118, 76))
    # лампы в решётке
    for x in (300, 800):
        c.rect(x - 40, 60, x + 40, 110, (60, 60, 60), r=10)
        c.ell(x, 85, 24, 18, (255, 236, 170))
    return img


PHOTOS = [  # (x, y, волосы, фон)
    (500, 440, (90, 60, 30), (190, 210, 220)), (640, 470, (200, 200, 200), (210, 200, 190)),
    (790, 440, (30, 30, 30), (200, 215, 200)), (940, 470, (160, 90, 40), (220, 205, 210)),
    (500, 640, (60, 40, 30), (205, 205, 220)), (940, 650, (40, 30, 20), (215, 215, 200)),
    (500, 840, (150, 150, 150), (200, 220, 215)), (640, 860, (90, 50, 20), (220, 210, 195)),
    (790, 850, (20, 20, 20), (205, 200, 220)), (940, 860, (170, 120, 60), (210, 220, 205)),
]


def photo(c, x, y, hair, back, s=1.0, vlad=False):
    c.rect(x - 58 * s + 5, y - 70 * s + 6, x + 58 * s + 5, y + 74 * s + 6, (0, 0, 0, 60))
    c.rect(x - 58 * s, y - 70 * s, x + 58 * s, y + 74 * s, (248, 248, 244))
    c.rect(x - 48 * s, y - 60 * s, x + 48 * s, y + 40 * s, back)
    if vlad:
        char(c, x, y + 98 * s, 200 * s, VLAD, expr="neutral")
    else:
        c.rect(x - 30 * s, y + 10 * s, x + 30 * s, y + 40 * s, tuple(int(q * 0.6) for q in back))
        c.ell(x, y - 14 * s, 26 * s, 26 * s, SKIN)
        c.chord(x, y - 16 * s, 28 * s, 180, 360, hair)
        for e in (-1, 1):
            c.ell(x + e * 9 * s, y - 12 * s, 3.4 * s, 3.4 * s, (20, 20, 24))
        c.line([(x - 6 * s, y + 2 * s), (x + 6 * s, y + 2 * s)], (60, 30, 30), 2.2 * s)
    c.ell(x, y - 64 * s, 7 * s, 7 * s, (210, 40, 40))


def bunker_world(t):
    img = bg_bunker(OY[0]).copy()
    c = C(img)
    pins = [(px, py - 64) for px, py, *_ in PHOTOS]
    for a, b in ((0, 4), (4, 6), (2, 8), (3, 5), (1, 7), (5, 9)):
        c.line([pins[a], (720, 586)], (200, 30, 30), 2.5)
    for x, y, hair, back in PHOTOS:
        photo(c, x, y, hair, back)
    photo(c, 720, 650, None, (190, 215, 225), s=1.15, vlad=True)
    if t >= EV["label"]:
        k = ease((t - EV["label"]) / 0.5)
        c.rect(640, 760, 640 + 165 * k, 808, (245, 240, 200))
        if k > 0.5:
            c.text(722, 784, "ОБЪЕКТ № 17", FB, 27, (200, 30, 30))
        a1 = 360 * ease((t - EV["label"] - 0.3) / 0.8)
        if a1 > 1:
            c.arc(720, 655, 96, 110, -90, -90 + a1, (215, 30, 30), 7)
    return img, c


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


def lab_world(t, screen="off", siren=0.0, zombies=0.0, kneel=0.0):
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
                char(c, x, fy + 120 * sc + rise, 330 * sc, zombie_spec(i + ri), t, zombie=True, kneel=kd, bow=kd,
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


# ================================================================ шоты
def tags_from(f, items, t):
    out = []
    for who, pos in items:
        name, sub, col = TAGS[who](t)
        out.append((f(*pos), name, sub, col))
    return out


def shot_hut(t, lt):
    img, c = hut_world(t, door_k=clamp((t - EV["door_off"]) / 0.45) if t >= EV["door_off"] else 0)
    grab = ease((t - EV["grab"]) / 0.3)
    scared = "scared" if t > EV["door_off"] else "angry"
    p1, _ = char(c, 330, 1240, 560, VLAD, t, talk=talking("vlad", t), expr=scared,
                 arms=(8, lerp(8, 62, grab)), look=(1 if t < EV["door_off"] else 1.2, 0))
    p2, _ = char(c, 600, 1240, 540, VIKA, t, talk=talking("vika", t), expr="scared",
                 arms=(lerp(8, 62, grab), 8), look=(-1 if talking("vlad", t) else 1, 0))
    if t < EV["door_off"]:
        frame, f = cam(img, lerp(1.0, 1.12, ease(lt / 9)), 560, 1040)
    else:
        frame, f = cam(img, 1.05, 620, 1020)
    items = [("vlad", p1), ("vika", p2)]
    if t >= EV["door_off"]:
        items.append(("rescuer", (885, 1250 - 520 * 1.1)))
    return frame, tags_from(f, items, t)


def shot_hatch(t, lt):
    img, c = hut_world(t, door_k=1.0, hatch_open=clamp((t - EV["hatch_open"]) / 0.5))
    jt = t - EV["jump"]
    poses = []
    for who, sp, x0, delay in (("vika", VIKA, 600, 0.0), ("vlad", VLAD, 330, 0.25)):
        k = ease((t - EV["hatch_open"] - 0.3) / 1.0)
        x = lerp(x0, 540 + (25 if who == "vika" else -25), k)
        y = lerp(1240, 1395, k)
        fall = max(0.0, jt - delay)
        y += 1800 * fall * fall
        run = 1.0 if 0.02 < k < 0.98 else 0.0
        p, _ = char(c, x, y, 520, sp, t, run=run, expr="scared", arms=(150, 150) if fall > 0 else (20, 20))
        if fall < 0.25:
            poses.append((who, p))
    img.paste(bg_hut(OY[0]), (0, 0), floor_front_mask(OY[0]))
    c = C(img)
    c.poly([(390, 1350), (690, 1350), (720, 1470), (360, 1470)], (10, 8, 8, 0))
    frame, f = cam(img, lerp(1.0, 1.4, ease(lt / 2.5)), 560, 1220)
    return frame, tags_from(f, poses, t)


def shot_below(t, lt):
    img = Image.new("RGB", (WW, WH), (22, 16, 14))
    c = C(img)
    c.rect(270, 230, 810, 760, (60, 44, 34))
    c.rect(290, 250, 790, 740, (196, 146, 96))
    lamp_glow(c, 540, 300, 260)
    k = ease((t - EV["below"] - 0.3) / 0.8)
    char(c, 540, lerp(2250, 1800, k), 1500, RESCUER, t, talk=talking("rescuer", t),
         expr="creepy", mouth="creepy" if t >= EV["smile"] else "neutral", look=(0, 1.2), arms=(10, 10))
    c.rect(0, 0, 1080, 250, (22, 16, 14))
    c.rect(0, 740, 1080, 1920, (22, 16, 14))
    c.rect(0, 0, 290, 1920, (22, 16, 14))
    c.rect(790, 0, 1080, 1920, (22, 16, 14))
    lid = ease((t - EV["lid_close"]) / 0.2)
    if lid > 0:
        c.rect(290, 250, 290 + 500 * lid, 740, (84, 56, 36))
    # мы внизу смотрим вверх
    p1, _ = char(c, 300, 2050, 760, VLAD, t, expr="shock", look=(0.5, -1.4), arms=(20, 20))
    p2, _ = char(c, 780, 2060, 740, VIKA, t, expr="scared", look=(-0.5, -1.4), arms=(20, 20))
    frame, f = cam(img, lerp(1.0, 1.08, lt / 6), 540, 960)
    items = []
    items += [("vlad", p1), ("vika", p2)]
    return frame, tags_from(f, items, t)


def shot_dark(t, lt):
    img = Image.new("RGB", (WW, WH), (8, 8, 10))
    c = C(img)
    lamp_glow(c, 700, 1100, 520, (120, 140, 200))
    p, _ = char(c, 540, 2350, 1500, VLAD, t, expr="scared", look=(0.3, 0.2), arms=(10, 10))
    frame, f = cam(img, lerp(1.0, 1.1, lt / 2.2), 540, 1000)
    frame = to_img(np.asarray(frame, np.float32) * 0.75)
    return frame, tags_from(f, [("vlad", p)], t)


def tunnel_scroll(t):
    # бег с постоянной скоростью до остановки у двери
    t0, t1 = EV["tunnel"], EV["door_stop"]
    sp = 520
    if t < t1 - 0.5:
        return sp * (t - t0)
    tt = t1 - 0.5 - t0
    k = clamp((t - (t1 - 0.5)) / 0.5)
    return sp * tt + sp * 0.5 * (k - k * k / 2) * 1.0


def shot_tunnel(t, lt):
    scroll = tunnel_scroll(t)
    stop = tunnel_scroll(EV["door_stop"] + 1)
    door_x = 860 + (stop - scroll)
    img, c = tunnel_world(t, scroll, door_x if door_x < 1400 else None)
    run = 1.0 if t < EV["door_stop"] - 0.2 else 0.0
    roar = t >= EV["roar"]
    # зомби, спрыгнувший в люк, вдалеке слева
    if t >= EV["drop"] - 0.3:
        k = ease((t - EV["drop"] + 0.3) / 0.3)
        char(c, 80, lerp(700, 1220, k), 360, zombie_spec(2), t, zombie=True, arms=(150, 150))
    lookx = -1.4 if roar else 1.0
    p1, _ = char(c, 330, 1215, 520, VLAD, t, run=run, expr="scared" if roar else "angry", look=(lookx, 0))
    p2, hands = char(c, 590, 1215, 500, VIKA, t + 0.13, run=run, expr="scared",
                     look=(lookx, 0), arms=(10, 120), phone=True)
    if t < EV["door_stop"]:
        frame, f = cam(img, 1.0, 540, 960)
    elif t < EV["door_stop"] + 3.6:
        k = ease((t - EV["door_stop"]) / 1.0)
        frame, f = cam(img, lerp(1.0, 1.9, k), lerp(540, 860, k), lerp(960, 820, k))
    else:
        k = ease((t - EV["door_stop"] - 3.6) / 0.8)
        z = lerp(1.9, 1.15, k)
        frame, f = cam(img, z, lerp(860, 480, k), lerp(820, 1000, k))
    # темнота и фонарик телефона
    hx, hy = f(*hands[1])
    a = np.asarray(frame, np.float32)
    yy, xx = np.ogrid[0:H:4, 0:W:4]
    dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
    beam = np.clip(1 - np.abs((yy - hy) / np.maximum(xx - hx, 1)) * 1.6, 0, 1) * (xx > hx) * np.clip(1 - (xx - hx) / 1100, 0, 1)
    light = np.clip(0.28 + np.exp(-(dist / 380) ** 2) * 0.75 + beam * 0.6, 0, 1.1)
    if roar:
        light = light + np.exp(-((xx - 0) ** 2 + (yy - 1000) ** 2) / 400 ** 2) * 0.5 * clamp((t - EV["roar"]) / 0.3)
    light = np.asarray(Image.fromarray((np.clip(light, 0, 1.2) * 200).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                       np.float32) / 200
    tint = np.array([1.0, 0.75, 0.75], np.float32) if roar else np.array([1, 1, 1], np.float32)
    frame = to_img(a * light[:, :, None] * tint)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_bunker(t, lt):
    img, c = bunker_world(t)
    k_door = ease((t - EV["bunker"]) / 0.25)
    d = C(img)
    d.rect(34, 574, 34 + 202 * k_door, 1222, (90, 96, 104))
    vk = ease((t - EV["talk_bunker"]) / 0.5)
    look_v = (1, 0) if t < EV["board"] else (1.2, -0.6)
    p1, _ = char(c, lerp(300, 380, vk), 1210, 540, VLAD, t, talk=talking("vlad", t),
                 expr="shock" if t > EV["label"] else "angry", look=look_v if t < EV["talk_bunker"] else (1, 0))
    away = t >= EV["look_away"]
    p2, _ = char(c, lerp(470, 640, vk), 1210, 520, VIKA, t, talk=talking("vika", t), expr="sad" if away else "neutral",
                 look=(1.3, 0.6) if away else (-1, 0))
    if t < EV["board"]:
        frame, f = cam(img, 1.05, 400, 1000)
    elif t < EV["photo_zoom"]:
        k = ease((t - EV["board"]) / 1.2)
        frame, f = cam(img, lerp(1.05, 1.5, k), lerp(400, 720, k), lerp(1000, 760, k))
    elif t < EV["talk_bunker"]:
        k = ease((t - EV["photo_zoom"]) / 2.0)
        frame, f = cam(img, lerp(1.5, 2.6, k), 720, lerp(760, 690, k))
        return frame, []
    else:
        k = ease((t - EV["talk_bunker"]) / 0.8)
        frame, f = cam(img, lerp(1.5, 1.15, k), lerp(720, 510, k), lerp(760, 900, k))
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def screen_frame(t, mode):
    """Крупный план экрана старого компьютера."""
    img = Image.new("RGB", (WW, WH), (30, 36, 40))
    c = C(img)
    c.rect(30, 420, 1050, 1500, (214, 206, 180), r=60)
    c.rect(90, 480, 990, 1260, (18, 22, 20), r=40)
    if mode == "sci":
        c.rect(110, 500, 970, 1240, (150, 170, 175), r=30)
        c.rect(110, 1000, 970, 1240, (120, 136, 140))
        c.rect(150, 560, 390, 760, (120, 136, 140))
        char(c, 540, 1500, 900, SCI, t, talk=talking("sci", t), expr="sad", arms=(10, 10))
        c.text(170, 540, "● REC", FB, 30, (220, 40, 40), anchor="lm")
        c.text(930, 540, "ЗАПИСЬ 07", FM, 26, (240, 240, 240), anchor="rm")
    elif mode in ("found", "protocol"):
        c.rect(110, 500, 970, 1240, (10, 24, 14), r=30)
        blink = int(t * 4) % 2 == 0
        c.text(540, 760, "ОБЪЕКТ № 17", FB, 64, (90, 240, 120))
        c.text(540, 840, "ОБНАРУЖЕН", FB, 64, (90, 240, 120))
        if mode == "protocol":
            c.rect(150, 930, 930, 1150, (150, 20, 20) if blink else (100, 10, 10), r=10)
            c.text(540, 990, "ЗАПУСК ПРОТОКОЛА", FB, 50, (255, 240, 240))
            c.text(540, 1080, "УНИЧТОЖЕНИЯ", FB, 64, (255, 240, 240))
    elif mode == "boot":
        c.rect(110, 500, 970, 1240, (10, 24, 14), r=30)
        if int(t * 3) % 2:
            c.text(160, 560, "_", FM, 40, (90, 240, 120), anchor="lm")
    # строки развёртки
    for y in range(500, 1240, 10):
        c.rect(110, y, 970, y + 3, (0, 0, 0, 40))
    c.ell(930, 1380, 20, 20, (90, 220, 110) if mode != "off" else (60, 60, 60))
    return img


def shot_lab(t, lt):
    if t < EV["screen"]:
        img, c = lab_world(t, screen="on" if t >= EV["pc_on"] else "off")
        p1, _ = char(c, 400, 1210, 540, VLAD, t, expr="angry", look=(1.2, 0))
        p2, _ = char(c, 760, 1210, 520, VIKA, t, expr="neutral", look=(1.2, -0.4), arms=(8, 40))
        frame, f = cam(img, lerp(1.05, 1.15, ease(lt / 1.8)), 640, 900)
        return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)
    if t < EV["two_lab"]:
        img = screen_frame(t, "sci")
        frame, f = cam(img, lerp(1.0, 1.15, ease((t - EV["screen"]) / 8)), 540, 880)
        return frame, tags_from(f, [("sci", (540, 1500 - 900 * 1.12))], t)
    img, c = lab_world(t, screen="on")
    turn = ease((t - EV["turn"]) / 0.4)
    gun = ease((t - EV["gun"]) / 0.35)
    p1, _ = char(c, 400, 1210, 540, VLAD, t, expr="shock" if t > EV["gun"] else "angry",
                 look=(lerp(-0.3, 1.3, turn), 0), arms=(8 + 30 * gun, 8 + 30 * gun))
    p2, _ = char(c, 720, 1210, 520, VIKA, t, talk=talking("vika", t), expr="sad", look=(-1.2, 0),
                 arms=(lerp(8, 88, gun), 8), gun=-1 if gun > 0.05 else None)
    frame, f = cam(img, lerp(1.1, 1.2, ease((t - EV["two_lab"]) / 5)), 560, 900)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_alert(t, lt):
    if t < EV["siren"]:
        mode = "boot" if t < EV["found"] else ("found" if t < EV["protocol"] else "protocol")
        img = screen_frame(t, mode)
        frame, f = cam(img, lerp(1.0, 1.2, ease(lt / 8)), 540, 900)
        return frame, []
    img, c = lab_world(t, screen="red", siren=1.0)
    p1, _ = char(c, 400, 1210, 540, VLAD, t, expr="shock", look=(0, -1), arms=(30, 30))
    p2, _ = char(c, 720, 1210, 520, VIKA, t, expr="scared", look=(0, -1), arms=(88, 8), gun=-1)
    frame, f = cam(img, 1.15, 560, 900)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_zombies(t, lt):
    zk = (t - EV["zombies"]) / 1.6
    kn = (t - EV["kneel"]) / 1.4
    img, c = lab_world(t, screen="red", siren=1.0, zombies=clamp(zk), kneel=clamp(kn))
    drop = t - EV["drop_gun"]
    creator = t >= EV["creator"]
    p1, _ = char(c, 470, 1250, 580, VLAD, t, expr="shock" if not creator else "sad",
                 look=(0, 1.6) if creator else (-0.2, -0.6), arms=(55, 55) if creator else (25, 25))
    p2, _ = char(c, 800, 1230, 520, VIKA, t, talk=talking("vika", t), expr="shock" if drop > 0 else "scared",
                 look=(-1.3, 0), arms=(lerp(88, 15, ease(drop / 0.3)) if drop > 0 else 88, 8),
                 gun=-1 if drop < 0 else None)
    if drop > 0:
        fy = min(1250, 1100 + 900 * drop * drop)
        bounce = 30 * abs(math.sin(drop * 10)) * math.exp(-drop * 4) if fy >= 1250 else 0
        c.poly([(660, fy - 22 - bounce), (712, fy - 22 - bounce), (712, fy - 10 - bounce), (676, fy - 10 - bounce),
                (672, fy + 6 - bounce), (660, fy + 6 - bounce)], (35, 35, 40))
    if creator:
        k = ease((t - EV["creator"]) / 1.4)
        frame, f = cam(img, lerp(1.0, 1.35, k), lerp(540, 520, k), lerp(980, 820, k))
    else:
        frame, f = cam(img, lerp(1.0, 1.06, lt / 13), 540, 980)
    return frame, tags_from(f, [("vlad", p1), ("vika", p2)], t)


def shot_black(t, lt):
    return Image.new("RGB", (W, H)), []


def shot_final(t, lt):
    img = Image.new("RGB", (W, H), (10, 10, 14))
    d = ImageDraw.Draw(img)
    a = ease(lt / 0.5) * (1 - ease((lt - 4.2) / 0.6))
    words = FINAL_TEXT.split()
    f = font(FB, 92)
    for i, wd in enumerate(words):
        k = ease((lt - i * 0.25) / 0.35)
        col = tuple(int(255 * a * k) for _ in range(3))
        d.text((540, 840 + i * 120 - (1 - k) * 30), wd, font=f, fill=col, anchor="mm", stroke_width=0)
    d.rectangle([290, 1110, 290 + 500 * ease((lt - 0.8) / 0.8), 1122], fill=(int(226 * a), int(46 * a), int(46 * a)))
    return img, []


SHOTS = [
    (0.0, EV["hatch_shot"], shot_hut), (EV["hatch_shot"], EV["below"], shot_hatch),
    (EV["below"], EV["dark"], shot_below), (EV["dark"], EV["tunnel"], shot_dark),
    (EV["tunnel"], EV["bunker"], shot_tunnel), (EV["bunker"], EV["lab"], shot_bunker),
    (EV["lab"], EV["alert"], shot_lab), (EV["alert"], EV["zombies_shot"], shot_alert),
    (EV["zombies_shot"], EV["black"], shot_zombies), (EV["black"], EV["final"], shot_black),
    (EV["final"], DURATION, shot_final),
]

# шоты со сдвигом сцены вниз, чтобы над головами помещались бейджи
SHIFTED = {shot_hut: 250, shot_hatch: 250, shot_tunnel: 250, shot_bunker: 250, shot_lab: 250,
           shot_alert: 250, shot_zombies: 250}

FLASHES = [(EV["door_off"], (255, 255, 255), 0.12), (EV["lid_close"], (0, 0, 0), 0.15), (EV["roar"], (255, 40, 30), 0.25),
           (EV["label"], (220, 30, 30), 0.25), (EV["gun"], (255, 255, 255), 0.1), (EV["creator"], (255, 255, 255), 0.25)]


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
    if EV["siren"] <= t < EV["black"] and fn is not shot_alert or (fn is shot_alert and t >= EV["siren"]):
        p = 0.5 + 0.5 * math.sin(t * 9)
        a = a * (1 - 0.25 * p) + np.array([255, 20, 20], np.float32) * 0.25 * p
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

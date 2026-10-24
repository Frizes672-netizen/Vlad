"""Рендер кадров хоррор-шорта «Ночь только начинается».

Все изображения рисуются процедурно (PIL + numpy), кадры передаются в ffmpeg.
Запуск: python3 render.py build/video.mp4 [--preview СЕКУНДЫ ...]
"""
import math
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from timeline import (CHAPTERS, DURATION, EV, FINAL_START, FPS, H, SPEAKERS,
                      SUBS, W)

FD = "/usr/share/fonts/truetype/"
F_SANS_B = FD + "dejavu/DejaVuSans-Bold.ttf"
F_SANS_BI = FD + "liberation/LiberationSans-BoldItalic.ttf"
F_SERIF_B = FD + "dejavu/DejaVuSerif-Bold.ttf"
F_MONO_B = FD + "dejavu/DejaVuSansMono-Bold.ttf"


# ================================================================ утилиты
@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return ease((t - a) / (b - a))


def lerp(a, b, k):
    return a + (b - a) * k


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def vgrad(w, h, stops):
    ys = np.linspace(0, 1, h)
    pos = [p for p, _ in stops]
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(ys, pos, [col[c] for _, col in stops])[:, None]
    return out


def smooth_noise(w, h, cell, seed):
    r = np.random.default_rng(seed)
    small = r.random((max(2, h // cell), max(2, w // cell)))
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255


def fbm(w, h, seed, cells=(400, 160, 60, 20, 6)):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for i, c in enumerate(cells):
        out += smooth_noise(w, h, c, seed + i) * amp
        tot += amp
        amp *= 0.55
    return out / tot


def radial(w, h, cx, cy, r):
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r
    return np.clip(1 - d, 0, 1) ** 2


def add_glow(img, cx, cy, r, col, strength=1.0):
    a = np.asarray(img, np.float32)
    g = radial(a.shape[1], a.shape[0], cx, cy, r)[:, :, None] * np.array(col, np.float32) * strength
    return to_img(a + g)


# ================================================================ камера / тряска
SHAKES = [  # (время, амплитуда px, затухание с)
    (EV["roar"], 26, 1.2), (EV["pursuers"], 14, 0.6), (EV["slam"], 42, 0.45), (EV["table"], 8, 0.9),
    (EV["table"] + 1.0, 10, 0.3), (EV["thud"], 48, 0.55), (EV["lunge"], 34, 0.45), (EV["wall_text"], 16, 0.8),
    *[(k, 24 + i * 3, 0.25) for i, k in enumerate(EV["knocks"])], (EV["head_turn"] + 0.3, 14, 0.5),
    *[(EV["bend"] + d, a, 0.35) for d, a in ((0.1, 22), (0.9, 30), (1.7, 40), (2.4, 50), (2.9, 60))],
    (EV["final"], 18, 0.7),
]


def shake(t):
    dx = dy = 0.0
    for i, (te, a, dec) in enumerate(SHAKES):
        if t >= te:
            k = a * math.exp(-(t - te) / dec)
            dx += k * math.sin(t * 61 + i * 1.7)
            dy += k * math.cos(t * 53 + i * 2.3)
    return dx, dy


CUR_T = [0.0]


def cam(img, zoom=1.0, cx=0.5, cy=0.5, rot=0.0):
    """Вырезает «окно камеры» из мира размера W×H с учётом тряски."""
    dx, dy = shake(CUR_T[0])
    if abs(dx) + abs(dy) > 0.5:
        zoom = max(zoom, 1.0 + (abs(dx) + abs(dy)) * 2.2 / W + 0.02)
    ww, hh = W / zoom, H / zoom
    x0 = clamp(cx * W - ww / 2 + dx / zoom, 0, W - ww)
    y0 = clamp(cy * H - hh / 2 + dy / zoom, 0, H - hh)
    if rot:
        img = img.rotate(rot, Image.BICUBIC, center=(cx * W, cy * H))
    if zoom == 1.0 and x0 == 0 and y0 == 0:
        return img
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + ww, y0 + hh))


# ================================================================ персонажи
def limb(d, p, a1, l1, a2, l2, w, col):
    """Конечность из двух сегментов; углы от вертикали вниз, в радианах."""
    k = (p[0] + math.sin(a1) * l1, p[1] + math.cos(a1) * l1)
    e = (k[0] + math.sin(a1 + a2) * l2, k[1] + math.cos(a1 + a2) * l2)
    for a, b in ((p, k), (k, e)):
        d.line([a, b], fill=col, width=int(w))
    for q in (p, k, e):
        d.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=col)
    return e


def person(d, x, feet, s, phase=0.0, run=0.0, col=(10, 9, 10), lean=0.0,
           hair=False, facing=1, arm_reach=None, blood=None, eyes=None, scale_w=1.0):
    """Силуэт человека. s — рост в px, run — степень бега 0..1."""
    w = s * 0.075 * scale_w
    hip = (x, feet - s * 0.47 - abs(math.sin(phase)) * s * 0.03 * run)
    sh = (x + lean * s * 0.18 * facing, hip[1] - s * 0.33)
    head = (sh[0] + lean * s * 0.05 * facing, sh[1] - s * 0.11)
    hr = s * 0.068
    # ноги
    for side in (0, 1):
        ph = phase + side * math.pi
        a1 = math.sin(ph) * 0.75 * run * facing
        a2 = -max(0.0, math.cos(ph)) * 1.2 * run * facing
        limb(d, hip, a1, s * 0.25, a2, s * 0.24, w * 1.15, col)
    # торс
    tw = s * 0.11 * scale_w
    d.polygon([(sh[0] - tw, sh[1]), (sh[0] + tw, sh[1]), (hip[0] + tw * 0.75, hip[1]),
               (hip[0] - tw * 0.75, hip[1])], fill=col)
    d.ellipse([hip[0] - tw * 0.8, hip[1] - tw * 0.5, hip[0] + tw * 0.8, hip[1] + tw * 0.5], fill=col)
    # руки
    ends = []
    for side in (0, 1):
        ph = phase + side * math.pi + math.pi
        if arm_reach is not None and side == 0:
            ang = math.atan2(arm_reach[0] - sh[0], arm_reach[1] - sh[1])
            dist = min(math.hypot(arm_reach[0] - sh[0], arm_reach[1] - sh[1]), s * 0.36)
            ends.append(limb(d, (sh[0], sh[1] + w * 0.4), ang, dist / 2, 0, dist / 2, w, col))
            continue
        a1 = math.sin(ph) * 0.9 * run * facing + 0.08 * (1 - run) * (1 if side else -1)
        a2 = -0.9 * run * facing - 0.1
        ends.append(limb(d, (sh[0], sh[1] + w * 0.4), a1, s * 0.16, a2, s * 0.16, w * 0.9, col))
    # шея и голова
    d.line([sh, head], fill=col, width=int(w))
    d.ellipse([head[0] - hr, head[1] - hr * 1.15, head[0] + hr, head[1] + hr * 1.05], fill=col)
    if hair:
        d.polygon([(head[0] - hr * 1.15, head[1] - hr * 0.4), (head[0] + hr * 1.15, head[1] - hr * 0.4),
                   (head[0] + hr * 1.35, head[1] + hr * 2.8), (head[0] - hr * 1.3, head[1] + hr * 2.7)], fill=col)
        d.ellipse([head[0] - hr * 1.2, head[1] - hr * 1.3, head[0] + hr * 1.2, head[1] + hr * 0.6], fill=col)
    if blood:
        r = np.random.default_rng(int(x))
        for _ in range(7):
            bx = x + r.uniform(-tw, tw)
            by = r.uniform(sh[1], hip[1] + s * 0.2)
            br = r.uniform(0.08, 0.25) * tw
            d.ellipse([bx - br, by - br, bx + br, by + br * 1.6], fill=blood)
        d.ellipse([head[0] - hr * 0.6, head[1] + hr * 0.1, head[0] + hr * 0.5, head[1] + hr * 0.9], fill=blood)
    if eyes:
        for ex in (-0.35, 0.35):
            cx, cy = head[0] + ex * hr, head[1] - hr * 0.15
            rr = max(2, hr * 0.13)
            d.ellipse([cx - rr, cy - rr * 0.7, cx + rr, cy + rr * 0.7], fill=eyes)
    return head, hr, ends


def rim_person(img, x, feet, s, rim=(90, 110, 140), off=(5, -3), **kw):
    """Силуэт с контровым светом: светлый контур + тёмная фигура."""
    d = ImageDraw.Draw(img)
    kw2 = dict(kw)
    kw2.pop("blood", None)
    kw2.pop("eyes", None)
    person(d, x + off[0], feet + off[1], s * 1.01, col=rim, **kw2)
    return person(d, x, feet, s, **kw)


def leaf(d, x, y, length, ang, width, col):
    pts_l, pts_r = [], []
    for i in range(13):
        k = i / 12
        px = x + math.cos(ang) * length * k
        py = y + math.sin(ang) * length * k + math.sin(k * math.pi) * length * 0.08
        wv = math.sin(k * math.pi) ** 0.8 * width
        nx, ny = -math.sin(ang), math.cos(ang)
        pts_l.append((px + nx * wv, py + ny * wv))
        pts_r.append((px - nx * wv, py - ny * wv))
    d.polygon(pts_l + pts_r[::-1], fill=col)


def frond(d, x, y, length, ang, col, n=14, scale=1.0):
    d.line([(x, y), (x + math.cos(ang) * length, y + math.sin(ang) * length)], fill=col, width=max(2, int(6 * scale)))
    for i in range(1, n):
        k = i / n
        px, py = x + math.cos(ang) * length * k, y + math.sin(ang) * length * k
        ll = length * 0.45 * math.sin(k * math.pi) * scale + 10
        for sgn in (-1, 1):
            leaf(d, px, py, ll, ang + sgn * 1.0 + 0.3, ll * 0.16, col)


# ================================================================ фоны
@lru_cache(None)
def bg_beach():
    a = vgrad(W, H, [(0, (4, 6, 14)), (0.25, (14, 14, 28)), (0.355, (95, 40, 22)),
                     (0.365, (30, 22, 30)), (0.6, (10, 14, 22)), (0.6001, (34, 26, 22)),
                     (1.0, (14, 10, 9))])
    sand = fbm(W, H, 3)[:, :, None]
    a[int(H * 0.6):] *= (0.75 + sand[int(H * 0.6):] * 0.6)
    a += radial(W, H, 700, 690, 650)[:, :, None] * np.array([150, 60, 15], np.float32)
    clouds = fbm(W, H, 9)
    a[: int(H * 0.36)] += (clouds[: int(H * 0.36), :, None] ** 3) * np.array([40, 30, 40], np.float32)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    # горящий корабль на горизонте
    hy = int(H * 0.36)
    d.polygon([(560, hy - 10), (870, hy - 38), (850, hy + 4), (590, hy + 6)], fill=(6, 5, 6))
    for mx, mh, tilt in ((640, 230, 12), (730, 300, -6), (810, 190, 20)):
        d.line([(mx, hy - 15), (mx + tilt, hy - 15 - mh)], fill=(6, 5, 6), width=7)
        d.line([(mx - 50 + tilt * 0.6, hy - mh * 0.6), (mx + 50 + tilt * 0.6, hy - mh * 0.65)], fill=(6, 5, 6), width=4)
    # обломки и камни на песке
    d.polygon([(0, 1700), (240, 1640), (380, 1690), (420, 1920), (0, 1920)], fill=(8, 7, 7))
    d.polygon([(700, 1560), (1010, 1500), (1030, 1530), (720, 1595)], fill=(22, 15, 11))
    d.polygon([(820, 1760), (1080, 1690), (1080, 1920), (860, 1920)], fill=(7, 6, 6))
    return img


@lru_cache(None)
def jungle_layer(depth, seed):
    """Полоса джунглей шириной 3W (RGBA) для параллакса."""
    r = np.random.default_rng(seed)
    w = W * 3
    img = Image.new("RGBA", (w, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = {0: (24, 38, 38, 255), 1: (10, 18, 15, 255), 2: (1, 2, 2, 255)}[depth]
    n_trunks = {0: 26, 1: 10, 2: 3}[depth]
    for _ in range(n_trunks):
        x = r.uniform(0, w)
        tw = r.uniform(*{0: (15, 35), 1: (40, 80), 2: (90, 150)}[depth])
        bend = r.uniform(-60, 60)
        d.polygon([(x - tw / 2, H), (x + tw / 2, H), (x + bend + tw * 0.35, 0), (x + bend - tw * 0.35, 0)], fill=col)
        if depth < 2:
            for _ in range(3):
                vy = r.uniform(0, H * 0.7)
                d.line([(x, vy), (x + r.uniform(-200, 200), vy + r.uniform(200, 700))], fill=col, width=4)
    n_fr = {0: 50, 1: 22, 2: 7}[depth]
    for _ in range(n_fr):
        x = r.uniform(0, w)
        y = r.choice([r.uniform(-100, H * 0.35), r.uniform(H * 0.7, H + 100)])
        ln = r.uniform(*{0: (150, 300), 1: (300, 550), 2: (600, 1000)}[depth])
        frond(d, x, y, ln * (0.7 if depth == 2 else 1), r.uniform(0, 2 * math.pi), col, scale=1 + depth * 0.25)
    if depth == 2:
        img = img.filter(ImageFilter.GaussianBlur(10))
    return img


@lru_cache(None)
def bg_jungle_sky():
    a = vgrad(W, H, [(0, (8, 14, 18)), (0.4, (34, 52, 58)), (1, (12, 18, 18))])
    a += radial(W, H, 760, 380, 800)[:, :, None] * np.array([80, 110, 125], np.float32)
    mist = fbm(W, H, 21)
    a += (mist[:, :, None] ** 2) * np.array([25, 35, 38], np.float32) * np.linspace(0.3, 1, H)[:, None, None]
    return to_img(a).convert("RGBA")


def jungle_frame(t, speeds=(140, 620, 1900), tint=None, offset=0):
    base = bg_jungle_sky().copy()
    for depth, sp in enumerate(speeds):
        layer = jungle_layer(depth, 40 + depth)
        x = int((t * sp + offset * (depth + 1)) % (W * 2))
        base.alpha_composite(layer.crop((x, 0, x + W, H)))
    img = base.convert("RGB")
    if tint is not None:
        a = np.asarray(img, np.float32) * np.array(tint, np.float32)
        img = to_img(a)
    return img


@lru_cache(None)
def bg_hut(open_door_view=False):
    a = vgrad(W, H, [(0, (20, 15, 11)), (0.5, (34, 25, 18)), (0.78, (18, 13, 10)),
                     (0.7801, (14, 10, 8)), (1, (5, 4, 3))])
    grain = fbm(W, H, 77, cells=(300, 80, 20, 5))
    planks = np.asarray(Image.fromarray((np.random.default_rng(5).random((1, 12)) * 255).astype(np.uint8))
                        .resize((W, 1), Image.NEAREST), np.float32)[0] / 255
    a *= (0.7 + grain[:, :, None] * 0.5) * (0.8 + planks[None, :, None] * 0.35)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 90):
        d.line([(x, 0), (x, int(H * 0.78))], fill=(6, 4, 3), width=5)
    for y in range(int(H * 0.78), H, 70):
        d.line([(0, y), (W, y + 20)], fill=(3, 2, 2), width=3)
    # дверной проём
    d.rectangle([330, 500, 750, 1505], fill=(8, 6, 5))
    d.rectangle([340, 515, 740, 1500], fill=(3, 4, 6))
    # лучи лунного света сквозь щели
    beams = Image.new("L", (W, H), 0)
    bd = ImageDraw.Draw(beams)
    for bx in (60, 230, 820, 960):
        bd.polygon([(bx, 0), (bx + 26, 0), (bx + 520, H), (bx + 420, H)], fill=55)
    beams = beams.filter(ImageFilter.GaussianBlur(25))
    a = np.asarray(img, np.float32) + (np.asarray(beams, np.float32) / 255)[:, :, None] * np.array([90, 110, 140], np.float32)
    return to_img(a)


@lru_cache(None)
def door_leaf():
    """Текстура дверного полотна 400×985."""
    w, h = 400, 985
    a = np.full((h, w, 3), (52, 37, 25), np.float32)
    g = fbm(w, h, 12, cells=(200, 60, 12, 4))
    a *= (0.6 + g[:, :, None] * 0.7)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for x in range(0, w, 80):
        d.line([(x, 0), (x, h)], fill=(18, 12, 8), width=5)
    for y in (140, 840):
        d.rectangle([10, y, w - 10, y + 50], fill=(40, 28, 19), outline=(15, 10, 7), width=3)
    d.line([(20, 190), (w - 20, 840)], fill=(38, 26, 18), width=40)
    d.ellipse([w - 60, 480, w - 30, 510], fill=(70, 60, 50))
    return img


@lru_cache(None)
def outside_view():
    a = vgrad(420, 1000, [(0, (12, 20, 34)), (0.6, (30, 44, 60)), (1, (20, 26, 30))])
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for i in range(6):
        x = i * 80 + 20
        d.polygon([(x, 1000), (x + 30, 1000), (x + 25, 380 + (i * 47) % 120), (x + 5, 380 + (i * 47) % 120)], fill=(8, 12, 14))
        frond(d, x + 15, 400 + (i * 47) % 120, 160, -2.0 + i * 0.4, (8, 12, 14), n=8)
    return img


def table(d, x, y, s=1.0):
    col, dark = (48, 33, 22), (22, 15, 10)
    d.polygon([(x - 260 * s, y), (x + 260 * s, y), (x + 300 * s, y + 70 * s), (x - 300 * s, y + 70 * s)], fill=col)
    d.rectangle([x - 300 * s, y + 70 * s, x + 300 * s, y + 95 * s], fill=dark)
    for lx in (-270, 250):
        d.rectangle([x + lx * s, y + 95 * s, x + (lx + 25) * s, y + 260 * s], fill=dark)


SCRATCHES = [  # (время появления, x, y, длина, угол)
    (ts, x, y, ln, ang) for ts, (x, y, ln, ang) in zip(EV["scratches"], (
        (420, 980, 300, 1.25), (600, 900, 340, 1.9), (470, 1180, 260, 1.4),
        (640, 1120, 300, 2.0), (520, 820, 380, 1.6)))
]


def hut_world(t, door_angle=0.0, table_k=1.0, gap_light=1.0, feet_x=None, bulge=0.0, scratches=True):
    """Интерьер хижины в мировых координатах."""
    img = bg_hut().copy()
    d = ImageDraw.Draw(img)
    # полоска света под дверью и тени шагов
    gl = int(120 * gap_light)
    d.rectangle([340, 1496, 740, 1506], fill=(gl // 2, gl * 2 // 3, gl))
    if feet_x is not None:
        for fx in feet_x:
            d.rectangle([fx - 40, 1494, fx + 40, 1508], fill=(2, 2, 3))
    if door_angle > 0.01:
        img.paste(outside_view(), (330, 505))
        leaf_w = max(4, int(400 * math.cos(door_angle)))
        img.paste(door_leaf().resize((leaf_w, 985)), (340, 515))
        d = ImageDraw.Draw(img)
    else:
        img.paste(door_leaf(), (340, 515))
        d = ImageDraw.Draw(img)
        if scratches:
            for (ts, x, y, ln, ang) in SCRATCHES:
                k = clamp((t - ts) / 0.4)
                if k <= 0:
                    continue
                for j in range(4):
                    ox = j * 26
                    x2 = x + ox + math.cos(ang) * ln * k
                    y2 = y + j * 10 + math.sin(ang) * ln * k
                    d.line([(x + ox + 3, y + j * 10 + 3), (x2 + 3, y2 + 3)], fill=(12, 8, 6), width=9)
                    d.line([(x + ox, y + j * 10), (x2, y2)], fill=(150, 120, 92), width=6)
    if table_k > 0:
        tx = lerp(-400, 540, table_k)
        table(d, tx, 1395)
    return img


def bulge_warp(img, strength, cx=540, cy=1000, r=700):
    a = np.asarray(img)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = x - cx, y - cy
    dist = np.sqrt(dx * dx + dy * dy) / r
    f = 1 - strength * np.clip(1 - dist, 0, 1) ** 2
    sx = np.clip(cx + dx * f, 0, W - 1).astype(np.int32)
    sy = np.clip(cy + dy * f, 0, H - 1).astype(np.int32)
    return Image.fromarray(a[sy, sx])


# ================================================================ лица
@lru_cache(None)
def face_bg():
    img = cam(bg_hut(), 1.6, 0.3, 0.45).filter(ImageFilter.GaussianBlur(18))
    return to_img(np.asarray(img, np.float32) * 0.55)


def draw_face(t, mouth=0.0, eyes_open=1.0, blood=0.0, look_up=0.0, pale=0.0, tremble=0.0):
    img = face_bg().copy()
    cx, cy = 540 + math.sin(t * 23) * tremble * 6, 930 + math.cos(t * 19) * tremble * 4
    # волосы сзади
    hl = Image.new("L", (W, H), 0)
    hd = ImageDraw.Draw(hl)
    hd.ellipse([cx - 340, cy - 450, cx + 340, cy + 300], fill=255)
    hd.polygon([(cx - 360, cy - 50), (cx + 360, cy - 50), (cx + 420, H), (cx - 430, H)], fill=255)
    img.paste((14, 10, 9), (0, 0), hl.filter(ImageFilter.GaussianBlur(6)))
    # шея и плечи
    d = ImageDraw.Draw(img)
    d.polygon([(cx - 110, cy + 200), (cx + 110, cy + 200), (cx + 140, cy + 520), (cx - 140, cy + 520)], fill=(78, 66, 60))
    d.polygon([(cx - 520, H), (cx - 420, cy + 520), (cx + 420, cy + 520), (cx + 520, H)], fill=(30, 36, 30))
    # лицо с освещением луной слева
    fm = Image.new("L", (W, H), 0)
    ImageDraw.Draw(fm).ellipse([cx - 245, cy - 330, cx + 245, cy + 330], fill=255)
    fm = fm.filter(ImageFilter.GaussianBlur(4))
    skin = np.array([215, 180, 158], np.float32) * (1 - pale * 0.15) + np.array([0, 8, 14], np.float32) * pale
    xs = np.clip(1.25 - (np.arange(W) - (cx - 245)) / 490 * 0.65, 0.5, 1.3)[None, :, None]
    ys = np.linspace(1.05, 0.75, H)[:, None, None]
    shade = to_img(skin * xs * ys * np.array([0.97, 0.98, 1.02], np.float32) + np.zeros((H, W, 3), np.float32))
    img.paste(shade, (0, 0), fm)
    d = ImageDraw.Draw(img)
    # грязь на лице
    r = np.random.default_rng(3)
    for _ in range(6):
        x, y = cx + r.uniform(-200, 200), cy + r.uniform(-250, 280)
        rr = r.uniform(6, 22)
        d.ellipse([x - rr, y - rr * 0.6, x + rr, y + rr * 0.6], fill=(150, 120, 104))
    # глаза
    for side in (-1, 1):
        ex, ey = cx + side * 105, cy - 40
        op = 34 * eyes_open
        d.polygon([(ex - 70, ey + 8), (ex - 20, ey - 30), (ex + 30, ey - 34), (ex + 70, ey + 4), (ex + 20, ey + 26),
                   (ex - 30, ey + 26)], fill=(60, 45, 42))
        if op > 2:
            d.ellipse([ex - 58, ey - op * 0.75, ex + 58, ey + op * 0.65], fill=(215, 210, 205))
            iy = ey - look_up * 14
            d.ellipse([ex - 26, iy - 26, ex + 26, iy + 26], fill=(55, 42, 35))
            d.ellipse([ex - 12, iy - 12, ex + 12, iy + 12], fill=(5, 5, 5))
            d.ellipse([ex + 4 * side, iy - 14, ex + 14 * side + (0 if side > 0 else 10), iy - 4], fill=(240, 240, 240))
        d.line([(ex - 75, ey - 62 - look_up * 8), (ex + 60, ey - 76 - look_up * 10)], fill=(25, 18, 15), width=14)
    # нос
    d.line([(cx + 12, cy - 10), (cx + 30, cy + 95), (cx - 5, cy + 112)], fill=(110, 90, 80), width=7)
    # рот
    mo = 6 + mouth * 50
    d.ellipse([cx - 62, cy + 175 - mo * 0.3, cx + 62, cy + 175 + mo], fill=(110, 55, 55))
    if mouth > 0.15:
        d.ellipse([cx - 45, cy + 178, cx + 45, cy + 175 + mo * 0.8], fill=(25, 10, 10))
    # пряди волос спереди
    for sx in (-1, 1):
        d.polygon([(cx + sx * 40, cy - 345), (cx + sx * 270, cy - 260), (cx + sx * 280, cy + 260),
                   (cx + sx * 215, cy - 120), (cx + sx * 120, cy - 250)], fill=(16, 11, 10))
    # кровь из-под волос
    if blood > 0:
        bx, by = cx - 175, cy - 230
        ln = 560 * blood
        pts = [(bx + math.sin(k / 60) * 18 + k * 0.05, by + k) for k in range(0, int(ln) + 1, 8)]
        if len(pts) > 1:
            d.line(pts, fill=(105, 6, 8), width=16)
            d.line(pts, fill=(150, 12, 14), width=8)
            ex, ey = pts[-1]
            d.ellipse([ex - 12, ey - 6, ex + 12, ey + 22], fill=(130, 8, 10))
    return img


# ================================================================ спасатель
def rescuer(d, x, feet, s, turn=0.0, glow=0.0):
    """Спасатель в оранжевой форме; turn 0 — профиль, 1 — смотрит в камеру."""
    orange, dark_o, stripe = (190, 82, 22), (120, 48, 14), (200, 205, 200)
    pants = (25, 28, 32)
    hip_y, sh_y = feet - s * 0.47, feet - s * 0.8
    tw = s * 0.12
    for lx in (-0.45, 0.45):
        d.rectangle([x + lx * tw - s * 0.04, hip_y, x + lx * tw + s * 0.04, feet - s * 0.03], fill=pants)
        d.rectangle([x + lx * tw - s * 0.05, feet - s * 0.05, x + lx * tw + s * 0.06, feet], fill=(10, 10, 10))
        d.rectangle([x + lx * tw - s * 0.04, feet - s * 0.22, x + lx * tw + s * 0.04, feet - s * 0.2], fill=stripe)
    d.polygon([(x - tw * 1.2, sh_y), (x + tw * 1.2, sh_y), (x + tw * 1.1, hip_y + s * 0.04), (x - tw * 1.1, hip_y + s * 0.04)], fill=orange)
    d.polygon([(x + tw * 0.2, sh_y), (x + tw * 1.2, sh_y), (x + tw * 1.1, hip_y + s * 0.04), (x + tw * 0.3, hip_y + s * 0.04)], fill=dark_o)
    d.rectangle([x - tw * 1.15, hip_y - s * 0.12, x + tw * 1.12, hip_y - s * 0.1], fill=stripe)
    for side in (-1, 1):
        ax = x + side * tw * 1.3
        d.polygon([(ax - s * 0.04, sh_y + s * 0.01), (ax + s * 0.04, sh_y + s * 0.01), (ax + s * 0.035, hip_y + s * 0.02),
                   (ax - s * 0.035, hip_y + s * 0.02)], fill=orange if side < 0 else dark_o)
        d.rectangle([ax - s * 0.04, sh_y + s * 0.17, ax + s * 0.04, sh_y + s * 0.19], fill=stripe)
        d.ellipse([ax - s * 0.03, hip_y, ax + s * 0.03, hip_y + s * 0.06], fill=(150, 150, 145))
    # голова
    hr = s * 0.065
    hx, hy = x, sh_y - hr * 1.3
    d.rectangle([hx - hr * 0.4, sh_y - hr * 0.5, hx + hr * 0.4, sh_y + 2], fill=(140, 140, 135))
    d.ellipse([hx - hr, hy - hr * 1.15, hx + hr, hy + hr * 1.1], fill=(165, 165, 158))
    d.chord([hx - hr * 1.08, hy - hr * 1.3, hx + hr * 1.08, hy + hr * 0.3], 180, 360, fill=(30, 30, 30))
    off = (1 - turn) * -hr * 0.55
    eye_col = tuple(int(lerp(c, 255, glow)) for c in (8, 8, 8))
    for ex in ((-0.38, 0.38) if turn > 0.5 else (-0.45,)):
        cx = hx + ex * hr + off
        er = hr * 0.17
        d.ellipse([cx - er, hy - er * 0.8, cx + er, hy + er * 0.8], fill=eye_col)
    d.line([(hx + off - hr * 0.2, hy + hr * 0.55), (hx + off + hr * 0.2, hy + hr * 0.55)], fill=(60, 50, 50), width=max(2, int(hr * 0.08)))
    return hx, hy, hr


@lru_cache(None)
def bg_clearing():
    a = vgrad(W, H, [(0, (5, 9, 20)), (0.55, (24, 36, 52)), (0.62, (16, 22, 26)), (1, (8, 10, 11))])
    a += radial(W, H, 760, 260, 420)[:, :, None] * np.array([70, 80, 95], np.float32)
    fog = fbm(W, H, 31)
    band_ = np.exp(-((np.arange(H) - H * 0.8) / 260) ** 2)[:, None]
    a += (fog * band_)[:, :, None] * np.array([60, 70, 80], np.float32)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    d.ellipse([700, 200, 820, 320], fill=(215, 220, 225))
    r = np.random.default_rng(8)
    for i in range(16):
        x = i * 75 + r.uniform(-20, 20)
        top = H * 0.42 + r.uniform(-80, 80)
        d.polygon([(x, H * 0.64), (x + 18, H * 0.64), (x + 14, top), (x + 4, top)], fill=(6, 9, 11))
        frond(d, x + 9, top, r.uniform(140, 240), r.uniform(-2.8, -0.4), (6, 9, 11), n=9)
    d.rectangle([0, int(H * 0.64), W, int(H * 0.66)], fill=(6, 9, 11))
    return img


@lru_cache(None)
def crack_mask():
    m = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(m)
    r = np.random.default_rng(4)
    left, right = [], []
    for y in range(-40, H + 80, 80):
        wv = 150 + 40 * math.sin(y / 300)
        left.append((540 - wv + r.uniform(-25, 25), y))
        right.append((540 + wv + r.uniform(-25, 25), y))
    d.polygon(left + right[::-1], fill=0)
    return m.filter(ImageFilter.GaussianBlur(3))


@lru_cache(None)
def crack_wood():
    a = np.full((H, W, 3), (30, 21, 14), np.float32)
    a *= (0.5 + fbm(W, H, 55, cells=(400, 50, 10, 3))[:, :, None] * 0.8)
    a *= np.clip(1 - np.abs(np.arange(W) - 540) / 700, 0.15, 1)[None, :, None]
    return to_img(a)


# ================================================================ рация
@lru_cache(None)
def bg_radio():
    a = np.full((H, W, 3), (26, 32, 24), np.float32)
    a *= 0.6 + fbm(W, H, 61, cells=(300, 40, 8, 3))[:, :, None] * 0.7
    img = to_img(a)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 1180, W, 1380], fill=(10, 10, 10))
    d.rectangle([0, 1192, W, 1196], fill=(30, 30, 30))
    d.rectangle([0, 1364, W, 1368], fill=(30, 30, 30))
    # корпус
    d.rounded_rectangle([340, 640, 740, 1480], 50, fill=(18, 18, 20), outline=(45, 45, 50), width=6)
    d.rectangle([600, 250, 640, 650], fill=(12, 12, 12))
    d.rounded_rectangle([590, 230, 650, 280], 15, fill=(20, 20, 20))
    d.rounded_rectangle([380, 680, 700, 830], 14, fill=(30, 46, 32), outline=(10, 10, 10), width=6)
    for gx in range(400, 690, 36):
        for gy in range(900, 1250, 36):
            d.ellipse([gx, gy, gx + 18, gy + 18], fill=(4, 4, 4))
    d.rounded_rectangle([700, 760, 760, 960], 20, fill=(28, 28, 30))
    d.rectangle([340, 1180, 740, 1380], fill=(8, 8, 8))
    d.rectangle([380, 1190, 700, 1370], fill=(14, 14, 14))
    return add_glow(img, 540, 740, 600, (20, 40, 25), 0.6)


def radio_frame(t, lt):
    img = bg_radio().copy()
    d = ImageDraw.Draw(img)
    talking = EV["radio_talk"][0] <= t < EV["radio_talk"][1]
    on = talking or any(0 <= t - q < 0.3 for q in EV["squelch"])
    led = (40, 255, 80) if on and int(t * 8) % 2 == 0 else ((200, 20, 20) if on else (60, 10, 10))
    d.ellipse([610, 285, 630, 305], fill=led)
    img = add_glow(img, 620, 295, 80, led, 0.6 if on else 0.1)
    d = ImageDraw.Draw(img)
    scr = (90, 230, 110) if on else (50, 110, 60)
    d.text((402, 700), "CH 03  ◉ RX", font=font(F_MONO_B, 40), fill=scr)
    if talking:
        r = np.random.default_rng(int(t * 30))
        for i in range(14):
            h = r.uniform(5, 60)
            d.rectangle([405 + i * 20, 815 - h, 418 + i * 20, 815], fill=scr)
        if r.random() > 0.8:
            d.text((402, 750), "▒▒ ERR ▒▒", font=font(F_MONO_B, 36), fill=scr)
    else:
        d.text((402, 760), "-- --- --", font=font(F_MONO_B, 36), fill=scr)
    return cam(img, lerp(1.05, 1.35, ease(lt / 7.5)), 0.5, 0.45)


# ================================================================ кадры (шоты)
def shot_wake(t, lt):
    base = beach_frame(1.6, 0.0)
    o = 0.0
    if lt < 0.45:
        o = ease(lt / 0.45) * 0.3
    elif lt < 0.75:
        o = 0.3 * (1 - ease((lt - 0.45) / 0.3)) + 0.05
    else:
        o = 0.05 + ease((lt - 0.75) / 0.85) * 1.2
    blur = 26 * (1 - ease(lt / 1.6))
    if blur > 0.5:
        base = base.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(blur / 4)).resize((W, H), Image.BICUBIC)
    m = Image.new("L", (W // 4, H // 4), 0)
    ry = o * H * 0.45 / 4
    ImageDraw.Draw(m).ellipse([-60, H / 8 - ry, W / 4 + 60, H / 8 + ry], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BICUBIC)
    return Image.composite(base, Image.new("RGB", (W, H)), m)


def beach_frame(t, lt):
    img = bg_beach().copy()
    d = ImageDraw.Draw(img, "RGBA")
    hy = int(H * 0.36)
    # дым
    for i in range(10):
        k = ((t * 0.12 + i / 10) % 1)
        sx = 720 + k * 260 + math.sin(t + i) * 20
        sy = hy - 60 - k * 650
        sr = 40 + k * 190
        d.ellipse([sx - sr, sy - sr * 0.7, sx + sr, sy + sr * 0.7], fill=(12, 10, 12, int(160 * (1 - k))))
    # огонь
    fire = Image.new("RGBA", (500, 400), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fire)
    r = np.random.default_rng(int(t * 24))
    for i in range(22):
        fx = 120 + r.uniform(0, 260)
        fh = r.uniform(40, 170) * (0.7 + 0.3 * math.sin(t * 9 + i))
        fw = r.uniform(14, 36)
        col = (255, int(r.uniform(90, 200)), 20, 230)
        fd.polygon([(fx - fw, 330), (fx + fw, 330), (fx + r.uniform(-15, 15), 330 - fh)], fill=col)
    fire = fire.filter(ImageFilter.GaussianBlur(5))
    img.paste(fire, (470, hy - 320), fire)
    flick = 0.8 + 0.2 * math.sin(t * 13) * math.sin(t * 7.3)
    img = add_glow(img, 720, hy - 60, 260, (255, 120, 30), 0.55 * flick)
    d = ImageDraw.Draw(img, "RGBA")
    # искры
    for i in range(40):
        k = (t * 0.3 + i * 0.137) % 1
        ex = 720 + math.sin(i * 12.9) * 160 + k * 120
        ey = hy - 60 - k * 700
        d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(255, 170, 60, int(255 * (1 - k))))
    # волны и пена
    for i in range(26):
        y = int(H * 0.37) + i ** 1.55 * 6
        ph = t * (0.6 + i * 0.04) + i
        alpha = int(30 + i * 3)
        for seg in range(5):
            x0 = (seg * 260 + ph * 80 + i * 47) % (W + 300) - 150
            d.line([(x0, y), (x0 + 120 + i * 4, y)], fill=(150, 90, 60, alpha) if i < 6 else (90, 110, 130, alpha), width=2 + i // 7)
    wl = H * 0.6 + 40 * math.sin(t * math.pi / 3.3) ** 2
    pts = [(x, wl + 14 * math.sin(x / 70 + t * 2)) for x in range(0, W + 40, 40)]
    d.polygon(pts + [(W, H * 0.6 - 10), (0, H * 0.6 - 10)], fill=(40, 52, 64, 150))
    d.line(pts, fill=(170, 180, 190, 170), width=6)
    return img


def shot_beach(t, lt):
    img = beach_frame(t, lt)
    return cam(img, lerp(1.0, 1.14, ease(lt / 5.8)), 0.6, 0.42)


@lru_cache(None)
def jungle_edge_bg():
    img = jungle_frame(0.0, offset=500)
    return to_img(np.asarray(img, np.float32) * 0.65)


def shot_edge(t, lt):
    img = jungle_edge_bg().copy()
    d = ImageDraw.Draw(img)
    for k, (ts, x, y) in enumerate(((0.1, 380, 1050), (0.35, 720, 980), (0.6, 520, 1260), (0.85, 860, 1180))):
        if lt > ts and not (1.1 < lt < 1.18 and k == 1):
            for ex in (-22, 22):
                d.ellipse([x + ex - 9, y - 5, x + ex + 9, y + 5], fill=(255, 40, 30))
            img = add_glow(img, x, y, 110, (220, 20, 10), 0.6 * min(1, (lt - ts) * 4))
            d = ImageDraw.Draw(img)
    return cam(img, lerp(1.05, 1.35, ease(lt / 1.3)), 0.55, 0.6)


def shot_run(t, lt):
    img = jungle_frame(t)
    bob = math.sin(t * 2 * math.pi * 1.85) * 22
    return cam(img, 1.12, 0.5, 0.5 + bob / H, rot=math.sin(t * 2 * math.pi * 0.93) * 1.6)


def shot_pursuers(t, lt):
    img = jungle_frame(13.2 + lt * 0.1, tint=(1.15, 0.55, 0.5))
    d = ImageDraw.Draw(img)
    k = ease(lt / 2.9)
    for i, (x0, ph) in enumerate(((290, 0.0), (560, 1.3), (820, 2.1))):
        s = lerp(520, 1500, k) * (0.92 + 0.08 * i)
        x = 540 + (x0 - 540) * lerp(0.7, 1.6, k)
        feet = lerp(1350, 2350, k) + i * 20
        person(d, x, feet, s, phase=t * 15 + ph, run=1.0, col=(16, 6, 6), lean=0.35,
               blood=(120, 8, 10), eyes=(255, 230, 220), facing=1 if i != 1 else -1)
    return cam(img, 1.05, 0.5, 0.5, rot=math.sin(t * 9) * 1.2)


def shot_slam(t, lt):
    ang = 1.25 * (1 - ease((t - EV["slam"] + 0.22) / 0.22))
    tk = ease((t - EV["table"]) / 0.95)
    img = hut_world(t, door_angle=ang, table_k=tk, scratches=False)
    if ang > 0.05:
        d = ImageDraw.Draw(img)
        person(d, 620, 1480, 380, phase=t * 15, run=1.0, col=(20, 6, 6), blood=(100, 8, 8), eyes=(255, 200, 200), lean=0.3)
    return cam(img, 1.15, 0.5, 0.52)


def shot_scratch(t, lt):
    shadows = [600 + 60 * math.sin(t * 3), 450 + 40 * math.sin(t * 2.3 + 1)] if t < EV["silence"] - 0.4 else None
    img = hut_world(t, feet_x=shadows)
    return cam(img, lerp(1.45, 1.6, lt / 3.1), 0.5, 0.53)


def shot_silence(t, lt):
    img = hut_world(t, gap_light=0.6)
    return cam(img, lerp(1.25, 1.4, lt / 1.2), 0.5, 0.5)


def shot_thud(t, lt):
    img = hut_world(t)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(11)
    for i in range(160):
        x = r.uniform(0, W)
        v = r.uniform(400, 1200)
        y = r.uniform(-200, 100) + v * lt + 300 * lt * lt
        sz = r.uniform(2, 6)
        d.ellipse([x - sz, y - sz, x + sz, y + sz], fill=(120, 105, 85))
    return cam(img, 1.25, 0.5, 0.45)


def shot_lunge(t, lt):
    img = hut_world(t)
    img = to_img(np.asarray(img, np.float32) * 0.45)
    jump = EV["lunge"] - EV["lunge_shot"]
    if lt > jump - 0.25:
        k = ease((lt - jump + 0.25) / 0.55)
        rim_person(img, lerp(300, 620, k), lerp(1500, 2600, k), lerp(700, 2300, k), phase=1.0, run=0.6,
                   hair=True, lean=0.5, arm_reach=(lerp(500, 900, k), lerp(1100, 1600, k)), rim=(80, 95, 120))
    img = cam(img, 1.2, 0.5 + (0.08 * ease((lt - jump - 0.25) / 0.4) if lt > jump + 0.25 else 0), 0.5)
    # нож на переднем плане
    d = ImageDraw.Draw(img)
    kx, ky = 760, 1500 + (1 - ease(lt / 0.4)) * 400
    d.polygon([(kx - 40, ky + 120), (kx + 10, ky + 130), (kx + 110, ky - 300), (kx + 85, ky - 330)], fill=(150, 155, 165))
    d.line([(kx + 85, ky - 330), (kx - 40, ky + 120)], fill=(210, 215, 225), width=6)
    d.rectangle([kx - 70, ky + 120, kx + 40, ky + 150], fill=(30, 30, 30))
    d.polygon([(kx - 50, ky + 150), (kx + 25, ky + 150), (kx + 10, ky + 420), (kx - 70, ky + 420)], fill=(25, 18, 14))
    if 0.15 < lt < 0.7:
        g = math.sin((lt - 0.15) / 0.55 * math.pi)
        gy = ky + 100 - (lt - 0.15) / 0.55 * 400
        gx = kx - 20 + (lt - 0.15) / 0.55 * 110
        d.line([(gx - 90 * g, gy), (gx + 90 * g, gy)], fill=(255, 255, 255), width=4)
        d.line([(gx, gy - 90 * g), (gx, gy + 90 * g)], fill=(255, 255, 255), width=4)
        img = add_glow(img, gx, gy, 140, (200, 210, 255), g * 0.8)
    return img


def two_shot(t, girl_reach=None, me_x=330, zoom=1.12, cx=0.5, cy=0.5, girl_head_tilt=0.0):
    img = hut_world(t, gap_light=0.7)
    img = to_img(np.asarray(img, np.float32) * 0.8)
    br = math.sin(t * 2.2) * 6
    rim_person(img, me_x, 2050, 1500 + br, rim=(70, 85, 110), off=(-6, -4), scale_w=1.2, facing=1)
    rim_person(img, 780, 2010, 1350 - br, rim=(110, 130, 160), off=(6, -4), hair=True, lean=-0.05 + girl_head_tilt,
               arm_reach=girl_reach, facing=-1)
    return cam(img, zoom, cx, cy)


def shot_dialogue(t, lt):
    return two_shot(t, zoom=lerp(1.1, 1.22, lt / 4.7), cx=lerp(0.48, 0.55, lt / 4.7), cy=0.5)


def shot_wall(t, lt):
    img = wall_bg().copy()
    d = ImageDraw.Draw(img)
    marks = 47
    t0, t1 = EV["wall"], EV["wall_text"]
    shown = int(marks * ease((t - t0) / (t1 - t0 - 0.6)))
    for i in range(shown):
        grp, j = divmod(i, 5)
        row, col = divmod(grp, 3)
        x0, y0 = 120 + col * 300, 300 + row * 145
        if j < 4:
            d.line([(x0 + j * 45 + 3, y0 + 3), (x0 + j * 45 + 13, y0 + 110 + 3)], fill=(10, 7, 5), width=10)
            d.line([(x0 + j * 45, y0), (x0 + j * 45 + 10, y0 + 110)], fill=(165, 140, 112), width=7)
        else:
            d.line([(x0 - 15, y0 + 80), (x0 + 175, y0 + 25)], fill=(165, 140, 112), width=7)
    if t > t1:
        img = blood_text(img, t)
    zoom = lerp(1.0, 1.08, ease((t - t0) / (t1 - t0))) if t < t1 else lerp(1.08, 1.16, ease((t - t1) / 1.8))
    cy = lerp(0.4, 0.5, ease((t - t0) / (t1 - t0))) if t < t1 else lerp(0.5, 0.68, ease((t - t1) / 1.8))
    return cam(img, zoom, 0.5, cy)


@lru_cache(None)
def wall_bg():
    a = np.full((H, W, 3), (62, 46, 33), np.float32)
    a *= 0.55 + fbm(W, H, 91, cells=(500, 120, 30, 6))[:, :, None] * 0.7
    beams = Image.new("L", (W, H), 0)
    ImageDraw.Draw(beams).polygon([(200, 0), (420, 0), (1080, 1300), (1080, 1700)], fill=90)
    beams = beams.filter(ImageFilter.GaussianBlur(60))
    a += (np.asarray(beams, np.float32) / 255)[:, :, None] * np.array([60, 75, 100], np.float32)
    img = to_img(a)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 180):
        d.line([(x, 0), (x, H)], fill=(14, 10, 7), width=8)
    return img


def blood_text(img, t):
    f = font(F_SERIF_B, 150)
    layer = Image.new("L", (W, H), 0)
    ld = ImageDraw.Draw(layer)
    lines = [("ОНИ", 1200), ("УЧАТСЯ", 1440)]
    for txt, y in lines:
        tw = ld.textlength(txt, font=f)
        ld.text(((W - tw) / 2, y), txt, font=f, fill=255)
    # раскрытие слева направо, построчно
    k = ease((t - EV["wall_text"]) / 1.0)
    reveal = Image.new("L", (W, H), 0)
    rd = ImageDraw.Draw(reveal)
    k1, k2 = clamp(k * 2), clamp(k * 2 - 1)
    rd.rectangle([0, 1150, 120 + k1 * 840, 1440], fill=255)
    rd.rectangle([0, 1440, 60 + k2 * 960, 1720], fill=255)
    mask = Image.fromarray(np.minimum(np.asarray(layer), np.asarray(reveal)))
    # подтёки
    dr = ImageDraw.Draw(mask)
    r = np.random.default_rng(17)
    for i in range(18):
        x = r.uniform(120, 960)
        y0 = 1400 if i % 2 else 1650
        ln = r.uniform(40, 200) * clamp((t - EV["wall_text"] - 0.4 - i * 0.05) / 1.5)
        if ln > 1:
            dr.line([(x, y0), (x, y0 + ln)], fill=255, width=int(r.uniform(6, 12)))
            dr.ellipse([x - 8, y0 + ln - 6, x + 8, y0 + ln + 12], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    img.paste((120, 6, 8), (0, 0), mask)
    return img


def shot_gap(t, lt):
    step = sum(t >= q for q in EV["steps"])
    fx = 300 + step * 110
    feet = [fx, fx + 90] if step > 0 else None
    img = hut_world(t, feet_x=feet, gap_light=1.8, table_k=0.0)
    return cam(img, 3.0, 0.5, 0.78)


def shot_girl_pale(t, lt):
    img = draw_face(t, eyes_open=1.1, pale=ease(lt / 1.5), tremble=0.6, mouth=0.25 if 43.0 < t < 44.9 and int(t * 7) % 2 else 0.05)
    return cam(img, lerp(1.05, 1.25, lt / 4.0), 0.5, 0.5)


def shot_knock(t, lt):
    img = hut_world(t)
    for kt in EV["knocks"]:
        if 0 <= t - kt < 0.15:
            img = bulge_warp(img, 0.06 * (1 - (t - kt) / 0.15), 560, 1000, 500)
    return cam(img, lerp(1.35, 1.6, lt / 2.8), 0.5, 0.5)


def crack_pov(t, zoom, cy, turn=0.0, glow=0.0):
    world = bg_clearing().copy()
    d = ImageDraw.Draw(world)
    rescuer(d, 540, 1600, 1050, turn=turn, glow=glow)
    if glow > 0:
        hy = 1600 - 1050 * 0.8 - 1050 * 0.065 * 1.3
        for ex in (-0.38, 0.38):
            world = add_glow(world, 540 + ex * 1050 * 0.065, hy, 28, (255, 255, 240), glow * 0.9)
    view = cam(world, zoom, 0.5, cy)
    return Image.composite(crack_wood(), view, crack_mask())


def shot_crack(t, lt):
    return crack_pov(t, lerp(1.0, 1.15, ease(lt / 7.5)), lerp(0.55, 0.5, lt / 7.5))


def shot_grab(t, lt):
    k = ease((t - EV["grab"]) / 0.4)
    reach = (lerp(690, 450, k), lerp(1300, 1250, k))
    return two_shot(t, girl_reach=reach, zoom=1.25, cx=0.5, cy=0.52)


def shot_turn(t, lt):
    turn = ease((t - EV["head_turn"]) / 0.6)
    z = lerp(1.15, 3.6, ease((t - EV["head_turn"] - 0.3) / 2.0))
    glow = ease((t - EV["glow"]) / 0.6)
    return crack_pov(t, z, lerp(0.5, 0.34, ease((t - EV["head_turn"] - 0.3) / 2.0)), turn=turn, glow=glow)


def shot_radio(t, lt):
    return radio_frame(t, lt)


def shot_turn_to_her(t, lt):
    k = ease(lt / 2.5)
    return two_shot(t, zoom=lerp(1.15, 1.9, k), cx=lerp(0.4, 0.66, k), cy=lerp(0.5, 0.35, k), me_x=lerp(330, 300, k))


def shot_blood(t, lt):
    talk = 76.2 < t < 78.5 and int(t * 6) % 2
    img = draw_face(t, blood=ease((lt - 1.2) / 3.0), look_up=1.0, pale=0.6, eyes_open=0.85, mouth=0.18 if talk else 0.05)
    img = to_img(np.asarray(img, np.float32) * np.array([1.12, 0.85, 0.85], np.float32))
    return cam(img, lerp(1.2, 1.5, lt / 4.5), 0.5, 0.45)


def shot_door_bend(t, lt):
    img = hut_world(t)
    k = ease(lt / 3.0)
    img = bulge_warp(img, 0.08 + 0.28 * k + 0.05 * math.sin(t * 40), 540, 1000, 600)
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(23)
    for i in range(int(12 * k)):
        x, y = r.uniform(380, 700), r.uniform(600, 1400)
        pts = [(x, y)]
        for _ in range(5):
            x += r.uniform(-30, 30)
            y += r.uniform(20, 70)
            pts.append((x, y))
        d.line(pts, fill=(5, 3, 2), width=5)
    return cam(img, 1.5, 0.5, 0.52)


def shot_black(t, lt):
    return Image.new("RGB", (W, H))


def shot_final(t, lt):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    f = font(F_SERIF_B, 92)
    lines = ["НОЧЬ ТОЛЬКО", "НАЧИНАЕТСЯ…"]
    alpha = ease(lt / 0.6) * (1 - ease((lt - 4.4) / 0.8))
    glitch = lt < 0.5 or 2.2 < lt < 2.32
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f)
        x, y = (W - tw) / 2, 840 + i * 130
        if glitch:
            d.text((x - 10, y), ln, font=f, fill=(int(200 * alpha), 0, 0))
            d.text((x + 10, y + 3), ln, font=f, fill=(0, int(60 * alpha), int(120 * alpha)))
        d.text((x, y), ln, font=f, fill=(int(225 * alpha), int(215 * alpha), int(210 * alpha)))
    d.line([(240, 1110), (240 + 600 * ease(lt / 1.5), 1110)], fill=(int(160 * alpha), 0, 0), width=4)
    return img


SHOTS = [
    (0.0, 1.6, shot_wake), (1.6, 7.4, shot_beach), (7.4, EV["run"], shot_edge),
    (EV["run"], EV["pursuers"], shot_run), (EV["pursuers"], EV["slam_shot"], shot_pursuers),
    (EV["slam_shot"], EV["scratch_shot"], shot_slam), (EV["scratch_shot"], EV["silence"], shot_scratch),
    (EV["silence"], EV["thud"], shot_silence), (EV["thud"], EV["lunge_shot"], shot_thud),
    (EV["lunge_shot"], EV["dialogue"], shot_lunge), (EV["dialogue"], EV["wall"], shot_dialogue),
    (EV["wall"], EV["gap"], shot_wall), (EV["gap"], EV["knock_shot"], shot_gap),
    (EV["knock_shot"], EV["pale"], shot_knock), (EV["pale"], EV["crack"], shot_girl_pale),
    (EV["crack"], EV["grab_shot"], shot_crack), (EV["grab_shot"], EV["turn_shot"], shot_grab),
    (EV["turn_shot"], EV["radio"], shot_turn), (EV["radio"], EV["turn_her"], shot_radio),
    (EV["turn_her"], EV["blood"], shot_turn_to_her), (EV["blood"], EV["bend"], shot_blood),
    (EV["bend"], EV["black"], shot_door_bend), (EV["black"], FINAL_START, shot_black),
    (FINAL_START, DURATION, shot_final),
]

# цветокоррекция по шотам (множители RGB)
GRADES = {
    shot_beach: (1.05, 0.95, 0.95), shot_wake: (1.0, 0.95, 0.95), shot_edge: (0.8, 1.0, 1.0),
    shot_run: (0.85, 1.05, 1.0), shot_pursuers: (1.0, 0.9, 0.9), shot_crack: (0.9, 1.0, 1.1),
    shot_turn: (0.9, 1.0, 1.1), shot_radio: (0.9, 1.05, 0.95),
}

FLASHES = [(EV["pursuers"], (255, 30, 20), 0.25), (EV["slam"], (255, 255, 255), 0.12),
           (EV["lunge"], (255, 255, 255), 0.2), (EV["wall_text"], (180, 0, 0), 0.4),
           (EV["head_turn"] + 0.3, (255, 255, 255), 0.15), (EV["final"], (120, 0, 0), 0.5)]


# ================================================================ постобработка
@lru_cache(None)
def vignette():
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
    return np.clip(1.15 - d ** 2.2, 0.15, 1).astype(np.float32)[:, :, None]


@lru_cache(None)
def grain_bank():
    r = np.random.default_rng(99)
    return [r.normal(0, 1, (H // 2, W // 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:, :, None]
            for _ in range(6)]


def brightness(t):
    b = 1.0
    if t < 0.4:
        b = t / 0.4
    if EV["silence"] <= t < EV["thud"]:
        b *= 0.55
    if EV["run"] <= t < EV["pursuers"]:  # лунные вспышки между деревьями
        b *= 0.85 + 0.25 * max(0, math.sin(t * 7.1) * math.sin(t * 3.3))
    if EV["bend"] + 2.0 <= t < EV["black"]:
        b *= 0.6 + 0.6 * (int(t * 20) % 2)
    return b


def post(img, t, shot):
    a = np.asarray(img, np.float32)
    g = GRADES.get(shot, (1.0, 0.97, 0.97))
    a = a * np.array(g, np.float32) * vignette()
    for ft, col, dur in FLASHES:
        if 0 <= t - ft < dur:
            k = 1 - (t - ft) / dur
            a = a * (1 - k * 0.7) + np.array(col, np.float32) * k * 0.7
    if shot is not shot_final and shot is not shot_black:
        a = a + grain_bank()[int(t * FPS) % 6] * 9
    a *= brightness(t)
    return to_img(a)


# ================================================================ надписи
def wrap(d, text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if d.textlength(test, font=f) <= maxw:
            cur = test
        else:
            lines.append(cur)
            cur = w_
    lines.append(cur)
    return lines


def outlined(d, xy, text, f, fill, stroke=6):
    d.text(xy, text, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))


def overlay_text(img, t):
    d = ImageDraw.Draw(img)
    for (a, b, spk, text) in SUBS:
        if not (a <= t < b):
            continue
        k = min(1, (t - a) / 0.12, (b - t) / 0.12)
        name, col = SPEAKERS[spk]
        italic = spk in ("VO", "GIRL_WHISPER", "GIRL_SILENT")
        f = font(F_SANS_BI if italic else F_SANS_B, 60)
        if spk == "RADIO":
            f = font(F_MONO_B, 54)
        lines = wrap(d, "«" + text + "»", f, W - 140)
        y = 1480 - (len(lines) - 1) * 40
        fn = font(F_SANS_B, 36)
        nw = d.textlength(name, font=fn)
        c = tuple(int(v * k) for v in col)
        outlined(d, ((W - nw) / 2, y - 62), name, fn, c, 4)
        for i, ln in enumerate(lines):
            txt = ln
            if spk == "RADIO" and int(t * 12) % 9 == 0:
                txt = ln.replace("о", "▒", 1)
            lw = d.textlength(txt, font=f)
            fc = tuple(int(v * k) for v in ((235, 235, 235) if spk != "RADIO" else (150, 245, 165)))
            if spk == "GIRL_SILENT":
                fc = tuple(int(v * k) for v in (190, 190, 200))
            outlined(d, ((W - lw) / 2, y + i * 78), txt, f, fc)
    for (a, b, title) in CHAPTERS:
        if a <= t < b:
            k = min(1, (t - a) / 0.4, (b - t) / 0.5)
            f = font(F_SERIF_B, 44)
            spaced = " ".join(title)
            tw = d.textlength(spaced, font=f)
            c = (int(200 * k), int(190 * k), int(185 * k))
            outlined(d, ((W - tw) / 2, 170), spaced, f, c, 3)
            lw = tw * ease((t - a) / 0.8)
            d.line([((W - lw) / 2, 240), ((W + lw) / 2, 240)], fill=(int(150 * k), 0, 0), width=3)
    return img


# ================================================================ рендер
def shot_at(t):
    for a, b, fn in SHOTS:
        if a <= t < b:
            return fn, t - a
    return SHOTS[-1][2], t - SHOTS[-1][0]


def frame(i):
    t = i / FPS
    CUR_T[0] = t
    fn, lt = shot_at(t)
    img = fn(t, lt)
    if img.size != (W, H):
        img = img.resize((W, H))
    img = post(img.convert("RGB"), t, fn)
    img = overlay_text(img, t)
    return img.tobytes()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/video.mp4"
    if "--preview" in sys.argv:
        for s in sys.argv[sys.argv.index("--preview") + 1:]:
            i = int(float(s) * FPS)
            Image.frombytes("RGB", (W, H), frame(i)).save(f"{out}_{s}.jpg", quality=88)
        return
    total = int(DURATION * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "27", "-tune", "grain", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool() as pool:
        for n, buf in enumerate(pool.imap(frame, range(total), chunksize=6)):
            ff.stdin.write(buf)
            if n % (FPS * 5) == 0:
                print(f"  кадр {n}/{total}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

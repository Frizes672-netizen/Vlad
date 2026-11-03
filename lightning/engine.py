"""Движок ролика «Человек на всех фотографиях»: утилиты, камера, люди с суставами,
старые фотографии, постобработка. Сцены — в render.py.

Мир рисуется в ×S (2160×3840) и сжимается камерой до 1080×1920.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from timeline import FPS, H, W, v

FONT_DIR = "../president/fonts"
S = 2
WW, WH = W * S, H * S

RED = (214, 44, 44)
CREAM = (238, 232, 220)
PAPER = (232, 224, 206)
INK = (8, 9, 12)


# ================================================================ шрифты
@lru_cache(None)
def _font(name, size, wght):
    f = ImageFont.truetype(f"{FONT_DIR}/{name}", int(size))
    f.set_variation_by_axes([wght])
    f._name, f._w = name, wght
    return f


def resize_font(f, size):
    """Тот же шрифт и вес другого размера (font_variant теряет вес у variable-шрифтов)."""
    return _font(f._name, int(size), f._w)


def MS(size, w=700):
    return _font("Montserrat[wght].ttf", size, w)


def PF(size, w=700):
    return _font("PlayfairDisplay[wght].ttf", size, w)


def PFI(size, w=500):
    return _font("PlayfairDisplay-Italic[wght].ttf", size, w)


# ================================================================ утилиты
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def eio(x):
    """Мягкий разгон и торможение (как у человеческого движения)."""
    x = clamp(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def eout(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back(x, s=1.6):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(a, b, k)) for a, b in zip(c1, c2))


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def env(t, a, b, fi=0.3, fo=0.3):
    """Огибающая появления: 0 до a, 1 в середине, 0 после b."""
    return min(ease((t - a) / fi), ease((b - t) / fo))


@lru_cache(None)
def words():
    out = []
    for line in open("voice/words.tsv", encoding="utf-8"):
        s, e, w = line.rstrip("\n").split("\t")
        out.append((v(float(s)), v(float(e)), w.lower().strip(".,!?«»—…-")))
    return out


def word_t(word, after=0.0):
    """Время ролика, когда звучит слово (первое после after секунд озвучки)."""
    for s, e, w in words():
        if s >= v(after) - 0.01 and w.startswith(word):
            return s
    raise KeyError(word)


class C:
    """Холст в координатах кадра 1080×1920, рисует в мире ×S."""

    def __init__(self, img, s=S):
        self.img = img
        self.s = s
        self.d = ImageDraw.Draw(img, "RGBA" if img.mode in ("RGB", "RGBA") else None)

    def rect(self, x0, y0, x1, y1, fill, r=0, outline=None, w=0):
        s = self.s
        b = [min(x0, x1) * s, min(y0, y1) * s, max(x0, x1) * s, max(y0, y1) * s]
        if r:
            self.d.rounded_rectangle(b, r * s, fill=fill, outline=outline, width=int(w * s))
        else:
            self.d.rectangle(b, fill=fill, outline=outline, width=int(w * s))

    def ell(self, cx, cy, rx, ry, fill, outline=None, w=0):
        s = self.s
        self.d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], fill=fill,
                       outline=outline, width=int(w * s))

    def poly(self, pts, fill):
        self.d.polygon([(x * self.s, y * self.s) for x, y in pts], fill=fill)

    def line(self, pts, fill, w):
        self.d.line([(x * self.s, y * self.s) for x, y in pts], fill=fill, width=max(1, int(w * self.s)),
                    joint="curve")

    def arc(self, cx, cy, rx, ry, a0, a1, fill, w):
        s = self.s
        self.d.arc([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], a0, a1, fill=fill,
                   width=max(1, int(w * s)))

    def text(self, x, y, txt, f, fill, anchor="mm"):
        ff = resize_font(f, f.size * self.s)
        self.d.text((x * self.s, y * self.s), txt, font=ff, fill=fill, anchor=anchor)


def grad(top, bottom, w=WW, h=WH, mid=None):
    ys = np.linspace(0, 1, h)[:, None]
    stops = [(0, top), (1, bottom)] if mid is None else [(0, top), (mid[0], mid[1]), (1, bottom)]
    pos = [p for p, _ in stops]
    a = np.zeros((h, w, 3), np.float32)
    for ch in range(3):
        a[:, :, ch] = np.interp(ys, pos, [c[ch] for _, c in stops])
    return a


def radial(cx, cy, r, w=WW, h=WH, s=S, p=2.0):
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx * s) ** 2 + (y - cy * s) ** 2) / (r * s)
    return (np.clip(1 - d, 0, 1) ** p).astype(np.float32)


def glow(img, shapes, blur=40, strength=1.0):
    """Мягкий аддитивный свет: shapes(draw, k) рисует фигуры в масштабе k от кадра."""
    k = 0.25
    lay = Image.new("RGB", (int(img.width / S * k), int(img.height / S * k)))
    shapes(ImageDraw.Draw(lay), k)
    lay = lay.filter(ImageFilter.GaussianBlur(blur * k)).resize(img.size, Image.BILINEAR)
    if strength != 1.0:
        lay = to_img(np.asarray(lay, np.float32) * strength)
    return ImageChops.add(img, lay)


def fbm(w, h, cells, seed, s=1):
    """Плавный шум (сумма октав) размером w×h, значения 0..1."""
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for c in cells:
        cw, ch = max(2, int(c * w / max(w, h))), max(2, int(c * h / max(w, h)))
        small = (r.random((ch, cw)) * 255).astype(np.uint8)
        out += np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), np.float32) / 255 * amp
        tot += amp
        amp *= 0.5
    return out / tot


def dust(c, t, n=40, seed=1, col=(230, 220, 200), area=(0, 0, W, H), speed=6, size=(1.2, 3), alpha=110):
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = area
    for i in range(n):
        x = r.uniform(x0, x1) + math.sin(t * 0.3 + i) * 25
        y = (r.uniform(y0, y1) - t * speed * r.uniform(0.4, 1.4)) % (y1 - y0) + y0
        a = int(alpha * (0.5 + 0.5 * math.sin(t * 1.1 + i * 2.3)))
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


def handheld(t, amp=4.0):
    """Лёгкое «ручное» покачивание камеры."""
    return (amp * (math.sin(t * 0.9) + 0.5 * math.sin(t * 2.3 + 1)),
            amp * (math.cos(t * 0.7 + 2) + 0.4 * math.sin(t * 1.9)))


def cam(world, zoom=1.0, cx=540, cy=960, rot=0.0, hh_amp=0.0):
    dx, dy = shake(CUR_T[0])
    if hh_amp:
        hx, hy = handheld(CUR_T[0], hh_amp)
        dx, dy = dx + hx, dy + hy
    zoom = max(zoom, 1.0 + (abs(dx) + abs(dy)) * 2.2 / W + (0.012 if (dx or dy) else 0))
    ww_, hh_ = W / zoom, H / zoom
    x0 = clamp(cx - ww_ / 2 + dx / zoom, 0, W - ww_)
    y0 = clamp(cy - hh_ / 2 + dy / zoom, 0, H - hh_)
    if rot:
        world = world.rotate(rot, Image.BICUBIC, center=(cx * S, cy * S))
    return world.resize((W, H), Image.BILINEAR, box=(x0 * S, y0 * S, (x0 + ww_) * S, (y0 + hh_) * S),
                        reducing_gap=1.5)


# ================================================================ люди
# Скелет в долях роста h. Углы в градусах: 0 — вниз (для конечностей) / вверх (для корпуса),
# положительный — вперёд, по направлению взгляда.
L_THIGH, L_SHIN, L_TORSO, L_NECK = 0.245, 0.235, 0.29, 0.035
L_UARM, L_FARM = 0.165, 0.15
HEAD_RX, HEAD_RY = 0.054, 0.066

POSE0 = dict(lean=0.0, neck=0.0, nod=0.0, an=(4.0, 8.0), af=(-4.0, 8.0), ln=(0.0, 2.0), lf=(0.0, 2.0),
             turn=0.0, hand_n=None, hand_f=None)


def pose(**kw):
    p = dict(POSE0)
    p.update(kw)
    return p


def blend(p1, p2, k):
    out = {}
    for key in POSE0:
        a, b = p1[key], p2[key]
        if isinstance(a, tuple):
            out[key] = tuple(lerp(x, y, k) for x, y in zip(a, b))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[key] = lerp(a, b, k)
        else:
            out[key] = b if k > 0.5 else a
    return out


def keyed(t, keys):
    """keys: [(время, поза)] — плавная интерполяция поз (как у живого человека)."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t < t1:
            return blend(p0, p1, eio((t - t0) / (t1 - t0)))
    return keys[-1][1]


def walk(ph, amp=1.0, old=False):
    """Поза шага; ph — фаза (радианы)."""
    a = 21 * amp * (0.7 if old else 1.0)

    def leg(p):
        th = a * math.sin(p)
        kn = 4 + 50 * amp * max(0.0, math.cos(p - 0.15)) ** 2
        return th, kn

    def arm(p):
        return (-17 * amp * math.sin(p) * (0.6 if old else 1.0), 14 + 12 * max(0.0, -math.sin(p)))
    return pose(lean=(9 if old else 3), neck=(10 if old else 2), ln=leg(ph), lf=leg(ph + math.pi),
                an=arm(ph + math.pi), af=arm(ph))


def breathe(p, t, rate=0.23, amp=1.0):
    k = math.sin(t * 2 * math.pi * rate)
    q = dict(p)
    q["lean"] = p["lean"] + 0.8 * k * amp
    q["nod"] = p["nod"] + 0.6 * k * amp
    q["an"] = (p["an"][0] + 1.0 * k * amp, p["an"][1])
    return q


def _dir(ang, f):
    r = math.radians(ang)
    return f * math.sin(r), math.cos(r)


def skeleton(x, feet, h, p, f=1, kind="man"):
    """Суставы фигуры в виде сбоку (f=±1 — куда смотрит)."""
    legs = {}
    drop = 0.0
    for side in ("ln", "lf"):
        th, kn = p[side]
        dx1, dy1 = _dir(th, f)
        dx2, dy2 = _dir(th - kn, f)
        knee = (dx1 * L_THIGH * h, dy1 * L_THIGH * h)
        ankle = (knee[0] + dx2 * L_SHIN * h, knee[1] + dy2 * L_SHIN * h)
        legs[side] = (knee, ankle)
        drop = max(drop, ankle[1])
    hip = (x, feet - drop - 0.012 * h)
    J = {"hip": hip}
    for side, (knee, ankle) in legs.items():
        J[side + "_knee"] = (hip[0] + knee[0], hip[1] + knee[1])
        J[side + "_ankle"] = (hip[0] + ankle[0], hip[1] + ankle[1])
    lx, ly = _dir(p["lean"], f)
    sh = (hip[0] + lx * L_TORSO * h, hip[1] - ly * L_TORSO * h)
    J["sh"] = sh
    nx, ny = _dir(p["lean"] + p["neck"], f)
    nt = (sh[0] + nx * L_NECK * h, sh[1] - ny * L_NECK * h)
    J["neck"] = nt
    hx, hy = _dir(p["lean"] + p["neck"] + p["nod"], f)
    J["head"] = (nt[0] + hx * HEAD_RY * h * 0.95, nt[1] - hy * HEAD_RY * h * 0.95)
    J["head_ang"] = p["lean"] + p["neck"] + p["nod"]
    for side in ("an", "af"):
        s_, e_ = p[side]
        a0 = (sh[0] - lx * 0.02 * h, sh[1] + ly * 0.02 * h)
        dx1, dy1 = _dir(s_ + 0, f)
        el = (a0[0] + dx1 * L_UARM * h, a0[1] + dy1 * L_UARM * h)
        dx2, dy2 = _dir(s_ + e_, f)
        wr = (el[0] + dx2 * L_FARM * h, el[1] + dy2 * L_FARM * h)
        J[side + "_sh"], J[side + "_el"], J[side + "_wr"] = a0, el, wr
    return J


def capsule(r0, r1, p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    pts = []
    for k in range(9):
        a = math.pi * k / 8
        ca, sa = math.cos(a), math.sin(a)
        pts.append((x0 + (nx * ca - ux * sa) * r0, y0 + (ny * ca - uy * sa) * r0))
    for k in range(9):
        a = math.pi * k / 8
        ca, sa = math.cos(a), math.sin(a)
        pts.append((x1 + (-nx * ca + ux * sa) * r1, y1 + (-ny * ca + uy * sa) * r1))
    return pts


def rot_pts(pts, cx, cy, ang):
    r = math.radians(ang)
    ca, sa = math.cos(r), math.sin(r)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]


def head_side(hx, hy, h, ang, f, kind):
    """Голова в профиль: полигон черепа с носом и подбородком + детали (волосы/шляпа)."""
    rx, ry = HEAD_RX * h, HEAD_RY * h
    prof = [(-1.0, -0.15), (-0.92, -0.62), (-0.6, -0.95), (-0.1, -1.05), (0.42, -0.9), (0.78, -0.55),
            (0.86, -0.22), (0.92, -0.08), (1.12, 0.12), (0.9, 0.22), (0.92, 0.36), (0.86, 0.46),
            (0.82, 0.62), (0.6, 0.92), (0.2, 0.98), (-0.25, 0.7), (-0.55, 0.62), (-0.9, 0.35)]
    pts = [(hx + f * px * rx, hy + py * ry) for px, py in prof]
    out = [("poly", rot_pts(pts, hx, hy, f * ang), "skin")]
    if kind in ("man", "man_old"):
        brim = [(hx + f * px * rx, hy + py * ry) for px, py in
                [(-1.35, -0.48), (1.45, -0.62), (1.42, -0.5), (-1.3, -0.34)]]
        crown = [(hx + f * px * rx, hy + py * ry) for px, py in
                 [(-0.9, -0.42), (-0.85, -1.15), (-0.3, -1.38), (0.35, -1.3), (0.85, -1.05), (0.92, -0.55)]]
        out += [("poly", rot_pts(crown, hx, hy, f * ang), "hat"), ("poly", rot_pts(brim, hx, hy, f * ang), "hat")]
    elif kind == "grandma":
        hair = [(hx + f * px * rx, hy + py * ry) for px, py in
                [(-1.08, 0.2), (-1.08, -0.6), (-0.6, -1.08), (0.2, -1.12), (0.7, -0.8), (0.55, -0.55),
                 (-0.2, -0.6), (-0.55, -0.1)]]
        out += [("poly", rot_pts(hair, hx, hy, f * ang), "hair")]
        bx, by = rot_pts([(hx - f * 1.1 * rx, hy - 0.55 * ry)], hx, hy, f * ang)[0]
        out += [("ell", (bx, by, 0.42 * rx, 0.42 * ry), "hair")]
    elif kind == "woman":
        hair = [(hx + f * px * rx, hy + py * ry) for px, py in
                [(-1.1, 1.4), (-1.15, -0.5), (-0.6, -1.1), (0.3, -1.12), (0.8, -0.7), (0.5, -0.5),
                 (-0.1, -0.55), (-0.4, 0.4), (-0.6, 1.4)]]
        out += [("poly", rot_pts(hair, hx, hy, f * ang), "hair")]
    elif kind in ("young", "suit", "boy"):
        hair = [(hx + f * px * rx, hy + py * ry) for px, py in
                [(-1.05, 0.1), (-1.02, -0.6), (-0.55, -1.1), (0.3, -1.15), (0.85, -0.75), (0.7, -0.55),
                 (0.0, -0.62), (-0.5, -0.2)]]
        out += [("poly", rot_pts(hair, hx, hy, f * ang), "hair")]
    if kind == "young":
        # капюшон, лежащий на плечах
        hood = [(hx + f * px * rx, hy + py * ry) for px, py in
                [(-1.25, 0.7), (-1.5, 1.4), (-0.8, 1.75), (0.1, 1.5), (-0.4, 0.9)]]
        out.insert(0, ("poly", hood, "coat"))
    return out


def body_side(J, h, f, kind, p):
    """Части тела (в порядке рисования) для вида сбоку."""
    parts = []
    girl = kind in ("grandma", "woman")
    thick = 1.12 if kind in ("man", "man_old") else 1.0
    if kind == "boy":
        thick = 1.05
    # дальние конечности
    for side in ("af",):
        parts.append(("cap", (J[side + "_sh"], J[side + "_el"], 0.032 * h * thick, 0.027 * h * thick), "far_sleeve"))
        parts.append(("cap", (J[side + "_el"], J[side + "_wr"], 0.026 * h * thick, 0.02 * h * thick), "far_sleeve"))
        parts.append(("ell", (J[side + "_wr"][0], J[side + "_wr"][1], 0.021 * h, 0.024 * h), "far_skin"))
    for side in ("lf",):
        parts.append(("cap", (J["hip"], J[side + "_knee"], 0.045 * h, 0.034 * h), "far_pants"))
        parts.append(("cap", (J[side + "_knee"], J[side + "_ankle"], 0.034 * h, 0.023 * h), "far_pants"))
        ax, ay = J[side + "_ankle"]
        parts.append(("poly", [(ax - f * 0.03 * h, ay - 0.02 * h), (ax + f * 0.085 * h, ay - 0.004 * h),
                               (ax + f * 0.09 * h, ay + 0.014 * h), (ax - f * 0.035 * h, ay + 0.014 * h)], "far_shoes"))
    # ближняя нога
    for side in ("ln",):
        parts.append(("cap", (J["hip"], J[side + "_knee"], 0.047 * h, 0.035 * h), "pants"))
        parts.append(("cap", (J[side + "_knee"], J[side + "_ankle"], 0.035 * h, 0.024 * h), "pants"))
        ax, ay = J[side + "_ankle"]
        parts.append(("poly", [(ax - f * 0.03 * h, ay - 0.02 * h), (ax + f * 0.085 * h, ay - 0.004 * h),
                               (ax + f * 0.09 * h, ay + 0.014 * h), (ax - f * 0.035 * h, ay + 0.014 * h)], "shoes"))
    # корпус
    hip, sh = J["hip"], J["sh"]
    ax_, ay_ = sh[0] - hip[0], sh[1] - hip[1]
    L = math.hypot(ax_, ay_)
    ux, uy = ax_ / L, ay_ / L          # вверх по корпусу
    nx, ny = -uy * f, ux * f            # вперёд (по взгляду)
    if f < 0:
        nx, ny = uy, -ux
    prof = [(0.0, -0.06, 0.05), (0.35, -0.062, 0.05), (0.7, -0.068, 0.07 if girl else 0.06),
            (0.92, -0.06, 0.05), (1.04, -0.03, 0.035)]
    if kind == "boy":
        prof = [(0.0, -0.055, 0.05), (0.5, -0.06, 0.06), (0.92, -0.055, 0.05), (1.04, -0.03, 0.035)]
    front = [(hip[0] + ux * L * k + nx * fw * h, hip[1] + uy * L * k + ny * fw * h) for k, bw, fw in prof]
    backs = [(hip[0] + ux * L * k + nx * bw * h, hip[1] + uy * L * k + ny * bw * h) for k, bw, fw in prof]
    torso = backs + front[::-1]
    # пальто / юбка / куртка: подол учитывает положение ног
    if kind in ("man", "man_old", "grandma", "woman"):
        hem_y = hip[1] + (0.3 if kind in ("man", "man_old") else (0.36 if kind == "grandma" else 0.4)) * h
        xs = []
        for side in ("ln", "lf"):
            kx, ky = J[side + "_knee"]
            ax2, ay2 = J[side + "_ankle"]
            k_ = clamp((hem_y - hip[1]) / max(1, (ky - hip[1])), 0, 1.6)
            if k_ <= 1:
                xs.append(lerp(hip[0], kx, k_))
            else:
                xs.append(lerp(kx, ax2, (hem_y - ky) / max(1, ay2 - ky)))
        rel = [clamp((x_ - hip[0]) * f, -0.075 * h, 0.09 * h) for x_ in xs]
        fx = hip[0] + f * (max(rel) + 0.07 * h)
        bx = hip[0] + f * (min(rel) - 0.075 * h)
        top_f = (hip[0] + ux * L * 0.85 + nx * 0.06 * h, hip[1] + uy * L * 0.85 + ny * 0.06 * h)
        top_b = (hip[0] + ux * L * 0.85 - nx * 0.065 * h, hip[1] + uy * L * 0.85 - ny * 0.065 * h)
        mid_f = (hip[0] + nx * 0.065 * h, hip[1] + ny * 0.065 * h)
        mid_b = (hip[0] - nx * 0.07 * h, hip[1] - ny * 0.07 * h)
        coat = [top_b, mid_b, (bx, hem_y), (fx, hem_y + 0.006 * h), mid_f, top_f]
        if max(p["ln"][0], p["lf"][0]) > 40:
            kx, ky = J["ln_knee"]
            drape = 0.16 * h if kind in ("grandma", "woman") else 0.07 * h
            coat = [top_b, mid_b, (hip[0] - f * 0.085 * h, hip[1] + 0.055 * h),
                    (kx - f * 0.01 * h, ky + drape), (kx + f * 0.05 * h, ky + drape),
                    (kx + f * 0.05 * h, ky - 0.04 * h), mid_f, top_f]
        parts.append(("poly", coat, "skirt" if kind in ("grandma", "woman") else "coat"))
    if kind == "boy":
        bag = [(sh[0] - nx * 0.05 * h - ux * 0.02 * h, sh[1] - ny * 0.05 * h - uy * 0.02 * h)]
        bx0, by0 = bag[0]
        parts.append(("poly", [(bx0, by0), (bx0 - f * 0.08 * h, by0 + 0.01 * h), (bx0 - f * 0.085 * h, by0 + 0.2 * h),
                               (bx0 + f * 0.0 * h, by0 + 0.21 * h)], "bag"))
    parts.append(("poly", torso, "coat"))
    # шея и голова
    parts.append(("cap", (sh, J["neck"], 0.024 * h, 0.022 * h), "skin"))
    parts += head_side(J["head"][0], J["head"][1], h, J["head_ang"], f, kind)
    if kind in ("man", "man_old"):
        # воротник пальто
        parts.append(("poly", [(sh[0] - nx * 0.04 * h, sh[1] + 0.005 * h), (J["neck"][0] - nx * 0.02 * h, J["neck"][1]),
                               (sh[0] + nx * 0.035 * h, sh[1] + 0.02 * h)], "coat"))
    # ближняя рука
    for side in ("an",):
        parts.append(("cap", (J[side + "_sh"], J[side + "_el"], 0.034 * h * thick, 0.028 * h * thick), "sleeve"))
        parts.append(("cap", (J[side + "_el"], J[side + "_wr"], 0.028 * h * thick, 0.021 * h * thick), "sleeve"))
        parts.append(("ell", (J[side + "_wr"][0], J[side + "_wr"][1], 0.022 * h, 0.025 * h), "skin"))
    return parts


def body_front(x, feet, h, p, kind, back=False):
    """Вид спереди / со спины: p['an'], p['af'] — (отведение руки, сгиб локтя) для правой/левой руки,
    p['turn'] — поворот головы в профиль (−1 влево, +1 вправо, 0 — анфас)."""
    parts = []
    girl = kind in ("grandma", "woman")
    sw = 0.105 * h if kind != "boy" else 0.09 * h
    leg_len = (L_THIGH + L_SHIN) * h
    hip_y = feet - leg_len - 0.012 * h
    lean = p["lean"]
    sh_y = hip_y - L_TORSO * h * math.cos(math.radians(lean)) * 1.0
    bob = 0.0
    for s, key in ((-1, "lf"), (1, "ln")):
        spread = p[key][0]
        hx = x + s * 0.055 * h
        ax = hx + s * math.sin(math.radians(spread)) * leg_len
        kx = (hx + ax) / 2
        parts.append(("cap", ((hx, hip_y), (kx, hip_y + L_THIGH * h), 0.05 * h, 0.036 * h), "pants"))
        parts.append(("cap", ((kx, hip_y + L_THIGH * h), (ax, feet - 0.02 * h), 0.036 * h, 0.025 * h), "pants"))
        parts.append(("ell", (ax + s * 0.012 * h, feet - 0.008 * h, 0.034 * h, 0.016 * h), "shoes"))
    # корпус
    torso = [(x - sw * 0.82, hip_y), (x - sw * 0.8, hip_y - 0.12 * h), (x - sw, sh_y + 0.02 * h), (x - sw * 0.85, sh_y - 0.012 * h),
             (x - 0.03 * h, sh_y - 0.025 * h), (x + 0.03 * h, sh_y - 0.025 * h), (x + sw * 0.85, sh_y - 0.012 * h),
             (x + sw, sh_y + 0.02 * h), (x + sw * 0.8, hip_y - 0.12 * h), (x + sw * 0.82, hip_y)]
    if kind in ("man", "man_old", "grandma", "woman"):
        hem = hip_y + (0.3 if kind in ("man", "man_old") else (0.36 if kind == "grandma" else 0.42)) * h
        flare = 0.13 * h if girl else 0.11 * h
        parts.append(("poly", [(x - sw * 0.84, hip_y - 0.1 * h), (x - flare, hem), (x + flare, hem),
                               (x + sw * 0.84, hip_y - 0.1 * h)], "skirt" if girl else "coat"))
    parts.append(("poly", torso, "coat"))
    if kind == "suit" and not back:
        parts.append(("poly", [(x - 0.03 * h, sh_y - 0.02 * h), (x + 0.03 * h, sh_y - 0.02 * h), (x, sh_y + 0.12 * h)], "shirt"))
    if kind == "woman" and not back:
        parts.append(("poly", [(x - 0.04 * h, sh_y - 0.02 * h), (x + 0.04 * h, sh_y - 0.02 * h), (x, sh_y + 0.06 * h)], "skin"))
    # руки: (отведение, сгиб) — сгиб внутрь, к поясу
    for s, key in ((-1, "af"), (1, "an")):
        ab, el = p[key]
        a0 = (x + s * sw * 0.9, sh_y + 0.02 * h)
        r1 = math.radians(ab)
        elb = (a0[0] + s * math.sin(r1) * L_UARM * h, a0[1] + math.cos(r1) * L_UARM * h)
        r2 = math.radians(ab - el)
        wr = (elb[0] + s * math.sin(r2) * L_FARM * h, elb[1] + math.cos(r2) * L_FARM * h)
        parts.append(("cap", (a0, elb, 0.034 * h, 0.028 * h), "sleeve"))
        parts.append(("cap", (elb, wr, 0.028 * h, 0.021 * h), "sleeve"))
        parts.append(("ell", (wr[0], wr[1], 0.022 * h, 0.025 * h), "skin"))
    # шея и голова
    hy = sh_y - 0.035 * h - HEAD_RY * h * 0.95
    parts.append(("cap", ((x, sh_y), (x, sh_y - 0.04 * h), 0.026 * h, 0.024 * h), "skin"))
    turn = p.get("turn", 0.0)
    if p.get("look"):
        parts += head_front(x, hy, h, kind, False, p["nod"])
    elif abs(turn) > 0.5:
        parts += head_side(x, hy, h, p["nod"], 1 if turn > 0 else -1, kind)
    else:
        parts += head_front(x, hy, h, kind, back, p["nod"])
    return parts


def head_front(hx, hy, h, kind, back=False, nod=0.0):
    rx, ry = HEAD_RX * h * 0.95, HEAD_RY * h
    out = []
    if kind == "woman":
        out.append(("poly", [(hx - rx * 1.2, hy - ry * 0.4), (hx + rx * 1.2, hy - ry * 0.4), (hx + rx * 1.3, hy + ry * 1.5),
                             (hx - rx * 1.3, hy + ry * 1.5)], "hair"))
    out.append(("ell", (hx, hy, rx, ry), "hair" if back else "skin"))
    if kind in ("man", "man_old"):
        out.append(("poly", [(hx - rx * 0.92, hy - ry * 0.42), (hx - rx * 0.86, hy - ry * 1.3), (hx, hy - ry * 1.42),
                             (hx + rx * 0.86, hy - ry * 1.3), (hx + rx * 0.92, hy - ry * 0.42)], "hat"))
        out.append(("ell", (hx, hy - ry * 0.45, rx * 1.55, ry * 0.16), "hat"))
    elif kind == "grandma":
        out.append(("poly", [(hx - rx * 1.05, hy), (hx - rx * 0.95, hy - ry * 0.75), (hx, hy - ry * 1.08),
                             (hx + rx * 0.95, hy - ry * 0.75), (hx + rx * 1.05, hy), (hx + rx * 0.8, hy - ry * 0.5),
                             (hx - rx * 0.8, hy - ry * 0.5)], "hair"))
        out.append(("ell", (hx, hy - ry * 1.1, rx * 0.45, ry * 0.32), "hair"))
    elif kind in ("young", "suit", "boy"):
        out.append(("poly", [(hx - rx * 1.02, hy - ry * 0.1), (hx - rx * 0.95, hy - ry * 0.8), (hx, hy - ry * 1.12),
                             (hx + rx * 0.95, hy - ry * 0.8), (hx + rx * 1.02, hy - ry * 0.1), (hx + rx * 0.7, hy - ry * 0.6),
                             (hx - rx * 0.7, hy - ry * 0.6)], "hair"))
    elif kind == "woman":
        out.append(("poly", [(hx - rx * 1.1, hy + ry * 0.2), (hx - rx, hy - ry * 0.8), (hx, hy - ry * 1.12),
                             (hx + rx, hy - ry * 0.8), (hx + rx * 1.1, hy + ry * 0.2), (hx + rx * 0.5, hy - ry * 0.6),
                             (hx - rx * 0.6, hy - ry * 0.5)], "hair"))
    if back:
        out.append(("ell", (hx, hy - ry * 0.1, rx * 0.98, ry * 0.95), "hair"))
    else:
        for s in (-1, 1):
            out.append(("ell", (hx + s * rx * 0.38, hy - ry * 0.05 + nod * 0.3, rx * 0.11, ry * 0.07), "eyes"))
        out.append(("ell", (hx, hy + ry * 0.5, rx * 0.25, ry * 0.05), "mouth"))
    return out


def person_parts(x, feet, h, p, f=1, kind="man", view="side"):
    if view == "side":
        return body_side(skeleton(x, feet, h, p, f, kind), h, f, kind, p)
    return body_front(x, feet, h, p, kind, back=(view == "back"))


def draw_parts(d, parts, colors, s=S, ox=0.0, oy=0.0, default=255):
    for kind_, geo, part in parts:
        col = colors.get(part, colors.get(part.replace("far_", ""), default)) if isinstance(colors, dict) else colors
        if col is None:
            continue
        if part.startswith("far_") and isinstance(colors, dict) and "far" in colors and part not in colors:
            base = colors.get(part.replace("far_", ""), default)
            if isinstance(base, tuple):
                col = tuple(int(c * colors["far"]) for c in base[:3]) + base[3:]
        if kind_ == "poly":
            d.polygon([((x - ox) * s, (y - oy) * s) for x, y in geo], fill=col)
        elif kind_ == "ell":
            cx, cy, rx, ry = geo
            d.ellipse([(cx - rx - ox) * s, (cy - ry - oy) * s, (cx + rx - ox) * s, (cy + ry - oy) * s], fill=col)
        elif kind_ == "cap":
            p0, p1, r0, r1 = geo
            d.polygon([((x - ox) * s, (y - oy) * s) for x, y in capsule(r0, r1, p0, p1)], fill=col)


def silhouette(img, parts, base=(10, 10, 14), rim=(220, 200, 170), light=(1.0, -0.4), rim_w=5.0,
               fill_light=None, alpha=1.0, blur=0.0, extra=None):
    """Силуэт с контровым светом: маска фигуры, светлая кромка со стороны источника.
    extra(draw, ox, oy) — дорисовать предметы в маску."""
    xs = []
    for kind_, geo, _ in parts:
        if kind_ == "poly":
            xs += geo
        elif kind_ == "ell":
            xs += [(geo[0] - geo[2], geo[1] - geo[3]), (geo[0] + geo[2], geo[1] + geo[3])]
        else:
            xs += [geo[0], geo[1]]
    pad = 60
    x0 = int(min(x for x, _ in xs) - pad)
    y0 = int(min(y for _, y in xs) - pad)
    x1 = int(max(x for x, _ in xs) + pad)
    y1 = int(max(y for _, y in xs) + pad)
    if extra:
        x0, y0, x1, y1 = x0 - 200, y0 - 200, x1 + 200, y1 + 200
    bw, bh = (x1 - x0) * S, (y1 - y0) * S
    m = Image.new("L", (bw, bh), 0)
    d = ImageDraw.Draw(m)
    draw_parts(d, parts, 255, S, x0, y0)
    if extra:
        extra(d, x0, y0)
    if blur:
        m = m.filter(ImageFilter.GaussianBlur(blur * S))
    a = np.asarray(m, np.float32) / 255
    lx, ly = light
    n = math.hypot(lx, ly) or 1
    sx, sy = int(round(lx / n * rim_w * S)), int(round(ly / n * rim_w * S))
    sh = np.roll(np.roll(a, sy, 0), sx, 1)
    edge = np.clip(a - sh, 0, 1)
    edge = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2 * S)),
                      np.float32) / 255 * a
    col = np.zeros((bh, bw, 3), np.float32) + np.array(base, np.float32)
    if fill_light:
        # мягкая подсветка со стороны источника по всей фигуре
        yy = np.linspace(0, 1, bh)[:, None]
        xx = np.linspace(0, 1, bw)[None, :]
        g = np.clip(0.5 + (xx - 0.5) * (lx / n) * 1.2 + (yy - 0.5) * (ly / n) * 1.2, 0, 1)
        col = col + g[:, :, None] * np.array(fill_light, np.float32)
    col = col * (1 - edge[:, :, None]) + np.array(rim, np.float32) * edge[:, :, None]
    # вставка в мир с обрезкой по краям
    X0, Y0 = x0 * S, y0 * S
    cx0, cy0 = max(0, X0), max(0, Y0)
    cx1, cy1 = min(img.width, X0 + bw), min(img.height, Y0 + bh)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = (slice(cy0 - Y0, cy1 - Y0), slice(cx0 - X0, cx1 - X0))
    al = a[sub][:, :, None] * alpha
    region = np.asarray(img.crop((cx0, cy0, cx1, cy1)), np.float32)[:, :, :3]
    out = region * (1 - al) + col[sub] * al
    img.paste(to_img(out), (cx0, cy0))


def ground_shadow(c, x, feet, w, alpha=110):
    for k in range(6):
        c.ell(x, feet + 2, w * (1 - k * 0.12), w * 0.11 * (1 - k * 0.12), (0, 0, 0, alpha // 6))


# ================================================================ фотографии
PS = 2              # масштаб растра фотографии
PW, PH = 800, 600   # координаты сцены внутри фото

PALETTE = dict(
    man=dict(coat=(112, 114, 118), sleeve=(106, 108, 112), pants=(52, 52, 58), shoes=(26, 24, 24), skin=(206, 168, 140),
             hat=(70, 72, 78), far=0.82),
    man_old=dict(coat=(118, 118, 120), sleeve=(110, 110, 114), pants=(56, 56, 60), shoes=(26, 24, 24),
                 skin=(204, 170, 146), hat=(84, 84, 90), far=0.82),
    grandma=dict(coat=(120, 70, 74), sleeve=(116, 66, 70), skirt=(70, 60, 72), pants=(70, 60, 60), shoes=(30, 26, 26),
                 skin=(214, 176, 150), hair=(150, 140, 132), eyes=(150, 112, 96), mouth=(176, 120, 110), far=0.85),
    boy=dict(coat=(236, 236, 230), sleeve=(236, 236, 230), pants=(40, 44, 70), shoes=(20, 20, 20), skin=(226, 186, 160),
             hair=(120, 86, 50), bag=(180, 40, 40), eyes=(160, 120, 104), mouth=(190, 130, 120), far=0.85),
    suit=dict(coat=(40, 42, 50), sleeve=(40, 42, 50), pants=(36, 38, 46), shoes=(14, 14, 16), skin=(216, 176, 150),
              hair=(70, 50, 34), shirt=(240, 240, 236), eyes=(150, 110, 92), mouth=(176, 120, 106), far=0.85),
    bride=dict(coat=(246, 244, 238), sleeve=(240, 238, 232), skirt=(250, 248, 244), pants=(240, 238, 232),
               shoes=(240, 238, 232), skin=(226, 190, 166), hair=(110, 70, 40), eyes=(160, 120, 104), mouth=(196, 120, 120),
               far=0.9),
    woman=dict(coat=(70, 110, 150), sleeve=(70, 110, 150), skirt=(60, 96, 132), pants=(60, 96, 132), shoes=(40, 30, 30),
               skin=(226, 190, 166), hair=(90, 60, 36), eyes=(160, 120, 104), mouth=(196, 120, 120), far=0.88),
    dad=dict(coat=(200, 200, 196), sleeve=(200, 200, 196), pants=(60, 66, 90), shoes=(30, 30, 30),
             skin=(214, 172, 142), hair=(70, 50, 34), eyes=(150, 110, 92), mouth=(176, 120, 106), far=0.85),
    kid=dict(coat=(220, 170, 60), sleeve=(220, 170, 60), pants=(60, 90, 150), shoes=(200, 60, 60), skin=(232, 196, 170),
             hair=(130, 90, 50), eyes=(166, 126, 110), mouth=(196, 130, 124), far=0.85),
)


def shaded(img, parts, pal, feet=None, light=(-1.0, -0.5), s=None, shadow=0.35):
    """Фигура со светотенью: градиент освещения, светлая кромка, мягкая тень на земле."""
    s = s or PS        # масштаб текущего растра (снимки бывают ×2, ×3, ×4)
    xs = []
    for kind_, geo, _ in parts:
        if kind_ == "poly":
            xs += geo
        elif kind_ == "ell":
            xs += [(geo[0] - geo[2], geo[1] - geo[3]), (geo[0] + geo[2], geo[1] + geo[3])]
        else:
            xs += [geo[0], geo[1]]
    x0 = int(min(x for x, _ in xs) - 10)
    y0 = int(min(y for _, y in xs) - 10)
    x1 = int(max(x for x, _ in xs) + 10)
    y1 = int(max(y for _, y in xs) + 10)
    bw, bh = (x1 - x0) * s, (y1 - y0) * s
    lay = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw_parts(ImageDraw.Draw(lay), parts, pal, s, x0, y0)
    a = np.asarray(lay, np.float32)
    al = a[:, :, 3:] / 255
    lx, ly = light
    yy = np.linspace(0, 1, bh)[:, None]
    xx = np.linspace(0, 1, bw)[None, :]
    g = 0.82 + 0.3 * np.clip(0.5 + (xx - 0.5) * lx * 0.9 + (yy - 0.5) * ly * 0.5, 0, 1)
    rgb = a[:, :, :3] * g[:, :, None]
    m = al[:, :, 0]
    sh = np.roll(np.roll(m, int(-ly * 3 * s), 0), int(-lx * 3 * s), 1)
    edge = np.clip(m - sh, 0, 1)[:, :, None]
    rgb = rgb + edge * 40
    if shadow and feet is not None:
        fx, fy = feet
        c = C(img, s)
        for k in range(5):
            c.ell(fx - lx * 20, fy + 2, (x1 - x0) * (0.55 - k * 0.07), 10 - k, (0, 0, 0, int(255 * shadow / 5)))
    lay = to_img(rgb).convert("RGBA")
    lay.putalpha(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6 * s)))
    img.paste(lay, (x0 * s, y0 * s), lay)


def ppl(d, x, feet, h, p, f, kind, pal, view="side"):
    parts = person_parts(x, feet, h, p, f, "man" if kind == "man_old" else kind, view)
    shaded(d._image, parts, pal, (x, feet))
    return parts


def photo_post(img, seed, tone="90s", date=None, fade_k=0.0):
    """Плёночный вид: мягкость, выцветание, зерно, виньетка, пыль, штамп даты."""
    w, h = img.size
    tex = fbm(w, h, [6, 20, 70, 200], seed + 7)
    img = to_img(np.asarray(img, np.float32) * (0.9 + 0.2 * tex[:, :, None]))
    img = img.filter(ImageFilter.GaussianBlur(1.6))
    a = np.asarray(img, np.float32)
    if tone == "sepia":
        g = a.mean(axis=2, keepdims=True)
        a = g * np.array([1.08, 0.94, 0.74], np.float32) + np.array([18, 12, 0], np.float32)
        a = a * 0.86 + 24
    elif tone == "90s":
        a = a * np.array([1.04, 0.98, 0.86], np.float32) + np.array([16, 10, 4], np.float32)
        a = 255 - (255 - a) * 0.88
    elif tone == "2000s":
        a = a * np.array([1.0, 1.0, 0.96], np.float32) + 6
    elif tone == "blue":
        a = a * np.array([0.92, 0.98, 1.06], np.float32)
    a = lerp(a, a.mean(axis=2, keepdims=True), 0.2 + fade_k)
    r = np.random.default_rng(seed)
    grain = r.normal(0, 9, (h // 2, w // 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:h, :w, None]
    a = a + grain
    yy, xx = np.ogrid[:h, :w]
    dd = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h / 2) / (h * 0.62)) ** 2)
    a = a * np.clip(1.12 - dd ** 2.4 * 0.7, 0.45, 1)[:, :, None]
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(14):
        x, y = r.uniform(0, w), r.uniform(0, h)
        d.ellipse([x, y, x + r.uniform(1, 4), y + r.uniform(1, 4)], fill=(250, 246, 236, int(r.uniform(60, 160))))
    for _ in range(3):
        x = r.uniform(0, w)
        d.line([(x, 0), (x + r.uniform(-20, 20), h)], fill=(240, 236, 226, 40), width=1)
    if date:
        f = _font("Montserrat[wght].ttf", int(26 * PS), 600)
        d.text((w - 30 * PS, h - 26 * PS), date, font=f, fill=(255, 150, 40, 230), anchor="rs")
    return img


def bg_school(d, c):
    sky = grad((176, 196, 214), (220, 222, 216), PW * PS, PH * PS)
    d._image.paste(to_img(sky))
    c.rect(-10, 70, 820, 330, (214, 196, 170))            # фасад школы
    for i in range(9):
        for j in range(2):
            c.rect(30 + i * 88, 105 + j * 105, 90 + i * 88, 175 + j * 105, (110, 130, 150))
            c.rect(58 + i * 88, 105 + j * 105, 62 + i * 88, 175 + j * 105, (230, 226, 216))
    c.rect(330, 250, 470, 330, (120, 82, 60))
    c.rect(-10, 330, 820, 610, (128, 124, 118))            # асфальт
    for k in range(3):                                     # деревья (не там, где стоят люди)
        x = [250, 470, 560][k]
        c.rect(x - 6, 200, x + 6, 340, (80, 62, 46))
        c.ell(x, 190, 70, 80, (88, 112, 64))
        c.ell(x - 30, 220, 50, 50, (76, 100, 56))
    c.rect(-10, 300, 820, 306, (40, 44, 40))                # ограда
    for i in range(0, 820, 16):
        c.rect(i, 262, i + 4, 336, (40, 44, 40))
    c.rect(-10, 268, 820, 272, (40, 44, 40))


def bg_zags(d, c):
    sky = grad((190, 206, 222), (232, 230, 222), PW * PS, PH * PS)
    d._image.paste(to_img(sky))
    c.rect(150, 60, 720, 360, (226, 216, 200))
    for i in range(6):
        c.rect(190 + i * 88, 90, 220 + i * 88, 360, (240, 234, 222))
    c.poly([(140, 60), (730, 60), (435, -20)], (210, 200, 186))
    c.rect(-10, 360, 820, 610, (150, 146, 140))
    for k in range(5):
        c.rect(260 - k * 20, 360 + k * 14, 610 + k * 20, 374 + k * 14, (176, 170, 160))
    c.rect(-10, 250, 140, 262, (34, 34, 32))
    for i in range(-10, 140, 14):
        c.rect(i, 230, i + 4, 380, (34, 34, 32))
    for x in (40, 790):
        c.ell(x, 180, 80, 110, (78, 104, 60))


def bg_sea(d, c):
    sky = grad((150, 186, 214), (226, 230, 226), PW * PS, PH * PS)
    d._image.paste(to_img(sky))
    c.rect(-10, 230, 820, 380, (70, 120, 150))
    for k in range(12):
        c.line([(k * 70, 260 + k % 3 * 30), (k * 70 + 40, 260 + k % 3 * 30)], (170, 200, 214), 2)
    c.rect(-10, 380, 820, 610, (214, 196, 160))
    for i in range(8):                                     # сваи в воде
        x = 480 + i * 46
        c.rect(x, 344, x + 8, 392, (70, 56, 44))
    c.rect(460, 326, 820, 346, (128, 104, 80))              # настил пирса
    c.rect(460, 342, 820, 348, (90, 72, 56))
    for i in range(462, 820, 14):
        c.line([(i, 327), (i, 341)], (110, 88, 66), 1)
    c.rect(460, 290, 820, 294, (90, 72, 56))                # перила
    for i in range(8):
        c.rect(470 + i * 48, 290, 474 + i * 48, 328, (90, 72, 56))


def bg_room(d, c, wall=(196, 170, 140)):
    a = grad(wall, mix(wall, (0, 0, 0), 0.25), PW * PS, PH * PS)
    d._image.paste(to_img(a))
    for i in range(0, 820, 40):
        c.rect(i, 0, i + 2, 420, mix(wall, (255, 255, 255), 0.1))
    c.rect(-10, 420, 820, 610, (120, 80, 54))
    # окно: вид на улицу
    x0, y0, x1, y1 = 520, 60, 780, 330
    c.rect(x0, y0, x1, y1, (170, 196, 216))                         # небо
    c.rect(x0, 120, 600, 250, (176, 170, 160))                      # дом напротив
    for i in range(3):
        for j in range(3):
            c.rect(x0 + 8 + i * 26, 132 + j * 38, x0 + 22 + i * 26, 152 + j * 38, (120, 136, 150))
    c.rect(x0, 250, x1, 282, (96, 96, 98))                          # дорога
    c.rect(x0, 282, x1, y1, (150, 146, 138))                        # тротуар
    c.rect(x0, 280, x1, 283, (200, 196, 186))                       # бордюр
    c.rect(737, 190, 743, 284, (80, 62, 46))                        # дерево
    c.ell(740, 175, 42, 52, (84, 112, 62))
    c.ell(722, 195, 28, 30, (74, 100, 56))
    c.poly([(x0 + 20, y0), (x0 + 60, y0), (x0 + 10, y0 + 120), (x0, y0 + 120)], (255, 255, 255, 40))   # блик
    fr = (236, 232, 222)                                            # рама с переплётом
    c.rect(x0 - 14, y0 - 14, x1 + 14, y0, fr)
    c.rect(x0 - 14, y1, x1 + 14, y1 + 6, fr)
    c.rect(x0 - 14, y0, x0, y1, fr)
    c.rect(x1, y0, x1 + 14, y1, fr)
    c.rect((x0 + x1) / 2 - 6, y0, (x0 + x1) / 2 + 6, y1, fr)
    c.rect(x0, 130, x1, 138, fr)
    c.rect(x0 - 26, y1 + 4, x1 + 26, y1 + 18, (226, 222, 212))     # подоконник
    c.rect(x0 - 26, y1 + 18, x1 + 26, y1 + 24, (180, 176, 166))
    for xa, xb in ((x0 - 60, x0 + 6), (x1 - 6, x1 + 60)):          # шторы
        c.poly([(xa, y0 - 26), (xb, y0 - 26), (xb + (6 if xa < x0 else -6), y1 + 60), (xa, y1 + 60)], (150, 74, 64))
        for k in range(3):
            xx = xa + (xb - xa) * (k + 0.5) / 3
            c.line([(xx, y0 - 20), (xx, y1 + 56)], (124, 58, 50), 3)
    c.rect(x0 - 70, y0 - 34, x1 + 70, y0 - 24, (110, 90, 70))


def bg_park(d, c, winter=False):
    sky = grad((200, 210, 220) if winter else (160, 196, 220), (230, 230, 228), PW * PS, PH * PS)
    d._image.paste(to_img(sky))
    ground = (236, 238, 240) if winter else (110, 140, 76)
    c.rect(-10, 330, 820, 610, ground)
    for x in (250, 340, 450, 550):
        c.rect(x - 5, 200, x + 5, 340, (70, 56, 44))
        c.ell(x, 190, 60, 90, (200, 206, 212) if winter else (70, 104, 56))


BG = dict(school=bg_school, zags=bg_zags, sea=bg_sea, room=bg_room, park=bg_park,
          winter=lambda d, c: bg_park(d, c, True))


# Каталог снимков: фон, люди (вид, x, ноги, рост, поза-параметры), позиция незнакомца, где бабушка.
# Незнакомец: (x, feet, h, вид, поза: 'stand' | 'sit' | 'cam')
PHOTOS = {
    "first_class": dict(bg="school", tone="90s", date="'90 9 1", man=(700, 372, 130, "cam"),
                        people=[("boy", 360, 560, 300, "front"), ("grandma", 150, 575, 390, "gaze")]),
    "wedding": dict(bg="zags", tone="2000s", date="'05 6 18", man=(70, 382, 124, "stand"),
                    people=[("suit", 420, 575, 420, "front"), ("bride", 530, 575, 400, "front"),
                            ("grandma", 690, 580, 380, "gaze")]),
    "sea": dict(bg="sea", tone="2000s", date="'11 7 23", man=(660, 333, 96, "stand"),
                people=[("dad", 300, 590, 400, "front"), ("woman", 420, 590, 380, "front"),
                        ("kid", 360, 594, 220, "front"), ("grandma", 120, 594, 360, "gaze")]),
}
EXTRA = [("room", "'93 12 31", 1993), ("park", "'95 5 9", 1995), ("winter", "'97 1 7", 1997),
         ("school", "'00 5 25", 2000), ("room", "'02 3 8", 2002), ("park", "'08 8 2", 2008),
         ("winter", "'14 1 1", 2014), ("room", "'17 11 4", 2017), ("park", "'21 6 12", 2021),
         ("sea", "'99 8 14", 1999), ("school", "'13 9 1", 2013), ("room", "'25 10 3", 2025)]


def man_age(year):
    """0 — молодой (1985), 1 — старик (2026)."""
    return clamp((year - 1985) / 41)


def stranger(d, x, feet, h, mode, f=-1, age=0.2, pal=None):
    pal = dict(pal or PALETTE["man"])
    g = int(lerp(0, 40, age))
    pal["hat"] = tuple(min(255, c + g // 2) for c in pal["hat"])
    if mode == "sit":
        p = pose(lean=8 + 10 * age, neck=10, ln=(80, 85), lf=(84, 88), an=(30, 70), af=(26, 76))
    elif mode == "cam":
        p = pose(lean=4 + 10 * age, neck=6 + 6 * age, an=(35, 95), af=(28, 102), ln=(3, 3), lf=(-3, 3))
    else:
        p = pose(lean=3 + 12 * age, neck=4 + 10 * age, an=(4, 10), af=(-3, 12), ln=(2, 2), lf=(-2, 3))
    parts = ppl(d, x, feet, h, p, f, "man", pal)
    J = skeleton(x, feet, h, p, f)
    if mode == "cam":
        wx, wy = J["an_wr"]
        cc = C(d._image, PS)
        cc.rect(wx - 0.05 * h, wy - 0.05 * h, wx + 0.05 * h, wy + 0.02 * h, (24, 24, 26))
    return J


def people_in(d, spec, man_head):
    for kind, x, feet, h, mode in spec:
        pal = PALETTE["bride" if kind == "bride" else kind]
        k = "woman" if kind == "bride" else ("young" if kind in ("dad", "kid") else kind)
        if kind == "boy":
            k = "boy"
        if mode == "gaze":
            # бабушка: корпус к камере, голова повернута к незнакомцу
            tx = man_head[0]
            p = pose(turn=1 if tx > x else -1, an=(10, 30), af=(10, 30))
            shaded(d._image, body_front(x, feet, h, p, "grandma"), pal, (x, feet))
        else:
            p = pose(an=(8, 14), af=(8, 14))
            shaded(d._image, body_front(x, feet, h, p, k), pal, (x, feet))
            if kind == "boy":
                cc = C(d._image, PS)   # букет гладиолусов
                for i in range(5):
                    cc.line([(x + 30, feet - h * 0.42), (x + 10 + i * 12, feet - h * 0.75 - i % 2 * 20)], (60, 110, 50), 3)
                    cc.ell(x + 10 + i * 12, feet - h * 0.78 - i % 2 * 20, 12, 18, (220, 60, 90))


@lru_cache(None)
def photo(name, scale=2):
    """Растр снимка PW×PH×scale. Возвращает (картинка, позиция головы незнакомца, позиция головы бабушки)."""
    global PS
    old, PS = PS, scale
    try:
        return _photo(name)
    finally:
        PS = old


def _photo(name):
    spec = PHOTOS[name]
    img = Image.new("RGB", (PW * PS, PH * PS))
    d = ImageDraw.Draw(img, "RGBA")
    c = C(img, PS)
    BG[spec["bg"]](d, c)
    mx, mf, mh, mode = spec["man"]
    J = stranger(d, mx, mf, mh, mode, f=-1 if mx > 400 else 1, age=man_age(int("19" + spec["date"][1:3]) if spec["date"][1] == "9" else int("20" + spec["date"][1:3])))
    man_head = J["head"]
    people_in(d, spec["people"], man_head)
    gm = [p for p in spec["people"] if p[0] == "grandma"][0]
    gm_head = (gm[1], gm[2] - gm[3] * 0.93)
    img = photo_post(img, hash(name) % 1000, spec["tone"], spec["date"])
    return img, man_head, gm_head


@lru_cache(None)
def extra_photo(i):
    bg, date, year = EXTRA[i]
    r = np.random.default_rng(100 + i)
    img = Image.new("RGB", (PW * PS, PH * PS))
    d = ImageDraw.Draw(img, "RGBA")
    c = C(img, PS)
    BG[bg](d, c)
    left = r.random() < 0.5
    mx = r.uniform(60, 180) if left else r.uniform(620, 740)
    feet = {"school": 372, "sea": 412, "park": 352, "winter": 352}.get(bg, 352)
    mh = 120
    if bg == "room":
        mx = 690          # на тротуаре за окном
        feet, mh = 312, 70
    J = stranger(d, mx, feet, mh, "stand" if r.random() < 0.6 else "cam", f=1 if left else -1, age=man_age(year))
    kinds = ["dad", "woman", "kid", "suit", "boy"]
    n = int(r.integers(2, 4))
    xs = np.linspace(280, 520, n)
    for k in range(n):
        kd = kinds[int(r.integers(0, len(kinds)))]
        hh = 300 if kd in ("kid", "boy") else 400
        p = pose(an=(8, 14), af=(8, 14))
        shaded(img, body_front(xs[k], 590, hh, p, "young" if kd in ("dad", "kid") else ("boy" if kd == "boy" else kd)),
               PALETTE[kd], (xs[k], 590))
    gx = 140 if not left else 700
    p = pose(turn=1 if mx > gx else -1, an=(10, 30), af=(10, 30))
    shaded(img, body_front(gx, 594, 360, p, "grandma"), PALETTE["grandma"], (gx, 594))
    img = photo_post(img, 200 + i, "90s" if year < 2004 else "2000s", date)
    return img, J["head"], (gx, 594 - 360 * 0.93)


def framed(img, border=26, back_col=(246, 242, 232)):
    """Фото с белой рамкой как у отпечатка."""
    w, h = img.size
    out = Image.new("RGB", (w + border * 2 * PS, h + border * 2 * PS), back_col)
    out.paste(img, (border * PS, border * PS))
    return out


def place_photo(world, img, cx, cy, width, ang=0.0, shadow=True, alpha=1.0):
    """Кладёт снимок в мир (координаты кадра). Возвращает функцию перевода точки снимка в мир."""
    k = width * S / img.width
    sm = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS if k < 0.6 else Image.BILINEAR)
    rot = sm.convert("RGBA").rotate(ang, Image.BICUBIC, expand=True)
    if alpha < 1:
        rot.putalpha(rot.split()[3].point(lambda p: int(p * alpha)))
    px, py = int(cx * S - rot.width / 2), int(cy * S - rot.height / 2)
    if shadow:
        sh = Image.new("RGBA", rot.size, (0, 0, 0, 0))
        sh.putalpha(rot.split()[3].point(lambda p: int(p * 0.55 * alpha)))
        sh = sh.filter(ImageFilter.GaussianBlur(10 * S))
        world.paste(sh, (px + 10 * S, py + 16 * S), sh)
    world.paste(rot, (px, py), rot)

    def to_world(u, vv, border=0):
        # u, vv — координаты в системе снимка (PW×PH) без рамки
        ps = img.width / (PW + 2 * border)          # масштаб растра этого снимка
        bx = (u + border) * ps * k / S - sm.width / S / 2
        by = (vv + border) * ps * k / S - sm.height / S / 2
        r = math.radians(-ang)
        return cx + bx * math.cos(r) - by * math.sin(r), cy + bx * math.sin(r) + by * math.cos(r)
    return to_world


def red_circle(c, x, y, r, k, w=7, seed=0, col=RED):
    """Красный круг маркером, рисуется по мере k (0..1)."""
    if k <= 0:
        return
    rng = np.random.default_rng(seed)
    a0 = rng.uniform(0, 360)
    n = int(60 * min(1, k) * 1.1) + 2
    pts = []
    for i in range(n):
        a = math.radians(a0 + i / 60 * 380)
        rr = r * (1 + 0.06 * math.sin(i * 0.4 + seed))
        pts.append((x + math.cos(a) * rr * 1.1, y + math.sin(a) * rr * 0.92))
    c.line(pts, col + (235,), w)


# ================================================================ постобработка
@lru_cache(None)
def vignette(strength=1.0):
    y, x = np.ogrid[:H, :W]
    d = np.sqrt(((x - W / 2) / (W * 0.72)) ** 2 + ((y - H / 2) / (H * 0.72)) ** 2)
    return np.clip(1.12 - d ** 2.2 * strength, 0.25, 1).astype(np.float32)[:, :, None]


@lru_cache(None)
def grain_bank():
    r = np.random.default_rng(9)
    return [r.normal(0, 1, (H // 2, W // 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:, :, None] for _ in range(8)]


def post(img, t, warm=0.0, grain=7.0, vig=1.0):
    a = np.asarray(img, np.float32)
    a = a * vignette(vig)
    lum = a.mean(axis=2, keepdims=True) / 255
    a = a + (1 - lum) * np.array([-3, 2, 8], np.float32) + lum * np.array([6 + warm * 8, 3 + warm * 3, -5 - warm * 8], np.float32)
    a = a + grain_bank()[int(t * FPS) % 8] * grain
    return to_img(a)


def squash(parts, cx, sx):
    """Сжатие по горизонтали вокруг cx — для разворота фигуры."""
    out = []
    for kind_, geo, part in parts:
        if kind_ == "poly":
            geo = [(cx + (x - cx) * sx, y) for x, y in geo]
        elif kind_ == "ell":
            geo = (cx + (geo[0] - cx) * sx, geo[1], max(0.5, geo[2] * abs(sx)), geo[3])
        else:
            p0, p1, r0, r1 = geo
            geo = ((cx + (p0[0] - cx) * sx, p0[1]), (cx + (p1[0] - cx) * sx, p1[1]), r0, r1)
        out.append((kind_, geo, part))
    return out


def bg_street(d, c, khr=True):
    sky = grad((170, 190, 208), (226, 226, 220), PW * PS, PH * PS)
    d._image.paste(to_img(sky))
    for k, (x0, x1, top, col) in enumerate([(-20, 300, 40, (196, 190, 178)), (330, 640, 70, (186, 182, 172)),
                                            (660, 840, 30, (200, 194, 182))]):
        c.rect(x0, top, x1, 360, col)
        for i in range(int(x0) + 20, int(x1) - 20, 46):
            for j in range(int(top) + 20, 330, 52):
                c.rect(i, j, i + 24, j + 30, (120, 136, 150))
    c.rect(-10, 330, 820, 610, (130, 126, 120))
    for x in (120, 480, 760):
        c.rect(x - 6, 220, x + 6, 340, (80, 62, 46))
        c.ell(x, 210, 64, 80, (90, 116, 66))


def fence_front(c, x0, x1, top=0, bottom=600, gap=70, w=22, col=(22, 24, 22)):
    for x in range(int(x0), int(x1), gap):
        c.rect(x, top, x + w, bottom, col)
    c.rect(x0, top + 90, x1, top + 110, col)
    c.rect(x0, bottom - 140, x1, bottom - 120, col)


REVERSE = {
    "first_class": dict(bg="street", tone="90s", date="'90 9 1", fence=True,
                        people=[("boy", 430, 470, 220, "back"), ("grandma", 300, 480, 280, "look"),
                                ("woman", 600, 470, 260, "back"), ("dad", 690, 480, 280, "back")],
                        shooter=(470, 380, 120)),
    "wedding": dict(bg="park", tone="2000s", date="'05 6 18", fence=True,
                    people=[("suit", 420, 500, 300, "back"), ("bride", 500, 500, 290, "back"),
                            ("grandma", 600, 505, 270, "look")], shooter=(470, 380, 120)),
    "sea": dict(bg="beach_rev", tone="2000s", date="'11 7 23", fence=False,
                people=[("dad", 360, 520, 250, "back"), ("woman", 440, 520, 240, "back"),
                        ("kid", 400, 524, 140, "back"), ("grandma", 250, 524, 230, "look")], shooter=(420, 410, 110)),
}


def bg_beach_rev(d, c):
    sky = grad((160, 192, 216), (228, 230, 226), PW * PS, PH * PS)
    d._image.paste(to_img(sky))
    c.rect(-10, 200, 820, 300, (90, 130, 90))
    for x in range(0, 820, 70):
        c.ell(x, 200, 50, 60, (80, 116, 70))
    c.rect(-10, 290, 820, 610, (216, 198, 162))
    c.poly([(-10, 470), (820, 520), (820, 610), (-10, 610)], (120, 96, 72))   # доски пирса
    for k in range(8):
        c.line([(-10, 480 + k * 18), (820, 530 + k * 12)], (96, 76, 58), 3)
    c.rect(-10, 430, 820, 440, (100, 80, 60))


BG["street"] = bg_street
BG["beach_rev"] = bg_beach_rev


@lru_cache(None)
def photo_rev(name, scale=2):
    """Тот же день, снятый с места незнакомца: все спиной, бабушка оглядывается в объектив."""
    global PS
    old, PS = PS, scale
    try:
        spec = REVERSE[name]
        img = Image.new("RGB", (PW * PS, PH * PS))
        d = ImageDraw.Draw(img, "RGBA")
        c = C(img, PS)
        BG[spec["bg"]](d, c)
        sx, sf, sh = spec["shooter"]       # родственник с фотоаппаратом лицом к нам
        p = pose(an=(30, 120), af=(30, 120))
        shaded(img, body_front(sx, sf, sh, p, "young"), PALETTE["dad"], (sx, sf))
        c.rect(sx - 0.07 * sh, sf - 0.86 * sh, sx + 0.07 * sh, sf - 0.79 * sh, (20, 20, 22))
        gm_head = None
        for kind, x, feet, h, mode in spec["people"]:
            pal = PALETTE["bride" if kind == "bride" else kind]
            k = {"dad": "young", "kid": "young", "bride": "woman"}.get(kind, kind)
            if mode == "look":
                pp = pose(look=True, an=(10, 30), af=(10, 30))
                shaded(img, body_front(x, feet, h, pp, k, back=True), pal, (x, feet))
                gm_head = (x, feet - h * 0.93)
            else:
                shaded(img, body_front(x, feet, h, pose(an=(8, 14), af=(8, 14)), k, back=True), pal, (x, feet))
        if spec["fence"]:
            fl = Image.new("RGBA", img.size, (0, 0, 0, 0))
            fence_front(C(fl, PS), -40, 860, 0, 640, 150, 34, (18, 20, 18, 255))
            fl = fl.filter(ImageFilter.GaussianBlur(5 * PS))
            img.paste(fl, (0, 0), fl)
        img = photo_post(img, hash(name) % 1000 + 50, spec["tone"], spec["date"])
        return img, gm_head
    finally:
        PS = old


@lru_cache(None)
def stranger_card(year, w=420, h=560, seed=0):
    """Кроп снимка: незнакомец крупнее (для «с каждым годом — старше»)."""
    global PS
    old, PS = PS, 2
    try:
        img = Image.new("RGB", (w * PS, h * PS))
        d = ImageDraw.Draw(img, "RGBA")
        c = C(img, PS)
        a = grad((150, 170, 160), (110, 112, 104), w * PS, h * PS)
        d._image.paste(to_img(a))
        r = np.random.default_rng(year)
        for k in range(5):
            x = r.uniform(0, w)
            c.ell(x, r.uniform(40, 200), r.uniform(60, 120), r.uniform(80, 140), (70, 96, 62))
        c.rect(0, h * 0.72, w, h, (124, 120, 112))
        img = img.filter(ImageFilter.GaussianBlur(6 * PS))
        d = ImageDraw.Draw(img, "RGBA")
        age = man_age(year)
        J = stranger(d, w * 0.55, h * 0.96, h * 0.86, "cam", f=-1, age=age)
        if age > 0.7:      # трость
            C(img, PS).line([(w * 0.55 + 40, h * 0.6), (w * 0.55 + 56, h * 0.96)], (40, 30, 24), 5)
        img = photo_post(img, year, "90s" if year < 2004 else "2000s", None, fade_k=0.1)
        return img
    finally:
        PS = old

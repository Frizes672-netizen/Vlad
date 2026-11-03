"""Рендер шортса «Struck by Lightning 7 Times» (Рой Салливан, смотритель парка Шенандоа).

Грозовые ночные сцены, молнии с ветвлением и вспышкой, дождь, мокрая земля, силуэты с контровым светом.
Запуск: python3 render.py build/video.mp4 [--preview СЕКУНДЫ ...] [--range ПЕРВЫЙ ПОСЛЕДНИЙ]
"""
import math
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import lib1
from engine import (CREAM, CUR_T, MS, PFI, RED, S, SHAKES, WH, WW, C, body_front, breathe, cam, clamp, dust, ease,
                    eio, env, eout, fbm, framed, glow, grad, keyed, lerp, mix, person_parts, pose, post, radial,
                    silhouette, skeleton, to_img, walk, word_t)
from hands import pov_hand
from lib1 import WHITE, caption, shadow_text
from props import bloom, rain, reflect, tree, volumetric
from render_helpers import rot_parts
from subs import draw_subs
from timeline import DURATION, FPS, H, SCENES, W, v

lib1.pov_hand = pov_hand
BOLT = (220, 230, 255)
YEL = (255, 214, 60)


def wt(word, after):
    return word_t(word, after)


# ================================================================ гроза: небо, молния, вспышка
@lru_cache(None)
def storm_sky(seed=1, ground=1420, hills=True):
    n = fbm(WW, WH, [3, 7, 18, 50], seed)
    yy = np.linspace(0, 1, WH)[:, None]
    base = np.array([20, 24, 36], np.float32) * (1 - yy[:, :, None] * 0.5)
    clouds = np.clip((n - 0.35) * 1.8, 0, 1) * (yy < ground / H)
    a = base + clouds[:, :, None] * np.array([46, 50, 64], np.float32)
    img = to_img(a)
    c = C(img)
    if hills:
        c.poly([(0, ground - 140), (220, ground - 300), (430, ground - 190), (700, ground - 330), (1080, ground - 160),
                (1080, ground), (0, ground)], (12, 14, 20))
        for x in range(0, W, 46):                       # лес на склонах
            h = 70 + (x * 37) % 60
            c.poly([(x - 26, ground - 120), (x, ground - 120 - h), (x + 26, ground - 120)], (8, 10, 14))
    c.rect(0, ground, W, H, (10, 12, 14))
    return img


def bolt_points(x0, y0, x1, y1, seed, detail=7):
    r = np.random.default_rng(seed)
    pts = [(x0, y0), (x1, y1)]
    disp = abs(y1 - y0) * 0.18
    for _ in range(detail):
        new = [pts[0]]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            mx, my = (ax + bx) / 2 + r.uniform(-disp, disp), (ay + by) / 2 + r.uniform(-disp * 0.2, disp * 0.2)
            new += [(mx, my), (bx, by)]
        pts = new
        disp *= 0.55
    return pts


def lightning(img, t, at, x0, x1, y1, seed=1, life=0.35):
    """Молния: ствол + ветви, яркое ядро, свечение; возвращает силу вспышки 0..1."""
    k = (t - at) / life
    if k < 0 or k > 1:
        return 0.0
    flick = 1.0 if k < 0.25 else (0.6 if int(k * 12) % 2 else 1.0) * (1 - k)
    pts = bolt_points(x0, -40, x1, y1, seed)
    branches = []
    r = np.random.default_rng(seed + 9)
    for _ in range(5):
        i = int(r.integers(len(pts) // 6, len(pts) * 3 // 4))
        bx, by = pts[i]
        branches.append(bolt_points(bx, by, bx + r.uniform(-260, 260), by + r.uniform(150, 420), seed + 20 + i, 5))

    def shapes(d, kk):
        for p, w in [(pts, 26)] + [(b, 12) for b in branches]:
            d.line([(x * kk, y * kk) for x, y in p], fill=(90, 110, 200), width=int(w * kk * 2))
    out = glow(img, shapes, 30, 1.6 * flick)
    img.paste(out)
    c = C(img)
    for b in branches:
        c.line(b, BOLT + (int(200 * flick),), 3)
    c.line(pts, (255, 255, 255, int(255 * flick)), 7)
    c.line(pts, BOLT + (int(255 * flick),), 3)
    return flick


def flash_frame(frame, f):
    if f <= 0:
        return frame
    a = np.asarray(frame, np.float32)
    return to_img(a + (255 - a) * 0.55 * f + np.array([10, 14, 30]) * f)


def ranger_hat(c, hx, hy, h, f=1, col=(40, 34, 26), rim=None):
    """Шляпа смотрителя (campaign hat): широкие поля и «защипнутая» тулья."""
    r = 0.054 * h
    brim = [(hx - r * 1.9, hy - r * 0.5), (hx + r * 1.9, hy - r * 0.55), (hx + r * 1.85, hy - r * 0.32), (hx - r * 1.85, hy - r * 0.28)]
    crown = [(hx - r * 0.95, hy - r * 0.5), (hx - r * 0.7, hy - r * 1.4), (hx - r * 0.2, hy - r * 1.15), (hx, hy - r * 1.55),
             (hx + r * 0.2, hy - r * 1.15), (hx + r * 0.7, hy - r * 1.4), (hx + r * 0.95, hy - r * 0.55)]
    if rim:
        c.poly([(x + 3, y - 3) for x, y in brim + crown[::-1]], rim)
    c.poly(crown, col)
    c.poly(brim, col)
    c.rect(hx - r * 0.95, hy - r * 0.68, hx + r * 0.95, hy - r * 0.52, (90, 70, 40))


def ranger(img, x, feet, h, p, f=1, view="side", rim=(200, 210, 255), light=(1, -0.6), hat=True, base=(8, 9, 12)):
    parts = person_parts(x, feet, h, p, f, "suit", view)
    silhouette(img, parts, base=base, rim=rim, light=light, rim_w=4, fill_light=(4, 6, 10))
    if view == "side":
        J = skeleton(x, feet, h, p, f)
        hx, hy = J["head"]
    else:
        J = None
        hx, hy = x, feet - h * (0.48 + 0.29 + 0.035 + 0.066 * 0.95) - 0.012 * h
    if hat:
        ranger_hat(C(img), hx, hy, h, f, rim=rim)
    return J, (hx, hy)


def strike_hud(im, t, n):
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([740, 130, 1010, 230], 22, fill=(0, 0, 0, 150))
    d.text((875, 180), f"⚡ {n}/7".replace("⚡ ", ""), font=MS(60, 800), fill=YEL, anchor="mm")
    d.polygon([(760, 150), (790, 150), (775, 180), (795, 180), (762, 214), (772, 186), (752, 186)], fill=YEL)


# ================================================================ 1. хук
def sc_hook(t, lt, dur):
    img = storm_sky(1).copy()
    ranger(img, 540, 1420, 620, breathe(pose(an=(8, 14), af=(8, 14)), t), view="front")
    f = lightning(img, t, 0.35, 520, 540, 1420 - 620, 3) + lightning(img, t, 1.6, 700, 545, 1420 - 620, 5) * 0.8
    f += lightning(img, t, 3.6, 300, 540, 1420 - 620, 7) * 0.7
    reflect(img, 1424, 400, 0.5, 4, ripple=2, t=t)
    rain(C(img), t, 200, (0, 0, W, H), 3, (180, 190, 220), 80)
    frame = flash_frame(cam(img, lerp(1.15, 1.0, lt / dur), 540, 1100, hh_amp=3), f)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        n = min(7, 1 + int(lt / (dur * 0.7) * 7))
        k = eout(lt / 0.4)
        shadow_text(d, (540, 320), f"STRUCK {n}x", MS(130, 800), YEL + (int(255 * k),))
        k2 = eout((t - wt("survived", 3.6)) / 0.4)
        shadow_text(d, (540, 450), "AND SURVIVED", MS(84, 800), WHITE + (int(255 * k2),))
    return frame, txt


# ================================================================ 2. Рой Салливан
@lru_cache(None)
def forest_bg():
    img = to_img(grad((40, 56, 52), (10, 14, 12)))
    c = C(img)
    for i in range(14):
        x = (i * 83) % 1100
        c.poly([(x - 80, 1500), (x, 500 + (i * 53) % 300), (x + 80, 1500)], (14 + i % 3 * 4, 26 + i % 3 * 4, 20))
    c.rect(0, 1500, W, H, (20, 24, 18))
    return img


def sc_roy(t, lt, dur):
    img = forest_bg().copy()
    img = volumetric(img, [(700, 0), (900, 0), (620, 1500), (300, 1500)], (60, 70, 50), 60)
    p = breathe(pose(an=(10, 30), af=(14, 40), lean=2), t)
    ranger(img, 540, 1700, 1100, p, view="front", rim=(220, 210, 170), light=(0.4, -1), base=(30, 34, 26))
    c = C(img)
    c.poly([(470, 860), (500, 840), (530, 860), (520, 900), (480, 900)], (200, 170, 60))     # значок
    frame = cam(img, lerp(1.2, 1.05, lt / dur), 540, 1000)

    def txt(im):
        caption(im, t, SHOT_START["roy"] + 0.3, 99, "ROY SULLIVAN", "park ranger · Virginia", y=330, size=110)
    return frame, txt


# ================================================================ 3–9. удары
@lru_cache(None)
def tower_bg():
    img = storm_sky(2).copy()
    c = C(img)
    for s in (-1, 1):                                   # пожарная вышка на гребне
        c.line([(540 + s * 160, 1300), (540 + s * 70, 720)], (18, 16, 14), 12)
    for k in range(5):
        y = 760 + k * 110
        w = 70 + (y - 720) / 580 * 90
        c.line([(540 - w, y), (540 + w, y)], (18, 16, 14), 6)
        c.line([(540 - w, y), (540 + w, y + 110)], (18, 16, 14), 4)
    c.rect(440, 600, 640, 720, (24, 20, 18))
    c.poly([(420, 600), (660, 600), (540, 520)], (20, 18, 16))
    c.rect(460, 620, 620, 690, (200, 160, 90))
    return img


def sc_s1(t, lt, dur):
    t_hit = wt("blows", 16.0) - 0.15
    t_toe = wt("nail", 17.2) - 0.2
    if t < t_toe:
        img = tower_bg().copy()
        f = lightning(img, t, t_hit, 520, 540, 560, 11, 0.45)
        rain(C(img), t, 160, (0, 0, W, H), 5, (180, 190, 220), 70)
        frame = flash_frame(cam(img, lerp(1.0, 1.15, lt / 5), 540, 900, hh_amp=2), f)
    else:                                               # дымящийся ботинок
        img = to_img(grad((30, 26, 24), (10, 9, 8)))
        c = C(img)
        c.rect(300, 700, 520, 1240, (46, 52, 46))                        # штанина
        c.poly([(280, 1180), (540, 1180), (560, 1260), (300, 1260)], (60, 44, 30))   # голенище
        c.poly([(260, 1250), (640, 1250), (820, 1310), (880, 1380), (880, 1440), (260, 1440)], (70, 50, 34))
        for k in range(5):                                                # шнуровка
            c.line([(330 + k * 50, 1200), (360 + k * 50, 1260)], (200, 180, 140), 4)
        c.poly([(700, 1270), (840, 1300), (870, 1380), (780, 1360)], (100, 74, 50))   # блик на носке
        c.rect(240, 1440, 900, 1474, (24, 18, 14))                        # подошва
        c.ell(820, 1330, 40, 30, (255, 140, 60))
        img = glow(img, lambda d, k: d.ellipse([740 * k, 1260 * k, 900 * k, 1400 * k], fill=(160, 70, 20)), 40)
        for i in range(8):                              # дым
            ph = (t * 0.5 + i * 0.12) % 1
            C(img).ell(820 + math.sin(t * 2 + i) * 30 * ph, 1300 - ph * 600, 30 + ph * 80, 30 + ph * 80, (140, 140, 140, int(90 * (1 - ph))))
        frame = cam(img, 1.1, 560, 1250)

    def txt(im):
        strike_hud(im, t, 1)
        caption(im, t, wt("1942", 12), t_toe - 0.05, "1942", "fire lookout tower", y=330, size=130)
        caption(im, t, t_toe, 99, "TOENAIL: GONE", y=330, size=100, accent=YEL)
    return frame, txt


@lru_cache(None)
def road_bg():
    img = storm_sky(3).copy()
    c = C(img)
    c.poly([(380, 1420), (700, 1420), (1080, 1920), (0, 1920)], (24, 24, 28))
    for k in range(6):
        y = 1450 + k * 80
        c.rect(535 - k * 4, y, 545 + k * 4, y + 40, (200, 190, 120))
    return img


def pickup(c, x, y, s=1.0):
    c.poly([(x - 260 * s, y), (x - 250 * s, y - 110 * s), (x - 60 * s, y - 120 * s), (x - 20 * s, y - 210 * s),
            (x + 120 * s, y - 210 * s), (x + 150 * s, y - 120 * s), (x + 260 * s, y - 110 * s), (x + 270 * s, y)], (60, 70, 60))
    c.poly([(x - 10 * s, y - 120 * s), (x + 10 * s, y - 196 * s), (x + 110 * s, y - 196 * s), (x + 134 * s, y - 120 * s)], (30, 36, 44))
    for sx in (-170, 170):
        c.ell(x + sx * s, y, 46 * s, 46 * s, (14, 14, 14))
        c.ell(x + sx * s, y, 22 * s, 22 * s, (120, 120, 120))
    c.ell(x + 262 * s, y - 70 * s, 14 * s, 14 * s, (255, 240, 200))


def sc_s2(t, lt, dur):
    img = road_bg().copy()
    c = C(img)
    x = lerp(300, 620, lt / dur)
    pickup(c, x, 1560, 1.0)
    img = volumetric(img, [(x + 262, 1490), (x + 262, 1500), (x + 900, 1700), (x + 900, 1300)], (90, 84, 60), 30)
    f = lightning(img, t, wt("eyebrows", 22.2) - 0.25, 600, x - 20, 1360, 13, 0.45)
    rain(C(img), t, 180, (0, 0, W, H), 6, (180, 190, 220), 70)
    reflect(img, 1424, 300, 0.35, 4)
    frame = flash_frame(cam(img, 1.05, 540, 1300, hh_amp=2), f)

    def txt(im):
        strike_hud(im, t, 2)
        caption(im, t, wt("eyebrows", 22.2), 99, "EYEBROWS: BURNED OFF", y=330, size=78, accent=YEL)
    return frame, txt


def sc_s3(t, lt, dur):
    img = storm_sky(4).copy()
    c = C(img)
    c.rect(620, 1000, 1000, 1420, (30, 28, 30))         # дом
    c.poly([(600, 1000), (1020, 1000), (810, 840)], (24, 22, 24))
    c.rect(700, 1100, 800, 1200, (210, 170, 100))
    for i in range(0, W, 60):                           # забор
        c.rect(i, 1330, i + 12, 1440, (40, 36, 34))
    c.line([(0, 1360), (W, 1360)], (40, 36, 34), 6)
    p = breathe(pose(an=(10, 20), af=(10, 20)), t)
    ranger(img, 380, 1440, 560, p, view="front", hat=False)
    f = lightning(img, t, wt("shoulder", 28.0) - 0.25, 400, 420, 1440 - 560 * 0.78, 17, 0.45)
    rain(C(img), t, 160, (0, 0, W, H), 7, (180, 190, 220), 70)
    frame = flash_frame(cam(img, 1.1, 500, 1150, hh_amp=2), f)

    def txt(im):
        strike_hud(im, t, 3)
        caption(im, t, wt("shoulder", 28.0), 99, "SHOULDER: SCORCHED", y=330, size=80, accent=YEL)
    return frame, txt


def sc_s4(t, lt, dur):
    t_bucket = wt("dump", 34.0) - 0.15
    img = to_img(grad((34, 30, 28), (10, 10, 10)))
    p = breathe(pose(an=(150, 30) if t > t_bucket else (40, 120), af=(150, 30) if t > t_bucket else (40, 120)), t, 0.5)
    J, (hx, hy) = ranger(img, 540, 1750, 1100, p, view="front", hat=False, rim=(255, 160, 80), light=(0, -1))
    c = C(img)
    if t < t_bucket + 0.4:                               # волосы горят
        fire = 1 - clamp((t - t_bucket - 0.2) / 0.3)
        for i in range(10):
            ph = (t * 3 + i * 0.1) % 1
            c.ell(hx + math.sin(i * 2 + t * 9) * 50, hy - 60 - ph * 160, 36 * (1 - ph) + 10, 50 * (1 - ph) + 14,
                  (255, int(160 - 80 * ph), 40, int(220 * fire * (1 - ph))))
        img = glow(img, lambda d, k: d.ellipse([(hx - 150) * k, (hy - 260) * k, (hx + 150) * k, (hy + 40) * k], fill=(160, 70, 20)), 50, fire)
    if t > t_bucket:                                     # ведро и всплеск
        k = clamp((t - t_bucket) / 0.6)
        c = C(img)
        c.poly([(hx - 90, hy - 300 + k * 60), (hx + 90, hy - 300 + k * 60), (hx + 70, hy - 170 + k * 60), (hx - 70, hy - 170 + k * 60)], (110, 110, 120))
        r_ = np.random.default_rng(int(t * 10))
        for i in range(40):
            a = r_.uniform(math.pi * 0.1, math.pi * 0.9)
            d_ = k * r_.uniform(80, 420)
            c.ell(hx + math.cos(a) * d_ * (1 if i % 2 else -1), hy - 120 + math.sin(a) * d_, 10, 14, (150, 190, 240, int(220 * (1 - k * 0.6))))
    frame = cam(img, 1.0, 540, 1000)

    def txt(im):
        strike_hud(im, t, 4)
        caption(im, t, wt("hair", 31.8), t_bucket, "HAIR: ON FIRE", y=330, size=100, accent=(255, 120, 60))
        caption(im, t, t_bucket + 0.2, 99, "BUCKET OF WATER", y=330, size=90)
    return frame, txt


def sc_s5(t, lt, dur):
    img = storm_sky(6, hills=False).copy()
    x = lerp(150, 900, lt / dur)
    p = walk(lt * 4.5, 0.9)
    ranger(img, x, 1420, 520, p, 1)
    c = C(img)
    for i in range(8):                                   # туча идёт за ним
        c.ell(x - 180 + i * 50, 520 + math.sin(i + t) * 20, 160, 110, (14, 14, 20, 245))
    f = lightning(img, t, wt("following", 39.8) + 0.3, x, x + 10, 1420 - 520 * 0.95, 19, 0.4)
    rain(C(img), t, 120, (x - 300, 560, x + 300, H), 8, (170, 180, 210), 90)
    reflect(img, 1424, 400, 0.4, 4)
    frame = flash_frame(cam(img, 1.0, 540, 1000), f)

    def txt(im):
        strike_hud(im, t, 5)
        caption(im, t, wt("swears", 38), 99, "THE CLOUD FOLLOWED HIM", y=1650, size=76, accent=YEL)
    return frame, txt


def sc_avoid(t, lt, dur):
    img = storm_sky(8, hills=False).copy()
    ranger(img, 540, 1420, 560, breathe(pose(an=(8, 14), af=(8, 14)), t), view="front")
    k = eio(lt / 1.8)
    for i, s in enumerate((-1, 1, -1, 1)):
        x = 540 + s * (150 + i // 2 * 140) + s * 260 * k
        p = walk(lt * 4 + i, 0.6) if 0.2 < lt < 1.8 else pose(an=(8, 14), af=(8, 14))
        silhouette(img, person_parts(x, 1420 - i // 2 * 30, 520 - i // 2 * 40, p, s, "young"), base=(8, 9, 12),
                   rim=(150, 160, 200), light=(-s, -0.5), rim_w=3)
    rain(C(img), t, 160, (0, 0, W, H), 9, (170, 180, 210), 70)
    reflect(img, 1424, 400, 0.4, 4)
    frame = cam(img, 1.05, 540, 1100)

    def txt(im):
        caption(im, t, SHOT_START["avoid"] + 0.4, 99, "NOBODY STOOD NEXT TO HIM", "not even coworkers", y=330, size=70)
    return frame, txt


def sc_s6(t, lt, dur):
    img = to_img(grad((24, 24, 30), (8, 8, 10)))
    c = C(img)
    c.rect(380, 300, 560, 1400, (30, 34, 40))           # нога в брюках крупно
    c.poly([(360, 1400), (580, 1400), (760, 1480), (760, 1560), (360, 1560)], (34, 26, 20))
    c.rect(0, 1560, W, H, (14, 14, 16))
    r_ = np.random.default_rng(int(t * 24))
    for i in range(24):                                   # искры
        a = r_.uniform(0, 2 * math.pi)
        d_ = r_.uniform(20, 220)
        c.line([(560, 1420), (560 + math.cos(a) * d_, 1420 + math.sin(a) * d_)], (255, 230, 150, 220), 3)
    img = glow(img, lambda d, k: d.ellipse([440 * k, 1300 * k, 700 * k, 1540 * k], fill=(160, 140, 220)), 40, 1.2)
    frame = cam(img, 1.0, 540, 1100, hh_amp=6)

    def txt(im):
        strike_hud(im, t, 6)
        caption(im, t, SHOT_START["s6"] + 0.2, 99, "ANKLE", y=330, size=120, accent=YEL)
    return frame, txt


@lru_cache(None)
def river_bg():
    img = storm_sky(10).copy()
    c = C(img)
    c.poly([(0, 1420), (W, 1420), (W, 1560), (0, 1620)], (24, 30, 40))
    for k in range(10):
        c.line([(0, 1440 + k * 16), (W, 1450 + k * 14)], (40, 50, 64), 2)
    c.rect(0, 1600, W, H, (16, 18, 14))
    return img


def bear(c, x, y, s=1.0, k=0.0):
    for dx, col in ((-5, (120, 110, 140)), (0, (26, 18, 14))):    # контровой свет от молний, потом тело
        _bear(c, x + dx, y - abs(dx) * 0.6, s, k, col)


def _bear(c, x, y, s, k, col):
    c.ell(x, y - 120 * s, 170 * s, 110 * s, col)
    c.ell(x + 150 * s, y - 180 * s - 30 * k * s, 70 * s, 60 * s, col)
    c.ell(x + 120 * s, y - 235 * s - 30 * k * s, 18 * s, 18 * s, col)
    c.ell(x + 200 * s, y - 165 * s - 30 * k * s, 30 * s, 22 * s, col)
    for sx in (-110, -40, 60, 120):
        c.rect(x + sx * s - 22 * s, y - 60 * s, x + sx * s + 22 * s, y, col)


def sc_s7(t, lt, dur):
    t_hit = wt("lightning", 52.6)
    t_bear = wt("bear", 55.2) - 0.3
    t_stick = wt("fights", 57.7) - 0.1
    img = river_bg().copy()
    c = C(img)
    p = breathe(pose(an=(70, 40), af=(60, 50), lean=6), t)
    if t > t_stick:
        p = pose(an=(150 - 60 * math.sin((t - t_stick) * 10) ** 2, 10), af=(40, 50), lean=12)
    J, _ = ranger(img, 360, 1600, 620, p, 1)
    c = C(img)
    wx, wy = J["an_wr"]
    if t < t_stick:                                     # удочка и леска
        c.line([(wx, wy), (wx + 420, wy - 380)], (40, 34, 28), 5)
        c.line([(wx + 420, wy - 380), (wx + 460, 1500)], (200, 200, 200, 160), 1.5)
    else:                                               # палка
        c.line([(wx, wy), (wx + 260, wy - 160)], (60, 44, 30), 12)
    if t > t_bear:
        k = eio((t - t_bear) / 0.8)
        bx = lerp(1300, 760, k)
        bear(c, bx, 1610, 1.0, math.sin(t * 6) * 0.5)
        if t < t_stick:
            c.ell(bx + 210, 1440, 40, 16, (160, 170, 180))      # рыба в пасти
    f = lightning(img, t, t_hit, 380, 362, 1600 - 620 * 0.95, 23, 0.45)
    rain(C(img), t, 160, (0, 0, W, H), 10, (180, 190, 220), 70)
    frame = flash_frame(cam(img, 1.05, 560, 1300, hh_amp=2), f)

    def txt(im):
        strike_hud(im, t, 7)
        caption(im, t, SHOT_START["s7"] + 0.3, t_bear, "FISHING", y=330, size=110)
        caption(im, t, t_bear, t_stick, "+ A BEAR", y=330, size=120, accent=YEL)
        caption(im, t, t_stick, 99, "WITH A STICK", y=330, size=110)
    return frame, txt


# ================================================================ 10. шансы
def sc_odds(t, lt, dur):
    img = to_img(grad((10, 12, 24), (2, 2, 6)))
    r_ = np.random.default_rng(3)
    c = C(img)
    for i in range(140):                                 # звёзды-нули
        x, y = r_.uniform(0, W), r_.uniform(0, H)
        c.ell(x, y, 2, 2, (200, 210, 255, int(120 + 100 * math.sin(t * 2 + i))))
    frame = cam(img, 1.0, 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = eout((t - wt("one", 61.9)) / 0.4)
        shadow_text(d, (540, 720), "1 IN", MS(110, 800), WHITE + (int(255 * k),))
        zeros = "22," + ",".join(["000"] * 7)
        n = int(clamp((t - wt("22", 62.6)) / 1.2) * len(zeros))
        f = MS(58, 800)
        lines = [zeros[:n][i:i + 16] for i in range(0, max(1, n), 16)]
        for i, ln in enumerate(lines):
            shadow_text(d, (540, 880 + i * 80), ln, f, YEL + (255,))
        caption(im, t, wt("septillion", 63.3), 99, "22 SEPTILLION", y=1450, size=110, accent=YEL)
    return frame, txt


# ================================================================ 11. шляпы в музее
def sc_hats(t, lt, dur):
    img = to_img(grad((30, 26, 22), (8, 7, 6)))
    c = C(img)
    c.rect(160, 700, 920, 1400, (40, 46, 50, 120))        # витрина
    c.rect(160, 700, 920, 1400, None, outline=(160, 170, 180), w=6)
    c.rect(120, 1400, 960, 1560, (50, 40, 32))
    for i, x in enumerate((350, 730)):
        ranger_hat(c, x, 1250, 2700, 1, col=(80, 62, 42))
        c.ell(x + 30, 1060, 46, 30, (20, 16, 12, 210))   # прожжённое пятно
        c.ell(x - 60, 1090, 26, 18, (30, 22, 16, 180))
        c.rect(x - 90, 1300, x + 90, 1340, (210, 200, 170))   # табличка
        c.text(x, 1320, f"STRIKE {[4, 5][i]}", MS(22, 800), (40, 34, 26))
    c.poly([(220, 720), (300, 720), (200, 1100)], (255, 255, 255, 40))
    img = volumetric(img, [(480, 0), (600, 0), (980, 1400), (100, 1400)], (90, 80, 60), 50)
    frame = cam(img, lerp(1.0, 1.12, lt / dur), 540, 1100)

    def txt(im):
        caption(im, t, SHOT_START["hats"] + 0.3, 99, "IN A MUSEUM", "his burned ranger hats", y=330, size=100)
    return frame, txt


# ================================================================ 12. вопрос
def sc_question(t, lt, dur):
    img = storm_sky(1).copy()
    ranger(img, 460, 1420, 620, breathe(pose(an=(8, 14), af=(8, 14)), t), view="front")
    c = C(img)
    for k in range(18):                                   # пустое место рядом — пунктир
        a = k / 18 * 2 * math.pi
        c.ell(700 + math.cos(a) * 90, 1150 + math.sin(a) * 250, 6, 6, (255, 255, 255, 160))
    f = lightning(img, t, SHOT_START["question"] + 1.2, 500, 470, 1420 - 620 * 0.95, 29, 0.4)
    rain(C(img), t, 180, (0, 0, W, H), 3, (180, 190, 220), 80)
    reflect(img, 1424, 400, 0.5, 4, ripple=2, t=t)
    frame = flash_frame(cam(img, 1.0, 560, 1100), f)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        k = eout((lt - 0.2) / 0.5)
        shadow_text(d, (540, 330), "WOULD YOU STAND", MS(84, 800), WHITE + (int(255 * k),))
        shadow_text(d, (540, 440), "NEXT TO HIM?", MS(100, 800), YEL + (int(255 * k),))
    return frame, txt


# ================================================================ монтаж
SC = {name: globals()["sc_" + name] for _, name in SCENES}
SHOTS = []
SHOT_START = {}
for i, (a, name) in enumerate(SCENES):
    start = max(0.0, v(a))
    end = v(SCENES[i + 1][0]) if i + 1 < len(SCENES) else DURATION
    SHOTS.append((start, end, name))
    SHOT_START[name] = start

SHAKES += [(0.35, 14, 0.35), (1.6, 12, 0.35), (word_t("blows", 16) - 0.15, 16, 0.4), (word_t("eyebrows", 22.2) - 0.25, 14, 0.4),
           (word_t("shoulder", 28) - 0.25, 14, 0.4), (word_t("lightning", 52.6), 16, 0.4)]

WARM = {"hook": -0.6, "roy": 0.2, "s1": -0.5, "s2": -0.5, "s3": -0.5, "s4": 0.4, "s5": -0.6, "avoid": -0.6,
        "s6": -0.3, "s7": -0.5, "odds": -0.4, "hats": 0.6, "question": -0.6}


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
    img = bloom(img.convert("RGB"), 190, 0.5)
    img = post(img, t, WARM.get(name, 0.0), grain=7)
    if txt:
        txt(img)
    draw_subs(img, t, y=1250)
    return img.tobytes()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/video.mp4"
    if "--preview" in sys.argv:
        for s in sys.argv[sys.argv.index("--preview") + 1:]:
            Image.frombytes("RGB", (W, H), frame(int(float(s) * FPS))).save(f"{out}_{s}.jpg", quality=86)
        return
    total = int(DURATION * FPS)
    first, last = 0, total
    if "--range" in sys.argv:
        i = sys.argv.index("--range")
        first, last = int(sys.argv[i + 1]), min(total, int(sys.argv[i + 2]))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "17", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool(3, maxtasksperchild=60) as pool:
        for n, buf in enumerate(pool.imap(frame, range(first, last), chunksize=4), first):
            ff.stdin.write(buf)
            if n % (FPS * 10) == 0:
                print(f"  кадр {n}/{total}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

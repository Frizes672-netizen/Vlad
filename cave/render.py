"""Рендер шортса «Два месяца в пещере без часов» (Мишель Сифр, 1962).

Ледяная пещера, налобный фонарь, пар изо рта, капли, часы, телефон наверх, календари.
Движок прошлых роликов: engine/props/hands/lib1.
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
from engine import (CREAM, CUR_T, MS, PALETTE, PF, PFI, RED, S, SHAKES, WH, WW, C, body_front, breathe, cam, clamp,
                    dust, ease, eio, env, eout, fbm, framed, glow, grad, keyed, lerp, mix, person_parts, photo_post,
                    place_photo, pose, post, radial, rot_pts, shaded, silhouette, skeleton, to_img, walk, word_t)
from hands import pov_hand
from lib1 import WHITE, caption, draw_quote, shadow_text, year_tag
from props import bloom, volumetric
from timeline import DURATION, FPS, H, QUOTES, SCENES, W, v
from subs import draw_subs

lib1.pov_hand = pov_hand
SKIN = (206, 168, 142)
ICE = (120, 170, 210)
LAMP = (255, 228, 170)


def wt(word, after):
    return word_t(word, after)


def bw(img, k=1.0, warm=0.08):
    a = np.asarray(img, np.float32)
    g = a.mean(axis=2, keepdims=True)
    return to_img(a * (1 - k) + g * np.array([1 + warm, 1.0, 1 - warm], np.float32) * k)


# ================================================================ ледяная пещера
@lru_cache(None)
def ice_cave(seed=1, tint=ICE):
    """Грот во льду: неровные стены, сосульки, блики, тьма в глубине."""
    n = fbm(WW, WH, [4, 10, 30, 90], seed)
    n2 = fbm(WW, WH, [3, 8], seed + 5)
    yy, xx = np.mgrid[0:WH, 0:WW].astype(np.float32)
    # своды: тьма в центре-глубине, лёд по краям
    depth = np.clip(1 - np.sqrt(((xx - WW * 0.5) / (WW * 0.55)) ** 2 + ((yy - WH * 0.48) / (WH * 0.42)) ** 2), 0, 1)
    ice = np.array(tint, np.float32) * (0.25 + 0.55 * n[:, :, None]) * (1 - depth[:, :, None] * 0.9)
    streak = np.sin(xx / (40 * S) + n2 * 9) * 0.5 + 0.5
    ice = ice * (0.94 + 0.1 * streak[:, :, None])
    fine = fbm(WW, WH, [40, 120, 300], seed + 9)
    ice = ice * (0.85 + 0.3 * fine[:, :, None])
    spec = np.clip((n - 0.62) * 6, 0, 1) * (1 - depth) * 0.8
    ice = ice + spec[:, :, None] * np.array([150, 190, 220], np.float32)
    floor = np.clip((yy - WH * 0.78) / (WH * 0.22), 0, 1)
    ice = ice * (1 - floor[:, :, None]) + floor[:, :, None] * np.array([40, 48, 60], np.float32) * (0.6 + 0.6 * n[:, :, None])
    img = to_img(ice)
    c = C(img)
    r = np.random.default_rng(seed)
    for i in range(46):                               # сосульки
        x = r.uniform(0, W)
        L = r.uniform(40, 220)
        w = r.uniform(6, 20)
        y0 = r.uniform(-20, 160)
        c.poly([(x - w, y0), (x + w, y0), (x + r.uniform(-4, 4), y0 + L)], (170, 210, 235, 200))
        c.line([(x - w * 0.3, y0), (x, y0 + L * 0.8)], (235, 248, 255, 140), 2)
    return img


def headlamp_glow(img, x, y, ang, length=900, spread=0.32, col=(150, 130, 90), strength=1.0):
    a0, a1 = ang - spread, ang + spread
    poly = [(x, y), (x + math.cos(a0) * length, y + math.sin(a0) * length), (x + math.cos(a1) * length, y + math.sin(a1) * length)]
    return volumetric(img, poly, col, 36, strength)


def caver(img, x, feet, h, p, f=1, rim=LAMP, view="side", lamp=True, light=(1, -0.4)):
    """Спелеолог: силуэт, каска, налобный фонарь."""
    parts = person_parts(x, feet, h, p, f, "young", view)
    silhouette(img, parts, base=(10, 12, 16), rim=rim, light=light, rim_w=4, fill_light=(6, 8, 12))
    if view == "side":
        J = skeleton(x, feet, h, p, f)
        hx, hy = J["head"]
    else:
        hx, hy = x, feet - h * 0.86
    c = C(img)
    c.ell(hx, hy - h * 0.03, h * 0.066, h * 0.055, (200, 160, 40))   # каска
    c.rect(hx - h * 0.07, hy - h * 0.01, hx + h * 0.07, hy + h * 0.004, (150, 120, 30))
    lx = hx + f * h * 0.05 if view == "side" else hx
    if lamp:
        c.ell(lx, hy - h * 0.035, h * 0.016, h * 0.016, (255, 250, 220))
    return lx, hy - h * 0.035


def breath(c, x, y, t, f=1):
    """Пар изо рта в холоде."""
    k = (t * 0.35) % 1
    for i in range(5):
        kk = (k + i * 0.08) % 1
        c.ell(x + f * kk * 90, y - kk * 40, 14 + kk * 50, 10 + kk * 30, (220, 230, 240, int(70 * (1 - kk))))


def drips(c, t, n=10, seed=2, area=(0, 0, W, 1500)):
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = area
    for i in range(n):
        x = r.uniform(x0, x1)
        ph = (t * r.uniform(0.6, 1.2) + r.uniform(0, 1)) % 1
        y = y0 + 120 + ph ** 2 * (y1 - y0)
        c.line([(x, y), (x, y + 16)], (200, 230, 250, 200), 3)


# ================================================================ 1. хук: часы трескаются, падение в темноту
def watch(c, x, y, r, t, crack=0.0, stop=None, strap=True):
    c.ell(x, y + 10, r + 30, r + 30, (0, 0, 0, 120))
    if strap:
        c.rect(x - r * 0.45, y - r - 260, x + r * 0.45, y + r + 260, (60, 40, 30), r=20)       # ремешок
        for k in range(6):
            c.line([(x - r * 0.4, y - r - 240 + k * 30), (x + r * 0.4, y - r - 240 + k * 30)], (50, 34, 24), 2)
    c.ell(x, y, r + 22, r + 22, (180, 180, 186))
    c.ell(x, y, r + 8, r + 8, (90, 90, 96))
    c.ell(x, y, r, r, (232, 228, 214))
    for k in range(60):
        a = k * math.pi / 30
        L = 0.12 if k % 5 == 0 else 0.05
        c.line([(x + math.cos(a) * r * (0.92 - L), y + math.sin(a) * r * (0.92 - L)),
                (x + math.cos(a) * r * 0.92, y + math.sin(a) * r * 0.92)], (30, 30, 30), 5 if k % 5 == 0 else 2)
    for k in range(12):
        a = k * math.pi / 6 - math.pi / 2
        c.text(x + math.cos(a) * r * 0.7, y + math.sin(a) * r * 0.7, str(k or 12), MS(int(r * 0.13), 700), (30, 30, 30))
    tt = stop if stop is not None else t
    sec = tt * 6
    for ang, L, w, col in ((tt / 3600 * 30 + 300, 0.45, 12, (20, 20, 20)), (tt / 60 * 6 + 60, 0.68, 8, (20, 20, 20)),
                           (math.floor(sec) * 6 if stop is None else sec, 0.8, 3, (190, 30, 30))):
        a = math.radians(ang - 90)
        c.line([(x, y), (x + math.cos(a) * r * L, y + math.sin(a) * r * L)], col, w)
    c.ell(x, y, 12, 12, (20, 20, 20))
    c.poly([(x - r * 0.6, y - r * 0.7), (x - r * 0.2, y - r * 0.9), (x - r * 0.5, y - r * 0.3)], (255, 255, 255, 60))
    if crack > 0:                                     # трещины по стеклу
        r_ = np.random.default_rng(5)
        for k in range(9):
            a = r_.uniform(0, 2 * math.pi)
            pts = [(x + 40, y - 60)]
            for j in range(6):
                a += r_.uniform(-0.5, 0.5)
                L = r * 0.18 * crack
                pts.append((pts[-1][0] + math.cos(a) * L, pts[-1][1] + math.sin(a) * L))
            c.line(pts, (255, 255, 255, 220), 3)


@lru_cache(None)
def mountain_bg():
    img = to_img(grad((70, 90, 120), (190, 200, 210)))
    c = C(img)
    c.poly([(0, 900), (260, 420), (480, 760), (700, 300), (1080, 860), (1080, 1920), (0, 1920)], (60, 66, 80))
    c.poly([(220, 500), (260, 420), (300, 500), (280, 520)], (236, 240, 246))
    c.poly([(650, 380), (700, 300), (760, 400), (720, 420)], (236, 240, 246))
    n = fbm(WW, WH, [6, 20, 60], 9)
    a = np.asarray(img, np.float32) * (0.85 + 0.3 * n[:, :, None])
    img = to_img(a)
    c = C(img)
    c.ell(540, 1250, 230, 300, (6, 6, 8))              # вход в пещеру
    c.ell(540, 1290, 190, 260, (0, 0, 0))
    return img


def sc_hook(t, lt, dur):
    t_fall = wt("пещере", 2) - 0.35
    if t < t_fall:
        img = to_img(grad((30, 30, 34), (10, 10, 12)))
        c = C(img)
        crack = eio((t - 0.9) / 0.6)
        watch(c, 540, 900, 300, 14 * 3600 + 32 * 60 + t * 60, crack, stop=None if crack < 0.5 else 14 * 3600 + 32 * 60 + 54)
        frame = cam(img, lerp(1.0, 1.2, lt / t_fall), 540, 900)
        frame = flash_if(frame, t, 1.25, 0.2)

        def txt(im):
            caption(im, t, 0.05, t_fall, "2 МЕСЯЦА БЕЗ ВРЕМЕНИ", "что будет с человеком?", y=1600, size=84)
        return frame, txt
    k = eio((t - t_fall) / 2.2)                        # проваливаемся во вход пещеры
    img = mountain_bg()
    frame = cam(img, lerp(1.0, 7.5, k), 540, lerp(960, 1290, k))
    frame = to_img(np.asarray(frame, np.float32) * (1 - 0.92 * clamp((k - 0.6) / 0.4)))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    oc = C(ov, 1)
    t_w, t_s, t_z = wt("часов", 3.5), wt("солнца", 4.8), wt("звука", 6.5)
    if t_w < t < t_s:                                   # остановившиеся часы
        watch(oc, 540, 1350, 190, 0, 1.0, stop=14 * 3600 + 32 * 60 + 54, strap=False)
    elif t_s <= t < t_z:                                # солнце гаснет
        kk = clamp((t - t_s) / 1.2)
        for j in range(6):
            oc.ell(540, 1350, 150 + j * 40, 150 + j * 40, (255, 200, 90, int(60 * (1 - kk) / (j + 1))))
        oc.ell(540, 1350, 150, 150, (255, int(220 - 120 * kk), int(120 - 100 * kk), int(255 * (1 - kk * 0.9))))
    elif t >= t_z:                                      # звук затихает в ровную линию
        kk = clamp((t - t_z) / 1.0)
        pts = [(x, 1350 + math.sin(x / 18 + t * 20) * 120 * (1 - kk) * math.sin(x / 1080 * math.pi)) for x in range(80, 1000, 6)]
        oc.line(pts, (255, 90, 80, 255), 6)
    frame.paste(ov, (0, 0), ov)
    t_no = [(wt("часов", 3.5), "БЕЗ ЧАСОВ"), (wt("солнца", 4.8), "БЕЗ СОЛНЦА"), (wt("звука", 6.5), "БЕЗ ЗВУКА")]

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        for i, (ta, lab_) in enumerate(t_no):
            kk = eout((t - ta + 0.1) / 0.3)
            if kk > 0:
                shadow_text(d, (540, 420 + i * 170), lab_, MS(104, 800), (255, 90, 80, int(255 * kk)) if i == 2 else WHITE + (int(255 * kk),))
    return frame, txt


def flash_if(im, t, at, d):
    k = 1 - clamp((t - at) / d)
    if t < at or k <= 0:
        return im
    return to_img(np.asarray(im, np.float32) * (1 - k) + 255 * k)


# ================================================================ 2. 1962: Мишель Сифр
@lru_cache(None)
def archive_photo():
    import engine
    ps = 2
    img = Image.new("RGB", (800 * ps, 600 * ps))
    c = C(img, ps)
    img.paste(to_img(grad((170, 176, 180), (110, 112, 110), 800 * ps, 600 * ps)))
    c.poly([(0, 330), (200, 120), (380, 280), (560, 90), (800, 300), (800, 600), (0, 600)], (90, 96, 100))
    c.poly([(170, 160), (200, 120), (230, 170)], (230, 232, 236))
    c.poly([(530, 130), (560, 90), (600, 150)], (230, 232, 236))
    c.rect(0, 470, 800, 600, (130, 126, 118))
    pal = dict(PALETTE["dad"], coat=(70, 74, 60), sleeve=(70, 74, 60), pants=(60, 60, 54))
    shaded(img, body_front(400, 560, 330, pose(an=(10, 40), af=(40, 70)), "young"), pal, (400, 560), s=ps)
    c.ell(400, 560 - 330 * 0.9, 26, 20, (200, 170, 60))             # каска
    return photo_post(img, 62, "sepia", None)


def sc_y1962(t, lt, dur):
    img = to_img(grad((20, 20, 20), (6, 6, 6)))
    k = eio(lt / dur)
    place_photo(img, framed(archive_photo(), 18), 540, 960, 980, lerp(-2, 0, k))
    frame = bw(cam(img, lerp(1.0, 1.4, k), 540, lerp(960, 880, k)), 1.0)
    t_vol = wt("добровольно", 12.8)

    def txt(im):
        year_tag(im, t, SHOT_START["y1962"] + 0.15, t_vol - 0.1, "1962", "МИШЕЛЬ СИФР · ФРАНЦИЯ", y=250)
        if t > t_vol:
            d = ImageDraw.Draw(im, "RGBA")
            kk = clamp((t - t_vol) / 0.25)
            s = lerp(1.8, 1.0, eout(kk))
            d.rounded_rectangle([540 - 330 * s, 1500 - 70 * s, 540 + 330 * s, 1500 + 70 * s], 14, outline=(220, 40, 40, int(230 * kk)), width=int(10 * s))
            d.text((540, 1500), "ДОБРОВОЛЬНО", font=MS(int(70 * s), 800), fill=(220, 40, 40, int(230 * kk)), anchor="mm")
        caption(im, t, wt("23", 14.5), 99, "ЕМУ 23", y=250, size=110)
    return frame, txt


# ================================================================ 3. спуск и палатка во льду
@lru_cache(None)
def shaft_bg():
    n = fbm(WW, WH, [4, 12, 40], 31)
    yy, xx = np.mgrid[0:WH, 0:WW].astype(np.float32)
    walls = np.clip(np.abs(xx - WW / 2) / (WW * 0.32), 0, 1) ** 2
    a = np.array(ICE, np.float32) * (0.2 + 0.7 * walls[:, :, None]) * (0.6 + 0.6 * n[:, :, None])
    a *= (0.4 + 0.6 * (1 - yy / WH))[:, :, None]
    return to_img(a)


def sc_descent(t, lt, dur):
    t_tent = wt("поладка", 19.5) - 0.15
    if t < t_tent:                                      # спуск по верёвке в ледяной колодец
        img = shaft_bg().copy()
        scroll = lt * 260
        a = np.asarray(img, np.float32)
        a = np.roll(a, int(-scroll * S) % WH, axis=0)
        img = to_img(a)
        c = C(img)
        c.line([(540, 0), (540, H)], (200, 180, 140), 5)
        p = breathe(pose(an=(170, 30), af=(160, 40), ln=(20, 40), lf=(-10, 30), lean=-4), t)
        lx, ly = caver(img, 540, 1250, 640, p, 1, view="front", light=(0, -1))
        img = headlamp_glow(img, lx, ly, math.pi / 2 + 0.3 * math.sin(t * 0.8), 1100, 0.3, (130, 120, 90))
        frame = cam(img, 1.0, 540, 960, hh_amp=3)

        def txt(im):
            d = ImageDraw.Draw(im, "RGBA")
            depth = min(130, int(lerp(0, 130, clamp(lt / (t_tent - SHOT_START["descent"])))))
            d.rounded_rectangle([700, 140, 1020, 260], 24, fill=(0, 0, 0, 150))
            d.text((860, 175), "ГЛУБИНА", font=MS(26, 700), fill=(200, 210, 220), anchor="mm")
            d.text((860, 222), f"−{depth} м", font=MS(56, 800), fill=(120, 200, 255), anchor="mm")
            caption(im, t, SHOT_START["descent"] + 0.2, t_tent, "ПОД ЛЕДНИК", y=1700, size=96)
        return frame, txt
    img = ice_cave(1).copy()                            # палатка, фонарь, 0°, темнота
    c = C(img)
    c.poly([(300, 1500), (540, 1080), (780, 1500)], (150, 70, 40))
    c.poly([(540, 1080), (780, 1500), (700, 1500)], (110, 50, 30))
    c.poly([(470, 1500), (540, 1300), (610, 1500)], (40, 20, 14))
    c.rect(820, 1380, 860, 1470, (60, 60, 64))           # фонарь-лампа
    c.ell(840, 1370, 22, 22, LAMP)
    img = glow(img, lambda d, k: d.ellipse([600 * k, 1180 * k, 1080 * k, 1600 * k], fill=(150, 110, 60)), 70,
               0.8 + 0.2 * math.sin(t * 9) * 0.3)
    p = breathe(pose(lean=20, neck=20, ln=(80, 100), lf=(70, 110), an=(60, 60), af=(50, 70)), t)
    lx, ly = caver(img, 270, 1520, 560, p, 1, light=(1, -0.3))
    c = C(img)
    breath(c, lx + 20, ly + 40, t)
    drips(c, t, 8, 3)
    c.rect(120, 1100, 170, 1350, (230, 230, 230), r=20)  # термометр 0°
    c.rect(137, 1230, 153, 1335, (200, 30, 30))
    c.ell(145, 1340, 20, 20, (200, 30, 30))
    frame = cam(img, lerp(1.1, 1.0, (t - t_tent) / 4), 540, 1150, hh_amp=2)
    frame = to_img(np.asarray(frame, np.float32) * (1 - 0.6 * clamp((t - wt("темнота", 22.8) - 0.2) / 0.6)))

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        items = [(wt("поладка", 19.5), "ПАЛАТКА"), (wt("фонарь", 20.3), "ФОНАРЬ"), (wt("ноль", 21), "0°"), (wt("темнота", 22.8), "ТЕМНОТА")]
        for i, (ta, lab) in enumerate(items):
            kk = eout((t - ta) / 0.3)
            if kk > 0:
                shadow_text(d, (540, 300 + i * 120), lab, MS(84, 800), ((120, 200, 255) if i == 2 else WHITE) + (int(255 * kk),))
    return frame, txt


# ================================================================ 4. правило: телефон наверх, время не говорят
def field_phone(c, x, y, s=1.0, ring=0.0):
    sh = math.sin(ring * 60) * 6 * (ring > 0)
    x += sh
    c.rect(x - 150 * s, y - 120 * s, x + 150 * s, y + 120 * s, (50, 46, 40), r=16 * s)
    c.rect(x - 130 * s, y - 100 * s, x + 130 * s, y + 100 * s, (70, 64, 56), r=12 * s)
    c.ell(x, y + 20 * s, 60 * s, 60 * s, (30, 28, 26))
    c.ell(x, y + 20 * s, 44 * s, 44 * s, (180, 170, 150))
    c.rect(x - 170 * s, y - 170 * s, x + 170 * s, y - 120 * s, (24, 22, 20), r=24 * s)   # трубка
    c.ell(x - 150 * s, y - 145 * s, 40 * s, 34 * s, (24, 22, 20))
    c.ell(x + 150 * s, y - 145 * s, 40 * s, 34 * s, (24, 22, 20))
    c.line([(x + 150 * s, y + 60 * s), (x + 260 * s, y - 400 * s), (x + 300 * s, -100)], (40, 40, 40), 6 * s)   # провод вверх


def sc_rule(t, lt, dur):
    t_time = wt("но", 29.5) - 0.1
    img = ice_cave(4).copy()
    c = C(img)
    field_phone(c, 520, 1300, 1.2)
    p = breathe(pose(lean=10, neck=10, an=(110, 90), af=(30, 40)), t)
    lx, ly = caver(img, 250, 1600, 760, p, 1, light=(1, -0.3))
    img = headlamp_glow(img, lx, ly, 0.5, 900, 0.3, (120, 100, 70))
    c = C(img)
    breath(c, lx + 30, ly + 50, t)
    frame = cam(img, 1.05, 520, 1200, hh_amp=2)
    if t > t_time:                                      # наверху: часы есть; внизу — нет
        top = to_img(grad((60, 60, 56), (30, 30, 28), WW, WH // 2))
        ct = C(top)
        watch(ct, 540, 470, 210, 14 * 3600 + 32 * 60 + t * 60)
        top = top.resize((W, H // 2))
        frame = frame.copy()
        k = eio((t - t_time) / 0.4)
        frame.paste(top.crop((0, 0, W, int(H / 2 * k))), (0, 0))
        d = ImageDraw.Draw(frame, "RGBA")
        d.rectangle([0, H // 2 - 4, W, H // 2 + 4], fill=(0, 0, 0, 255))

    def txt(im):
        caption(im, t, SHOT_START["rule"] + 0.2, t_time - 0.05, "ПРОСНУЛСЯ — ЗВОНИ НАВЕРХ", "единственное правило", y=330, size=70)
        if t > t_time:
            d = ImageDraw.Draw(im, "RGBA")
            kk = eout((t - t_time - 0.3) / 0.4)
            shadow_text(d, (540, 880), "НАВЕРХУ ЗНАЮТ ВРЕМЯ", MS(56, 800), WHITE + (int(255 * kk),))
            shadow_text(d, (540, 1060), "ЕМУ — НЕТ", MS(96, 800), (255, 90, 80, int(255 * kk)))
    return frame, txt


# ================================================================ 5. первая неделя: палочки на стене
def sc_week1(t, lt, dur):
    img = ice_cave(7).copy()
    c = C(img)
    marks = [wt("встал", 34.5), wt("поел", 35.2), wt("уснул", 35.7), wt("сутки", 36.8)] + [wt("логично", 37.8) + i * 0.25 for i in range(3)]
    n = sum(1 for m in marks if t > m)
    for i in range(n):
        x = 330 + i * 70 + (40 if i >= 4 else 0)
        c.line([(x, 760), (x + 6, 1060)], (230, 245, 255), 9)
        c.line([(x + 3, 760), (x + 9, 1060)], (60, 80, 100), 3)
    if n >= 5:
        c.line([(310, 1000), (640, 800)], (230, 245, 255), 9)
    img = headlamp_glow(img, 900, 1900, -2.2, 1400, 0.35, (150, 130, 90), 1.2)
    c = C(img)
    tip = (330 + max(0, n - 1) * 70 + 60, 1060 + 30 * math.sin(t * 20))
    c.line([(tip[0] + 100, tip[1] + 120), tip], (200, 200, 210), 10)            # нож
    pov_hand(c, (1150, 2300), (tip[0] + 160, tip[1] + 220), -2.3, 0.7, 1.0, skin=SKIN, sleeve=(70, 74, 60))
    frame = cam(img, 1.05, 560, 960, hh_amp=2)

    def txt(im):
        caption(im, t, SHOT_START["week1"] + 0.2, wt("логично", 37.8) - 0.05, "СЧИТАЕТ ДНИ", "первая неделя", y=330, size=96)
        caption(im, t, wt("логично", 37.8), 99, "ЛОГИЧНО ЖЕ", y=330, size=96)
    return frame, txt


# ================================================================ 6. путаница: часы сходят с ума, дни одинаковые
def sc_confusion(t, lt, dur):
    t_month = wt("через", 45.8) - 0.15
    t_same = wt("каждый", 49.4) - 0.1
    if t < t_month:                                     # спит → просыпается рывком, часы крутятся
        img = ice_cave(1).copy()
        c = C(img)
        c.poly([(300, 1500), (540, 1080), (780, 1500)], (150, 70, 40))
        c.poly([(470, 1500), (540, 1300), (610, 1500)], (40, 20, 14))
        lying = [(kind, geo, part) for kind, geo, part in person_parts(560, 1500, 520, pose(), 1, "young", "front")]
        from render_helpers import rot_parts
        sil = rot_parts(lying, 560, 1500, -90)
        silhouette(img, [(k, g, p) for k, g, p in sil], base=(12, 12, 16), rim=LAMP, light=(0, -1), rim_w=3)
        frame = cam(img, 1.1, 560, 1300, hh_amp=2)
        lamp_on = int(lt * 2.5) % 2 == 0
        frame = to_img(np.asarray(frame, np.float32) * (1.0 if lamp_on else 0.35))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        oc = C(ov, 1)
        watch(oc, 540, 520, 170, 0, 0.0, stop=lt * 3600 * 4, strap=False)
        frame.paste(ov, (0, 0), ov)

        def txt(im):
            caption(im, t, wt("начинается", 39.7), wt("ему", 41.8) - 0.05, "ПОЛНАЯ ХЕРНЯ", y=1650, size=100, accent=(255, 90, 80))
            caption(im, t, wt("ему", 41.8), t_month, "«ПОСПАЛ ПАРУ ЧАСОВ»", "а прошло намного больше", y=1650, size=74)
        return frame, txt
    # месяц: утро или ночь? одинаковые дни-повторы
    img = ice_cave(9).copy()
    loop = (t - t_month) % 1.3                          # день повторяется как петля
    p = breathe(pose(lean=16, neck=24, nod=16, an=(40, 60), af=(36, 64), ln=(80, 100), lf=(70, 110)), loop * 3)
    lx, ly = caver(img, 540, 1600, 760, p, 1, light=(0, -1))
    img = headlamp_glow(img, lx, ly, 1.2, 800, 0.3, (130, 110, 80))
    c = C(img)
    breath(c, lx + 20, ly + 50, loop * 3)
    drips(c, loop * 3, 12, 5)
    frame = cam(img, lerp(1.0, 1.15, (t - t_month) / 6), 540, 1200, hh_amp=3)
    if loop < 0.08:                                    # «склейка» дня
        frame = to_img(np.asarray(frame, np.float32) * 0.3)

    def txt(im):
        caption(im, t, t_month + 0.2, t_same - 0.05, "УТРО ИЛИ НОЧЬ?", "через месяц", y=330, size=96)
        caption(im, t, t_same, 99, "КАЖДЫЙ ДЕНЬ ОДИНАКОВЫЙ", y=330, size=74)
    return frame, txt


# ================================================================ 7. звонок сверху
def sc_call(t, lt, dur):
    t_pick = wt("все", 53.5) - 0.4
    t_shock = wt("он", 55.8)
    img = ice_cave(4).copy()
    c = C(img)
    ring = lt if t < t_pick else 0.0
    field_phone(c, 540, 1250, 1.3, ring)
    img = glow(img, lambda d, k: d.ellipse([300 * k, 1000 * k, 780 * k, 1500 * k], fill=(120, 100, 60)), 70)
    frame = cam(img, lerp(1.0, 1.25, lt / dur) + (0.4 * eout((t - t_shock) / 0.3) if t > t_shock else 0), 540, 1200, hh_amp=4)

    def txt(im):
        caption(im, t, SHOT_START["call"] + 0.2, t_pick, "ЗВОНЯТ СВЕРХУ", y=330, size=96)
        caption(im, t, t_shock, 99, "ОН В АХУЕ", y=1700, size=110, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 8. даты: 20 августа vs 14 сентября
def calendar(c, x, y, month, day, col=(190, 40, 40), s=1.0):
    c.rect(x - 230 * s, y - 260 * s, x + 230 * s, y + 300 * s, (0, 0, 0, 120))
    c.rect(x - 220 * s, y - 250 * s, x + 220 * s, y + 290 * s, (238, 232, 220), r=10 * s)
    c.rect(x - 220 * s, y - 250 * s, x + 220 * s, y - 130 * s, col, r=10 * s)
    c.text(x, y - 190 * s, month, MS(int(40 * s), 800), (255, 255, 255))
    c.text(x, y + 60 * s, str(day), MS(int(200 * s), 800), (30, 30, 30))


def sc_dates(t, lt, dur):
    t_real = wt("а", 59.9) - 0.05
    img = to_img(grad((24, 26, 32), (8, 8, 10)))
    c = C(img)
    if t < t_real:
        calendar(c, 540, 900, "АВГУСТ 1962", 20, (60, 110, 180))
    else:
        k = clamp((t - t_real) / 1.6)
        day = 20 + int(k * 25)
        mon = "АВГУСТ 1962" if day <= 31 else "СЕНТЯБРЬ 1962"
        d_ = day if day <= 31 else day - 31
        calendar(c, 540, 900, mon, d_, (190, 40, 40) if day > 31 else (60, 110, 180))
    frame = cam(img, 1.0, 540, 960, hh_amp=2)

    def txt(im):
        caption(im, t, SHOT_START["dates"] + 0.1, t_real - 0.05, "ОН ДУМАЛ: 20 АВГУСТА", y=330, size=76)
        caption(im, t, t_real + 1.0, 99, "НА САМОМ ДЕЛЕ: 14 СЕНТЯБРЯ", y=330, size=66, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 9. потерянные 25 дней
def sc_lost(t, lt, dur):
    img = to_img(grad((10, 10, 14), (2, 2, 4)))
    c = C(img)
    for i in range(25):                                # листки-дни осыпаются в темноту
        k = clamp((lt - i * 0.12) / 2.2)
        x = 170 + (i % 5) * 185
        y = 520 + (i // 5) * 190 + k ** 2 * 1400
        rot = k * (i % 2 - 0.5) * 60
        a = int(255 * (1 - k))
        c.rect(x - 70, y - 70, x + 70, y + 70, (238, 232, 220, a), r=6)
        c.text(x, y + 6, str(21 + i if 21 + i <= 31 else 21 + i - 31), MS(56, 800), (40, 40, 40, a))
    frame = cam(img, 1.0, 540, 960)

    def txt(im):
        d = ImageDraw.Draw(im, "RGBA")
        kk = eout((t - wt("25", 65.5)) / 0.4)
        shadow_text(d, (540, 330), "−25 ДНЕЙ", MS(150, 800), (255, 90, 80, int(255 * kk)))
        caption(im, t, SHOT_START["lost"] + 0.2, 99, "МОЗГ БЕЗ СОЛНЦА", "просто потерял их", y=1700, size=84)
        caption(im, t, wt("целый", 67.4), 99, "МЕСЯЦ ЖИЗНИ ИСЧЕЗ", y=1500, size=80)
    return frame, txt


# ================================================================ 10. тизер: 1972, полгода
def sc_teaser(t, lt, dur):
    img = ice_cave(13, (150, 70, 60)).copy()
    c = C(img)
    p = breathe(pose(lean=6, an=(10, 20), af=(10, 20), nod=10), t, 0.3)
    lx, ly = caver(img, 540, 1650, 900, p, 1, rim=(255, 120, 100), view="front", light=(0, -1))
    img = headlamp_glow(img, lx, ly, math.pi / 2, 900, 0.25, (140, 50, 40), 1.0 + 0.3 * math.sin(t * 13))
    c = C(img)
    mx = 300 + 40 * math.sin(t * 2)                     # мышь у ног
    c.ell(mx, 1640, 26, 14, (20, 16, 16))
    c.ell(mx - 26, 1632, 10, 9, (20, 16, 16))
    c.line([(mx + 24, 1642), (mx + 70, 1636)], (20, 16, 16), 3)
    frame = cam(img, lerp(1.0, 1.25, lt / dur), 540, 1150, hh_amp=3)
    flick = 0.75 + 0.25 * (math.sin(t * 31) > -0.2)
    frame = to_img(np.asarray(frame, np.float32) * flick)

    def txt(im):
        year_tag(im, t, SHOT_START["teaser"] + 0.2, wt("и", 74.6) - 0.05, "1972", "ПОЛГОДА ПОД ЗЕМЛЁЙ", y=250)
        caption(im, t, wt("и", 74.6), 99, "ПО-НАСТОЯЩЕМУ СТРАШНО", y=300, size=74, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 11. призыв
def sc_cta(t, lt, dur):
    img = to_img(grad((6, 6, 8), (0, 0, 0)))
    dust(C(img), t, 30, seed=4, alpha=60)
    frame = cam(img, 1.0, 540, 960)

    def txt(im):
        k = eout((lt - 0.1) / 0.5)
        d = ImageDraw.Draw(im, "RGBA")
        shadow_text(d, (540, 850), "ЧАСТЬ 2?", MS(150, 800), (255, 90, 80, int(255 * k)))
        shadow_text(d, (540, 1010), "ПИШИ В КОММЕНТАРИЯХ 👇".replace(" 👇", ""), MS(64, 800), WHITE + (int(255 * k),))
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

SHAKES += [(1.25, 10, 0.4), (word_t("добровольно", 12.8), 8, 0.3), (word_t("херня", 40.5), 10, 0.4),
           (word_t("он", 55.8), 14, 0.4), (word_t("а", 59.9) + 1.0, 10, 0.4), (word_t("25", 65.5), 12, 0.4)]

WARM = {"hook": 0.2, "y1962": 0.4, "descent": -0.6, "rule": -0.5, "week1": -0.4, "confusion": -0.6, "call": -0.3,
        "dates": 0.2, "lost": -0.2, "teaser": 0.3, "cta": 0.0}


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
    img = bloom(img.convert("RGB"), 190, 0.45)
    img = post(img, t, WARM.get(name, 0.0), grain=7)
    if txt:
        txt(img)
    draw_quote(img, t)
    draw_subs(img, t, hide=[(v(a) - 0.1, v(b)) for a, b, _, _ in QUOTES])
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

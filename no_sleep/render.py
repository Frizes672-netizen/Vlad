"""Рендер шортса «Что будет, если не спать 11 дней» (реальная история Рэнди Гарднера, 1964).

Движок «Человека на всех фотографиях» (engine/props/hands/lib1) + макроплан глаза (eye.py).
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
from eye import eye
from hands import pov_hand
from lib1 import WHITE, caption, shadow_text, year_tag
from props import bloom, reflect, tree, volumetric
from timeline import DURATION, FPS, H, SCENES, W, v

lib1.pov_hand = pov_hand
YOUNG_SKIN = (210, 170, 144)
CYAN = (120, 220, 255)


def wt(word, after):
    return word_t(word, after)


def rot_parts(parts, cx, cy, ang):
    out = []
    for kind_, geo, part in parts:
        if kind_ == "poly":
            geo = rot_pts(geo, cx, cy, ang)
        elif kind_ == "ell":
            x, y = rot_pts([(geo[0], geo[1])], cx, cy, ang)[0]
            geo = (x, y, geo[2], geo[3])
        else:
            p0, p1, r0, r1 = geo
            geo = (rot_pts([p0], cx, cy, ang)[0], rot_pts([p1], cx, cy, ang)[0], r0, r1)
        out.append((kind_, geo, part))
    return out


def bw(img, k=1.0, warm=0.08):
    """Архивный чёрно-белый вид (1964)."""
    a = np.asarray(img, np.float32)
    g = a.mean(axis=2, keepdims=True)
    tone = g * np.array([1.0 + warm, 1.0, 1.0 - warm], np.float32)
    return to_img(a * (1 - k) + tone * k)


# ================================================================ HUD: день и часы без сна
DAYS = [("day1", 1, 24), ("day2", 2, 48), ("day3", 3, 72), ("day4", 4, 96), ("day5", 6, 144), ("day11", 11, 264),
        ("record", 11, 264)]


def hud(im, t, name):
    rec = [d for d in DAYS if d[0] == name]
    if not rec:
        return
    _, day, hours = rec[0]
    a = SHOT_START[name]
    k = eout((t - a) / 0.4)
    prev = {"day1": 0, "day2": 24, "day3": 48, "day4": 72, "day5": 96, "day11": 144, "record": 264}[name]
    hrs = lerp(prev, hours, eio((t - a) / 1.6))
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([60, 120, 1020, 270], 26, fill=(0, 0, 0, int(150 * k)))
    d.text((100, 160), f"ДЕНЬ {day}" if name != "day5" else "ДЕНЬ 5–6", font=MS(60, 800), fill=WHITE + (int(255 * k),), anchor="lm")
    d.text((990, 150), "БЕЗ СНА", font=MS(26, 700), fill=(200, 200, 200, int(255 * k)), anchor="rm")
    col = (255, 90, 80) if hours >= 96 else (255, 210, 120)
    d.text((990, 196), f"{int(hrs)} ч", font=MS(58, 800), fill=col + (int(255 * k),), anchor="rm")
    fk = hrs / 264
    d.rounded_rectangle([100, 230, 980, 246], 8, fill=(255, 255, 255, int(40 * k)))
    d.rounded_rectangle([100, 230, 100 + 880 * fk, 246], 8, fill=mix((255, 210, 120), (230, 40, 40), fk) + (int(230 * k),))


# ================================================================ 1. хук: глаз крупно
def big_timer(im, t, h, y=1500, col=WHITE, size=150, a=255):
    d = ImageDraw.Draw(im, "RGBA")
    s = int(h * 3600)
    txt = f"{s // 3600:03}:{s // 60 % 60:02}:{s % 60:02}"
    shadow_text(d, (540, y), txt, MS(size, 800), col + (a,))


def eye_frame(t, red, open_k, pupil, look=(0, 0), zoom=1.0):
    s = 1920
    e = eye(s, t, red, open_k, pupil, look)
    x0 = (s - int(W / zoom)) // 2
    y0 = (s - int(H / zoom)) // 2
    crop = e.crop((x0, max(0, y0), x0 + int(W / zoom), max(0, y0) + int(H / zoom)))
    return crop.resize((W, H), Image.BILINEAR)


def sc_hook(t, lt, dur):
    red = lerp(0.1, 0.95, clamp(lt / 3.0))
    blink = 1.0
    for tb, d_ in ((0.9, 0.5), (2.2, 0.35)):        # веко пытается закрыться и рывком открывается
        k = (lt - tb) / d_
        if 0 < k < 1:
            blink = min(blink, 1 - 0.8 * math.sin(math.pi * k) ** 0.7)
    pupil = 0.34 + 0.05 * math.sin(lt * 9)
    frame = eye_frame(t, red, blink, pupil, (0.15 * math.sin(lt * 5), 0), lerp(1.0, 1.25, lt / dur))
    a = np.asarray(frame, np.float32)
    if int(lt * 24) % 17 == 0:                       # вспышка-сбой
        a = a * 1.3
    frame = to_img(a)

    def txt(im):
        big_timer(im, t, min(264, (lt / 3.2) ** 2 * 264), y=1560, col=(255, 90, 80))
        caption(im, t, 0.05, 99, "11 ДНЕЙ БЕЗ СНА", "что будет с мозгом?", y=330, size=104)
    return frame, txt


# ================================================================ 2. 1964: школьник и конкурс
@lru_cache(None)
def school_photo():
    """Архивный снимок: школьник у стенда научного конкурса."""
    import engine
    ps = 2
    img = Image.new("RGB", (800 * ps, 600 * ps))
    d = ImageDraw.Draw(img, "RGBA")
    c = C(img, ps)
    a = grad((150, 150, 146), (90, 90, 88), 800 * ps, 600 * ps)
    img.paste(to_img(a))
    c.rect(-10, 430, 820, 610, (120, 116, 110))
    c.rect(80, 230, 720, 440, (180, 176, 168))            # стол со стендом
    c.rect(200, 60, 600, 300, (220, 216, 206))
    c.text(400, 110, "НАУЧНЫЙ КОНКУРС", MS(30, 800), (40, 40, 40))
    c.text(400, 160, "ДОЛГО ЛИ МОЖНО", MS(22, 700), (60, 60, 60))
    c.text(400, 190, "НЕ СПАТЬ?", MS(34, 800), (150, 30, 30))
    for k in range(4):
        c.rect(240, 220 + k * 16, 560 - k * 40, 228 + k * 16, (150, 146, 140))
    shaded(img, body_front(560, 590, 420, pose(an=(10, 20), af=(30, 80)), "young"), PALETTE["dad"], (560, 590), s=ps)
    return photo_post(img, 64, "sepia", None)


def calendar(c, x, y, n, tear):
    c.rect(x - 230, y - 260, x + 230, y + 300, (0, 0, 0, 120))
    c.rect(x - 220, y - 250, x + 220, y + 290, (238, 232, 220), r=10)
    c.rect(x - 220, y - 250, x + 220, y - 130, (190, 40, 40), r=10)
    c.text(x, y - 190, "ДЕКАБРЬ 1963" if n < 4 else "ЯНВАРЬ 1964", MS(36, 800), (255, 255, 255))
    c.text(x, y + 60, f"ДЕНЬ {n}", MS(110, 800), (30, 30, 30))
    if tear > 0:                                       # отрывной лист улетает
        dy = tear * 900
        c.poly([(x - 220, y - 130 - dy), (x + 220, y - 130 - dy * 1.2), (x + 220 + tear * 200, y + 290 - dy * 1.4),
                (x - 220 + tear * 100, y + 290 - dy)], (238, 232, 220, int(255 * (1 - tear))))


def sc_y1964(t, lt, dur):
    t_cal = wt("он", 12.4) - 0.2
    if t < t_cal:
        img = to_img(grad((20, 20, 20), (6, 6, 6)))
        k = eio(lt / 7.0)
        place_photo(img, framed(school_photo(), 18), 540, 980, 980, lerp(-3, 0, k))
        frame = cam(img, lerp(1.0, 1.35, k), lerp(540, 620, k), lerp(980, 900, k))
        frame = bw(frame, 1.0)

        def txt(im):
            year_tag(im, t, wt("1964", 3.8), wt("17", 6) - 0.1, "1964", "САН-ДИЕГО, США", y=260)
            caption(im, t, wt("17", 6), t_cal, "17 ЛЕТ", "школьник решил не спать", y=260, size=110)
        return frame, txt
    img = to_img(grad((26, 24, 24), (8, 8, 8)))
    c = C(img)
    lt2 = t - t_cal
    n = min(11, 1 + int(lt2 / 0.16))
    calendar(c, 540, 960, n, (lt2 / 0.16) % 1 if n < 11 else 0)
    frame = cam(img, 1.0, 540, 960, hh_amp=2)

    def txt(im):
        caption(im, t, t_cal + 0.1, 99, "11 ДНЕЙ", "он продержался", y=1650, size=120, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 3. мозг
@lru_cache(None)
def brain_img():
    s = 1400
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx, cy = s / 2, s / 2
    shape = ((xx - cx) / (s * 0.42)) ** 2 + ((yy - cy) / (s * 0.3)) ** 2
    shape += 0.25 * np.exp(-((xx - cx * 1.2) / (s * 0.2)) ** 2 - ((yy - cy * 1.45) / (s * 0.1)) ** 2) * -1
    m = np.clip((1 - shape) * 20, 0, 1)
    n = fbm(s, s, [6, 14, 40], 77)
    gyri = 0.5 + 0.5 * np.sin(n * 60)
    col = np.array([214, 160, 168], np.float32) * (0.55 + 0.45 * gyri[:, :, None])
    col = col * (0.6 + 0.5 * np.clip(1 - shape, 0, 1)[:, :, None])
    fiss = np.exp(-((xx - cx) / 6) ** 2) * (yy < cy * 1.1)
    col = col * (1 - 0.7 * fiss[:, :, None])
    rgba = np.dstack([np.clip(col, 0, 255), m * 255]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def sc_brain(t, lt, dur):
    img = to_img(grad((6, 10, 22), (2, 2, 6)))
    b = brain_img()
    k = eio(lt / dur)
    sz = int(lerp(1500, 1900, k))
    bi = b.resize((sz, sz))
    img.paste(bi, (int(WW / 2 - sz / 2), int(WH / 2 - sz / 2)), bi)
    c = C(img)
    r = np.random.default_rng(int(t * 12))
    for i in range(40):                              # нервные импульсы
        a = r.uniform(0, 2 * math.pi)
        d = r.uniform(0, 380)
        x, y = 540 + math.cos(a) * d, 960 + math.sin(a) * d * 0.7
        c.ell(x, y, 6, 6, CYAN + (200,))
    img = glow(img, lambda d, kk: d.ellipse([140 * kk, 660 * kk, 940 * kk, 1260 * kk], fill=(30, 70, 110)), 80)
    frame = cam(img, lerp(1.0, 1.15, k), 540, 960)

    def txt(im):
        caption(im, t, SHOT_START["brain"] + 0.2, 99, "ЧТО ПРОИСХОДИЛО С МОЗГОМ", y=1650, size=70)
    return frame, txt


# ================================================================ 4. Рэнди, друзья, учёный
def sc_randy(t, lt, dur):
    t_fr = wt("друзья", 19.8) - 0.15
    t_sci = wt("учёные", 23.5) - 0.2
    if t < t_fr:
        img = to_img(grad((18, 18, 18), (6, 6, 6)))
        st, _ = lib1.portrait("father")
        place_photo(img, framed(st, 12), 540, 900, 760, -2)
        frame = bw(cam(img, lerp(1.0, 1.12, lt / 2.3), 540, 900), 1.0)

        def txt(im):
            caption(im, t, wt("ренди", 17.5) - 0.1, 99, "РЭНДИ ГАРДНЕР", "ему 17", y=1560, size=96)
        return frame, txt
    img = to_img(grad((40, 34, 30), (14, 12, 10)))   # комната ночью, настольная лампа
    c = C(img)
    c.rect(0, 1560, W, H, (24, 20, 18))
    c.rect(660, 1100, 1000, 1130, (60, 44, 34))
    c.rect(680, 1130, 700, 1560, (40, 30, 24))
    c.rect(960, 1130, 980, 1560, (40, 30, 24))
    c.line([(900, 1100), (880, 960), (930, 920)], (30, 30, 30), 6)
    c.poly([(900, 900), (980, 900), (960, 950), (920, 950)], (40, 40, 40))
    img = volumetric(img, [(910, 950), (970, 950), (1080, 1200), (760, 1200)], (120, 90, 40), 40)
    sit = breathe(pose(lean=14, neck=18, nod=12, ln=(90, 92), lf=(86, 92), an=(50, 50), af=(46, 54)), t, 0.18)
    nudge = (lt * 2.2) % 3 < 0.6
    if nudge:
        sit["lean"] -= 8
        sit["nod"] -= 10
    silhouette(img, person_parts(560, 1560, 820, sit, 1, "young"), base=(10, 9, 9), rim=(255, 200, 130), light=(1, -0.6),
               rim_w=5, fill_light=(10, 6, 4))
    fp = breathe(pose(lean=10, an=(70 + (20 if nudge else 0), 20), af=(10, 20)), t, 0.2)
    silhouette(img, person_parts(330, 1560, 860, fp, 1, "suit"), base=(10, 9, 9), rim=(255, 200, 130), light=(1, -0.6), rim_w=4)
    if t > t_sci:                                    # учёный с блокнотом входит
        x = lerp(1300, 900, eio((t - t_sci) / 1.2))
        sp = walk((t - t_sci) * 5, 0.8) if t - t_sci < 1.2 else breathe(pose(an=(60, 80), af=(50, 90)), t)
        silhouette(img, person_parts(x, 1560, 880, sp, -1, "suit"), base=(12, 12, 14), rim=(200, 220, 255), light=(-1, -0.4), rim_w=4)
        J = skeleton(x, 1560, 880, sp, -1)
        C(img).rect(J["an_wr"][0] - 50, J["an_wr"][1] - 70, J["an_wr"][0] + 10, J["an_wr"][1] + 10, (220, 216, 200))
    frame = cam(img, lerp(1.1, 1.0, lt / 6), 620, 1150, hh_amp=2)

    def txt(im):
        caption(im, t, t_fr + 0.2, t_sci - 0.1, "НЕ ДАВАЛИ УСНУТЬ", "друзья дежурили по очереди", y=330, size=84)
        caption(im, t, t_sci + 0.2, 99, "УЧЁНЫЙ ИЗ СТЭНФОРДА", "изучать, что будет", y=330, size=74)
    return frame, txt


# ================================================================ 5. день 1: баскетбол
@lru_cache(None)
def court_bg():
    img = to_img(grad((150, 186, 214), (230, 226, 210)))
    c = C(img)
    c.rect(0, 1300, W, H, (176, 120, 80))
    c.poly([(0, 1300), (W, 1300), (W, 1340), (0, 1340)], (220, 220, 220))
    for x in (120, 960):
        tree(c, x, 1300, 1.4, (60, 80, 50))
    c.rect(840, 560, 856, 1300, (60, 60, 64))          # стойка и щит
    c.rect(720, 520, 900, 640, (240, 240, 240))
    c.rect(770, 560, 840, 610, None, outline=(200, 40, 40), w=4)
    c.ell(790, 650, 46, 10, None, outline=(220, 80, 40), w=5)
    return img


def sc_day1(t, lt, dur):
    img = court_bg().copy()
    ph = (lt % 2.4) / 2.4
    jump = math.sin(min(1, ph / 0.35) * math.pi) * 60 if ph < 0.35 else 0
    shoot = pose(an=(170, 10), af=(160, 14), ln=(10, 30), lf=(-6, 30)) if ph < 0.35 else pose(an=(30, 40), af=(20, 50))
    silhouette(img, person_parts(330, 1460 - jump, 700, shoot, 1, "young"), base=(30, 26, 24), rim=(255, 240, 210),
               light=(-0.5, -1), rim_w=3, fill_light=(30, 24, 18))
    k = clamp((ph - 0.2) / 0.5)                        # мяч по дуге в кольцо
    if 0 < k < 1:
        bx, by = lerp(380, 790, k), lerp(820, 640, k) - math.sin(k * math.pi) * 380
    else:
        bx, by = (380, 820) if k <= 0 else (790, lerp(660, 1260, clamp((ph - 0.7) / 0.3)))
    c = C(img)
    c.ell(bx, by, 34, 34, (210, 100, 40))
    c.arc(bx, by, 34, 34, 0, 360, (120, 50, 20), 3)
    c.line([(bx - 34, by), (bx + 34, by)], (120, 50, 20), 3)
    frame = cam(img, 1.05, 560, 1000)
    frame = to_img(np.asarray(bw(frame, 0.55, 0.1), np.float32))
    return frame, None


# ================================================================ 6. день 2: буквы расплываются
def sc_day2(t, lt, dur):
    img = to_img(grad((28, 24, 22), (10, 9, 8)))
    c = C(img)
    c.rect(170, 620, 910, 1260, (60, 44, 32), r=40)    # ламповый телевизор
    c.rect(220, 670, 760, 1130, (20, 20, 20), r=60)
    c.rect(240, 690, 740, 1110, (150, 160, 150), r=50)
    for k in range(8):
        c.rect(240 + k * 62, 690, 300 + k * 62, 860, [(230, 230, 230), (230, 230, 80), (80, 230, 230), (80, 230, 80),
                                                       (230, 80, 230), (230, 80, 80), (80, 80, 230), (20, 20, 20)][k])
    c.text(490, 970, "ПЕРЕДАЧА", MS(50, 800), (20, 20, 20))
    c.text(490, 1040, "НАЧНЁТСЯ В 21:00", MS(36, 700), (30, 30, 30))
    for k in range(3):
        c.ell(830, 760 + k * 110, 34, 34, (40, 30, 24))
    c.rect(300, 1260, 340, 1400, (40, 30, 24))
    c.rect(740, 1260, 780, 1400, (40, 30, 24))
    img = glow(img, lambda d, k: d.rectangle([240 * k, 690 * k, 740 * k, 1110 * k], fill=(80, 90, 80)), 60)
    frame = cam(img, lerp(1.0, 1.2, lt / dur), 500, 900)
    k = clamp((t - wt("глаза", 35)) / 3.5)            # двоение и расфокус
    if k > 0:
        off = int(30 * k + 14 * math.sin(t * 3) * k)
        a = np.asarray(frame, np.float32)
        b = np.roll(a, off, axis=1) * 0.5 + np.roll(a, -off // 2, axis=0) * 0.5
        frame = to_img(a * (1 - 0.5 * k) + b * 0.5 * k).filter(ImageFilter.GaussianBlur(8 * k))
    frame = bw(frame, 0.4)

    def txt(im):
        caption(im, t, wt("глаза", 35), wt("смотреть", 39) - 0.1, "ГЛАЗА НЕ ФОКУСИРУЮТСЯ", y=1650, size=72)
        caption(im, t, wt("смотреть", 39), 99, "БУКВЫ РАСПЛЫВАЮТСЯ", y=1650, size=76)
    return frame, txt


# ================================================================ 7. день 3: руки не слушаются
def sc_day3(t, lt, dur):
    t_hands = wt("руки", 45.3) - 0.2
    if t < t_hands:                                  # раздражительность: он сидит, сжимает голову
        img = to_img(grad((44, 30, 28), (14, 10, 10)))
        p = breathe(pose(lean=24, neck=30, nod=20, ln=(90, 92), lf=(86, 92), an=(150, 120), af=(146, 124)), t, 0.6, 1.8)
        silhouette(img, person_parts(520, 1560, 900, p, 1, "young"), base=(10, 8, 8), rim=(255, 120, 90), light=(1, -0.6), rim_w=5)
        a = np.asarray(cam(img, 1.15, 560, 1100, hh_amp=6), np.float32)
        pulse = 0.5 + 0.5 * math.sin(t * 8)
        a[:, :, 0] *= 1 + 0.15 * pulse

        def txt(im):
            caption(im, t, wt("раздражительным", 43.5) - 0.2, 99, "РАЗДРАЖИТЕЛЬНОСТЬ", y=1650, size=84, accent=(255, 90, 80))
        return to_img(a), txt
    img = to_img(grad((50, 44, 40), (20, 18, 16)))   # от первого лица: руки и шнурки
    c = C(img)
    c.ell(540, 1200, 300, 180, (40, 30, 26))          # ботинок сверху
    c.ell(540, 1150, 220, 120, (60, 44, 36))
    for k in range(5):
        c.line([(440, 1080 + k * 30), (640, 1090 + k * 30)], (230, 226, 216), 6)
    sh = 18 * math.sin(t * 31) + 12 * math.sin(t * 23)
    c.line([(500, 1080), (380 + sh, 900), (300, 860 + sh)], (230, 226, 216), 6)
    c.line([(580, 1080), (700 - sh, 900), (790, 870 - sh)], (230, 226, 216), 6)
    pov_hand(c, (-150, 1900), (300 + sh, 980 + sh * 0.5), -0.3, 0.6, 1.0, skin=YOUNG_SKIN, sleeve=(70, 80, 110))
    pov_hand(c, (1230, 1900), (780 - sh, 990 - sh * 0.4), 0.3, 0.6, 1.0, skin=YOUNG_SKIN, sleeve=(70, 80, 110))
    frame = cam(img, 1.0, 540, 1050, hh_amp=4)

    def txt(im):
        caption(im, t, t_hands + 0.2, 99, "РУКИ НЕ СЛУШАЮТСЯ", "простые движения — с трудом", y=330, size=84)
    return frame, txt


# ================================================================ 8. день 4: галлюцинации
@lru_cache(None)
def night_road():
    img = to_img(grad((8, 10, 20), (2, 2, 6)))
    c = C(img)
    c.rect(0, 1320, W, H, (14, 14, 16))
    for i in range(10):
        c.rect(i * 140 - 40, 1600 + i % 2 * 4, i * 140 + 40, 1612, (180, 170, 120))
    tree(c, 120, 1320, 1.6, (6, 6, 8))
    tree(c, 980, 1320, 1.3, (6, 6, 8))
    return img


def sc_day4(t, lt, dur):
    t_sign = wt("знак", 53) - 0.2
    t_foot = wt("футболист", 58.5) - 0.6
    if t < t_foot:
        img = night_road().copy()
        c = C(img)
        k = eio((t - t_sign) / 1.4) if t > t_sign else 0.0
        wob = math.sin(t * 6) * 6 * k
        # дорожный знак: столб и круг; превращается в силуэт человека
        c.rect(536, 760, 548, 1330, (60, 60, 64, int(255 * (1 - k))))
        c.ell(542, 700, 90, 90, (230, 230, 230, int(255 * (1 - k))))
        c.ell(542, 700, 74, 74, (200, 40, 40, int(255 * (1 - k))))
        c.rect(492, 690, 592, 712, (230, 230, 230, int(255 * (1 - k))))
        if k > 0:
            p = breathe(pose(an=(6 + wob, 10), af=(6, 10)), t)
            silhouette(img, body_front(542, 1330, 700 + 40 * math.sin(t * 2) * k, p, "suit"), base=(26, 8, 10),
                       rim=(255, 110, 100), light=(0, -1), rim_w=7, alpha=k, fill_light=(40, 6, 6))
        img = glow(img, lambda d, kk: d.ellipse([360 * kk, 520 * kk, 720 * kk, 880 * kk], fill=(90, 20, 20)), 60)
        frame = cam(img, lerp(1.0, 1.3, lt / 6), 542, 960, hh_amp=4)
        a = np.asarray(frame, np.float32)
        if k > 0:                                    # волны искажения
            rows = np.arange(H)
            sh = (np.sin(rows / 40 + t * 4) * 10 * k).astype(int)
            for y in range(0, H, 4):
                a[y:y + 4] = np.roll(a[y:y + 4], sh[y], axis=1)
        frame = to_img(a)

        def txt(im):
            caption(im, t, wt("начинаются", 50.5) - 0.1, t_sign, "ГАЛЛЮЦИНАЦИИ", y=1650, size=96, accent=(255, 90, 80))
            caption(im, t, t_sign + 0.3, t_foot, "ЗНАК — ЭТО ЧЕЛОВЕК", y=1650, size=84)
        return frame, txt
    img = to_img(grad((10, 14, 26), (20, 30, 20)))   # стадион в его голове
    c = C(img)
    c.rect(0, 1200, W, H, (30, 80, 40))
    for x in (100, 980):
        c.rect(x - 8, 300, x + 8, 1200, (20, 20, 24))
        for k in range(6):
            c.ell(x - 50 + (k % 3) * 50, 300 + (k // 3) * 46, 20, 20, (255, 250, 230))
    img = glow(img, lambda d, kk: [d.ellipse([(x - 160) * kk, 200 * kk, (x + 160) * kk, 450 * kk], fill=(200, 200, 180))
                                   for x in (100, 980)], 60)
    r = np.random.default_rng(int(t * 8))
    c = C(img)
    for i in range(120):                              # вспышки фотоаппаратов на трибунах
        if r.random() < 0.3:
            c.ell(r.uniform(0, W), r.uniform(700, 1150), 3, 3, (255, 255, 255, 200))
    hero = pose(an=(170, 10), af=(165, 14))
    silhouette(img, person_parts(540, 1650, 900, breathe(hero, t), 1, "young", "front"), base=(10, 10, 12),
               rim=(255, 250, 230), light=(0, -1), rim_w=6)
    c = C(img)
    c.ell(540, 1650 - 900 * 0.86, 80, 74, (40, 40, 60))                     # шлем
    c.rect(470, 1650 - 900 * 0.83, 610, 1650 - 900 * 0.8, (200, 200, 200))
    frame = cam(img, lerp(1.2, 1.0, (t - t_foot) / 1.5), 540, 1100)

    def txt(im):
        caption(im, t, t_foot + 0.4, 99, "Я — ЗНАМЕНИТЫЙ ФУТБОЛИСТ", "он поверил всерьёз", y=330, size=70)
    return frame, txt


# ================================================================ 9. дни 5–6: путает слова, держат в движении
def sc_day5(t, lt, dur):
    t_walk = wt("чтобы", 65.5) - 0.2
    if t < t_walk:
        img = to_img(grad((20, 20, 26), (6, 6, 8)))
        frame = cam(img, 1.0, 540, 960)
        r = np.random.default_rng(int(t * 6))
        words = ["Я", "ХОЧУ", "СПАТЬ"]
        mixk = clamp((t - wt("путает", 61.8)) / 2.5)

        def txt(im):
            d = ImageDraw.Draw(im, "RGBA")
            for i, w_ in enumerate(words):
                chars = list(w_)
                if mixk > 0:
                    for _ in range(int(len(chars) * mixk)):
                        a, b = r.integers(0, len(chars), 2)
                        chars[a], chars[b] = chars[b], chars[a]
                s = "".join(chars)
                dx = r.uniform(-30, 30) * mixk
                dy = r.uniform(-20, 20) * mixk
                shadow_text(d, (540 + dx, 760 + i * 170 + dy), s, MS(130, 800), WHITE + (int(255 * (1 - 0.4 * mixk)),))
            caption(im, t, wt("путает", 61.8), t_walk, "ПУТАЕТ СЛОВА", "и забывает, что сказал", y=1500, size=84)
        return frame, txt
    img = to_img(grad((24, 26, 34), (8, 8, 10)))     # ночью водят по улице, чтобы не уснул
    c = C(img)
    c.rect(0, 1400, W, H, (16, 16, 18))
    lt2 = t - t_walk
    x = lerp(150, 760, lt2 / 3.5)
    p1 = walk(lt2 * 4.2, 0.6)
    p1["lean"] += 10
    p1["nod"] = 20
    silhouette(img, person_parts(x, 1400, 640, p1, 1, "young"), base=(8, 8, 10), rim=(200, 210, 255), light=(-1, -0.5), rim_w=4)
    p2 = walk(lt2 * 4.2 + 1.5, 0.7)
    p2["an"] = (40, 50)
    silhouette(img, person_parts(x - 90, 1400, 680, p2, 1, "suit"), base=(8, 8, 10), rim=(200, 210, 255), light=(-1, -0.5), rim_w=4)
    reflect(img, 1404, 400, 0.4, 4)
    frame = cam(img, 1.05, 520, 1100)

    def txt(im):
        caption(im, t, t_walk + 0.2, 99, "ЕГО ДЕРЖАЛИ В ДВИЖЕНИИ", "чтобы не уснул", y=330, size=74)
    return frame, txt


# ================================================================ 10. день 11: лицо без эмоций, счёт
def blank_face(img, t):
    c = C(img)
    cx, cy, r = 540, 860, 330
    c.ell(cx, cy + 60, r * 0.9, r * 1.15, (176, 150, 136))
    c.ell(cx, cy - r * 0.55, r * 0.95, r * 0.6, (40, 30, 24))                   # волосы
    for s_ in (-1, 1):
        c.ell(cx + s_ * r * 0.88, cy + 60, r * 0.14, r * 0.24, (160, 132, 118))
        ex, ey = cx + s_ * r * 0.36, cy + 30
        c.ell(ex, ey + r * 0.12, r * 0.24, r * 0.12, (120, 90, 96, 160))         # синяки под глазами
        c.ell(ex, ey, r * 0.2, r * 0.085, (214, 190, 184))                       # белок с красноватым
        c.ell(ex, ey, r * 0.085, r * 0.085, (60, 70, 80))
        c.ell(ex, ey, r * 0.04, r * 0.04, (10, 10, 12))
        c.ell(ex + r * 0.03, ey - r * 0.03, r * 0.015, r * 0.015, (255, 255, 255))
        c.line([(ex - r * 0.2, ey - r * 0.08), (ex + r * 0.2, ey - r * 0.06)], (120, 90, 80), r * 0.03)   # веко тяжёлое
        c.line([(ex - r * 0.22, ey - r * 0.2), (ex + r * 0.18, ey - r * 0.2)], (60, 44, 36), r * 0.05)
    c.line([(cx, cy + 40), (cx + 20, cy + 180), (cx - 20, cy + 190)], (140, 110, 96), 5)
    c.line([(cx - 80, cy + 280), (cx + 80, cy + 280)], (130, 96, 90), 6)
    c.rect(cx - 140, cy + 380, cx + 140, cy + 700, (150, 124, 110))
    c.poly([(cx - 520, 1920), (cx - 380, cy + 560), (cx + 380, cy + 560), (cx + 520, 1920)], (60, 70, 90))


def sc_day11(t, lt, dur):
    t_count = wt("его", 74.6) - 0.1
    # пустой взгляд: глаз красный, веко тяжёлое, зрачок плывёт и не фокусируется
    drift = (0.25 * math.sin(t * 0.4), 0.15 * math.sin(t * 0.33))
    frame = eye_frame(t, 1.0, 0.5 + 0.05 * math.sin(t * 0.5), 0.3, drift, lerp(1.0, 1.2, lt / dur))
    frame = to_img(np.asarray(frame.filter(ImageFilter.GaussianBlur(1.2)), np.float32) * np.array([0.9, 0.86, 0.9]))
    nums = [100, 93, 86, 79, 72, 65]
    t65 = wt("шестидесяти", 78)

    def txt(im):
        caption(im, t, SHOT_START["day11"] + 0.4, wt("речь", 73) - 0.05, "ЛИЦО БЕЗ ЭМОЦИЙ", y=1650, size=84)
        caption(im, t, wt("речь", 73), t_count, "РЕЧЬ МЕДЛЕННАЯ", y=1650, size=84)
        if t > t_count:
            d = ImageDraw.Draw(im, "RGBA")
            span = max(0.5, t65 - t_count)
            n = min(len(nums), 1 + int((t - t_count) / span * (len(nums) - 1)))
            fade = 1 - clamp((t - wt("забыл", 81.3)) / 1.5)
            for i in range(n):
                k = eout((t - t_count - i * span / (len(nums) - 1)) / 0.3)
                col = (255, 90, 80) if nums[i] == 65 else WHITE
                shadow_text(d, (200 + (i % 3) * 340, 1400 + (i // 3) * 170), str(nums[i]), MS(120, 800),
                            col + (int(255 * k * fade),))
            if t > wt("останавливается", 79.5):
                k = clamp((t - wt("останавливается", 79.5)) / 0.4)
                shadow_text(d, (880, 1570), "…?", MS(130, 800), (255, 90, 80, int(255 * k * fade)))
            caption(im, t, t_count + 0.1, wt("он", 77.2), "100 − 7 − 7 − 7…", "считать в уме", y=320, size=84)
            caption(im, t, wt("забыл", 81.3), 99, "ЗАБЫЛ, ЧТО СЧИТАЛ", y=320, size=84, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 11. рекорд и сон
def sc_record(t, lt, dur):
    t_sleep = wt("после", 88.2) - 0.15
    if t < t_sleep:
        img = to_img(grad((20, 6, 6), (4, 2, 2)))
        frame = cam(img, 1.0, 540, 960)
        stop = (t - wt("264", 85)) > 0.6

        def txt(im):
            hrs = 264 if stop else lerp(250, 264, clamp((t - SHOT_START["record"]) / 1.6))
            blink = 255 if not stop or int(t * 3) % 2 else 120
            big_timer(im, t, hrs, y=960, col=(255, 80, 70), size=170, a=blink)
            caption(im, t, SHOT_START["record"] + 0.2, 99, "РЕКОРД", "без сна", y=700, size=110)
        return frame, txt
    lt2 = t - t_sleep                                # он падает на кровать, за окном светает
    dawn = clamp(lt2 / 5.0)
    img = to_img(grad(mix((14, 16, 30), (180, 160, 140), dawn), mix((6, 6, 10), (60, 50, 44), dawn)))
    c = C(img)
    c.rect(620, 300, 960, 760, mix((20, 30, 60), (230, 200, 150), dawn))
    c.rect(620, 300, 960, 760, None, outline=(40, 36, 34), w=14)
    c.rect(784, 300, 796, 760, (40, 36, 34))
    c.rect(80, 1250, 1000, 1400, (90, 80, 76), r=20)                # кровать
    c.rect(80, 1400, 110, 1560, (50, 40, 34))
    c.rect(970, 1400, 1000, 1560, (50, 40, 34))
    c.rect(0, 1560, W, H, (30, 26, 24))
    fall = eio(lt2 / 0.9)
    parts = rot_parts(person_parts(540, 1250, 860, pose(lean=4, an=(10, 20), af=(10, 20)), 1, "young"), 540, 1250, -82 * fall)
    silhouette(img, parts, base=(12, 10, 10), rim=mix((180, 190, 255), (255, 210, 150), dawn), light=(1, -1), rim_w=4)
    c = C(img)
    c.ell(200, 520, 90, 90, (40, 36, 34))                           # часы крутятся
    c.ell(200, 520, 80, 80, (220, 214, 200))
    a = math.radians(lt2 * 360 * 1.2)
    c.line([(200, 520), (200 + math.cos(a) * 64, 520 + math.sin(a) * 64)], (30, 30, 30), 4)
    c.line([(200, 520), (200 + math.cos(a / 12) * 44, 520 + math.sin(a / 12) * 44)], (30, 30, 30), 7)
    img = volumetric(img, [(620, 300), (960, 300), (1080, 1300), (300, 1300)], (int(90 * dawn), int(70 * dawn), int(40 * dawn)), 60)
    frame = cam(img, 1.05, 540, 1000)

    def txt(im):
        caption(im, t, t_sleep + 0.4, wt("и", 91.9) - 0.1, "ПОЧТИ 15 ЧАСОВ СНА", y=1720, size=80)
        caption(im, t, wt("и", 91.9), 99, "БУДТО НИЧЕГО НЕ БЫЛО", y=1720, size=80)
    return frame, txt


# ================================================================ 12. годы спустя: бессонница
def sc_later(t, lt, dur):
    img = to_img(grad((10, 12, 24), (4, 4, 8)))
    c = C(img)
    c.rect(140, 300, 940, 1700, (40, 44, 60), r=30)                # кровать сверху
    c.rect(220, 360, 860, 560, (200, 204, 214), r=40)               # подушка
    c.rect(160, 900, 920, 1680, (60, 70, 100), r=30)               # одеяло
    c.ell(540, 520, 130, 160, (110, 100, 110))                      # лицо в лунном свете смотрит в потолок
    c.ell(540, 420, 140, 90, (110, 110, 110))                       # седеющие волосы
    for s_ in (-1, 1):
        c.ell(540 + s_ * 50, 510, 22, 12, (230, 226, 220))
        c.ell(540 + s_ * 50, 510, 9, 9, (20, 20, 24))
    c.rect(780, 220, 960, 300, (10, 10, 12), r=10)                  # электронные часы
    c.text(870, 260, "03:17", MS(48, 800), (255, 60, 50))
    img = glow(img, lambda d, k: d.rectangle([780 * k, 220 * k, 960 * k, 300 * k], fill=(120, 20, 20)), 40)
    frame = cam(img, lerp(1.0, 1.15, lt / dur), 540, 700)

    def txt(im):
        caption(im, t, SHOT_START["later"] + 0.3, 99, "БЕССОННИЦА", "спустя годы", y=1600, size=110, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 13. Гиннесс
def sc_guinness(t, lt, dur):
    img = lib1.table_bg().copy()
    c = C(img)
    k = eio(lt / 1.6)
    w = lerp(760, 380, k)                                           # книга захлопывается
    c.rect(540 - 390, 700 + 20, 540 + 390, 1300 + 30, (0, 0, 0, 120))
    c.rect(540 - 10, 700, 540 + 380, 1300, (220, 200, 140))
    c.rect(540 - 380, 700, 540 + 10, 1300, (30, 60, 120), r=10)
    c.poly([(540, 700), (540 - w, 700 + 40 * (1 - k)), (540 - w, 1300 - 40 * (1 - k)), (540, 1300)],
           mix((236, 230, 214), (30, 60, 120), k))
    if k > 0.9:
        c.text(540 - 180, 900, "КНИГА", MS(48, 800), (240, 210, 120))
        c.text(540 - 180, 980, "РЕКОРДОВ", MS(48, 800), (240, 210, 120))
    t_st = wt("слишком", 103.8)
    sk = clamp((t - t_st) / 0.3)
    if sk > 0:
        s = lerp(1.8, 1.0, eout(sk))
        c.rect(540 - 300 * s, 1100 - 70 * s, 540 + 300 * s, 1100 + 70 * s, None, outline=(200, 30, 30, 230), w=10 * s)
        c.text(540, 1100, "СЛИШКОМ ОПАСНО", MS(int(50 * s), 800), (200, 30, 30, 230))
    frame = cam(img, lerp(1.1, 1.0, k), 540, 1000)

    def txt(im):
        caption(im, t, SHOT_START["guinness"] + 0.3, t_st - 0.05, "БОЛЬШЕ НЕ РЕГИСТРИРУЮТ", "Книга рекордов Гиннесса", y=330, size=72)
    return frame, txt


# ================================================================ 14. вопрос
def sc_question(t, lt, dur):
    open_k = lerp(1.0, 0.05, eio(lt / 2.6))
    frame = eye_frame(t, 0.5, open_k, 0.36, (0, 0), 1.15)
    frame = to_img(np.asarray(frame, np.float32) * lerp(1, 0.45, clamp(lt / 2)))

    def txt(im):
        k = eout((lt - 0.3) / 0.6)
        d = ImageDraw.Draw(im, "RGBA")
        shadow_text(d, (540, 330), "А ВЫ СКОЛЬКО", MS(92, 800), WHITE + (int(255 * k),))
        shadow_text(d, (540, 440), "ПРОДЕРЖИТЕСЬ?", MS(100, 800), (255, 90, 80, int(255 * k)))
        big_timer(im, t, lt / 3600 * 40, y=1600, col=WHITE, size=130, a=int(255 * k))
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

SHAKES += [(word_t("он", 12.4), 6, 0.3), (word_t("начинаются", 50.5), 10, 0.4), (word_t("знак", 53), 8, 0.5),
           (word_t("264", 85) + 0.6, 14, 0.4), (word_t("слишком", 103.8), 10, 0.3)]

WARM = {"hook": 0.0, "y1964": 0.4, "brain": -0.3, "randy": 0.4, "day1": 0.4, "day2": 0.2, "day3": 0.1, "day4": -0.5,
        "day5": -0.3, "day11": -0.2, "record": 0.0, "later": -0.5, "guinness": 0.5, "question": 0.0}


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
    hud(img, t, name)
    if txt:
        txt(img)
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

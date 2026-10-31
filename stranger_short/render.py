"""Рендер шортса «Человек на всех фотографиях».

Кинематографичный стиль: старые плёночные снимки, ночные интерьеры, силуэты людей
с контровым светом (как в «24 часа президентом»), живая пластика через позы с суставами.
Запуск: python3 render.py build/video.mp4 [--preview СЕКУНДЫ ...]
"""
import math
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine import (resize_font, CREAM, CUR_T, MS, PALETTE, PF, PFI, RED, REVERSE, S, SHAKES, WH, WW, C, back, blend,
                    body_front, breathe, cam, clamp, dust, ease, eio, env, eout, extra_photo, fbm, framed, glow,
                    grad, ground_shadow, keyed, lerp, mix, person_parts, photo, photo_rev, place_photo, pose, post,
                    radial, red_circle, shaded, silhouette, skeleton, squash, stranger_card, to_img, walk, word_t)
from timeline import DURATION, FPS, H, QUOTES, SCENES, VOICE_OFFSET, W, v
import props
from props import (abazhur, bloom, courtyard, curtains, hallway, night_street, peaked_cap, police_car,
                   rain, rain_in_cone, reflect, table_set, tree, wardrobe)

WHITE = (246, 244, 238)
NIGHT_RIM = (150, 170, 220)
LAMP_RIM = (255, 196, 130)


# ================================================================ подписи
def shadow_text(d, xy, txt, f, fill, anchor="mm", sh=6, sa=170):
    x, y = xy
    a = fill[3] if len(fill) == 4 else 255
    d.text((x + sh * 0.4, y + sh), txt, font=f, fill=(0, 0, 0, int(sa * a / 255)), anchor=anchor)
    d.text((x, y), txt, font=f, fill=fill, anchor=anchor)


def fit(d, txt, f, maxw):
    while d.textlength(txt, font=f) > maxw and f.size > 20:
        f = resize_font(f, f.size - 4)
    return f


def caption(im, t, a, b, big, small=None, y=1560, col=WHITE, size=84, accent=None):
    """Крупная подпись: маленькая строка курсивом + жирная строка капсом. a, b — время ролика."""
    k = env(t, a, b, 0.35, 0.3)
    if k <= 0:
        return
    d = ImageDraw.Draw(im, "RGBA")
    dy = (1 - eout((t - a) / 0.45)) * 30
    if small:
        shadow_text(d, (540, y - 92 + dy), small, PFI(50, 500), CREAM + (int(255 * k),))
    f = fit(d, big, MS(size, 800), 980)
    shadow_text(d, (540, y + dy), big, f, (accent or col) + (int(255 * k),))


def year_tag(im, t, a, b, year, label, y=300):
    k = env(t, a, b, 0.3, 0.25)
    if k <= 0:
        return
    d = ImageDraw.Draw(im, "RGBA")
    dx = (1 - eout((t - a) / 0.4)) * -40
    shadow_text(d, (540 + dx, y), year, MS(120, 800), WHITE + (int(255 * k),))
    w = d.textlength(label, font=MS(34, 700))
    d.rectangle([540 - w / 2 - 22, y + 78, 540 + w / 2 + 22, y + 132], fill=RED + (int(230 * k),))
    d.text((540, y + 105), label, font=MS(34, 700), fill=WHITE + (int(255 * k),), anchor="mm")


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


QUOTE_T = [(v(a), v(b), who, txt) for a, b, who, txt in QUOTES]


def draw_quote(im, t):
    for a, b, who, text in QUOTE_T:
        if a - 0.1 <= t < b:
            k = min(eout((t - a + 0.1) / 0.4), ease((b - t) / 0.3))
            d = ImageDraw.Draw(im, "RGBA")
            f = PFI(58, 500)
            lines = wrap(d, text, f, 860)
            y0 = 1450 - len(lines) * 36
            d.rectangle([0, y0 - 140, W, y0 + len(lines) * 74 + 70], fill=(4, 5, 8, int(160 * k)))
            d.text((100, y0 - 66), "“", font=PF(150, 800), fill=RED + (int(255 * k),), anchor="lm")
            d.text((186, y0 - 72), who, font=MS(30, 700), fill=RED + (int(255 * k),), anchor="lm")
            for i, ln in enumerate(lines):
                d.text((100, y0 + i * 74), ln, font=f, fill=CREAM + (int(255 * k),), anchor="lm")
            return


# ================================================================ общие фоны
@lru_cache(None)
def table_bg():
    """Деревянный стол сверху под тёплой лампой."""
    n = fbm(WW, WH, [3, 8], 31)
    xs = np.linspace(0, 1, WW)[None, :]
    grain = np.sin((xs * 60 + n * 9) * math.pi) * 0.5 + 0.5
    base = np.array([74, 46, 30], np.float32)
    a = base * (0.7 + 0.25 * grain[:, :, None] + 0.15 * n[:, :, None])
    a = a * (0.35 + 0.95 * radial(540, 900, 1250, p=1.4)[:, :, None])
    for k in range(1, 5):          # стыки досок
        x = int(k * WW / 5)
        a[:, x - 3:x + 3] *= 0.55
    return to_img(a)


def wt(word, after):
    return word_t(word, after)


# ================================================================ 1. хук: снимки на столе
HOOK_LAYOUT = [("first_class", 300, 520, 560, 7), ("wedding", 800, 470, 520, -6), ("sea", 560, 960, 640, 3),
               ("x1", 230, 1340, 500, -9), ("x3", 830, 1300, 520, 8), ("x5", 330, 1760, 520, 5),
               ("x9", 820, 1750, 520, -4), ("x0", 920, 900, 420, 12), ("x7", 120, 930, 420, -12)]


def get_photo(key, scale=2):
    if key.startswith("x"):
        img, mh, gh = extra_photo(int(key[1:]))
    else:
        img, mh, gh = photo(key, scale)
    return img, mh, gh


@lru_cache(None)
def table_with(layout):
    world = table_bg().copy()
    marks, gms = [], []
    for key, x, y, w, ang in layout:
        img, mh, gh = get_photo(key)
        to_w = place_photo(world, framed(img), x, y, w, ang)
        marks.append(to_w(mh[0], mh[1], 26))
        gms.append(to_w(gh[0], gh[1], 26))
    return world, marks, gms


def sc_hook(t, lt, dur):
    world, marks, _ = table_with(tuple(HOOK_LAYOUT))
    img = world.copy()
    c = C(img)
    t_one = wt("один", 3)
    for i, (x, y) in enumerate(marks):
        red_circle(c, x, y, 42, (t - t_one - i * 0.11) / 0.32, 6, seed=i)
    t_no = wt("никто", 5)
    if t < t_no:
        z = lerp(1.18, 1.0, eio(lt / 3.4))
        frame = cam(img, z, 540, 960, rot=lerp(2.0, 0.0, eio(lt / 3.4)))
    else:
        k = eio((t - t_no) / 1.3)
        mx, my = marks[2]
        frame = cam(img, lerp(1.0, 3.2, k), lerp(540, mx, k), lerp(960, my, k))

    def txt(im):
        year_tag(im, t, wt("40", 2) - 0.1, t_one - 0.05, "40 ЛЕТ", "ВСЕ СЕМЕЙНЫЕ ФОТО", y=260)
        caption(im, t, t_one, t_no - 0.1, "ОДИН И ТОТ ЖЕ ЧЕЛОВЕК", y=1600, size=78)
        caption(im, t, t_no, 99, "НИКТО ЕГО НЕ ЗНАЕТ", "и никто из нас…", y=1600, size=88, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 2. сканер и альбом
@lru_cache(None)
def desk_room():
    img = to_img(grad((18, 20, 28), (6, 6, 9)))
    c = C(img)
    c.rect(0, 1420, W, H, (16, 13, 12))
    c.rect(0, 1400, W, 1430, (30, 24, 20))
    c.rect(150, 560, 930, 1080, (10, 10, 12), r=14)        # монитор
    c.rect(500, 1080, 580, 1300, (12, 12, 14))
    c.rect(400, 1290, 680, 1320, (14, 14, 16), r=8)
    c.rect(660, 1330, 1040, 1420, (38, 40, 44), r=8)       # сканер
    c.rect(660, 1318, 1040, 1338, (52, 54, 60), r=6)
    c.ell(1000, 1370, 6, 6, (90, 255, 140))
    c.rect(300, 1340, 640, 1380, (30, 30, 34), r=6)        # клавиатура
    for i in range(12):
        for j in range(3):
            c.rect(312 + i * 27, 1346 + j * 11, 334 + i * 27, 1354 + j * 11, (44, 44, 50))
    for k in range(4):                                     # стопка альбомов
        c.rect(40 + k * 4, 1260 - k * 36, 250 - k * 4, 1296 - k * 36, [(90, 30, 30), (40, 60, 90), (70, 50, 30), (40, 70, 50)][k], r=4)
    c.rect(700, 1250, 760, 1318, (200, 196, 186), r=6)    # кружка
    c.arc(762, 1282, 18, 20, 270, 90, (200, 196, 186), 6)
    for i in range(6):                                     # полки с книгами
        c.rect(960 + (i % 2) * 40, 300 + i * 34, 1000 + (i % 2) * 40, 330 + i * 34, (60 + i * 8, 40, 36))
    return img


def sc_scanner(t, lt, dur):
    t_alb = wt("старые", 12) - 0.5
    if t < t_alb:
        img = desk_room().copy()
        # экран: снимок проявляется полосой сканирования
        im, _, _ = get_photo("x1")
        scr = framed(im).resize((int(720 * S), int(540 * S)))
        k = clamp((lt - 0.4) / 3.0)
        img.paste(scr.crop((0, 0, scr.width, max(1, int(scr.height * k)))), (int(180 * S), int(570 * S)))
        c = C(img)
        y = 570 + 540 * k
        c.rect(180, y - 2, 900, y + 3, (160, 255, 200))
        c.rect(180, 1030, 900, 1060, (24, 26, 30))
        c.rect(184, 1036, 184 + 712 * k, 1054, (80, 160, 120))
        c.text(540, 1045, f"СКАНИРОВАНИЕ… {int(k * 100)}%", MS(22, 700), (230, 236, 230))
        sk = (lt % 2.4) / 2.4
        img = glow(img, lambda d, kk: [d.rectangle([150 * kk, 560 * kk, 930 * kk, 1080 * kk], fill=(40, 60, 70)),
                                       d.rectangle([(670 + 360 * sk) * kk, 1322 * kk, (680 + 360 * sk) * kk, 1336 * kk],
                                                   fill=(120, 255, 170))], 60)
        # рассказчик со спины на переднем плане
        p = breathe(pose(nod=4 * math.sin(lt * 0.8), an=(14, 30), af=(14, 30)), t)
        silhouette(img, person_parts(300, 2500, 1500, p, 1, "young", "back"), base=(6, 6, 8), rim=(120, 170, 160),
                   light=(0.3, -1), rim_w=4)
        frame = cam(img, lerp(1.0, 1.1, lt / 4), 560, 980, hh_amp=2)
    else:
        frame = album(t - t_alb)

    def txt(im):
        caption(im, t, wt("бабушка", 9) + 0.05, t_alb - 0.1, "БАБУШКЕ — 80", "я оцифровывал старые альбомы", y=330)
    return frame, txt


def album(lt):
    img = table_bg().copy()
    c = C(img)
    # открытый альбом сверху
    c.rect(90, 420, 990, 1500, (30, 22, 18), r=10)
    c.rect(110, 440, 530, 1480, (36, 34, 32))
    c.rect(550, 440, 970, 1480, (36, 34, 32))
    for i, key in enumerate(["x0", "x2", "x4", "x6"]):
        im, _, _ = get_photo(key)
        place_photo(img, framed(im), 320 if i < 2 else 760, 690 + (i % 2) * 520, 360, 0, shadow=False)
    # перелистывание: правая страница поднимается и ложится налево
    k = eio(lt / 1.1)
    if k < 1:
        sx = math.cos(k * math.pi)
        x0 = 540
        w = 430 * sx
        lift = math.sin(k * math.pi) * 60
        shade = int(28 + 30 * abs(sx))
        pts = [(x0, 440), (x0 + w, 440 - lift), (x0 + w, 1480 + lift), (x0, 1480)]
        c.poly([(x + 12, y + 18) for x, y in pts], (0, 0, 0, 90))
        c.poly(pts, (shade, shade - 2, shade - 4))
        if abs(sx) > 0.15:
            im, _, _ = get_photo("x8" if sx > 0 else "x10")
            cx = x0 + w / 2
            sm = framed(im).resize((max(2, int(abs(w) * 0.8 * S)), int(260 * S)))
            img.paste(sm, (int((cx - abs(w) * 0.4) * S), int((690 - 130) * S)))
    return cam(img, lerp(1.15, 1.05, lt / 2), 540, 960)


# ================================================================ 3–5. три снимка крупно
MAIN3 = {"photo1": ("first_class", "1990", "ПАПА · ПЕРВЫЙ КЛАСС", "заднем", 16, "мужчина", 17),
         "photo2": ("wedding", "2005", "СВАДЬБА РОДИТЕЛЕЙ", "тот", 20.5, "ограды", 21.5),
         "photo3": ("sea", "2011", "МОРЕ · МНЕ 5 ЛЕТ", "сидит", 24.5, "пирсе", 25)}


@lru_cache(None)
def photo_world(key):
    world = to_img(grad((16, 16, 18), (6, 6, 8)))
    img, mh, gh = photo(key, 4)
    to_w = place_photo(world, framed(img, 18), 540, 980, 1040, 0)
    return world, to_w(mh[0], mh[1], 18), to_w(gh[0], gh[1], 18)


def photo_frame(name, t, lt):
    key, year, label, zw, za, cw, ca = MAIN3[name]
    world, (mx, my), _ = photo_world(key)
    img = world.copy()
    t_z, t_c = wt(zw, za), wt(cw, ca)
    if name == "photo1" and lt < 1.0:      # результат сканирования проявляется сверху вниз
        k = eio(lt / 0.9)
        y = lerp(560, 1400, k)
        c = C(img)
        c.rect(0, y, W, H, (10, 10, 12))
        img = glow(img, lambda d, kk: d.rectangle([0, (y - 6) * kk, W * kk, (y + 6) * kk], fill=(160, 230, 190)), 16, 1.6)
    c = C(img)
    red_circle(c, mx, my + 20, 54, (t - t_c) / 0.4, 6, seed=3)
    k = eio((t - t_z) / 1.4)
    return cam(img, lerp(1.04, 2.9, k) + 0.02 * lt, lerp(540, mx, k), lerp(980, my + 30, k)), (year, label, t_c)


def sc_photo(t, lt, dur, name):
    frame, (year, label, t_c) = photo_frame(name, t, lt)
    if name != "photo1" and lt < 0.35:     # смена снимка: прошлый уезжает влево
        prev = {"photo2": "photo1", "photo3": "photo2"}[name]
        pf, _ = photo_frame(prev, t, 99)
        k = eio(lt / 0.35)
        out = Image.new("RGB", (W, H))
        out.paste(pf, (int(-W * k), 0))
        out.paste(frame, (int(W * (1 - k)), 0))
        frame = out
    a = SHOT_START[name]

    def txt(im):
        year_tag(im, t, a + 0.15, a + 2.2, year, label, y=230)
        if name == "photo1":
            caption(im, t, t_c, a + dur - 0.05, "МУЖЧИНА В СЕРОМ ПАЛЬТО", y=1640, size=74)
        else:
            caption(im, t, t_c, a + dur - 0.05, "ОПЯТЬ ОН", y=1640, size=84, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 6. все снимки на полу
FLOOR = []
_r = np.random.default_rng(77)
_keys = ["first_class", "wedding", "sea"] + [f"x{i}" for i in range(12)] + ["x2", "x5", "x8", "x11", "x1"]
for i, key in enumerate(_keys):
    col, row = i % 4, i // 4
    FLOOR.append((key, 150 + col * 260 + _r.uniform(-25, 25), 260 + row * 330 + _r.uniform(-25, 25), 300,
                  _r.uniform(-14, 14)))
FLOOR_ORDER = list(_r.permutation(len(FLOOR)))


@lru_cache(None)
def floor_photo(i):
    key, x, y, w, ang = FLOOR[i]
    img, mh, gh = get_photo(key)
    fr = framed(img)
    k = w * S / fr.width
    sm = fr.resize((int(fr.width * k), int(fr.height * k)), Image.LANCZOS).convert("RGBA").rotate(ang, Image.BICUBIC, expand=True)
    sh = Image.new("RGBA", sm.size, (0, 0, 0, 0))
    sh.putalpha(sm.split()[3].point(lambda p: int(p * 0.5)))
    sh = sh.filter(ImageFilter.GaussianBlur(8 * S))
    r = math.radians(-ang)

    def to_w(u, vv, cx, cy):
        bx = ((u + 26) * 2 * k - fr.width * k / 2) / S
        by = ((vv + 26) * 2 * k - fr.height * k / 2) / S
        return cx + bx * math.cos(r) - by * math.sin(r), cy + bx * math.sin(r) + by * math.cos(r)
    return sm, sh, to_w, mh, gh


def floor_world(t, t0, step=0.13, circles_at=None, hands=True, lines_at=None):
    img = table_bg().copy()
    c = C(img)
    marks = []
    active = None
    for n, i in enumerate(FLOOR_ORDER):
        key, x, y, w, ang = FLOOR[i]
        ta = t0 + n * step
        k = eout((t - ta) / 0.5)
        if k <= 0:
            continue
        cx, cy = x, lerp(2300, y, k)
        sm, sh, to_w, mh, gh = floor_photo(i)
        px, py = int(cx * S - sm.width / 2), int(cy * S - sm.height / 2)
        img.paste(sh, (px + 8 * S, py + 12 * S), sh)
        img.paste(sm, (px, py), sm)
        marks.append(to_w(mh[0], mh[1], cx, cy))
        if k < 1:
            active = (cx, cy, n)
    c = C(img)
    if circles_at is not None:
        for n, (x, y) in enumerate(marks):
            red_circle(c, x, y, 26, (t - circles_at - n * 0.035) / 0.25, 5, seed=n)
    if hands and active:
        hand(img, active[0], active[1] + 60, left=active[2] % 2 == 0)
    return img, marks


def hand(img, x, y, left=True):
    """Рука сверху: рукав толстовки из-за кадра снизу и кисть с пальцами."""
    c = C(img)
    bx = 180 if left else 900
    p0, p1 = (bx, 2050), (x, y)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    c.poly(_cap(p0, p1, 78, 58), (30, 30, 34))
    c.poly(_cap(p0, p1, 70, 50), (40, 40, 46))
    hx, hy = x + math.cos(ang) * 40, y + math.sin(ang) * 40
    c.ell(hx, hy, 46, 52, (196, 150, 122))
    for k in range(4):
        a = ang + (k - 1.5) * 0.22
        c.poly(_cap((hx + math.cos(a) * 30, hy + math.sin(a) * 30), (hx + math.cos(a) * 82, hy + math.sin(a) * 82), 12, 10),
               (188, 142, 116))
    a = ang + (1.4 if left else -1.4)
    c.poly(_cap((hx, hy), (hx + math.cos(a) * 62, hy + math.sin(a) * 62), 14, 11), (186, 140, 114))


def _cap(p0, p1, r0, r1):
    from engine import capsule
    return capsule(r0, r1, p0, p1)


def sc_floor(t, lt, dur):
    t0 = SHOT_START["floor"] + 0.05
    t_each = wt("каждом", 29)
    img, marks = floor_world(t, t0, 0.15, circles_at=t_each)
    k = eio((t - wt("всегда", 30)) / 1.6)
    frame = cam(img, lerp(1.25, 1.0, k), 540, lerp(1150, 960, k))

    def txt(im):
        caption(im, t, t_each, t0 + dur, "ОН БЫЛ НА КАЖДОМ", "всегда далеко", y=1600)
    return frame, txt


# ================================================================ 7. с фотоаппаратом / старше
YEARS = [1990, 1997, 2005, 2011, 2018, 2025]


def sc_aging(t, lt, dur):
    t_old = wt("каждым", 34) - 0.25
    img = to_img(grad((22, 22, 24), (8, 8, 10)))
    if t < t_old:
        card = stranger_card(1990)
        place_photo(img, framed(card, 14), 540, 960, 760, 0)
        k = eio(lt / 2.2)
        frame = cam(img, lerp(1.0, 1.7, k), lerp(540, 470, k), lerp(960, 840, k))
    else:
        for i, y in enumerate(YEARS):
            k = eout((t - t_old - i * 0.28) / 0.35)
            if k <= 0:
                continue
            cx, cy = 200 + (i % 3) * 340, 640 + (i // 3) * 600
            place_photo(img, framed(stranger_card(y), 12), cx, cy + (1 - k) * 80, 300, (i % 2 - 0.5) * 4, alpha=k)
            ImageDraw.Draw(img).text((cx * S, (cy + 290) * S), str(y), font=MS(46 * S, 800), fill=(240, 236, 226), anchor="mm")
        frame = cam(img, lerp(1.06, 1.0, eio((t - t_old) / 2)), 540, 930)

    def txt(im):
        caption(im, t, SHOT_START["aging"] + 0.1, t_old - 0.05, "ВСЕГДА С ФОТОАППАРАТОМ", y=1650, size=74)
        caption(im, t, t_old + 0.2, SHOT_START["aging"] + dur, "С КАЖДЫМ ГОДОМ — СТАРШЕ", y=260, size=72)
    return frame, txt


# ================================================================ 8. жутко: похож на отца
PROFILE = [(-0.35, 0.95), (-0.38, 0.6), (-0.45, 0.2), (-0.42, -0.15), (-0.3, -0.38), (-0.1, -0.5), (0.12, -0.48),
           (0.28, -0.38), (0.36, -0.2), (0.38, -0.05), (0.36, 0.02), (0.42, 0.15), (0.47, 0.24), (0.40, 0.28),
           (0.39, 0.33), (0.41, 0.38), (0.38, 0.41), (0.40, 0.45), (0.36, 0.5), (0.37, 0.58), (0.30, 0.64),
           (0.10, 0.66), (0.12, 0.8), (0.15, 1.0), (0.6, 1.15), (0.85, 1.4), (0.95, 2.0), (-0.95, 2.0),
           (-0.85, 1.4), (-0.6, 1.12)]
HAT = [(-0.62, -0.18), (-0.5, -0.62), (-0.18, -0.78), (0.2, -0.74), (0.42, -0.5), (0.46, -0.24), (0.7, -0.26),
       (0.72, -0.18), (-0.66, -0.12)]


@lru_cache(None)
def portrait(kind):
    """Портрет в профиль, контровой свет (как силуэты «24 часа президентом»)."""
    pw, ph = 900, 1100
    if kind == "stranger":
        bg = grad((150, 140, 120), (70, 62, 52), pw * S, ph * S)
    else:
        bg = grad((60, 74, 96), (20, 24, 34), pw * S, ph * S)
    img = to_img(bg)
    c = C(img)
    s = 520
    cx, cy = 430, 520
    pts = [(cx + px * s, cy + py * s) for px, py in PROFILE]
    rim = (255, 230, 190) if kind == "stranger" else (200, 220, 255)
    c.poly([(x + 7, y - 4) for x, y in pts], rim)
    c.poly(pts, (16, 14, 14))
    if kind == "stranger":
        hp = [(cx + px * s, cy + py * s) for px, py in HAT]
        c.poly([(x + 6, y - 4) for x, y in hp], rim)
        c.poly(hp, (20, 18, 18))
    else:
        hair = [(cx + px * s, cy + py * s) for px, py in [(-0.46, 0.05), (-0.44, -0.3), (-0.3, -0.52), (0.0, -0.6),
                                                           (0.26, -0.5), (0.36, -0.3), (0.2, -0.36), (-0.1, -0.3)]]
        c.poly([(x + 6, y - 4) for x, y in hair], rim)
        c.poly(hair, (16, 14, 14))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    a = np.asarray(img, np.float32)
    if kind == "stranger":
        g = a.mean(axis=2, keepdims=True)
        a = g * np.array([1.08, 0.95, 0.78], np.float32) + 12
    r = np.random.default_rng(5 if kind == "stranger" else 6)
    a += r.normal(0, 8, a.shape[:2])[:, :, None]
    return to_img(a), pts


def sc_uncanny(t, lt, dur):
    t_cmp = wt("чем", 38) - 0.1
    if t < t_cmp:
        img = to_img(grad((14, 12, 12), (4, 4, 5)))
        place_photo(img, framed(stranger_card(2025), 14), 540, 960, 760, 0)
        k = eio(lt / (t_cmp - SHOT_START["uncanny"]))
        frame = cam(img, lerp(1.2, 2.4, k), lerp(540, 520, k), lerp(960, 650, k))
        a = np.asarray(frame, np.float32) * lerp(1.0, 0.55, k)
        a[:, :, 0] *= 1 + 0.15 * k
        return to_img(a), None
    img = to_img(grad((12, 12, 14), (4, 4, 6)))
    k = eio((t - wt("похож", 41) + 0.3) / 1.3)
    st, pts = portrait("stranger")
    fa, _ = portrait("father")
    w = 460
    lx = lerp(290, 540, k)
    rx = lerp(790, 540, k)
    place_photo(img, framed(st, 10), lx, 900, w, lerp(-3, 0, k), shadow=True)
    place_photo(img, framed(fa, 10), rx, 900, w, lerp(3, 0, k), shadow=True, alpha=lerp(1.0, 0.55, k))
    c = C(img)
    d = ImageDraw.Draw(img)
    for x, lab in ((lx, "ОН · 2025"), (rx, "ПАПА · 2025")):
        if k < 0.6:
            d.text((x * S, 1240 * S), lab, font=MS(34 * S, 700), fill=(230, 226, 216), anchor="mm")
    if k > 0.85:   # профили совпадают: красный контур
        sc = w / (900 + 20) * 1
        ox, oy = 540 - w / 2 + 10 * sc, 900 - (1100 + 20) * sc / 2 + 10 * sc
        pr = [(ox + x * sc, oy + y * sc) for x, y in pts]
        a_ = int(255 * clamp((k - 0.85) / 0.15) * (0.6 + 0.4 * math.sin(t * 14)))
        c.line(pr + [pr[0]], RED + (a_,), 4)
    frame = cam(img, lerp(1.0, 1.1, (t - t_cmp) / 4.5), 540, 900)

    def txt(im):
        caption(im, t, wt("похож", 41), SHOT_START["uncanny"] + dur, "ПОХОЖ НА МОЕГО ОТЦА", y=1560, size=80,
                accent=(255, 90, 80))
    return frame, txt


# ================================================================ трансформируемый холст (для «зума» без потери чёткости)
class TC(C):
    """Холст с камерой: точка сцены (x, y) -> ((x-ox)*sc+540, (y-oy)*sc+960)."""

    def __init__(self, img, ox=540, oy=960, sc=1.0):
        super().__init__(img)
        self.ox, self.oy, self.sc = ox, oy, sc

    def p(self, x, y):
        return (x - self.ox) * self.sc + 540, (y - self.oy) * self.sc + 960

    def rect(self, x0, y0, x1, y1, fill, r=0, outline=None, w=0):
        a, b = self.p(x0, y0)
        c_, d_ = self.p(x1, y1)
        super().rect(a, b, c_, d_, fill, r * self.sc, outline, w * self.sc)

    def ell(self, cx, cy, rx, ry, fill, outline=None, w=0):
        a, b = self.p(cx, cy)
        super().ell(a, b, rx * self.sc, ry * self.sc, fill, outline, w * self.sc)

    def poly(self, pts, fill):
        super().poly([self.p(x, y) for x, y in pts], fill)

    def line(self, pts, fill, w):
        super().line([self.p(x, y) for x, y in pts], fill, w * self.sc)

    def parts(self, parts):
        out = []
        for kind_, geo, part in parts:
            if kind_ == "poly":
                geo = [self.p(x, y) for x, y in geo]
            elif kind_ == "ell":
                geo = (*self.p(geo[0], geo[1]), geo[2] * self.sc, geo[3] * self.sc)
            else:
                p0, p1, r0, r1 = geo
                geo = (self.p(*p0), self.p(*p1), r0 * self.sc, r1 * self.sc)
            out.append((kind_, geo, part))
        return out


# ================================================================ руки незнакомца от первого лица
SLEEVE = (78, 80, 86)
SLEEVE_D = (52, 54, 58)
OLD_SKIN = (190, 156, 134)


def pov_hand(c, base, tip, ang_fingers=0.0, curl=0.0, scale=1.0, skin=OLD_SKIN, sleeve=SLEEVE):
    """Рука в сером рукаве пальто из-за нижнего края кадра."""
    (bx, by), (x, y) = base, tip
    s = scale
    c.poly(_cap((bx, by), (x, y), 104 * s, 74 * s), SLEEVE_D)
    c.poly(_cap((bx, by), (x, y), 96 * s, 66 * s), sleeve)
    a = math.atan2(y - by, x - bx)
    cx, cy = x + math.cos(a) * 30 * s, y + math.sin(a) * 30 * s
    c.poly(_cap((x, y), (cx + math.cos(a) * 40 * s, cy + math.sin(a) * 40 * s), 52 * s, 50 * s), skin)
    for k in range(4):
        fa = a + ang_fingers + (k - 1.5) * 0.2
        L = (86 - abs(k - 1.5) * 12) * s * (1 - curl * 0.5)
        x0, y0 = cx + math.cos(fa) * 56 * s, cy + math.sin(fa) * 56 * s
        c.poly(_cap((x0, y0), (x0 + math.cos(fa) * L, y0 + math.sin(fa) * L), 15 * s, 12 * s), mix(skin, (0, 0, 0), 0.08))
        c.line([(x0 + math.cos(fa) * L * 0.5 - 8 * s, y0 + math.sin(fa) * L * 0.5),
                (x0 + math.cos(fa) * L * 0.5 + 8 * s, y0 + math.sin(fa) * L * 0.5)], mix(skin, (0, 0, 0), 0.3) + (120,), 2)
    for k in range(3):     # морщины на тыльной стороне ладони
        c.line([(cx - 20 * s, cy - 14 * s + k * 14 * s), (cx + 24 * s, cy - 18 * s + k * 14 * s)],
               mix(skin, (0, 0, 0), 0.25) + (90,), 2)


def film_camera(c, x, y, s=1.0):
    """Плёночный фотоаппарат сзади (видоискатель, верхняя панель, хром)."""
    c.rect(x - 230 * s, y - 120 * s, x + 230 * s, y + 120 * s, (16, 16, 18), r=26 * s)
    c.rect(x - 230 * s, y - 120 * s, x + 230 * s, y - 70 * s, (170, 170, 172), r=20 * s)
    c.rect(x - 70 * s, y - 190 * s, x + 70 * s, y - 100 * s, (22, 22, 24), r=14 * s)
    c.rect(x - 34 * s, y - 160 * s, x + 34 * s, y - 120 * s, (60, 70, 80), r=8 * s)
    c.ell(x + 150 * s, y - 132 * s, 34 * s, 16 * s, (150, 150, 152))
    c.ell(x - 160 * s, y - 132 * s, 28 * s, 14 * s, (150, 150, 152))
    for k in range(10):
        c.line([(x - 220 * s + k * 44 * s, y - 60 * s), (x - 210 * s + k * 44 * s, y + 110 * s)], (26, 26, 28), 2)


def viewfinder(im, k=1.0, label=None):
    """Вид через видоискатель плёночной камеры: тёмные края, рамка, круг фокусировки."""
    if k <= 0:
        return im
    a = np.asarray(im, np.float32)
    y, x = np.ogrid[:H, :W]
    d = np.maximum(np.abs(x - W / 2) / (W * 0.44), np.abs(y - H / 2) / (H * 0.3))
    m = np.clip(1 - (d - 0.9) * 6, 0, 1)
    a = a * (1 - k + k * m[:, :, None])
    im = to_img(a)
    dr = ImageDraw.Draw(im, "RGBA")
    al = int(150 * k)
    dr.ellipse([540 - 90, 960 - 90, 540 + 90, 960 + 90], outline=(255, 255, 255, al), width=2)
    dr.ellipse([540 - 34, 960 - 34, 540 + 34, 960 + 34], outline=(255, 255, 255, al), width=2)
    dr.line([(540 - 34, 960), (540 + 34, 960)], fill=(255, 255, 255, al), width=2)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = 540 + sx * 420, 960 + sy * 520
            dr.line([(cx, cy), (cx - sx * 60, cy)], fill=(255, 255, 255, al), width=3)
            dr.line([(cx, cy), (cx, cy - sy * 60)], fill=(255, 255, 255, al), width=3)
    return im


def flash(im, t, at, d=0.35):
    k = 1 - clamp((t - at) / d)
    if t < at or k <= 0:
        return im
    return to_img(np.asarray(im, np.float32) * (1 - k) + 255 * k)


# ================================================================ 9. дед
@lru_cache(None)
def wall_frame():
    n = fbm(WW, WH, [4, 16, 60], 41)
    a = np.array([92, 78, 62], np.float32) * (0.7 + 0.35 * n[:, :, None])
    for i in range(0, WW, 90):                 # обои в полоску
        a[:, i:i + 6] *= 0.88
    a = a * (0.25 + 0.9 * radial(540, 820, 900, p=1.5)[:, :, None])
    img = to_img(a)
    c = C(img)
    c.rect(300, 520, 780, 1120, (40, 28, 20))
    c.rect(330, 550, 750, 1090, (150, 132, 108))          # светлое пятно: снимка нет
    c.rect(350, 570, 730, 1070, (136, 118, 96))
    c.poly([(690, 520), (780, 520), (780, 610)], (12, 12, 12))   # траурная лента
    c.poly([(720, 520), (780, 580), (780, 610), (690, 520)], (20, 20, 20))
    return img


def river(t, lt=0.0):
    img = to_img(grad((12, 16, 28), (6, 8, 14)))
    c = C(img)
    c.ell(760, 330, 46, 46, (226, 230, 236))
    img = glow(img, lambda d, k: d.ellipse([600 * k, 170 * k, 920 * k, 490 * k], fill=(60, 70, 90)), 60)
    c = C(img)
    # дальний берег с лесом
    for i in range(30):
        x = i * 40 - 10
        hgt = 90 + (i * 53) % 70
        c.poly([(x - 34, 780), (x, 780 - hgt), (x + 34, 780)], (8, 10, 14))
    c.rect(0, 770, W, 800, (8, 10, 14))
    a = np.asarray(img, np.float32)
    yy = np.linspace(0, 1, WH)[:, None]
    xx = np.linspace(0, 1, WW)[None, :]
    water = (yy > 800 / H).astype(np.float32)
    a = a * (1 - water[:, :, None]) + water[:, :, None] * np.array([8, 12, 20], np.float32)
    f = fbm(WW // 4, WH // 4, [6, 20, 60], 33)
    f = np.roll(f, int(lt * 10), 0)
    f = np.asarray(Image.fromarray((f * 255).astype(np.uint8)).resize((WW, WH), Image.BILINEAR), np.float32) / 255
    path = np.exp(-((xx - 0.7) / (0.06 + 0.25 * (yy - 0.42))) ** 2) * water
    glitter = np.clip((f - 0.55) * 4, 0, 1) * path
    a = a + glitter[:, :, None] * np.array([170, 180, 200], np.float32) + (f * water)[:, :, None] * np.array([10, 14, 22], np.float32)
    img = to_img(a)
    c = C(img)
    # шляпа на воде дрейфует, по ней скользит лунный свет
    hx, hy = 330 + lt * 40, 1240 + math.sin(t * 1.3) * 10
    tilt = math.sin(t * 0.9) * 4
    c.ell(hx, hy + 18, 190, 40, (16, 16, 20))
    c.ell(hx, hy + 14, 186, 34, (26, 26, 30))
    crown = [(hx - 110, hy + 10), (hx - 96, hy - 70), (hx - 20, hy - 92), (hx + 50, hy - 88), (hx + 104, hy - 66),
             (hx + 112, hy + 10)]
    c.poly([(x + 4, y - 4) for x, y in crown], (120, 130, 150))
    c.poly(crown, (30, 30, 34))
    c.rect(hx - 108, hy - 14, hx + 110, hy + 6, (14, 14, 16))
    for k in range(3):
        c.arc(hx, hy + 24, 220 + k * 70, 50 + k * 16, 0, 360, (90, 100, 120, 90 - k * 25), 3)
    # туман
    fg = fbm(WW // 4, WH // 4, [3, 7], 51)
    fg = np.roll(fg, int(t * 8), 1)
    fog = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).resize((WW, WH), Image.BILINEAR), np.float32) / 255
    band = np.exp(-((yy - 0.42) / 0.12) ** 2)
    a = np.asarray(img, np.float32) + (fog ** 2 * band)[:, :, None] * np.array([70, 76, 88], np.float32)
    return to_img(a)


def sc_grandpa(t, lt, dur):
    t_r = wt("он", 45) - 0.15
    if t < t_r:
        frame = cam(wall_frame(), lerp(1.0, 1.18, lt / 2.2), 540, 840)
    else:
        img = river(t, t - t_r)
        frame = cam(img, lerp(1.05, 1.15, (t - t_r) / 4), 540, 1000)
        tn = wt("тело", 47)
        k = eout((t - tn) / 0.5)
        if k > 0:      # газетная вырезка
            c = C(frame, 1)
            y = lerp(2000, 1440, k)
            c.rect(150, y, 930, y + 360, (214, 206, 186))
            c.text(540, y + 80, "ПРОПАЛ БЕЗ ВЕСТИ", PF(62, 900), (24, 22, 22))
            c.text(540, y + 150, "Тело так и не найдено. 1985", PFI(36, 500), (60, 56, 52))
            for i in range(3):
                c.line([(200, y + 210 + i * 40), (880 - i * 90, y + 210 + i * 40)], (150, 144, 132), 7)

    def txt(im):
        caption(im, t, SHOT_START["grandpa"] + 0.1, t_r - 0.1, "ДЕДА Я НИКОГДА НЕ ВИДЕЛ", y=1650, size=70)
        year_tag(im, t, wt("утонул", 45), wt("тело", 47) - 0.1, "1985", "УТОНУЛ", y=280)
    return frame, txt


# ================================================================ 10. фото в огне
def fire_layer(t, w=WW, h=WH):
    f = fbm(w // 4, h // 4, [4, 10, 24], 61)
    rows = (np.arange(h // 4)[:, None] + int(t * 60)) % (h // 4)
    f = f[rows, np.arange(w // 4)[None, :]]
    yy = np.linspace(0, 1, h // 4)[:, None]
    heat = np.clip(f * 1.2 + (yy - 0.62) * 1.7, 0, 1) ** 2.6
    rgb = np.stack([heat * 235, heat ** 1.9 * 120, heat ** 4 * 40], 2) + 4
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)


def sc_burn(t, lt, dur):
    img = fire_layer(t)
    c = C(img)
    for i in range(3):
        ta = i * 0.9
        k = clamp((lt - ta) / 2.6)
        if k <= 0:
            continue
        x, y = 300 + i * 250, lerp(500, 1250, eout(k * 1.4)) + i * 60
        ang = lerp(-10, 25, k) * (1 if i % 2 else -1)
        card = Image.new("RGBA", (360 * S, 460 * S), (230, 222, 204, 255))
        cc = C(card)
        cc.rect(20, 20, 340, 380, (150, 136, 112))
        silhouette(card, person_parts(180, 400, 330, pose(), 1, "man", "front"), base=(40, 34, 28), rim=(120, 110, 90))
        burn = clamp(k * 1.3 - 0.1)
        a = np.asarray(card, np.float32)
        n = fbm(360 * S, 460 * S, [5, 14], 70 + i)
        yy = np.linspace(1, 0, 460 * S)[:, None]
        front = n * 0.5 + yy * 0.8
        char = np.clip((burn * 1.4 - front) * 6, 0, 1)
        gone = np.clip((burn * 1.4 - front - 0.15) * 6, 0, 1)
        a[:, :, :3] = a[:, :, :3] * (1 - char[:, :, None]) + np.array([20, 14, 10]) * char[:, :, None]
        edge = np.clip(char - gone, 0, 1)
        a[:, :, :3] += edge[:, :, None] * np.array([255, 140, 40]) * 1.2
        a[:, :, 3] *= 1 - gone
        card = to_img(a).convert("RGBA") if False else Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
        card = card.rotate(ang, Image.BICUBIC, expand=True)
        img.paste(card, (int(x * S - card.width / 2), int(y * S - card.height / 2)), card)
    # рука, бросающая снимки (тёмный силуэт сверху)
    k = eio(lt / 0.8) if lt < 1.6 else eio((2.4 - lt) / 0.8)
    if k > 0:
        c = C(img)
        c.poly(_cap((1150, -150), (lerp(1150, 640, k), lerp(-150, 420, k)), 90, 64), (14, 10, 10))
    c = C(img)
    c.rect(0, 0, 150, H, (20, 18, 18))
    c.rect(930, 0, W, H, (20, 18, 18))
    c.rect(0, 0, W, 180, (20, 18, 18))
    c.rect(0, 1760, W, H, (20, 18, 18))
    c.rect(130, 160, 950, 190, (40, 36, 34))
    c.rect(130, 160, 160, 1780, (40, 36, 34))
    c.rect(920, 160, 950, 1780, (40, 36, 34))
    for i in range(8):                                     # колосник
        c.rect(180 + i * 100, 1700, 210 + i * 100, 1760, (30, 26, 24))
    rr = np.random.default_rng(int(t * 24))
    for i in range(30):                                    # искры
        x = rr.uniform(200, 900)
        y = (1700 - ((t * 300 + i * 97) % 1500))
        c.ell(x + math.sin(t * 3 + i) * 20, y, 3, 3, (255, 200, 120, 220))
    fr = Image.new("RGB", (WW, WH))
    fr.paste(img)
    glowed = glow(fr, lambda d, kk: d.rectangle([0, 1100 * kk, W * kk, H * kk], fill=(70, 26, 6)), 80)
    frame = cam(glowed, lerp(1.1, 1.0, lt / 4), 540, 960, hh_amp=3)

    def txt(im):
        caption(im, t, wt("сожгла", 51), SHOT_START["burn"] + dur, "СОЖГЛА ВСЕ ЕГО ФОТО", "по её словам", y=360)
    return frame, txt


# ================================================================ 11. кухня
FLOOR_Y = 1560


@lru_cache(None)
def kitchen_bg(lamp=True, window_dim_man=False):
    img = props.kitchen_room(FLOOR_Y).copy()
    night_window(img, 600, 480, 980, 1010, man=window_dim_man)
    c = C(img)
    c.rect(590, 1010, 990, 1040, (70, 66, 64))                    # подоконник с цветком
    c.poly([(880, 1010), (930, 1010), (922, 960), (888, 960)], (120, 70, 50))
    for k in range(6):
        a = -2.6 + k * 0.45
        c.poly(_cap((905, 962), (905 + math.cos(a) * 60, 962 + math.sin(a) * 50), 6, 10), (40, 70, 40))
    curtains(c, 600, 480, 980, 1010)
    return img


def night_window(img, x0, y0, x1, y1, man=False, man_k=0.0):
    """Окно ночью: дерево, фонарь через дорогу; man — фигура под фонарём."""
    c = C(img)
    c.rect(x0, y0, x1, y1, (14, 20, 34))
    sky = grad((18, 26, 46), (10, 14, 24), int((x1 - x0) * S), int((y1 - y0) * S))
    img.paste(to_img(sky), (int(x0 * S), int(y0 * S)))
    c = C(img)
    lx, ly = x0 + (x1 - x0) * 0.66, y0 + (y1 - y0) * 0.42
    c.rect(x0, y0 + (y1 - y0) * 0.78, x1, y1, (22, 22, 26))               # тротуар
    c.rect(lx - 3, ly, lx + 3, y0 + (y1 - y0) * 0.86, (8, 8, 10))           # фонарь
    c.poly([(lx - 18, ly), (lx + 18, ly), (lx + 10, ly - 14), (lx - 10, ly - 14)], (8, 8, 10))
    tx = x0 + (x1 - x0) * 0.22                                              # дерево
    c.poly([(tx - 8, y1), (tx - 4, y0 + 120), (tx + 4, y0 + 120), (tx + 10, y1)], (6, 6, 8))
    for k in range(6):
        a = -0.9 + k * 0.36
        c.line([(tx, y0 + 260 + k * 30), (tx + math.cos(a - 1.57) * 120, y0 + 140 + k * 26)], (6, 6, 8), 4)
    for k in range(9):
        c.ell(tx + (k % 3 - 1) * 70, y0 + 120 + (k // 3) * 60, 80, 60, (8, 10, 12, 230))
    img2 = glow(img, lambda d, k: d.ellipse([(lx - 120) * k, (ly - 40) * k, (lx + 120) * k, (y0 + (y1 - y0) * 0.9) * k],
                                            fill=(110, 80, 40)), 30)
    img.paste(img2)
    c = C(img)
    if man:
        fh = (y1 - y0) * 0.24
        silhouette(img, person_parts(lx - 30, y0 + (y1 - y0) * 0.86, fh, pose(lean=8, neck=12), 1, "man", "front"),
                   base=(14, 14, 16), rim=(160, 120, 80), rim_w=1.5, alpha=0.85)
    c = C(img)
    for x in (x0, x1):
        c.rect(x - 16, y0 - 16, x + 16, y1 + 16, (40, 40, 46))
    c.rect(x0 - 16, y0 - 16, x1 + 16, y0 + 8, (40, 40, 46))
    c.rect(x0 - 30, y1, x1 + 30, y1 + 26, (50, 50, 56))
    c.rect((x0 + x1) / 2 - 8, y0, (x0 + x1) / 2 + 8, y1, (40, 40, 46))


def clock_hand(c, t):
    a = math.radians(t * 6 % 360 - 90)
    c.line([(140, 520), (140 + math.cos(a) * 50, 520 + math.sin(a) * 50)], (150, 30, 30), 2)
    c.line([(140, 520), (140 + 40, 520 - 14)], (30, 26, 24), 6)
    c.line([(140, 520), (140 - 8, 520 - 40)], (30, 26, 24), 4)


def kitchen_world(t, nar=None, gm=None, photos_x=None, gm_f=-1, gm_x=820, gm_view="side", lamp_k=1.0, gm_parts=None,
                  man_in_window=False):
    img = kitchen_bg(True, man_in_window).copy()
    c = C(img)
    clock_hand(c, t)
    abazhur(img, 540, 760, lamp_k)
    c = C(img)
    # стул рассказчика
    c.rect(150, 1350, 330, 1372, (44, 32, 26))
    c.rect(150, 1080, 170, 1560, (40, 30, 24))
    c.rect(150, 1100, 200, 1240, (40, 30, 24), r=6)
    c.rect(310, 1372, 330, 1560, (40, 30, 24))
    if nar:
        silhouette(img, person_parts(270, FLOOR_Y, 820, nar, 1, "young"), base=(10, 10, 12), rim=LAMP_RIM,
                   light=(1, -0.8), rim_w=5, fill_light=(10, 8, 6))
    if gm or gm_parts:
        c = C(img)
        c.rect(860, 1360, 1010, 1382, (44, 32, 26))
        c.rect(990, 1080, 1010, 1560, (40, 30, 24))
        c.rect(960, 1100, 1010, 1240, (40, 30, 24), r=6)
        c.rect(860, 1382, 880, 1560, (40, 30, 24))
        parts = gm_parts or person_parts(gm_x, FLOOR_Y, 760, gm, gm_f, "grandma", gm_view)
        silhouette(img, parts, base=(10, 10, 12), rim=NIGHT_RIM, light=(1, -0.5), rim_w=5, fill_light=(8, 8, 12))
    c = C(img)
    table_set(c, 360, 720, 1230)
    if photos_x is not None:
        for i in range(4):
            x = photos_x + i * 14
            c.poly([(x, 1218 - i * 2), (x + 70, 1218 - i * 2), (x + 76, 1226 - i * 2), (x + 4, 1226 - i * 2)],
                   (226, 218, 202))
    return img


SIT_N = pose(lean=-2, neck=6, ln=(90, 92), lf=(86, 92), an=(52, 56), af=(48, 60))
SIT_G = pose(lean=6, neck=6, ln=(88, 90), lf=(84, 90), an=(60, 40), af=(56, 44))


def sc_kitchen(t, lt, dur):
    push = pose(lean=14, neck=2, ln=(90, 92), lf=(86, 92), an=(80, 4), af=(48, 60))
    nar = breathe(keyed(lt, [(0, SIT_N), (0.5, push), (1.1, push), (1.7, SIT_N)]), t)
    look = pose(lean=14, neck=16, nod=14, ln=(88, 90), lf=(84, 90), an=(66, 30), af=(60, 36))
    gm = breathe(keyed(lt, [(0, SIT_G), (1.2, SIT_G), (2.4, look)]), t, 0.18)
    px = lerp(420, 560, eio((lt - 0.4) / 0.8))
    img = kitchen_world(t, nar, gm, px)
    return cam(img, lerp(1.0, 1.12, lt / dur), 560, 1150), None


# ================================================================ 12. коробка из шкафа
@lru_cache(None)
def hallway_bg():
    img = to_img(grad((30, 30, 36), (14, 14, 16)))
    c = C(img)
    c.rect(0, FLOOR_Y, W, H, (20, 18, 18))
    c.rect(640, 300, 1040, 1560, (42, 30, 24))          # шкаф
    c.rect(660, 320, 1020, 1540, (52, 38, 30))
    c.rect(660, 760, 1020, 776, (30, 22, 18))           # полка внутри
    return img


def sc_box(t, lt, dur):
    t_hall = SHOT_START["box"] + 1.05
    t_top = wt("поставила", 59) - 0.45
    if t < t_hall:   # бабушка встаёт из-за стола
        up = pose(lean=30, neck=-6, ln=(40, 50), lf=(36, 46), an=(40, 30), af=(36, 34))
        stand = pose(lean=8, neck=8, an=(6, 12), af=(-4, 14))
        gm = keyed(lt, [(0, SIT_G), (0.45, up), (1.0, stand)])
        img = kitchen_world(t, breathe(SIT_N, t), gm)
        return cam(img, 1.12, 560, 1150), None
    if t < t_top:    # прихожая: идёт к шкафу, открывает, берёт коробку
        lt2 = t - t_hall
        img = hallway(FLOOR_Y).copy()
        c = C(img)
        door = clamp((lt2 - 0.9) / 0.4)
        taken = lt2 > 1.75
        wardrobe(c, 660, 320, 1020, 1540, eio(door), box=not taken)
        img = props.volumetric(img, [(0, 300), (120, 300), (700, 1560), (0, 1560)], (34, 26, 18), 60)
        x = lerp(220, 520, eio(lt2 / 0.95))
        ph = lt2 / 0.95 * 2 * math.pi * 1.5
        if lt2 < 0.95:
            p = walk(ph, 0.8, old=True)
        else:
            raise_ = pose(lean=2, neck=-8, an=(150, 10), af=(140, 14))
            hold = pose(lean=6, neck=6, an=(60, 50), af=(56, 54))
            p = keyed(lt2, [(0.95, pose(lean=8, neck=8)), (1.25, pose(lean=4, neck=-4, an=(20, 20))),
                            (1.7, raise_), (2.2, hold)])
        p = breathe(p, t)
        silhouette(img, person_parts(x, FLOOR_Y, 780, p, 1, "grandma"), base=(10, 10, 12), rim=(220, 180, 130),
                   light=(-1, -0.6), rim_w=5, fill_light=(8, 6, 6))
        if taken:
            J = skeleton(x, FLOOR_Y, 780, p, 1)
            wx, wy = J["an_wr"]
            c = C(img)
            c.rect(wx - 20, wy - 60, wx + 180, wy + 10, (150, 120, 90))
            c.rect(wx - 20, wy - 60, wx + 180, wy - 44, (120, 96, 70))
        img = glow(img, lambda d, k: d.ellipse([0, 400 * k, 500 * k, 1500 * k], fill=(70, 50, 30)), 80)
        return cam(img, 1.05, 600, 1100, hh_amp=2), None
    # сверху: коробку ставят на стол
    lt3 = t - t_top
    img = table_bg().copy()
    c = C(img)
    k = eio(lt3 / 0.55)
    y = lerp(-500, 940, k)
    sc_ = lerp(1.25, 1.0, k)
    c.rect(540 - 330 * sc_ + 16, y - 230 * sc_ + 24, 540 + 330 * sc_ + 16, y + 230 * sc_ + 24, (0, 0, 0, 120))
    c.rect(540 - 330 * sc_, y - 230 * sc_, 540 + 330 * sc_, y + 230 * sc_, (160, 128, 96), r=6)
    c.rect(540 - 300 * sc_, y - 200 * sc_, 540 + 300 * sc_, y + 200 * sc_, (172, 140, 106), r=4)
    c.text(540, y, "ОБУВЬ · 38", MS(int(40 * sc_), 700), (110, 86, 60))
    if lt3 < 1.0:      # старческие руки бабушки
        hk = 1 - clamp((lt3 - 0.6) / 0.4)
        for s in (-1, 1):
            pov_hand(c, (540 + s * 300, y - 700 * hk - 60), (540 + s * 330, y - 120), 0.0, 0.2, 0.9,
                     skin=(200, 166, 146), sleeve=(110, 64, 68))
    return cam(img, 1.0, 540, 960), None


# ================================================================ 13. что в коробке
@lru_cache(None)
def small_card(key, w=240):
    if key.startswith("r_"):
        img = photo_rev(key[2:])[0]
    else:
        img = get_photo(key)[0]
    fr = framed(img)
    k = w * S / fr.width
    return fr.resize((int(fr.width * k), int(fr.height * k)), Image.LANCZOS).convert("RGBA")


SPILL = []
_r2 = np.random.default_rng(91)
_ck = [f"x{i}" for i in range(12)] + ["r_first_class", "r_wedding", "r_sea"]
for i in range(34):
    a = _r2.uniform(0, 2 * math.pi)
    d = _r2.uniform(250, 900)
    SPILL.append((_ck[i % len(_ck)], 540 + math.cos(a) * d * 0.7, 940 + math.sin(a) * d, _r2.uniform(-40, 40),
                  _r2.uniform(0, 0.35)))

REV3 = [("r_first_class", "папин", 63, "ПАПИН ПЕРВЫЙ КЛАСС", "1990"), ("r_wedding", "свадьба", 65, "СВАДЬБА РОДИТЕЛЕЙ", "2005"),
        ("r_sea", "тот", 66.5, "ТОТ САМЫЙ ПИРС", "2011")]


@lru_cache(None)
def big_rev(key, w=900):
    img = photo_rev(key[2:], 3)[0]
    return framed(img, 16)


def sc_inside(t, lt, dur):
    img = table_bg().copy()
    c = C(img)
    t_in, t_hund = wt("внутри", 61), wt("сотни", 62)
    # открытая коробка
    c.rect(540 - 330 + 16, 940 - 230 + 24, 540 + 330 + 16, 940 + 230 + 24, (0, 0, 0, 120))
    c.rect(210, 710, 870, 1170, (120, 94, 70), r=6)
    c.rect(236, 736, 844, 1144, (40, 30, 24))
    for i in range(6):
        sm = small_card(_ck[i * 2], 420)
        img.paste(sm, (int((330 + i * 6) * S), int((780 + i * 8) * S)), sm)
    # разлёт снимков
    for key, x, y, ang, delay in SPILL:
        k = eout((t - t_hund - delay) / 0.7)
        if k <= 0:
            continue
        sm = small_card(key).rotate(ang * k, Image.BICUBIC, expand=True)
        px, py = lerp(540, x, k), lerp(940, y, k)
        img.paste(sm, (int(px * S - sm.width / 2), int(py * S - sm.height / 2)), sm)
    # крышка уходит к камере
    k = eio((t - t_in + 0.2) / 0.7)
    if k < 1:
        sc_ = lerp(1.0, 2.6, k)
        cy = lerp(940, -400, k)
        c = C(img)
        c.rect(540 - 345 * sc_, cy - 245 * sc_, 540 + 345 * sc_, cy + 245 * sc_, (160, 128, 96), r=6)
        c.text(540, cy, "ОБУВЬ · 38", MS(int(40 * sc_), 700), (110, 86, 60))
    # три снимка по очереди крупно
    top = None
    for i, (key, w_, a_, label, year) in enumerate(REV3):
        ta = wt(w_, a_) - 0.15
        k = eout((t - ta) / 0.45)
        if k > 0:
            top = (key, k, label, year, ta)
            place_photo(img, big_rev(key), 540, lerp(1400, 900 + i * 18, k), 900, (i - 1) * 3 * (1 - k * 0.5))
    frame = cam(img, 1.0 if not top else 1.0, 540, 960)

    def txt(im):
        caption(im, t, t_hund, REV3[0][2] and wt("папин", 63) - 0.15, "СОТНИ", y=1650, size=110)
        if top:
            key, k, label, year, ta = top
            year_tag(im, t, ta + 0.1, SHOT_START["inside"] + dur, year, label, y=230)
    return frame, txt


# ================================================================ 14. с его места (от первого лица)
@lru_cache(None)
def compare_world():
    img = to_img(grad((16, 16, 18), (6, 6, 8)))
    orig, mh, _ = photo("first_class", 3)
    to_w = place_photo(img, framed(orig, 16), 540, 560, 900, -1.5)
    rev, _ = photo_rev("first_class", 3)
    place_photo(img, framed(rev, 16), 540, 1340, 900, 1.5)
    return img, to_w(mh[0], mh[1], 16)


@lru_cache(None)
def yard_bg():
    """Школьный двор 1 сентября 1990-го глазами незнакомца: улица, дома, деревья."""
    img = to_img(grad((168, 188, 206), (222, 222, 216)))
    c = C(img)
    for x0, x1, top, col in [(-40, 420, 420, (196, 190, 178)), (430, 820, 470, (186, 182, 172)), (840, 1120, 400, (200, 194, 182))]:
        c.rect(x0, top, x1, 1050, col)
        for i in range(int(x0) + 26, int(x1) - 26, 62):
            for j in range(int(top) + 26, 1000, 70):
                c.rect(i, j, i + 32, j + 40, (120, 136, 150))
    c.rect(0, 1050, W, H, (132, 128, 122))
    for x in (160, 640, 990):
        c.rect(x - 9, 760, x + 9, 1070, (80, 62, 46))
        c.ell(x, 740, 110, 140, (90, 116, 66))
        c.ell(x - 50, 800, 80, 80, (78, 104, 58))
    n = fbm(WW, WH, [8, 30, 90], 12)
    a = np.asarray(img, np.float32) * (0.9 + 0.2 * n[:, :, None])
    return to_img(a).filter(ImageFilter.GaussianBlur(2 * S))


@lru_cache(None)
def fence_fg():
    lay = Image.new("RGBA", (WW, WH), (0, 0, 0, 0))
    c = C(lay)
    for x in range(-60, W + 100, 250):
        c.rect(x, 0, x + 70, H, (16, 18, 16, 255))
    c.rect(0, 380, W, 430, (16, 18, 16, 255))
    c.rect(0, 1560, W, 1610, (16, 18, 16, 255))
    return lay.filter(ImageFilter.GaussianBlur(9 * S))


def yard_people(img, t, look_k):
    """Люди во дворе (спиной к нам), бабушка оглядывается в объектив."""
    sway = math.sin(t * 1.3)
    sh = (540, 1120, 300)                  # родственник с фотоаппаратом лицом к нам
    shaded(img, body_front(sh[0], sh[1], sh[2], pose(an=(30, 120), af=(30, 120)), "young"), PALETTE["dad"],
           (sh[0], sh[1]), s=S)
    C(img).rect(sh[0] - 22, sh[1] - 262, sh[0] + 22, sh[1] - 236, (20, 20, 22))
    for kind, x, feet, h in [("woman", 740, 1340, 560), ("dad", 870, 1360, 600), ("boy", 560, 1330, 400)]:
        k = {"dad": "young", "woman": "woman"}.get(kind, kind)
        p = pose(an=(8 + 2 * sway, 14), af=(8, 14), nod=sway)
        shaded(img, body_front(x + sway * 4, feet, h, p, k, back=True), PALETTE[kind], (x, feet), s=S)
        if kind == "boy":
            c = C(img)
            for i in range(5):
                c.line([(x + 50, feet - h * 0.45), (x + 30 + i * 18, feet - h * 0.8 - i % 2 * 26)], (60, 110, 50), 5)
                c.ell(x + 30 + i * 18, feet - h * 0.84 - i % 2 * 26, 16, 24, (220, 60, 90))
    gx, gf, gh = 330, 1370, 600
    if look_k < 0.35:
        p = pose(an=(10, 30), af=(10, 30))
        parts = body_front(gx, gf, gh, p, "grandma", back=True)
    elif look_k < 0.6:
        p = pose(turn=-1, an=(10, 30), af=(10, 30))
        parts = body_front(gx, gf, gh, p, "grandma", back=True)
    else:
        p = pose(look=True, an=(10, 30), af=(10, 30))
        parts = body_front(gx, gf, gh, p, "grandma", back=True)
    shaded(img, parts, PALETTE["grandma"], (gx, gf), s=S)


def sc_reverse(t, lt, dur):
    t_pov = wt("с", 71) - 0.1
    if t < t_pov:
        img, (mx, my) = compare_world()
        img = img.copy()
        c = C(img)
        red_circle(c, mx, my, 40, (lt - 0.2) / 0.3, 6, seed=8)
        k = eio((lt - 0.6) / 0.8)
        if k > 0:
            ex, ey = lerp(mx, 540, k), lerp(my, 1100, k)
            c.line([(mx, my + 40), (ex, ey)], RED + (230,), 6)
            if k > 0.95:
                c.poly([(540, 1130), (515, 1085), (565, 1085)], RED + (230,))
        frame = cam(img, 1.0, 540, 960)

        def txt(im):
            d = ImageDraw.Draw(im, "RGBA")
            for y, lab in ((150, "КАК СНИМАЛИ МЫ"), (1830, "КАК СНИМАЛ ОН")):
                kk = env(t, SHOT_START["reverse"] + 0.2, t_pov, 0.3, 0.2)
                shadow_text(d, (540, y), lab, MS(46, 800), WHITE + (int(255 * kk),))
        return frame, txt
    # от первого лица: руки с фотоаппаратом за оградой, подъём к глазу, видоискатель
    lt2 = t - t_pov
    t_look = wt("стоял", 72)
    img = yard_bg().copy()
    yard_people(img, t, clamp((t - t_look) / 0.5))
    img.paste(fence_fg(), (0, 0), fence_fg())
    raise_k = eio((lt2 - 0.75) / 0.45)
    frame = cam(img, lerp(1.0, 1.5, raise_k), 540, lerp(960, 1050, raise_k), hh_amp=lerp(5, 2, raise_k))
    if raise_k < 1:
        c = C(frame, 1)
        y = lerp(1500, 2600, raise_k)
        film_camera(c, 540, y, 1.0)
        pov_hand(c, (60, 2100), (330, y + 40), -0.3, 0.6, 1.0)
        pov_hand(c, (1020, 2100), (750, y + 40), 0.3, 0.6, 1.0)
    frame = viewfinder(frame, raise_k)
    frame = flash(frame, t, wt("стоял", 72) + 0.85, 0.3)

    def txt(im):
        caption(im, t, t_pov + 0.1, t_pov + 1.0, "С ЕГО МЕСТА", y=300, size=70)
    return frame, txt


# ================================================================ 15. «это твой дедушка»
@lru_cache(None)
def kitchen_blur():
    return kitchen_bg(True).filter(ImageFilter.GaussianBlur(14 * S))


GM_PROFILE = PROFILE


def sc_reveal(t, lt, dur):
    img = kitchen_blur().copy()
    c = C(img)
    s = 760
    cx, cy = 640, 820 + math.sin(t * 1.4) * 4
    nod = math.sin(t * 0.9) * 0.6
    pts = [(cx - px * s, cy + py * s) for px, py in GM_PROFILE]
    pts = [(x, y) for x, y in pts]
    rim = (150, 180, 240)
    hair = [(cx - px * s, cy + py * s) for px, py in [(-0.46, 0.1), (-0.48, -0.3), (-0.3, -0.54), (0.0, -0.6),
                                                      (0.26, -0.5), (0.34, -0.36), (0.1, -0.4), (-0.2, -0.32)]]
    bun = (cx + 0.5 * s, cy - 0.42 * s, 0.17 * s, 0.15 * s)
    for dx, col in ((-9, rim), (0, (12, 12, 15))):
        c.poly([(x - dx, y - 4 * (dx != 0)) for x, y in pts], col)
        c.poly([(x - dx, y - 4 * (dx != 0)) for x, y in hair], col)
        c.ell(bun[0] - dx, bun[1], bun[2], bun[3], col)
    img = glow(img, lambda d, k: d.ellipse([(cx - 0.5 * s) * k, (cy - 0.4 * s) * k, (cx - 0.25 * s) * k, (cy + 0.6 * s) * k],
                                           fill=(50, 34, 18)), 40)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 560, 960)
    t_no = wt("он", 76.3)

    def txt(im):
        caption(im, t, t_no, SHOT_START["reveal"] + dur + 0.2, "ОН НЕ УТОНУЛ", y=1560, size=110, accent=(255, 80, 70))
    return frame, txt


# ================================================================ 16. 1985: из арки (от первого лица)
@lru_cache(None)
def court_bg():
    img = to_img(grad((8, 10, 18), (4, 5, 8)))
    c = C(img)
    c.rect(-20, 380, 1100, 1300, (30, 32, 38))             # пятиэтажка
    r = np.random.default_rng(85)
    for i in range(10):
        for j in range(5):
            x, y = 40 + i * 104, 430 + j * 160
            lit = r.random() < 0.18
            c.rect(x, y, x + 56, y + 80, (210, 170, 100) if lit else (14, 16, 22))
    c.rect(440, 1130, 600, 1300, (16, 16, 18))               # подъезд
    c.rect(-20, 1300, 1100, H, (18, 18, 20))
    return img


def arch_mask():
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.rectangle([0, 0, W, H], fill=255)
    d.rectangle([150, 700, 930, H], fill=0)
    d.ellipse([150, 300, 930, 1100], fill=0)
    return m


def sc_night85(t, lt, dur):
    img = courtyard().copy()
    ph = t * 2 * math.pi * 1.7
    kb = clamp(math.sin(ph) * 3)
    kr = clamp(-math.sin(ph) * 3)
    # синий/красный отсвет на стенах
    img = glow(img, lambda d, k: d.ellipse([100 * k, 700 * k, 1000 * k, 1500 * k],
                                           fill=(int(40 * kr), int(30 * kb), int(150 * kb + 30 * kr))), 90, 1.0)
    # фонарик милиционера шарит по окнам
    fx, fy = 380, 1250
    ang = -1.2 + 0.35 * math.sin(lt * 0.9)
    beam = [(fx, fy), (fx + math.cos(ang - 0.08) * 1100, fy + math.sin(ang - 0.08) * 1100),
            (fx + math.cos(ang + 0.08) * 1100, fy + math.sin(ang + 0.08) * 1100)]
    img = props.volumetric(img, beam, (120, 110, 80), 26, 1.0)
    c = C(img)
    police_car(img, 640, 1400, 1.0, kb, kr, facing=-1)
    # милиционеры в фуражках: один светит фонариком, второй переминается и показывает на окна
    p1 = breathe(pose(an=(120, 10), af=(10, 20), neck=-6), t)
    silhouette(img, person_parts(330, 1420, 380, p1, 1, "suit"), base=(10, 12, 18), rim=(110, 140, 255),
               light=(1, -0.3), rim_w=3, fill_light=(int(10 * kr), 4, int(14 * kb)))
    J = skeleton(330, 1420, 380, p1, 1)
    peaked_cap(C(img), J["head"][0], J["head"][1], 380, 1, rim=(110, 140, 255))
    point = pose(an=(150, 4), af=(6, 10))
    p2 = keyed(lt, [(0, pose(an=(6, 20), af=(-4, 18))), (1.6, point), (3.4, point), (4.2, pose(an=(20, 30)))])
    x2 = 1010 - 60 * eio(clamp((lt - 0.4) / 1.0))
    if 0.4 < lt < 1.4:
        p2 = walk((lt - 0.4) * 6, 0.6)
    p2 = breathe(p2, t)
    silhouette(img, person_parts(x2, 1430, 390, p2, -1, "suit"), base=(10, 12, 18), rim=(255, 120, 120),
               light=(-1, -0.3), rim_w=3, fill_light=(int(12 * kr), 4, int(14 * kb)))
    J = skeleton(x2, 1430, 390, p2, -1)
    peaked_cap(C(img), J["head"][0], J["head"][1], 390, -1, rim=(255, 120, 120))
    reflect(img, 1440, 420, 0.5, 5, ripple=2.5, t=t)
    c = C(img)
    rain(c, t, 140, (0, 0, W, H), 7, (170, 180, 210), 70)
    frame = cam(img, lerp(1.0, 1.1, lt / dur), 560, 1150, hh_amp=4)
    # арка вокруг: тёмная кирпичная рамка, рука на стене
    a = np.asarray(frame, np.float32)
    m = np.asarray(arch_mask().filter(ImageFilter.GaussianBlur(10)), np.float32)[:, :, None] / 255
    brick = 14 + 10 * (np.asarray(Image.fromarray((fbm(W, H, [20, 60], 3) * 255).astype(np.uint8)), np.float32)[:, :, None] / 255)
    a = a * (1 - m) + brick * m * np.array([1.1, 1.0, 1.2]) * (0.7 + 0.5 * (kb * np.array([0.5, 0.7, 1.6]) + kr * np.array([1.6, 0.5, 0.5])))
    frame = to_img(a)
    c = C(frame, 1)
    pov_hand(c, (-220, 1320), (110, 1190), -0.6, 0.5, 0.8)

    def txt(im):
        year_tag(im, t, SHOT_START["night85"] + 0.15, wt("15", 80) - 0.1, "1985", "ЕГО ДОЛЖНЫ БЫЛИ ПОСАДИТЬ", y=230)
        caption(im, t, wt("15", 80), wt("за", 81.8) - 0.05, "НА 15 ЛЕТ", y=300, size=110, accent=(255, 80, 70))
        caption(im, t, wt("за", 81.8), SHOT_START["night85"] + dur, "ЗА ТО, ЧЕГО ОН НЕ ДЕЛАЛ", y=300, size=70)
    return frame, txt


# ================================================================ 17. исчез / окно / не решился
@lru_cache(None)
def street_bg():
    img = to_img(grad((8, 10, 16), (2, 2, 4)))
    c = C(img)
    c.rect(0, 1320, W, H, (10, 10, 12))
    for x in (180, 620, 1060):
        c.rect(x - 5, 700, x + 5, 1330, (6, 6, 8))
        c.poly([(x - 30, 700), (x + 30, 700), (x + 18, 680), (x - 18, 680)], (6, 6, 8))
    img = glow(img, lambda d, k: [d.polygon([((x - 30) * k, 700 * k), ((x + 30) * k, 700 * k), ((x + 200) * k, 1330 * k),
                                             ((x - 200) * k, 1330 * k)], fill=(90, 70, 40)) for x in (180, 620, 1060)], 40)
    return img


@lru_cache(None)
def facade_bg():
    img = to_img(grad((10, 12, 20), (4, 4, 8)))
    c = C(img)
    c.rect(-20, 0, 1100, 1500, (30, 32, 38))
    n = fbm(WW, WH, [6, 24, 80], 29)
    a = np.asarray(img, np.float32)
    a[:1500 * S] *= (0.8 + 0.3 * n[:1500 * S, :, None])
    img = to_img(a)
    c = C(img)
    for j in range(8):
        c.line([(0, 30 + j * 210), (W, 30 + j * 210)], (24, 26, 32), 3)
    r = np.random.default_rng(7)
    for i in range(6):
        for j in range(7):
            x, y = 40 + i * 180, 60 + j * 210
            if (i, j) == (3, 3):
                continue
            lit = r.random() < 0.14
            c.rect(x - 6, y - 6, x + 106, y + 146, (46, 48, 54))
            c.rect(x, y, x + 100, y + 140, (196, 150, 90) if lit else (12, 14, 20))
            c.rect(x + 47, y, x + 53, y + 140, (46, 48, 54))
            c.rect(x, y + 56, x + 100, y + 62, (46, 48, 54))
            if lit:
                c.rect(x, y, x + 30, y + 140, (170, 120, 90))
            c.rect(x - 12, y + 146, x + 112, y + 156, (54, 56, 62))
    c.rect(440, 1300, 640, 1500, (18, 16, 16))
    c.rect(452, 1312, 628, 1500, (44, 32, 26))
    c.rect(420, 1280, 660, 1300, (54, 56, 62))
    c.ell(540, 1316, 8, 8, (255, 230, 170))
    c.rect(0, 1500, W, H, (12, 12, 14))
    img = glow(img, lambda d, k: d.ellipse([400 * k, 1250 * k, 680 * k, 1520 * k], fill=(120, 90, 50)), 40)
    return img


def sc_leave(t, lt, dur):
    t_win = wt("а", 84.5) + 0.1
    t_back = wt("вернуться", 88.8) - 0.15
    if t < t_win:     # он уходит в темноту
        img = night_street().copy()
        x = lerp(380, 1000, lt / 2.0)
        p = walk(lt * 5.2, 0.95)
        silhouette(img, person_parts(x, 1330, 520, p, 1, "man"), base=(6, 6, 8), rim=(220, 170, 110), light=(-1, -0.5),
                   rim_w=4, alpha=clamp(1 - (lt - 0.5) / 0.9) * 0.8 + 0.2)
        reflect(img, 1334, 500, 0.6, 4, ripple=2, t=t)
        rain(C(img), t, 160, (0, 0, W, H), 9, (180, 180, 200), 60)
        return cam(img, 1.05, 600, 1050), None
    img = facade_bg().copy()
    c = C(img)
    wx0, wy0, wx1, wy1 = 580, 690, 680, 830
    c.rect(wx0 - 70, wy0 - 70, wx1 + 70, wy1 + 70, (24, 26, 32))
    c.rect(wx0 - 60, wy0 - 60, wx1 + 60, wy1 + 60, (214, 168, 100))
    # бабушка с малышом в окне, покачивает
    rock = math.sin(t * 2.2) * 3
    fp = pose(an=(40, 110), af=(40, 110), nod=rock)
    silhouette(img, squash(body_front(630 + rock, 1010, 300, fp, "grandma"), 630, 1.0), base=(40, 26, 16),
               rim=(255, 210, 140), light=(0, -1), rim_w=1.5)
    c = C(img)
    c.ell(630 + rock * 1.5, 862, 38, 22, (230, 220, 200))
    c.rect(wx0 - 70, wy1 + 60, wx1 + 70, 1000, (24, 26, 32))
    c.rect(wx0 - 66, wy1 + 54, wx1 + 66, wy1 + 70, (40, 40, 46))
    c.rect(wx0 - 60, wy0 - 60, wx1 + 60, wy1 + 60, None, outline=(30, 30, 34), w=12)
    c.rect(wx0 - 60, (wy0 + wy1) / 2 - 6, wx1 + 60, (wy0 + wy1) / 2 + 6, (30, 30, 34))
    img = glow(img, lambda d, k: d.rectangle([(wx0 - 60) * k, (wy0 - 60) * k, (wx1 + 60) * k, (wy1 + 60) * k],
                                             fill=(120, 80, 30)), 50)
    tree(C(img), 230, 1500, 2.2, (6, 6, 8))
    reflect(img, 1500, 400, 0.45, 5, ripple=2, t=t)
    rain(C(img), t, 150, (0, 0, W, H), 11, (170, 170, 190), 55)
    if t < t_back:    # от первого лица: стоит в темноте и смотрит на окно
        k = (t - t_win) / (t_back - t_win)
        frame = cam(img, lerp(1.0, 1.45, eio(k)), 630, lerp(1000, 820, eio(k)), hh_amp=4)
    else:             # шаг к подъезду… и отворачивается
        k = t - t_back
        step = eio(k / 0.7)
        bob = math.sin(k * 7) * 10 * (k < 0.8)
        z = lerp(1.45, 1.6, step)
        frame = cam(img, z, 630, 820 + bob, hh_amp=3)
        turn = eio((k - 1.1) / 0.6)
        if turn > 0:
            out = Image.new("RGB", (W, H), (0, 0, 0))
            out.paste(frame, (int(W * 0.9 * turn), 0))
            frame = to_img(np.asarray(out, np.float32) * (1 - turn * 0.8))

    def txt(im):
        caption(im, t, t_win + 0.4, t_back - 0.1, "С СЫНОМ НА РУКАХ", "и тайной на всю жизнь", y=1650, size=70)
        caption(im, t, t_back + 0.2, SHOT_START["leave"] + dur, "ВЕРНУТЬСЯ НЕ РЕШИЛСЯ", y=1650, size=70)
    return frame, txt


# ================================================================ 18. открытки
CARDS = [("1 сентября", "8:30", "школа №12"), ("18 июня", "14:00", "ЗАГС на Ленина"), ("23 июля", "10:00", "Анапа, пирс")]


def postcard(c, x, y, w, idx, k_text=1.0, ang=0.0, hl=0.0):
    h = w * 0.66
    c.rect(x - w / 2 + 10, y - h / 2 + 16, x + w / 2 + 10, y + h / 2 + 16, (0, 0, 0, 110))
    c.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, (236, 228, 208), r=6)
    c.rect(x + w / 2 - w * 0.22, y - h / 2 + 24, x + w / 2 - 24, y - h / 2 + 24 + w * 0.22, (190, 90, 70), r=4)
    c.line([(x - w * 0.02, y - h / 2 + 30), (x - w * 0.02, y + h / 2 - 30)], (190, 180, 160), 2)
    lines = CARDS[idx]
    fs = int(w * 0.075)
    total = sum(len(s) for s in lines)
    n = int(total * clamp(k_text))
    for i, s in enumerate(lines):
        part = s[:max(0, n)]
        n -= len(s)
        if part:
            c.text(x - w * 0.45, y - h * 0.22 + i * fs * 1.5, part, PFI(fs, 600), (30, 40, 90), anchor="lm")
        if hl > 0 and i in (1, 2) and part == s:
            ww = PFI(fs, 600).getlength(s)
            c.line([(x - w * 0.45, y - h * 0.22 + i * fs * 1.5 + fs * 0.6),
                    (x - w * 0.45 + ww * clamp(hl * 2 - (i - 1)), y - h * 0.22 + i * fs * 1.5 + fs * 0.6)], RED + (230,), 5)


@lru_cache(None)
def park_bg():
    img = to_img(grad((164, 196, 222), (226, 228, 222)))
    c = C(img)
    c.rect(0, 1050, W, H, (104, 136, 74))
    r = np.random.default_rng(66)
    for i in range(7):
        x = r.uniform(0, W)
        c.rect(x - 10, 720, x + 10, 1070, (76, 60, 46))
        c.ell(x, 700, 140, 190, (74, 106, 60))
        c.ell(x - 60, 780, 100, 110, (64, 96, 52))
    n = fbm(WW, WH, [8, 30, 90], 14)
    a = np.asarray(img, np.float32) * (0.9 + 0.2 * n[:, :, None])
    return to_img(a).filter(ImageFilter.GaussianBlur(2.5 * S))


def sc_cards(t, lt, dur):
    t_pov = wt("отправляла", 92) + 0.35
    t_pose = wt("и", 95.8) - 0.1
    if t < t_pov:    # бабушка пишет открытку (сверху)
        img = table_bg().copy()
        c = C(img)
        k = (t - SHOT_START["cards"]) / (t_pov - SHOT_START["cards"] - 0.2)
        postcard(c, 540, 960, 760, 0, k * 1.1)
        px = 540 - 760 * 0.45 + 760 * 0.6 * clamp(k)
        py = 900 + (int(k * 3) % 3) * 70 + math.sin(t * 30) * 5
        c.line([(px, py), (px + 220, py - 300)], (30, 30, 34), 18)
        pov_hand(c, (1150, 1500), (px + 230, py - 120), -2.2, 0.8, 0.85, skin=(200, 166, 146), sleeve=(110, 64, 68))
        return cam(img, lerp(1.1, 1.0, k), 540, 960), None
    if t < t_pose:   # от первого лица: его руки держат открытки
        lt2 = t - t_pov
        img = to_img(grad((26, 24, 26), (8, 8, 10)))
        img = glow(img, lambda d, k: d.ellipse([200 * k, 500 * k, 900 * k, 1400 * k], fill=(90, 70, 40)), 80)
        c = C(img)
        flip = eio((lt2 - 1.5) / 0.4)
        postcard(c, 540, 960, 820, 1, 1.0)
        postcard(c, 540 - 1000 * flip, 940 + 60 * flip, 820, 0, 1.0, hl=(t - wt("где", 94)) / 0.6)
        pov_hand(c, (-100, 2100), (190 - 900 * flip * 0.0, 1180), -0.6, 0.5, 1.0)
        pov_hand(c, (1180, 2100), (890, 1180), 0.6, 0.5, 1.0)
        frame = cam(img, 1.0, 540, 960, hh_amp=4)

        def txt(im):
            caption(im, t, wt("где", 94), t_pose, "ГДЕ И ВО СКОЛЬКО", y=300, size=88)
        return frame, txt
    # «и сама решала, куда нам встать»
    lt3 = t - t_pose
    img = park_bg().copy()
    # незнакомец вдалеке у дерева
    shaded(img, body_front(560, 1110, 190, breathe(pose(lean=4, neck=6), t), "man"),
           dict(PALETTE["man"], eyes=(150, 120, 100), mouth=(160, 120, 110)), (560, 1110), s=S)
    move = eio((lt3 - 0.55) / 0.7)
    for kind, x0, feet, h in [("dad", 560, 1720, 900), ("woman", 720, 1720, 860), ("kid", 640, 1740, 520)]:
        k = {"dad": "young", "kid": "young"}.get(kind, kind)
        x = x0 - 300 * move
        stepk = math.sin(clamp((lt3 - 0.55) / 0.7) * math.pi)
        p = pose(an=(8, 14), af=(8, 14), ln=(6 * stepk, 2), lf=(-6 * stepk, 2), nod=stepk * 2)
        shaded(img, body_front(x, feet - 10 * stepk, h, p, k), PALETTE[kind], (x, feet), s=S)
    gest = keyed(lt3, [(0, pose(an=(10, 20), af=(10, 20))), (0.35, pose(an=(70, 10), af=(10, 20), turn=-1)),
                       (1.1, pose(an=(70, 10), af=(10, 20), turn=-1)), (1.4, pose(an=(10, 20), af=(10, 20), turn=1))])
    gest["turn"] = -1 if lt3 < 1.25 else 1
    shaded(img, body_front(930, 1730, 860, gest, "grandma"), PALETTE["grandma"], (930, 1730), s=S)
    frame = cam(img, 1.0, 540, 1100)
    frame = flash(frame, t, SHOT_START["cards"] + dur - 0.35, 0.3)

    def txt(im):
        caption(im, t, t_pose + 0.2, SHOT_START["cards"] + dur, "КУДА НАМ ВСТАТЬ", "и сама решала…", y=330, size=88)
    return frame, txt


# ================================================================ 19. бабушка смотрит на него
GAZE = [("first_class", 540, 330), ("wedding", 540, 960), ("sea", 540, 1590)]


@lru_cache(None)
def gaze_world():
    img = table_bg().copy()
    pts = []
    for key, x, y in GAZE:
        im, mh, gh = photo(key, 2)
        to_w = place_photo(img, framed(im), x, y, 780, 0)
        pts.append((to_w(mh[0], mh[1], 26), to_w(gh[0], gh[1], 26)))
    return img, pts


def sc_gaze(t, lt, dur):
    world, pts = gaze_world()
    img = world.copy()
    c = C(img)
    t_g, t_line, t_him = wt("бабушка", 99.8), wt("смотрит", 101), wt("смотрит", 102.9)
    for i, ((mx, my), (gx, gy)) in enumerate(pts):
        k = eout((t - t_g - i * 0.12) / 0.3)
        if k > 0:
            c.ell(gx, gy, 46 * k, 52 * k, None, outline=(255, 255, 255, int(220 * k)), w=4)
        k = eio((t - t_line - i * 0.12) / 0.8)
        if k > 0:
            ex, ey = lerp(gx, mx, k), lerp(gy, my, k)
            L = math.hypot(ex - gx, ey - gy)
            n = int(L / 26)
            solid = t > t_him
            for j in range(n):
                if solid or j % 2 == 0:
                    a, b = j / max(1, n), (j + 1) / max(1, n)
                    c.line([(lerp(gx, ex, a), lerp(gy, ey, a)), (lerp(gx, ex, b), lerp(gy, ey, b))], RED + (240,), 5)
        red_circle(c, mx, my, 34, (t - t_him - i * 0.1) / 0.3, 5, seed=20 + i)
    z = lerp(1.0, 1.06, lt / dur)
    frame = cam(img, z, 540, 960)

    def txt(im):
        caption(im, t, t_line, t_him - 0.05, "НЕ В ОБЪЕКТИВ", y=1880 - 60, size=64)
        caption(im, t, t_him, SHOT_START["gaze"] + dur, "ОНА СМОТРИТ НА НЕГО", y=1820, size=74, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 20. вся семья вместе
@lru_cache(None)
def wall_world():
    img = table_bg().copy()
    keys = ["first_class", "wedding", "sea"] + [f"x{i}" for i in range(12)] + ["x1", "x4", "x7", "x10", "x3"]
    marks = []
    for i, key in enumerate(keys):
        x, y = 150 + (i % 4) * 260, 200 + (i // 4) * 380
        im, mh, gh = get_photo(key)
        to_w = place_photo(img, framed(im), x, y, 240, (i * 37 % 7) - 3)
        marks.append(to_w(mh[0], mh[1], 26))
    return img, marks


def sc_together(t, lt, dur):
    world, marks = wall_world()
    img = world.copy()
    t_all = wt("вся", 109)
    k = eio(lt / dur)
    frame = cam(img, lerp(2.4, 1.0, k), lerp(150, 540, k), lerp(200, 960, k))
    if t > t_all - 0.1:
        g = clamp((t - t_all + 0.1) / 0.4)
        a = np.asarray(frame, np.float32)
        a = a * (1 - 0.45 * g)
        frame = to_img(a)

    def txt(im):
        year_tag(im, t, wt("сорок", 104) - 0.05, wt("это", 105) + 0.6, "40 ЛЕТ", "ОН ВСЕГДА БЫЛ РЯДОМ", y=260)
        caption(im, t, wt("единственные", 105.5), t_all - 0.1, "ЕДИНСТВЕННЫЕ СНИМКИ,", "где наша семья вместе", y=1640, size=70)
        caption(im, t, t_all, SHOT_START["together"] + dur, "ВСЯ.", y=980, size=170)
    return frame, txt


# ================================================================ 21. «сфотографируй меня»
def sc_ask(t, lt, dur):
    t_q = wt("спросил", 110)
    t_up = wt("только", 113.5)
    t_turn = wt("сфотографируй", 115) - 0.2
    ask = pose(lean=18, neck=0, ln=(90, 92), lf=(86, 92), an=(60, 40), af=(48, 60))
    nar = breathe(keyed(t, [(SHOT_START["ask"], SIT_N), (t_q, ask), (t_q + 2.5, ask), (t_q + 3.5, SIT_N)]), t)
    low = pose(lean=12, neck=20, nod=18, ln=(88, 90), lf=(84, 90), an=(60, 40), af=(56, 44))
    up = pose(lean=30, neck=-6, ln=(40, 50), lf=(36, 46), an=(40, 30), af=(36, 34))
    stand = pose(lean=6, neck=6, an=(6, 12), af=(-4, 14))
    if t < t_turn:
        gm = keyed(t, [(SHOT_START["ask"], SIT_G), (wt("ничего", 112), SIT_G), (wt("ответила", 112.5), low),
                       (t_up, low), (t_up + 0.45, up), (t_up + 1.0, stand)])
        img = kitchen_world(t, nar, breathe(gm, t, 0.18), photos_x=560, man_in_window=True)
    else:
        k = eio((t - t_turn) / 0.55)
        if k < 0.5:
            parts = squash(person_parts(820, FLOOR_Y, 760, breathe(stand, t), -1, "grandma"), 820, 1 - k * 1.6)
        else:
            fp = breathe(pose(an=(16, 70), af=(16, 70)), t, 0.18)
            parts = squash(person_parts(820, FLOOR_Y, 760, fp, 1, "grandma", "front"), 820, 0.2 + (k - 0.5) * 1.6)
        img = kitchen_world(t, nar, gm_parts=parts, photos_x=560, man_in_window=True)
    frame = cam(img, lerp(1.0, 1.15, lt / dur), 620, 1120)
    return frame, None


# ================================================================ 22. телефон: он смотрит в камеру
def man_face(c, hx, hy, r, up):
    """Лицо незнакомца под фонарём анфас. up: 0 — голова опущена (лицо в тени шляпы), 1 — смотрит в камеру.
    Свет сверху: лоб и скулы светлее, низ лица и глазницы в тени, блик фонаря в глазах."""
    fy = hy + r * 0.28 * (1 - up)
    fh = r * lerp(0.72, 1.02, up)
    skin, dark = (178, 146, 126), (112, 88, 78)
    c.rect(hx - r * 0.36, fy + fh * 0.6, hx + r * 0.36, fy + fh * 1.35, (120, 96, 84))            # шея
    c.ell(hx, fy + fh * 1.05, r * 0.45, r * 0.18, (70, 58, 52, 160))                             # тень подбородка
    for s in (-1, 1):
        c.ell(hx + s * r * 0.78, fy - fh * 0.02, r * 0.13, fh * 0.22, (150, 120, 104))            # уши
        c.poly([(hx + s * r * 0.62, fy - fh * 0.55), (hx + s * r * 0.8, fy - fh * 0.45), (hx + s * r * 0.78, fy + fh * 0.05),
                (hx + s * r * 0.66, fy + fh * 0.1)], (160, 158, 154))                             # седые виски
    c.ell(hx, fy, r * 0.74, fh, skin)
    c.ell(hx, fy - fh * 0.3, r * 0.6, fh * 0.55, (196, 166, 146, 120))                            # свет на лбу и скулах
    for s in (-1, 1):
        c.ell(hx + s * r * 0.56, fy + fh * 0.12, r * 0.2, fh * 0.42, dark + (90,))                # впалые щёки
    c.ell(hx, fy + fh * 0.62, r * 0.6, fh * 0.42, dark + (110,))                                  # низ лица в тени
    c.ell(hx, fy + fh * 0.6, r * 0.5, fh * 0.34, (160, 156, 150, 80))                             # седая щетина
    ey = fy - fh * 0.12
    if up > 0.45:
        k = clamp((up - 0.45) / 0.4)
        for s in (-1, 1):
            ex = hx + s * r * 0.29
            c.ell(ex, ey + fh * 0.02, r * 0.24, fh * 0.16, (96, 74, 66, 120))                    # глазница
            c.ell(ex, ey, r * 0.12, fh * 0.055 * k + 0.5, (186, 178, 168))                     # белок
            c.ell(ex, ey, r * 0.058, fh * 0.055 * k + 0.5, (40, 32, 28))                       # радужка
            c.ell(ex + r * 0.02, ey - fh * 0.02, r * 0.016, r * 0.016 * k + 0.3, (255, 250, 236))   # блик фонаря
            c.line([(ex - r * 0.17, ey + fh * 0.08), (ex + r * 0.12, ey + fh * 0.1)], dark + (150,), r * 0.02)  # мешки
            c.line([(ex - r * 0.18, ey - fh * 0.15), (ex + r * 0.16, ey - fh * 0.17 + s * fh * 0.01)], (150, 148, 144),
                   r * 0.055)                                                                   # седые брови
    c.poly([(hx - r * 0.05, ey), (hx + r * 0.05, ey), (hx + r * 0.1, fy + fh * 0.3), (hx - r * 0.1, fy + fh * 0.3)],
           (190, 160, 140))                                                                     # спинка носа
    c.ell(hx, fy + fh * 0.33, r * 0.15, fh * 0.06, dark + (170,))                                 # тень под носом
    for s in (-1, 1):
        c.line([(hx + s * r * 0.16, fy + fh * 0.3), (hx + s * r * 0.26, fy + fh * 0.6)], dark + (150,), r * 0.03)   # носогубные
    c.line([(hx - r * 0.2, fy + fh * 0.52), (hx, fy + fh * 0.54), (hx + r * 0.2, fy + fh * 0.52)], (96, 70, 64), r * 0.04)
    c.line([(hx - r * 0.12, fy + fh * 0.6), (hx + r * 0.12, fy + fh * 0.6)], (170, 140, 124, 120), r * 0.02)
    # тень от полей шляпы: при поднятой голове едва касается бровей
    top = fy - fh * 0.82
    sh_bottom = lerp(fy + fh * 0.75, ey - fh * 0.1, up)
    for j, (ext, al) in enumerate(((1.25, 50), (1.12, 70), (1.0, 120), (0.88, 200))):
        pts = [(hx + math.cos(a) * r * 0.8, top + math.sin(a) * (sh_bottom - top) * ext) for a in np.linspace(0, math.pi, 24)]
        c.poly(pts, (28, 22, 20, al))
    c.ell(hx, top, r * 1.32, r * 0.24, (66, 66, 72))
    c.ell(hx, top - r * 0.03, r * 1.28, r * 0.2, (84, 84, 90))
    c.poly([(hx - r * 0.78, top), (hx - r * 0.7, top - r * 0.74), (hx - r * 0.2, top - r * 0.86), (hx + r * 0.2, top - r * 0.86),
            (hx + r * 0.7, top - r * 0.74), (hx + r * 0.78, top)], (90, 90, 96))
    c.poly([(hx - r * 0.2, top - r * 0.86), (hx + r * 0.2, top - r * 0.86), (hx + r * 0.1, top - r * 0.6), (hx - r * 0.1, top - r * 0.6)],
           (110, 110, 116))
    c.rect(hx - r * 0.77, top - r * 0.2, hx + r * 0.77, top - r * 0.04, (36, 36, 40))


def man_under_lamp(img, tc, t, up):
    """Мужчина в сером пальто под фонарём (координаты сцены телефона)."""
    x, feet, h = 770, 1180, 230
    p = breathe(pose(lean=2, an=(6, 10), af=(6, 10)), t, 0.2)
    parts = tc.parts(body_front(x, feet, h, p, "man"))
    pal = dict(PALETTE["man"])
    pal["hat"] = None
    pal["skin"] = None
    pal["eyes"] = None
    pal["mouth"] = None
    shaded(img, [q for q in parts if q[2] not in ("skin", "hat", "eyes", "mouth") or q[0] == "cap"], pal,
           tc.p(x, feet), light=(0.0, -1.0), s=S, shadow=0.5)
    sy = feet - h * (0.245 + 0.235 + 0.012 + 0.29)
    tc.poly([(x - h * 0.1, sy - h * 0.01), (x - h * 0.02, sy + h * 0.13), (x - h * 0.035, sy - h * 0.025)], (70, 72, 78))
    tc.poly([(x + h * 0.1, sy - h * 0.01), (x + h * 0.02, sy + h * 0.13), (x + h * 0.035, sy - h * 0.025)], (70, 72, 78))
    tc.poly([(x - h * 0.03, sy - h * 0.02), (x + h * 0.03, sy - h * 0.02), (x, sy + h * 0.1)], (40, 38, 40))
    for k in range(3):
        tc.ell(x + h * 0.012, sy + h * (0.16 + k * 0.07), h * 0.008, h * 0.008, (40, 40, 44))
    hx, hy = x, feet - h * (0.245 + 0.235 + 0.012 + 0.29 + 0.035 + 0.066 * 0.95)
    X, Y = tc.p(hx, hy)
    man_face(C(img), X, Y, h * 0.066 * tc.sc * 1.1, up)


def phone_feed(t, zoom, focus, up):
    """Кадр камеры телефона: бабушка у окна, за окном дерево, фонарь и он."""
    tx, ty = lerp(540, 770, clamp((zoom - 1) / 1.5)), lerp(960, 1180 - 230 * 0.86, clamp((zoom - 1) / 1.5))
    if zoom > 3.2:
        ty = lerp(1180 - 230 * 0.86, 1180 - 230 * 0.9, clamp((zoom - 3.2) / 3))
    tc_out = TC(Image.new("RGB", (WW, WH)), tx, ty, zoom)
    img = tc_out.img
    # улица
    a = grad((14, 20, 36), (10, 10, 14))
    img.paste(to_img(a))
    c = tc_out
    c.rect(-2000, 1120, 3000, 3000, (24, 24, 28))
    for i in range(6):
        c.rect(-400 + i * 330, 560, -400 + i * 330 + 280, 1120, (18, 20, 26))
        for j in range(5):
            if (i * 7 + j * 3) % 6 == 0:
                c.rect(-370 + i * 330, 600 + j * 100, -340 + i * 330, 640 + j * 100, (200, 160, 90))
    # дерево
    tree(c, 560, 1140, 1.3, (10, 10, 12))
    # фонарь
    c.rect(836, 760, 846, 1180, (12, 12, 14))
    c.poly([(800, 770), (880, 770), (866, 748), (814, 748)], (12, 12, 14))
    lx, ly = tc_out.p(841, 772)
    img = glow(img, lambda d, k: [d.ellipse([(lx - 40 * zoom) * k, (ly - 10 * zoom) * k, (lx + 40 * zoom) * k, (ly + 20 * zoom) * k],
                                            fill=(255, 220, 150)),
                                  d.polygon([((lx - 30 * zoom) * k, ly * k), ((lx + 30 * zoom) * k, ly * k),
                                             ((lx + 200 * zoom) * k, (ly + 420 * zoom) * k), ((lx - 260 * zoom) * k, (ly + 420 * zoom) * k)],
                                            fill=(110, 84, 46))], 30 * min(zoom, 3), 1.0 / (1 + 0.12 * (zoom - 1)))
    tc_out.img = img
    tc_out.d = ImageDraw.Draw(img, "RGBA")
    man_under_lamp(img, tc_out, t, up)
    gy = tc_out.p(0, 1120)[1]
    if 0 < gy < H - 10:
        reflect(img, gy, min(500, H - gy), 0.5, 3 + zoom, ripple=1.5 * zoom, t=t)
    cone = [tc_out.p(811, 772), tc_out.p(871, 772), tc_out.p(1041, 1192), tc_out.p(581, 1192)]
    rain_in_cone(img, t, cone, n=int(160 + 40 * min(zoom, 4)), seed=5)
    out_blur = (1 - focus) * 6
    if out_blur > 0.3:
        img = img.filter(ImageFilter.GaussianBlur(out_blur * S))
    # интерьер: рама окна и бабушка
    if zoom < 2.2:
        lay = Image.new("RGBA", (WW, WH), (0, 0, 0, 0))
        ti = TC(lay, tx, ty, zoom)
        ti.rect(-2000, -2000, 3000, 300, (22, 22, 26, 255))
        ti.rect(-2000, 1250, 3000, 4000, (22, 22, 26, 255))
        ti.rect(-2000, -2000, 420, 4000, (22, 22, 26, 255))
        ti.rect(1000, -2000, 3000, 4000, (22, 22, 26, 255))
        for x0, x1 in ((404, 436), (690, 724), (984, 1016)):
            ti.rect(x0, 284, x1, 1266, (46, 46, 52, 255))
        ti.rect(404, 284, 1016, 316, (46, 46, 52, 255))
        ti.rect(390, 1250, 1030, 1290, (56, 56, 62, 255))
        gp = breathe(pose(an=(16, 70), af=(16, 70)), t, 0.18)
        parts = ti.parts(body_front(330, 1900, 1150, gp, "grandma"))
        shaded(lay, parts, dict(PALETTE["grandma"], eyes=(120, 90, 80), mouth=(140, 100, 90)), None, light=(0.6, -0.4), s=S)
        b = focus * 10
        if b > 0.3:
            lay = lay.filter(ImageFilter.GaussianBlur(b * S))
        img.paste(lay, (0, 0), lay)
    return img.resize((W, H), Image.BILINEAR)


def phone_ui(im, zoom_label, focus_box=None):
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 150], fill=(0, 0, 0, 120))
    d.rectangle([0, 1600, W, H], fill=(0, 0, 0, 140))
    d.ellipse([540 - 80, 1720 - 80, 540 + 80, 1720 + 80], outline=(255, 255, 255, 240), width=8)
    d.ellipse([540 - 64, 1720 - 64, 540 + 64, 1720 + 64], fill=(255, 255, 255, 235))
    for x, lab, act in ((300, "ВИДЕО", False), (540, "ФОТО", True), (780, "ПОРТРЕТ", False)):
        d.text((x, 1630), lab, font=MS(30, 700 if act else 500), fill=(255, 210, 60) if act else (230, 230, 230), anchor="mm")
    d.rounded_rectangle([480, 1500, 600, 1570], 35, fill=(0, 0, 0, 140))
    d.text((540, 1535), zoom_label, font=MS(34, 700), fill=(255, 210, 60), anchor="mm")
    d.ellipse([60, 60, 100, 100], outline=(255, 255, 255, 200), width=3)
    if focus_box:
        x, y, k = focus_box
        s = lerp(140, 100, k)
        d.rectangle([x - s, y - s, x + s, y + s], outline=(255, 210, 60, int(255 * (1 - k * 0.5))), width=4)


def sc_phone(t, lt, dur):
    t_screen = wt("экране", 119.9)
    t_road = wt("через", 120.7)
    t_lamp = wt("фонарем", 121.6)
    t_up = wt("впервые", 125.3) - 0.2
    zoom = 1.0 + 2.2 * eio((t - t_lamp) / 1.2) + 3.6 * eio((t - wt("пожилой", 122.8)) / 2.2) + 1.2 * eio((t - t_up) / 2.6)
    focus = eio((t - t_road) / 0.6)
    up = eio((t - t_up - 0.4) / 2.2)
    feed = phone_feed(t, zoom, focus, up)
    zl = "1×" if zoom < 1.5 else ("3×" if zoom < 4 else ("5×" if zoom < 6 else "8×"))
    fb = (735, 1000, clamp((t - t_road) / 0.6)) if t_road < t < t_road + 1.0 else None
    phone_ui(feed, zl, fb)
    k = eio((t - t_screen + 0.3) / 0.8)          # телефон заполняет кадр
    rise = eio(lt / 1.0)
    if k >= 1:
        frame = feed
    else:
        # комната от первого лица: рука с телефоном
        room = kitchen_blur().copy()
        frame = cam(room, 1.2, 760, 1000)
        pw = lerp(560, 1180, k)
        ph = pw * 2.0
        px, py = 540, lerp(lerp(2700, 1080, rise), 960, k)
        c = C(frame, 1)
        c.rect(px - pw / 2 - 14, py - ph / 2 - 14, px + pw / 2 + 14, py + ph / 2 + 14, (16, 16, 18), r=50)
        scr = feed.resize((int(pw), int(ph)))
        frame.paste(scr, (int(px - pw / 2), int(py - ph / 2)))
        c = C(frame, 1)
        pov_hand(c, (1150, 2300), (px + pw / 2 - 10, py + ph * 0.3), 2.6, 0.7, 1.1 * (1 - k * 0.4),
                 skin=(206, 168, 140), sleeve=(40, 40, 46))
    hb = 1 + 0.004 * math.sin(t * 2 * math.pi * 1.1) * (t > t_up)
    if hb != 1:
        frame = frame.resize((int(W * hb), int(H * hb)), Image.BILINEAR).crop((0, 0, W, H))
    return frame, None


# ================================================================ 23. вопрос
def sc_question(t, lt, dur):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    c = C(img, 1)
    dust(c, t, 25, seed=4, alpha=60)

    def txt(im):
        k = eout((lt - 0.25) / 0.6)
        d = ImageDraw.Draw(im, "RGBA")
        shadow_text(d, (540, 900), "ВЫЙТИ К НЕМУ?", MS(104, 800), WHITE + (int(255 * k),))
        k2 = eout((lt - 0.9) / 0.6)
        d.text((540, 1030), "напиши в комментариях", font=PFI(52, 500), fill=CREAM + (int(220 * k2),), anchor="mm")
    return img, txt


# ================================================================ монтаж
def _photo_scene(name):
    return lambda t, lt, dur: sc_photo(t, lt, dur, name)


SC = {name: (globals()["sc_" + name] if "sc_" + name in globals() else _photo_scene(name)) for _, name in SCENES}
SHOTS = []
SHOT_START = {}
for i, (a, name) in enumerate(SCENES):
    start = max(0.0, v(a))
    end = v(SCENES[i + 1][0]) if i + 1 < len(SCENES) else DURATION
    SHOTS.append((start, end, name))
    SHOT_START[name] = start

SHAKES += [(word_t("никто", 5), 6, 0.4), (word_t("сотни", 62), 10, 0.35), (word_t("он", 76.3), 16, 0.4),
           (word_t("15", 80), 8, 0.4), (word_t("вся", 109), 6, 0.6)]

WARM = {"hook": 0.6, "scanner": 0.0, "photo1": 0.4, "photo2": 0.4, "photo3": 0.4, "floor": 0.6, "aging": 0.3,
        "uncanny": 0.2, "grandpa": -0.6, "burn": 0.8, "kitchen": 0.3, "box": 0.4, "inside": 0.6, "reverse": 0.3,
        "reveal": 0.0, "night85": -0.8, "leave": -0.4, "cards": 0.5, "gaze": 0.6, "together": 0.8, "ask": 0.2,
        "phone": 0.0, "question": 0.0}


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
    img = post(img, t, WARM.get(name, 0.0), grain=6 if name != "phone" else 9)
    if txt:
        txt(img)
    draw_quote(img, t)
    if t < 0.12:
        img = to_img(np.asarray(img, np.float32) * (t / 0.12))
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
                           "-crf", "19", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    # 3 процесса и перезапуск воркеров: кэши сцен не раздувают память до OOM
    with Pool(3, maxtasksperchild=60) as pool:
        for n, buf in enumerate(pool.imap(frame, range(first, last), chunksize=4), first):
            ff.stdin.write(buf)
            if n % (FPS * 10) == 0:
                print(f"  кадр {n}/{total}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

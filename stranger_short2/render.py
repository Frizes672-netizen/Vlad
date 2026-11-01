"""Рендер шортса «Человек на всех фотографиях. Часть 2 (финал)».

Тот же стиль и движок, что в первой части (lib1.py): старые снимки, ночные сцены, силуэты
с контровым светом, дождь, мокрый асфальт, сцены от первого лица.
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
from engine import (CREAM, CUR_T, MS, PALETTE, PF, PFI, RED, S, SHAKES, WH, WW, C, back, body_front, breathe, cam,
                    clamp, dust, ease, eio, env, eout, fbm, framed, glow, grad, keyed, lerp, mix, person_parts, photo,
                    photo_post, photo_rev, place_photo, pose, post, radial, red_circle, rot_pts, shaded, silhouette,
                    skeleton, squash, to_img, walk, word_t)
from lib1 import (LAMP_RIM, NIGHT_RIM, WHITE, _cap, caption, draw_quote, facade_bg, film_camera, flash, phone_feed,
                  phone_ui, portrait, pov_hand, shadow_text, viewfinder, year_tag)
from props import bloom, hallway, night_street, rain, reflect, tree, volumetric
from timeline import DURATION, FPS, H, SCENES, W, v

YOUNG_SKIN = (204, 166, 140)


def wt(word, after):
    return word_t(word, after)


def rot_parts(parts, cx, cy, ang):
    """Поворот всей фигуры вокруг точки (для падения)."""
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


def under_lamp(t, zoom, up=1.0):
    """Он под фонарём под дождём (кадр без интерфейса камеры)."""
    return phone_feed(t, zoom, 1.0, up)


# ================================================================ 1. хук
def sc_hook(t, lt, dur):
    t_run, t_man = 1.45, 2.85
    t_she = wt("она", 7.5)
    if t < t_run:                                   # финальный кадр первой части
        frame = under_lamp(t, 8.0, 1.0)
        phone_ui(frame, "8×")
        if t > t_run - 0.3:                         # телефон опускается
            k = eio((t - (t_run - 0.3)) / 0.3)
            out = Image.new("RGB", (W, H))
            out.paste(frame.filter(ImageFilter.GaussianBlur(10 * k)), (0, int(H * k)))
            frame = out

        def txt(im):
            d = ImageDraw.Draw(im, "RGBA")
            k = env(t, 0.1, t_run - 0.2, 0.25, 0.2)
            shadow_text(d, (540, 300), "ЧАСТЬ 2", MS(120, 800), WHITE + (int(255 * k),))
            w = d.textlength("ФИНАЛ", font=MS(40, 800))
            d.rectangle([540 - w / 2 - 24, 380, 540 + w / 2 + 24, 440], fill=RED + (int(230 * k),))
            d.text((540, 410), "ФИНАЛ", font=MS(40, 800), fill=WHITE + (int(255 * k),), anchor="mm")
        return frame, txt
    if t < t_man:                                   # от первого лица: выбегаю под дождь
        k = (t - t_run) / (t_man - t_run)
        img = night_street().copy()
        reflect(img, 1334, 500, 0.6, 4, ripple=2, t=t)
        rain(C(img), t, 200, (0, 0, W, H), 3, (190, 196, 220), 90)
        frame = cam(img, lerp(1.0, 2.2, eio(k)), 640, lerp(1050, 980, k), hh_amp=14)
        if k < 0.35:                                # дверь подъезда распахивается
            dk = eio(k / 0.35)
            out = frame.copy()
            dd = ImageDraw.Draw(out)
            dd.rectangle([0, 0, int(540 * (1 - dk)), H], fill=(14, 12, 12))
            dd.rectangle([int(540 + 540 * dk), 0, W, H], fill=(14, 12, 12))
            frame = out
        return frame, None
    k = (t - t_man) / (t_she - t_man)
    zoom = lerp(3.6, 5.2, eio(k)) + 1.6 * eio((t - wt("тюрьму", 5.5)) / 1.8)
    frame = under_lamp(t, zoom, 1.0)
    if t > t_she - 0.05:                            # вспышка: бабушка в окне
        img = facade_bg().copy()
        c = C(img)
        c.rect(520, 630, 740, 890, (214, 168, 100))
        silhouette(img, body_front(630, 1000, 300, pose(an=(14, 40), af=(14, 40)), "grandma"), base=(30, 20, 14),
                   rim=(255, 210, 140), light=(0, -1), rim_w=1.5)
        c = C(img)
        c.rect(410, 890, 850, 1000, (30, 32, 38))
        c.rect(520, 630, 740, 890, None, outline=(30, 30, 34), w=12)
        img = glow(img, lambda d, kk: d.rectangle([460 * kk, 570 * kk, 800 * kk, 950 * kk], fill=(140, 90, 30)), 50)
        frame = cam(img, lerp(2.0, 2.3, (t - t_she) / 1.0), 630, 760)
        frame = flash(frame, t, t_she, 0.25)

    def txt(im):
        caption(im, t, t_she, 99, "НЕ Я. А ОНА.", "в тюрьму должна была сесть…", y=1560, size=120, accent=(255, 80, 70))
    return frame, txt


# ================================================================ 2. под фонарём
def sc_lamp(t, lt, dur):
    t_dad, t_win = wt("вблизи", 11.9) - 0.1, wt("и", 15.2) - 0.1
    if t < t_dad:
        return under_lamp(t, lerp(4.0, 5.6, lt / (t_dad - SHOT_START["lamp"])), 1.0), None
    if t < t_win:                                   # профиль: он и папа
        img = to_img(grad((12, 12, 14), (4, 4, 6)))
        k = eio((t - t_dad) / 1.4)
        st, pts = portrait("stranger")
        fa, _ = portrait("father")
        place_photo(img, framed(st, 10), lerp(290, 540, k), 900, 460, lerp(-3, 0, k))
        place_photo(img, framed(fa, 10), lerp(790, 540, k), 900, 460, lerp(3, 0, k), alpha=lerp(1.0, 0.55, k))
        frame = cam(img, lerp(1.0, 1.1, k), 540, 900)

        def txt(im):
            caption(im, t, t_dad + 0.2, t_win, "ЕЩЁ БОЛЬШЕ ПОХОЖ НА ПАПУ", y=1500, size=70)
        return frame, txt
    # через его плечо: окно, где стоит бабушка
    img = facade_bg().copy()
    c = C(img)
    c.rect(520, 630, 740, 890, (214, 168, 100))
    silhouette(img, body_front(630, 1000, 300, breathe(pose(an=(14, 40), af=(14, 40)), t), "grandma"), base=(30, 20, 14),
               rim=(255, 210, 140), light=(0, -1), rim_w=1.5)
    c = C(img)
    c.rect(410, 890, 850, 1000, (30, 32, 38))
    c.rect(520, 630, 740, 890, None, outline=(30, 30, 34), w=12)
    c.rect(520, 754, 740, 766, (30, 30, 34))
    img = glow(img, lambda d, kk: d.rectangle([460 * kk, 570 * kk, 800 * kk, 950 * kk], fill=(140, 90, 30)), 50)
    tree(C(img), 160, 1500, 2.2, (6, 6, 8))
    reflect(img, 1500, 400, 0.45, 5, ripple=2, t=t)
    p = breathe(pose(nod=-10, an=(6, 10), af=(6, 10)), t)
    silhouette(img, body_front(800, 2700, 1500, p, "man", back=True), base=(8, 8, 10), rim=(255, 200, 130),
               light=(-0.3, -1), rim_w=6)
    rain(C(img), t, 170, (0, 0, W, H), 11, (170, 170, 190), 60)
    frame = cam(img, lerp(1.0, 1.12, (t - t_win) / 4), 640, 960, hh_amp=3)
    return frame, None


# ================================================================ 3. ночная аптека, 1985
@lru_cache(None)
def pharmacy_bg():
    img = to_img(grad((44, 56, 58), (20, 24, 26)))
    c = C(img)
    for j in range(5):                              # полки с флаконами
        y = 520 + j * 150
        c.rect(420, y, 1080, y + 12, (70, 60, 50))
        r = np.random.default_rng(j)
        x = 430
        while x < 1060:
            w = r.uniform(18, 36)
            h = r.uniform(50, 110)
            col = [(150, 90, 50), (60, 110, 80), (200, 196, 186), (90, 60, 40), (120, 140, 160)][int(r.integers(0, 5))]
            c.rect(x, y - h, x + w, y, col + (220,), r=4)
            c.rect(x + w * 0.3, y - h - 12, x + w * 0.7, y - h, (60, 56, 50))
            x += w + r.uniform(4, 12)
    c.rect(0, 1560, W, H, (40, 44, 42))             # кафель на полу
    for i in range(0, W, 90):
        c.line([(i, 1560), (i - 200, H)], (34, 38, 36), 2)
    for j in range(1560, H, 70):
        c.line([(0, j), (W, j)], (34, 38, 36), 2)
    c.rect(30, 600, 330, 1560, (16, 24, 34))        # дверь со стеклом, за ней ночь
    c.rect(50, 620, 310, 1100, (24, 40, 60))
    c.rect(170, 620, 182, 1100, (30, 34, 40))
    c.text(180, 560, "АПТЕКА", MS(46, 800), (120, 230, 150))
    c.poly([(150, 420), (210, 420), (210, 380), (250, 380), (250, 420), (310, 420), (310, 470), (250, 470), (250, 510),
            (210, 510), (210, 470), (150, 470)], (60, 220, 110))
    for x in (300, 700):                            # лампы дневного света
        c.rect(x, 140, x + 300, 160, (230, 250, 240))
    img = glow(img, lambda d, k: [d.rectangle([300 * k, 120 * k, 1000 * k, 200 * k], fill=(110, 140, 130)),
                                  d.ellipse([120 * k, 350 * k, 340 * k, 560 * k], fill=(30, 140, 60))], 60)
    return img


def counter(c):
    c.rect(420, 1240, 1080, 1560, (70, 76, 74))
    c.rect(420, 1222, 1080, 1244, (150, 170, 176))  # стекло прилавка
    c.rect(436, 1260, 1064, 1380, (40, 54, 58))
    c.rect(880, 1150, 1000, 1224, (50, 50, 54), r=6)  # касса
    c.rect(896, 1162, 984, 1196, (110, 170, 120))


def sc_pharmacy(t, lt, dur):
    t_in = wt("ворвался", 23) - 0.15
    t_def = wt("защищалась", 26)
    t_fall = wt("упал", 27.2)
    img = pharmacy_bg().copy()
    c = C(img)
    flick = 0.85 + 0.15 * (math.sin(t * 37) > 0.7)
    if t > t_in - 0.1:                              # дверь распахнута
        dk = eio((t - t_in + 0.1) / 0.25)
        c.rect(30, 600, 30 + 300 * (1 - dk) + 20, 1560, (16, 24, 34))
    # бабушка (молодая) за прилавком
    g_idle = pose(an=(30, 60), af=(26, 64))
    g_back = pose(lean=-10, an=(70, 30), af=(66, 34), nod=-6)
    g_push = pose(lean=14, an=(95, 4), af=(88, 8))
    gp = keyed(t, [(SHOT_START["pharmacy"], g_idle), (t_in, g_idle), (t_in + 0.4, g_back), (t_def - 0.1, g_back),
                   (t_def + 0.25, g_push), (t_fall + 0.6, g_push), (t_fall + 1.4, pose(lean=4, an=(40, 100), af=(40, 100)))])
    silhouette(img, person_parts(860, 1560, 760, breathe(gp, t), -1, "grandma"), base=(14, 16, 18),
               rim=(200, 240, 230), light=(0.3, -1), rim_w=4, fill_light=(6, 10, 10))
    c = C(img)
    counter(c)
    # пьяный с ножом: врывается, шатается, бросается, падает
    if t > t_in:
        k = clamp((t - t_in) / (t_def - t_in))
        x = lerp(120, 560, eio(k)) + math.sin(t * 5) * 12
        if t < t_fall:
            p = walk((t - t_in) * 7, 0.8)
            p["lean"] += 10 + 4 * math.sin(t * 3)
            p["an"] = (95, 10)
            if t > t_def - 0.3:
                p = pose(lean=24, an=(110, 4), af=(40, 30), ln=(30, 20), lf=(-20, 10))
            parts = person_parts(x, 1560, 780, p, 1, "suit")
            J = skeleton(x, 1560, 780, p, 1)
        else:
            x = lerp(560, 520, clamp((t - t_fall) / 0.4))
            fk = eio((t - t_fall) / 0.55)
            p = pose(lean=-6, an=(150, 20), af=(120, 20), ln=(20, 10), lf=(-10, 10))
            parts = rot_parts(person_parts(x, 1560, 780, p, 1, "suit"), x - 40, 1560, -88 * fk)
            J = None
        silhouette(img, parts, base=(10, 10, 12), rim=(200, 240, 230), light=(0.4, -1), rim_w=4)
        if J is not None:                           # нож
            wx, wy = J["an_wr"]
            c = C(img)
            c.line([(wx, wy), (wx + 90, wy - 20)], (220, 230, 230), 6)
    frame = cam(img, lerp(1.0, 1.08, lt / dur), 560, 1150, hh_amp=3 if t > t_in else 1)
    frame = to_img(np.asarray(frame, np.float32) * flick)

    def txt(im):
        year_tag(im, t, SHOT_START["pharmacy"] + 0.1, t_in - 0.1, "1985", "НОЧНАЯ АПТЕКА", y=230)
        caption(im, t, t_in, t_def - 0.05, "ПЬЯНЫЙ С НОЖОМ", y=280, size=80, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 4. после: свидетелей нет / папе два года
def crib_room(t):
    img = to_img(grad((16, 18, 30), (6, 6, 10)))
    c = C(img)
    c.rect(0, 1500, W, H, (14, 12, 14))
    c.rect(240, 900, 840, 1420, None, outline=(60, 50, 44), w=10)
    for i in range(14):
        x = 270 + i * 42
        c.rect(x, 900, x + 10, 1420, (60, 50, 44))
    c.rect(220, 1400, 860, 1440, (60, 50, 44))
    c.ell(540, 1330, 150, 60, (200, 196, 210))      # малыш под одеялом
    c.ell(430, 1290, 46, 40, (220, 200, 186))
    a = t * 0.8                                      # мобиль над кроваткой
    c.line([(540, 600), (540, 720)], (120, 110, 100), 3)
    for k in range(4):
        x = 540 + math.cos(a + k * math.pi / 2) * 110
        c.line([(540, 720), (x, 760)], (120, 110, 100), 2)
        c.ell(x, 790, 22, 22, [(220, 120, 120), (120, 160, 220), (230, 210, 120), (140, 200, 140)][k])
    img = glow(img, lambda d, k: d.ellipse([860 * k, 1380 * k, 1040 * k, 1520 * k], fill=(160, 120, 60)), 80)
    return img


def sc_after(t, lt, dur):
    t_crib = wt("а", 33.3)
    if t < t_crib:
        img = pharmacy_bg().copy()
        p = breathe(pose(lean=4, an=(40, 120), af=(40, 120), nod=10), t, 0.5, 1.6)
        silhouette(img, person_parts(860, 1560, 760, p, -1, "grandma"), base=(14, 16, 18), rim=(200, 240, 230),
                   light=(0.3, -1), rim_w=4)
        c = C(img)
        counter(c)
        dead = rot_parts(person_parts(520, 1560, 780, pose(lean=-6, an=(150, 20), af=(120, 20)), 1, "suit"), 480, 1560, -88)
        silhouette(img, dead, base=(10, 10, 12), rim=(200, 240, 230), light=(0.4, -1), rim_w=4)
        flick = 0.6 + 0.4 * (math.sin(t * 23) * math.sin(t * 7) > -0.3)
        frame = cam(img, lerp(1.05, 1.18, lt / 2.5), 650, 1250, hh_amp=2)
        a = np.asarray(frame, np.float32) * flick
        t_jail = wt("ее", 32)
        k = eout((t - t_jail) / 0.35)
        if k > 0:                                    # тень решётки
            for i in range(-1, 8):
                x = int(i * 160 + (1 - k) * 600)
                a[:, max(0, x):max(0, x + 46)] *= 0.25
        frame = to_img(a)

        def txt(im):
            caption(im, t, SHOT_START["after"] + 0.1, t_jail - 0.05, "СВИДЕТЕЛЕЙ НЕ БЫЛО", y=1700, size=74)
            caption(im, t, t_jail, t_crib, "ЕЁ БЫ ПОСАДИЛИ", y=1700, size=88, accent=(255, 90, 80))
        return frame, txt
    frame = cam(crib_room(t), lerp(1.1, 1.2, (t - t_crib) / 2), 540, 1100)

    def txt(im):
        caption(im, t, t_crib + 0.2, 99, "ПАПЕ — 2 ГОДА", y=380, size=96)
    return frame, txt


# ================================================================ 5. дед стирает отпечатки и исчезает
def sc_wipe(t, lt, dur):
    t_own = wt("оставил", 38.5)
    t_gone = wt("исчез", 40.5)
    if t < t_gone:                                  # от первого лица: стекло прилавка сверху
        img = to_img(grad((60, 80, 84), (30, 40, 44)))
        a = np.asarray(img, np.float32) + fbm(WW, WH, [6, 30], 5)[:, :, None] * 20
        img = to_img(a)
        c = C(img)
        wipe_x = lerp(-200, 1300, eio((t - SHOT_START["wipe"]) / (t_own - SHOT_START["wipe"] - 0.2)))
        for i, (x, y) in enumerate(((380, 820), (620, 900), (470, 1180), (720, 1250))):
            if x > wipe_x:                          # её смазанные отпечатки
                for f_ in range(4):
                    c.ell(x - 60 + f_ * 36, y - 80 - abs(f_ - 1.5) * 16, 16, 24, (200, 220, 220, 70))
                c.ell(x, y, 60, 70, (200, 220, 220, 60))
        if t > t_own + 0.3:                         # его ладонь осталась на стекле
            for f_ in range(4):
                c.ell(500 - 70 + f_ * 46, 1450 - 120 - abs(f_ - 1.5) * 22, 20, 32, (210, 230, 230, 110))
            c.ell(500, 1450, 80, 90, (210, 230, 230, 90))
        if t < t_own:
            hx = wipe_x
            c.poly(_cap((hx, 1000), (hx + 40, 1100), 120, 120), (150, 150, 150))   # тряпка
            pov_hand(c, (hx + 300, 2200), (hx + 60, 1150), -0.4, 0.8, 1.1, skin=YOUNG_SKIN)
        else:
            k = eio((t - t_own) / 0.4)
            pov_hand(c, (800, 2300), (500, lerp(1900, 1450, k)), 0.0, 0.0, 1.3, skin=YOUNG_SKIN)
        frame = cam(img, 1.0, 540, 960, hh_amp=4)

        def txt(im):
            caption(im, t, SHOT_START["wipe"] + 0.2, t_own - 0.1, "СТЁР ЕЁ ОТПЕЧАТКИ", y=330, size=80)
            caption(im, t, t_own, t_gone, "ОСТАВИЛ СВОИ", y=330, size=96, accent=(255, 90, 80))
        return frame, txt
    lt2 = t - t_gone                                # уходит в ночь под дождём
    img = night_street().copy()
    x = lerp(300, 1000, lt2 / 3.2)
    silhouette(img, person_parts(x, 1330, 520, walk(lt2 * 5.0, 0.95), 1, "man"), base=(6, 6, 8), rim=(220, 170, 110),
               light=(-1, -0.5), rim_w=4, alpha=clamp(1 - (lt2 - 1.6) / 1.5) * 0.85 + 0.15)
    reflect(img, 1334, 500, 0.6, 4, ripple=2, t=t)
    rain(C(img), t, 160, (0, 0, W, H), 9, (180, 180, 200), 60)
    frame = cam(img, 1.05, 600, 1050)

    def txt(im):
        caption(im, t, wt("чтобы", 41.5), 99, "ОН ВЗЯЛ ВИНУ НА СЕБЯ", "чтобы искали его, а не её", y=1650, size=76)
    return frame, txt


# ================================================================ 6. сорок лет под чужим именем
def passport(c, x, y, k):
    c.rect(x - 360, y - 250, x + 360, y + 250, (0, 0, 0, 90))
    c.rect(x - 350, y - 240, x + 350, y + 240, (120, 30, 34), r=10)
    c.rect(x - 330, y - 220, x - 10, y + 220, (226, 220, 200))
    c.rect(x + 10, y - 220, x + 330, y + 220, (226, 220, 200))
    c.rect(x - 300, y - 170, x - 150, y + 20, (150, 140, 120))
    c.ell(x - 225, y - 100, 40, 48, (60, 56, 52))
    c.poly([(x - 290, y + 20), (x - 280, y - 30), (x - 170, y - 30), (x - 160, y + 20)], (60, 56, 52))
    c.text(x - 120, y - 150, "Фамилия", PFI(24, 500), (110, 100, 90), anchor="lm")
    c.text(x - 120, y - 110, "ГРОМОВ", MS(34, 800), (30, 30, 60), anchor="lm")
    c.text(x - 120, y - 50, "Имя", PFI(24, 500), (110, 100, 90), anchor="lm")
    c.text(x - 120, y - 10, "ИВАН", MS(34, 800), (30, 30, 60), anchor="lm")
    c.text(x + 170, y - 120, "СССР", MS(40, 800), (150, 40, 40))
    for i in range(5):
        c.line([(x + 50, y - 40 + i * 40), (x + 290, y - 40 + i * 40)], (180, 170, 150), 4)
    if k > 0:                                        # штамп
        s = lerp(1.6, 1.0, eout(k))
        c.ell(x + 170, y + 150, 80 * s, 80 * s, None, outline=(60, 70, 160, int(220 * min(1, k * 3))), w=6 * s)


def sc_forty(t, lt, dur):
    t_guard = wt("работал", 47.5) - 0.1
    t_cards = wt("перед", 49.4) + 0.1
    if t < t_guard:
        img = lib1.table_bg().copy()
        c = C(img)
        passport(c, 540, 900, (lt - 1.2) / 0.4)
        frame = cam(img, lerp(1.2, 1.0, eio(lt / 2.5)), 540, 900)

        def txt(im):
            year_tag(im, t, SHOT_START["forty"] + 0.1, t_guard, "40 ЛЕТ", "ПОД ЧУЖИМ ИМЕНЕМ", y=240)
        return frame, txt
    if t < t_cards:                                  # будка ночного сторожа, идёт снег
        img = to_img(grad((10, 12, 20), (4, 4, 8)))
        c = C(img)
        c.rect(0, 1400, W, H, (30, 32, 36))
        c.rect(300, 800, 800, 1400, (50, 46, 42))
        c.poly([(270, 800), (830, 800), (760, 720), (340, 720)], (40, 36, 34))
        c.rect(380, 900, 720, 1130, (200, 160, 90))
        p = breathe(pose(lean=10, neck=12, nod=10, ln=(88, 90), lf=(86, 90), an=(60, 50), af=(56, 54)), t)
        silhouette(img, person_parts(560, 1300, 420, p, 1, "man"), base=(30, 20, 12), rim=(255, 210, 140),
                   light=(0, -1), rim_w=2)
        c = C(img)
        c.rect(300, 1130, 800, 1400, (50, 46, 42))
        c.rect(380, 900, 720, 1130, None, outline=(40, 36, 34), w=14)
        img = glow(img, lambda d, k: d.rectangle([360 * k, 880 * k, 740 * k, 1150 * k], fill=(120, 80, 30)), 60)
        dust(C(img), t, 120, seed=8, col=(230, 230, 240), speed=-60, size=(2, 5), alpha=180)
        frame = cam(img, lerp(1.0, 1.1, (t - t_guard) / 3), 540, 1050)

        def txt(im):
            caption(im, t, t_guard + 0.2, t_cards, "НОЧНОЙ СТОРОЖ", y=1650, size=84)
        return frame, txt
    img = to_img(grad((26, 24, 26), (8, 8, 10)))     # его руки: открытки от бабушки
    img = glow(img, lambda d, k: d.ellipse([200 * k, 500 * k, 900 * k, 1400 * k], fill=(90, 70, 40)), 80)
    c = C(img)
    lt3 = t - t_cards
    for i in range(3):
        k = eio((lt3 - i * 0.9) / 0.4)
        if k > 0:
            lib1.postcard(c, lerp(1500, 540 + (i - 1) * 20, k), 960 + i * 14, 820, i % 3, 1.0)
    pov_hand(c, (-100, 2100), (190, 1180), -0.6, 0.5, 1.0)
    pov_hand(c, (1180, 2100), (890, 1180), 0.6, 0.5, 1.0)
    frame = cam(img, 1.0, 540, 960, hh_amp=4)

    def txt(im):
        caption(im, t, t_cards + 0.3, 99, "ВСТАТЬ ТАМ, ГДЕ ОНА СКАЖЕТ", "перед каждым праздником", y=320, size=68)
    return frame, txt


# ================================================================ 7. «почему сейчас?» — последняя открытка
def last_card(c, x, y, w, t):
    h = w * 0.66
    c.rect(x - w / 2 + 10, y - h / 2 + 16, x + w / 2 + 10, y + h / 2 + 16, (0, 0, 0, 110))
    c.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, (236, 228, 208), r=6)
    c.rect(x + w / 2 - w * 0.22, y - h / 2 + 24, x + w / 2 - 24, y - h / 2 + 24 + w * 0.22, (190, 90, 70), r=4)
    lines = [("Сегодня.", "сегодня", 61), ("23:00.", "23", 61.5), ("У окна.", "у", 62.5), ("Хватит прятаться.", "хватит", 64.5)]
    fs = int(w * 0.075)
    for i, (s, wd, after) in enumerate(lines):
        try:
            ta = wt(wd, after)
        except KeyError:
            ta = v(after)
        k = clamp((t - ta) / 0.5)
        n = int(len(s) * k)
        if n:
            col = (150, 30, 30) if i == 3 else (30, 40, 90)
            c.text(x - w * 0.42, y - h * 0.26 + i * fs * 1.45, s[:n], PFI(fs, 600), col, anchor="lm")
        if i == 3 and k >= 1:
            ww = PFI(fs, 600).getlength(s)
            kk = clamp((t - ta - 0.5) / 0.4)
            c.line([(x - w * 0.42, y - h * 0.26 + i * fs * 1.45 + fs * 0.62),
                    (x - w * 0.42 + ww * kk, y - h * 0.26 + i * fs * 1.45 + fs * 0.62)], RED + (230,), 5)


def sc_why(t, lt, dur):
    t_card = wt("достал", 57) - 0.1
    t_bg = wt("она", 66.5) - 0.1
    if t < t_card:                                   # вдвоём под фонарём
        img = night_street().copy()
        pn = breathe(pose(lean=4, an=(10, 20), af=(8, 22), nod=-4), t)
        pm = breathe(pose(lean=8, neck=8, an=(6, 14), af=(4, 16)), t, 0.2)
        silhouette(img, person_parts(470, 1330, 600, pn, 1, "young"), base=(6, 6, 8), rim=(220, 170, 110),
                   light=(1, -0.6), rim_w=5)
        silhouette(img, person_parts(700, 1330, 590, pm, -1, "man"), base=(6, 6, 8), rim=(220, 170, 110),
                   light=(-0.4, -1), rim_w=5)
        reflect(img, 1334, 500, 0.6, 4, ripple=2, t=t)
        rain(C(img), t, 170, (0, 0, W, H), 9, (190, 190, 210), 70)
        return cam(img, lerp(1.3, 1.45, lt / 3), 600, 1080, hh_amp=2), None
    if t < t_bg:                                     # от первого лица: его руки с открыткой
        img = to_img(grad((24, 24, 30), (8, 8, 10)))
        img = glow(img, lambda d, k: d.ellipse([150 * k, 400 * k, 950 * k, 1400 * k], fill=(110, 84, 46)), 80)
        c = C(img)
        k = eout((t - t_card) / 0.6)
        last_card(c, 540, lerp(1500, 940, k), 860, t)
        pov_hand(c, (-100, 2100), (150, lerp(1700, 1140, k)), -0.6, 0.5, 1.0)
        pov_hand(c, (1180, 2100), (930, lerp(1700, 1140, k)), 0.6, 0.5, 1.0)
        rain(C(img), t, 90, (0, 0, W, H), 4, (190, 190, 210), 50)
        frame = cam(img, lerp(1.0, 1.12, (t - t_card) / 8), 540, 940, hh_amp=3)

        def txt(im):
            caption(im, t, t_card + 0.3, wt("сегодня", 61) - 0.1, "ПОСЛЕДНЯЯ ОТКРЫТКА", "бабушкин почерк", y=320, size=74)
        return frame, txt
    # всегда на заднем плане: старый снимок, он крошечный
    world, (mx, my), _ = lib1.photo_world("first_class")
    img = world.copy()
    red_circle(C(img), mx, my + 20, 54, (t - t_bg - 0.3) / 0.4, 6, seed=3)
    k = eio((t - t_bg) / 3.5)
    frame = cam(img, lerp(2.9, 1.05, k), lerp(mx, 540, k), lerp(my + 30, 980, k))

    def txt(im):
        caption(im, t, t_bg + 0.6, 99, "НЕ НА ЗАДНИЙ ПЛАН", "впервые за 40 лет", y=1700, size=86, accent=(255, 90, 80))
    return frame, txt


# ================================================================ 8. фотоаппарат: остался один кадр
def counter_dial(c, x, y, n_float):
    c.ell(x, y, 220, 220, (20, 20, 22))
    c.ell(x, y, 200, 200, (170, 170, 172))
    c.ell(x, y, 180, 180, (30, 30, 32))
    for k in range(-3, 4):
        num = int(round(n_float)) + k
        a = math.radians(-90 + (num - n_float) * 42)
        tx, ty = x + math.cos(a) * 128, y + math.sin(a) * 128
        if 0 <= num <= 36 and abs(num - n_float) < 2.6:
            col = (255, 200, 60) if num == 36 else (230, 230, 230)
            c.text(tx, ty, str(num), MS(40 if num != int(round(n_float)) else 60, 800), col)
    c.poly([(x, y - 196), (x - 14, y - 222), (x + 14, y - 222)], RED)
    c.ell(x, y, 40, 40, (150, 150, 152))


def sc_camera(t, lt, dur):
    t_dial = wt("там", 75.8) - 0.15
    if t < t_dial:                                   # передаёт фотоаппарат: его руки сверху, мои снизу
        img = to_img(grad((22, 22, 28), (6, 6, 8)))
        img = glow(img, lambda d, k: d.ellipse([100 * k, 300 * k, 1000 * k, 1500 * k], fill=(100, 76, 40)), 90)
        c = C(img)
        k = eio(lt / 1.4)
        y = lerp(500, 1050, k)
        film_camera(c, 540, y, 1.3)
        pov_hand(c, (300, -400), (330, y - 200), 2.6, 0.6, 1.0)
        pov_hand(c, (800, -400), (760, y - 200), 2.2, 0.6, 1.0)
        k2 = eio((lt - 1.0) / 0.8)
        if k2 > 0:
            pov_hand(c, (60, 2300), (lerp(-200, 300, k2), y + 220), -0.4, 0.6, 1.0, skin=YOUNG_SKIN, sleeve=(40, 40, 46))
            pov_hand(c, (1020, 2300), (lerp(1280, 780, k2), y + 220), 0.4, 0.6, 1.0, skin=YOUNG_SKIN, sleeve=(40, 40, 46))
        rain(C(img), t, 90, (0, 0, W, H), 6, (190, 190, 210), 50)
        frame = cam(img, 1.0, 540, 960, hh_amp=3)

        def txt(im):
            caption(im, t, wt("тот", 73) - 0.1, t_dial, "ТОТ САМЫЙ ФОТОАППАРАТ", "со всех наших снимков", y=1660, size=70)
        return frame, txt
    img = to_img(grad((20, 20, 22), (6, 6, 8)))
    c = C(img)
    n = lerp(33.0, 35.0, eio((t - t_dial) / 1.2))
    counter_dial(c, 540, 860, n)
    frame = cam(img, lerp(1.0, 1.15, (t - t_dial) / 4), 540, 900)

    def txt(im):
        caption(im, t, t_dial + 0.6, 99, "ОСТАЛСЯ ОДИН КАДР", y=330, size=90, accent=(255, 200, 60))
    return frame, txt


# ================================================================ 9. встреча
@lru_cache(None)
def stair_bg():
    img = to_img(grad((40, 44, 40), (14, 14, 14)))
    c = C(img)
    for k in range(12):                              # ступени уходят вверх
        f = k / 12
        y = lerp(1920, 700, f)
        w = lerp(1100, 380, f)
        c.rect(540 - w / 2, y - 60 * (1 - f * 0.6), 540 + w / 2, y, (70 - k * 3, 70 - k * 3, 66 - k * 3))
        c.rect(540 - w / 2, y - 6, 540 + w / 2, y, (110 - k * 4, 110 - k * 4, 104 - k * 4))
    c.poly([(0, 0), (160, 0), (420, 700), (360, 700)], (54, 60, 54))
    c.poly([(1080, 0), (920, 0), (660, 700), (720, 700)], (54, 60, 54))
    c.rect(420, 380, 660, 700, (36, 28, 24))         # дверь квартиры наверху
    c.ell(630, 540, 8, 8, (200, 180, 120))
    c.ell(540, 220, 18, 18, (255, 230, 170))
    img = glow(img, lambda d, k: d.ellipse([340 * k, 120 * k, 740 * k, 480 * k], fill=(120, 100, 60)), 60)
    return img


def sc_meet(t, lt, dur):
    t_door = wt("бабушка", 81.5) - 0.1
    t_touch = wt("они", 87.8) - 0.1
    if t < t_door:                                   # поднимаемся по лестнице (от первого лица)
        img = stair_bg()
        bob = math.sin(lt * 9) * 14
        frame = cam(img, lerp(1.0, 1.9, eio(lt / (t_door - SHOT_START["meet"]))), 540, lerp(1300, 560, eio(lt / 2.6)) + bob)
        return frame, None
    if t < t_touch:                                  # дверь открывается: бабушка в выходном платье
        k = eio((t - t_door) / 0.7)
        img = to_img(grad((30, 26, 24), (10, 9, 9)))
        c = C(img)
        c.rect(260, 300, 820, 1700, (255, 214, 150))
        img = glow(img, lambda d, kk: d.rectangle([260 * kk, 300 * kk, 820 * kk, 1700 * kk], fill=(160, 110, 50)), 80, k)
        c = C(img)
        dress = pose(an=(14, 30), af=(14, 30))
        silhouette(img, body_front(540, 1700, 1250, breathe(dress, t, 0.25), "grandma"), base=(24, 18, 22),
                   rim=(255, 220, 160), light=(0, -1), rim_w=6, fill_light=(20, 8, 16))
        c = C(img)
        c.poly([(540 - 50, 700), (540 + 50, 700), (540, 780)], (230, 226, 216))     # белый воротничок
        c.ell(540, 820, 10, 10, (230, 210, 140))                                      # брошь
        w = 560 * (1 - k)
        c.rect(260, 300, 260 + w, 1700, (40, 30, 26))
        frame = cam(img, lerp(1.25, 1.1, k), 540, 1000)

        def txt(im):
            caption(im, t, wt("выходном", 84) - 0.1, 99, "ОНА ЖДАЛА 40 ЛЕТ", "в выходном платье", y=1780, size=74)
        return frame, txt
    # она касается его лица
    lt2 = t - t_touch
    img = hallway().copy()
    img = glow(img, lambda d, kk: d.ellipse([200 * kk, 400 * kk, 900 * kk, 1600 * kk], fill=(120, 84, 40)), 90)
    gp = keyed(lt2, [(0, pose(an=(14, 30), af=(14, 30))), (1.3, pose(an=(14, 30), af=(14, 30))),
                     (2.3, pose(an=(118, 30), af=(14, 30), nod=-6))])
    mp = keyed(lt2, [(0, pose(lean=4, neck=8, nod=6)), (2.2, pose(lean=2, neck=2, nod=-4))])
    silhouette(img, person_parts(430, 1560, 900, breathe(gp, t), 1, "grandma"), base=(14, 10, 12), rim=(255, 210, 150),
               light=(0.2, -1), rim_w=5, fill_light=(16, 8, 10))
    silhouette(img, person_parts(650, 1560, 1010, breathe(mp, t, 0.2), -1, "man"), base=(10, 10, 12),
               rim=(255, 210, 150), light=(-0.2, -1), rim_w=5)
    frame = cam(img, lerp(1.25, 1.45, lt2 / 4), 540, 900)

    def txt(im):
        caption(im, t, t_touch + 0.3, 99, "КОСНУЛАСЬ ЕГО ЛИЦА", "они ничего не сказали", y=1700, size=70)
    return frame, txt


# ================================================================ 10. щелчок
@lru_cache(None)
def together_photo(scale=3):
    """36-й кадр: они рядом, бабушка смотрит в объектив, он — на неё."""
    import engine
    old, engine.PS = engine.PS, scale
    try:
        ps = scale
        img = Image.new("RGB", (800 * ps, 600 * ps))
        d = ImageDraw.Draw(img, "RGBA")
        c = C(img, ps)
        a = grad((196, 160, 120), (120, 90, 64), 800 * ps, 600 * ps)
        img.paste(to_img(a))
        for i in range(0, 820, 34):
            c.rect(i, 0, i + 3, 430, (210, 176, 136))
        c.rect(-10, 430, 820, 610, (110, 72, 50))
        c.rect(560, 60, 760, 420, (60, 44, 34))           # дверь
        c.rect(40, 80, 200, 300, (70, 54, 40))            # рамка на стене
        c.rect(52, 92, 188, 288, (160, 140, 110))
        pal_g = dict(PALETTE["grandma"], coat=(40, 46, 80), sleeve=(40, 46, 80), skirt=(36, 40, 70),
                     eyes=(140, 104, 90), mouth=(170, 110, 104))
        shaded(img, body_front(340, 590, 420, pose(look=True, an=(10, 40), af=(10, 40)), "grandma"), pal_g, (340, 590))
        c.poly([(340 - 16, 262), (340 + 16, 262), (340, 284)], (240, 236, 226))
        pal_m = dict(PALETTE["man_old"], eyes=(140, 104, 90), mouth=(160, 110, 100))
        shaded(img, body_front(480, 592, 470, pose(turn=-1, an=(8, 14), af=(8, 14), lean=4), "man"), pal_m, (480, 592))
        return photo_post(img, 36, "2000s", "'26 11 1")
    finally:
        engine.PS = old


def sc_click(t, lt, dur):
    t_click = wt("щелчок", 93.5)
    img = to_img(grad((20, 18, 18), (6, 6, 6)))
    place_photo(img, together_photo(), 540, 980, 1500, 0, shadow=False)
    frame = cam(img, lerp(1.6, 1.5, lt / dur), 540, 980, hh_amp=3)
    frame = viewfinder(frame, 1.0)
    frame = flash(frame, t, t_click, 0.45)
    return frame, None


# ================================================================ 11. проявка
@lru_cache(None)
def life_photo(kind):
    """Снимки жизни внука издалека: сквозь листву / из-за угла, герой маленький."""
    import engine
    ps = 2
    img = Image.new("RGB", (800 * ps, 600 * ps))
    d = ImageDraw.Draw(img, "RGBA")
    c = C(img, ps)
    if kind == "step":
        engine.bg_park(d, c)
        shaded(img, body_front(420, 470, 120, pose(an=(60, 10), af=(60, 10)), "young"), PALETTE["kid"], (420, 470), s=ps)
        shaded(img, body_front(330, 480, 300, pose(an=(50, 40), af=(50, 40), lean=10), "woman"), PALETTE["woman"], (330, 480), s=ps)
    elif kind == "grad":
        engine.bg_school(d, c)
        shaded(img, body_front(400, 500, 260, pose(an=(10, 20), af=(10, 20)), "suit"), PALETTE["suit"], (400, 500), s=ps)
        c.poly([(385, 330), (415, 330), (420, 380), (400, 370), (380, 380)], (220, 40, 40))
    else:
        engine.bg_street(d, c)
        shaded(img, body_front(480, 470, 230, pose(an=(30, 60), af=(8, 14)), "young"), PALETTE["dad"], (480, 470), s=ps)
        c.rect(440, 330, 470, 370, (40, 30, 24))
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))   # листва на переднем плане: снято издалека
    cl = C(lay, ps)
    for k in range(9):
        cl.ell(60 + (k % 3) * 40, 80 + k * 70, 120, 90, (40, 70, 36, 255))
        cl.ell(760 - (k % 3) * 30, 520 - k * 30, 110, 80, (36, 64, 32, 255))
    lay = lay.filter(ImageFilter.GaussianBlur(9 * ps))
    img.paste(lay, (0, 0), lay)
    dates = {"step": "'08 5 14", "grad": "'23 6 25", "work": "'25 9 1"}
    return photo_post(img, hash(kind) % 100, "2000s", dates[kind])


LIFE = [("step", "ПЕРВЫЙ ШАГ", "первый"), ("grad", "ВЫПУСКНОЙ", "выпускной"), ("work", "ПЕРВЫЙ ДЕНЬ НА РАБОТЕ", "мой")]


def sc_develop(t, lt, dur):
    img = to_img(grad((60, 8, 8), (16, 2, 2)))
    c = C(img)
    c.line([(0, 330), (W, 300)], (40, 10, 10), 4)   # верёвка с прищепками
    t_35 = wt("35", 97)
    prints = [("r_first_class", None), ("r_wedding", None), ("r_sea", None)]
    for i, (key, _) in enumerate(prints):
        k = eout((t - SHOT_START["develop"] - 0.3 - i * 0.25) / 0.4)
        if k > 0:
            im = lib1.small_card(key, 300)
            x = 180 + i * 360
            img.paste(im, (int((x - 150) * S), int((340 - (1 - k) * 200) * S)), im)
            c.rect(x - 10, 310, x + 10, 360, (180, 170, 150))
    focus = None
    for i, (kind, label, word) in enumerate(LIFE):
        ta = wt(word, 101.5) - 0.1
        k = eout((t - ta) / 0.45)
        if k > 0:
            focus = (i, label, ta)
            ph = framed(life_photo(kind))
            place_photo(img, ph, 540 + (i - 1) * 30, lerp(2300, 1180 + i * 20, k), 820, (i - 1) * 4)
    if t > t_35 and focus is None:                  # плёнка: 35 кадров
        k = clamp((t - t_35) / 3.0)
        for j in range(8):
            x = (j * 150 - k * 600) % 1400 - 160
            c.rect(x, 1000, x + 140, 1300, (30, 20, 16))
            c.rect(x + 10, 1040, x + 130, 1260, (90, 60, 40))
            for q in range(6):
                c.rect(x + 4, 1010 + q * 50, x + 14, 1024 + q * 50, (60, 40, 30))
        c.text(540, 1400, f"{min(35, int(k * 40))} / 35", MS(60, 800), (255, 140, 120))
    frame = cam(img, 1.0, 540, 960, hh_amp=1.5)
    a = np.asarray(frame, np.float32) * np.array([1.0, 0.75, 0.7], np.float32)
    frame = to_img(a)

    def txt(im):
        caption(im, t, SHOT_START["develop"] + 0.2, t_35 - 0.1, "ЧЕРЕЗ НЕДЕЛЮ", "проявка", y=1700, size=80)
        caption(im, t, t_35, wt("первый", 101.5) - 0.15, "35 КАДРОВ", "вся моя жизнь — издалека", y=1700, size=96)
        if focus:
            i, label, ta = focus
            caption(im, t, ta + 0.1, ta + 1.6 if i < 2 else wt("он", 104.5) - 0.05, label, y=1780, size=74)
        caption(im, t, wt("он", 104.5), 99, "ОН БЫЛ ВЕЗДЕ", y=1780, size=90, accent=(255, 120, 100))
    return frame, txt


# ================================================================ 12. 36-й кадр
@lru_cache(None)
def tray_world():
    img = to_img(grad((70, 10, 10), (20, 3, 3)))
    c = C(img)
    c.rect(60, 360, 1020, 1560, (30, 30, 34), r=30)
    c.rect(90, 390, 990, 1530, (60, 20, 20), r=20)
    return img


def sc_frame36(t, lt, dur):
    t_look = wt("бабушка", 109.5)
    t_him = wt("он", 112)
    img = tray_world().copy()
    k = clamp(lt / 2.6)
    ph = framed(together_photo(), 14)
    blank = Image.new("RGB", ph.size, (240, 236, 226))
    dev = Image.blend(blank, ph, ease(k))
    place_photo(img, dev, 540, 960, 860, -2, shadow=False)
    a = np.asarray(img, np.float32)
    ripple = 0.03 * np.sin(np.linspace(0, 30, WH)[:, None] + t * 3)
    a = a * (1 + ripple[:, :, None])
    img = to_img(a)
    # куда смотрим: сначала весь кадр, потом бабушка (в объектив), потом он (на неё)
    gx, gy = 540 + (340 - 400) * 860 / 800 * 0.94, 960 + (590 - 420 * 0.93 - 300) * 860 / 800 * 0.94
    mx, my = 540 + (480 - 400) * 860 / 800 * 0.94, 960 + (592 - 470 * 0.93 - 300) * 860 / 800 * 0.94
    kz1 = eio((t - t_look) / 1.0)
    kz2 = eio((t - t_him) / 0.9)
    cx = lerp(lerp(540, gx, kz1), mx, kz2)
    cy = lerp(lerp(960, gy + 40, kz1), my + 40, kz2)
    z = lerp(1.0, 2.6, kz1) + 0.3 * kz2
    frame = cam(img, z, cx, cy)
    if k < 1:
        frame = to_img(np.asarray(frame, np.float32) * np.array([1.0, lerp(0.7, 1.0, k), lerp(0.66, 1.0, k)], np.float32))

    def txt(im):
        caption(im, t, SHOT_START["frame36"] + 0.2, t_look - 0.1, "36-Й КАДР", "они стоят рядом", y=1750, size=96)
        caption(im, t, t_look, t_him - 0.05, "ВПЕРВЫЕ СМОТРИТ В ОБЪЕКТИВ", y=1750, size=66)
        caption(im, t, t_him, 99, "А ОН — НА НЕЁ", y=1750, size=96, accent=(255, 200, 140))
    return frame, txt


# ================================================================ 13. вопрос
def sc_question(t, lt, dur):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    dust(C(img, 1), t, 25, seed=4, alpha=60)

    def txt(im):
        k = eout((lt - 0.3) / 0.6)
        d = ImageDraw.Draw(im, "RGBA")
        shadow_text(d, (540, 860), "А ВЫ БЫ ВЗЯЛИ", MS(86, 800), WHITE + (int(255 * k),))
        shadow_text(d, (540, 970), "ВИНУ НА СЕБЯ?", MS(96, 800), (255, 90, 80, int(255 * k)))
        k2 = eout((lt - 1.2) / 0.6)
        d.text((540, 1090), "ради того, кого любите", font=PFI(52, 500), fill=CREAM + (int(220 * k2),), anchor="mm")
    return img, txt


# ================================================================ монтаж
SC = {name: globals()["sc_" + name] for _, name in SCENES}
SHOTS = []
SHOT_START = {}
for i, (a, name) in enumerate(SCENES):
    start = max(0.0, v(a))
    end = v(SCENES[i + 1][0]) if i + 1 < len(SCENES) else DURATION
    SHOTS.append((start, end, name))
    SHOT_START[name] = start

SHAKES += [(word_t("она", 7.5), 18, 0.45), (word_t("ворвался", 23), 10, 0.4), (word_t("упал", 27.2) + 0.5, 22, 0.35),
           (word_t("исчез", 40.5), 6, 0.4), (word_t("хватит", 64.5), 6, 0.5), (word_t("щелчок", 93.5), 8, 0.3)]

WARM = {"hook": 0.0, "lamp": 0.1, "pharmacy": -0.4, "after": -0.4, "wipe": -0.2, "forty": 0.3, "why": 0.2,
        "camera": 0.3, "meet": 0.7, "click": 0.6, "develop": 0.0, "frame36": 0.4, "question": 0.0}


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
    draw_quote(img, t)
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

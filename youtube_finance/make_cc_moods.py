"""Аватарки «CC» по мотивам трёх фото-референсов (сцены нарисованы заново):
1) человек в капюшоне у туманной воды, 2) лес и лучи сквозь туман,
3) рыцарь на коленях у руин замка на закате.

1080×1080 PNG. Запуск из корня репозитория: python3 youtube_finance/make_cc_moods.py
"""
import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "president", "fonts")
SIZE, S = 1080, 2
WS = SIZE * S


def font(name, size, w):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size * S))
    f.set_variation_by_axes([w])
    return f


def PF(size, w=700):
    return font("PlayfairDisplay[wght].ttf", size, w)


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def grad(stops):
    ys = np.linspace(0, 1, WS)[:, None]
    a = np.zeros((WS, WS, 3), np.float32)
    for ch in range(3):
        a[:, :, ch] = np.interp(ys, [p for p, _ in stops], [c[ch] for _, c in stops])
    return a


def noise(cells, seed):
    r = np.random.default_rng(seed)
    out = np.zeros((WS, WS), np.float32)
    amp = 1.0
    for i, c in enumerate(cells):
        small = r.random((c, c)).astype(np.float32)
        out += np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((WS, WS), Image.BICUBIC),
                          np.float32) / 255 * amp
        amp *= 0.5
    return out / out.max()


def P(pts):
    return [(x * S, y * S) for x, y in pts]


def soft_light(img, draw_fn, blur, strength=1.0):
    lay = Image.new("RGB", (SIZE // 2, SIZE // 2))
    draw_fn(ImageDraw.Draw(lay), 0.5)
    lay = lay.filter(ImageFilter.GaussianBlur(blur / 2)).resize(img.size, Image.BILINEAR)
    return ImageChops.add(img, to_img(np.asarray(lay, np.float32) * strength))


def grain(img, k=5):
    a = np.asarray(img, np.float32)
    a += np.random.default_rng(1).normal(0, k, a.shape[:2])[:, :, None]
    return to_img(a)


def finish(img, name):
    img = grain(img).resize((SIZE, SIZE), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(m).ellipse([0, 0, SIZE, SIZE], fill=255)
    prev = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


# ---------------------------------------------------------------- 1. капюшон
def hoodie():
    a = grad([(0, (196, 200, 204)), (0.42, (150, 156, 162)), (0.48, (92, 98, 106)), (1, (38, 42, 48))])
    fog = noise([6, 14, 40], 3)
    a[: int(WS * 0.5)] += (fog[: int(WS * 0.5), :, None] - 0.5) * 40
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(70):
        y = 540 + k ** 1.35 * 1.4
        x0 = (k * 137) % 900
        d.line(P([(x0 - 200, y), (x0 + 260 + k * 3, y)]), fill=(200, 205, 210, 22 + k // 3), width=S * (1 + k // 25))
    body = (14, 14, 16)
    # капюшон и плечи (вид со спины)
    d.polygon(P([(300, 1080), (290, 760), (330, 640), (420, 590), (470, 520), (470, 400), (500, 330), (560, 300),
                 (620, 320), (660, 390), (680, 470), (690, 560), (760, 600), (820, 680), (840, 800), (830, 1080)]), fill=body)
    d.polygon(P([(470, 520), (470, 400), (500, 330), (560, 300), (620, 320), (660, 390), (680, 470), (650, 560),
                 (560, 600), (490, 570)]), fill=(22, 22, 24))
    d.line(P([(500, 340), (470, 420), (480, 520)]), fill=(40, 40, 44), width=3 * S)
    d.line(P([(330, 760), (340, 1080)]), fill=(30, 30, 34), width=4 * S)
    d.line(P([(800, 760), (790, 1080)]), fill=(30, 30, 34), width=4 * S)
    img = img.filter(ImageFilter.GaussianBlur(0.6 * S))
    d = ImageDraw.Draw(img)
    d.text((565 * S, 740 * S), "CC", font=PF(130, 600), fill=(200, 200, 204), anchor="mm")
    d.text((565 * S, 830 * S), "Capital Crimes", font=PF(34, 500), fill=(170, 170, 176), anchor="mm")
    return finish(img, "cc_mood_1_hoodie")


# ---------------------------------------------------------------- 2. лес и лучи
def fern(d, x, y, ln, ang, col):
    ex, ey = x + math.cos(ang) * ln, y + math.sin(ang) * ln
    d.line(P([(x, y), (ex, ey)]), fill=col, width=3 * S)
    for i in range(1, 14):
        k = i / 14
        px, py = x + (ex - x) * k, y + (ey - y) * k
        ll = ln * 0.32 * math.sin(k * math.pi)
        for sg in (-1, 1):
            a2 = ang + sg * 1.2
            d.line(P([(px, py), (px + math.cos(a2) * ll, py + math.sin(a2) * ll)]), fill=col, width=4 * S)


def forest():
    sky = grad([(0, (30, 110, 220)), (0.6, (110, 160, 210)), (1, (90, 120, 120))])
    img = to_img(sky)
    # полог: неровная маска из шума, сверху плотнее
    n = noise([7, 16, 40], 21)
    ys = np.linspace(0, 1, WS)[:, None]
    dens = n + (1 - ys) * 0.15 - np.clip(ys - 0.55, 0, 1) * 1.5
    canopy = (dens > 0.62).astype(np.float32)
    canopy = np.asarray(Image.fromarray((canopy * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2 * S)), np.float32) / 255
    a = np.asarray(img, np.float32)
    leaf = np.array([22, 44, 30], np.float32) * (0.7 + 0.6 * n[:, :, None])
    a = a * (1 - canopy[:, :, None]) + leaf * canopy[:, :, None]
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    vx, vy = 600, -500
    rr = np.random.default_rng(7)
    trunks = sorted([(rr.uniform(-250, 1330), rr.uniform(50, 170)) for _ in range(9)], key=lambda q: q[1])
    for bx, w in trunks:
        tx = vx + (bx - vx) * 0.1
        dark = (20, 16, 14)
        lit = (70, 56, 46)
        d.polygon(P([(bx - w, 1200), (bx + w, 1200), (tx + w * 0.12, vy), (tx - w * 0.12, vy)]), fill=dark)
        d.polygon(P([(bx - w, 1200), (bx - w * 0.3, 1200), (tx - w * 0.03, vy), (tx - w * 0.12, vy)]), fill=lit)
    sx, sy = 250, 330

    def rays(dd, k):
        for i in range(18):
            ang = math.radians(-20 + i * 8)
            ln, w = 1800, 0.03
            dd.polygon([(sx * k, sy * k), ((sx + math.cos(ang - w) * ln) * k, (sy + math.sin(ang - w) * ln) * k),
                        ((sx + math.cos(ang + w) * ln) * k, (sy + math.sin(ang + w) * ln) * k)], fill=(140, 140, 130))
        dd.ellipse([(sx - 110) * k, (sy - 110) * k, (sx + 110) * k, (sy + 110) * k], fill=(255, 250, 230))
    img = soft_light(img, rays, 24, 0.85)
    fog = noise([5, 12, 30], 9)
    a = np.asarray(img, np.float32)
    a += fog[:, :, None] * 80 * np.clip(1.3 - np.abs(ys - 0.65) * 2.6, 0, 1)[:, :, None]
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    for i in range(22):
        x, y = rr.uniform(-80, 1160), rr.uniform(1000, 1120)
        for j in range(5):
            fern(d, x, y, rr.uniform(140, 240), math.radians(-150 + j * 30 + rr.uniform(-8, 8)),
                 (40 + int(rr.uniform(0, 40)), 100 + int(rr.uniform(0, 50)), 44, 255))

    def glow(dd, k):
        dd.text((540 * k, 600 * k), "CC", font=font("PlayfairDisplay[wght].ttf", 300 * k / S, 800), fill=(210, 210, 180), anchor="mm")
    img = soft_light(img, glow, 30, 0.9)
    ImageDraw.Draw(img).text((540 * S, 600 * S), "CC", font=PF(300, 800), fill=(252, 250, 236), anchor="mm")
    return finish(img, "cc_mood_2_forest")


# ---------------------------------------------------------------- 3. рыцарь на закате
def knight():
    a = grad([(0, (40, 30, 40)), (0.35, (160, 80, 50)), (0.52, (240, 150, 70)), (0.58, (120, 70, 50)), (1, (24, 16, 14))])
    clouds = noise([5, 11, 28], 4)
    a[: int(WS * 0.5)] *= (0.75 + clouds[: int(WS * 0.5), :, None] * 0.4)
    img = to_img(a)
    sunx, suny = 860, 580
    img = soft_light(img, lambda dd, k: dd.ellipse([(sunx - 160) * k, (suny - 160) * k, (sunx + 160) * k, (suny + 160) * k],
                                                   fill=(255, 170, 80)), 60)
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse(P([(sunx - 34, suny - 34), (sunx + 34, suny + 34)]), fill=(255, 236, 190))
    d.polygon(P([(380, 700), (560, 600), (760, 560), (980, 620), (1080, 660), (1080, 760), (380, 760)]), fill=(52, 32, 30))
    for x0, h in ((620, 120), (680, 160), (740, 110), (800, 140)):
        d.rectangle(P([(x0, 560 - h), (x0 + 40, 600)]), fill=(70, 46, 40))
        d.rectangle(P([(x0 + 10, 560 - h + 20), (x0 + 26, 560 - h + 50)]), fill=(200, 120, 70))
    d.rectangle(P([(600, 520), (860, 600)]), fill=(70, 46, 40))
    d.polygon(P([(560, 780), (1080, 740), (1080, 820), (600, 830)]), fill=(200, 110, 60, 160))
    d.polygon(P([(0, 920), (300, 860), (700, 900), (1080, 880), (1080, 1080), (0, 1080)]), fill=(26, 18, 16))

    # рыцарь на одном колене, голова склонена к рукояти меча
    def knight_shape(dx, dy, col):
        q = lambda pts: P([(x + dx, y + dy) for x, y in pts])
        d.polygon(q([(120, 1010), (170, 820), (250, 690), (330, 640), (380, 660), (420, 760), (440, 900), (470, 1010)]), fill=col)
        d.polygon(q([(330, 860), (480, 900), (520, 1000), (470, 1010), (440, 950), (330, 930)]), fill=col)
        d.polygon(q([(300, 650), (420, 640), (470, 700), (440, 860), (330, 870)]), fill=col)
        d.ellipse(q([(420, 580), (520, 680)]), fill=col)
        d.polygon(q([(430, 690), (500, 650), (530, 670), (470, 730)]), fill=col)
    knight_shape(5, -3, (250, 150, 80))
    knight_shape(0, 0, (14, 11, 12))
    d.rectangle(P([(526, 680), (538, 1010)]), fill=(205, 200, 194))
    d.rectangle(P([(498, 680), (566, 694)]), fill=(150, 120, 80))
    d.ellipse(P([(518, 654), (546, 682)]), fill=(150, 120, 80))
    d.ellipse(P([(505, 660), (545, 700)]), fill=(14, 11, 12))
    rr = np.random.default_rng(5)
    for _ in range(26):
        x, y = rr.uniform(0, 1080), rr.uniform(150, 1060)
        sz = rr.uniform(8, 18)
        ang = rr.uniform(0, math.pi)
        d.ellipse(P([(x - sz, y - sz * 0.55), (x + sz, y + sz * 0.55)]), fill=(190, 20, 30, 230))
    d.ellipse(P([(590, 975), (640, 1020)]), fill=(180, 16, 26))
    d.ellipse(P([(600, 985), (628, 1008)]), fill=(120, 8, 16))

    def glow(dd, kk):
        dd.text((540 * kk, 230 * kk), "CC", font=font("PlayfairDisplay[wght].ttf", 260 * kk / S, 800), fill=(200, 120, 40), anchor="mm")
    img = soft_light(img, glow, 26, 0.9)
    ImageDraw.Draw(img).text((540 * S, 230 * S), "CC", font=PF(260, 800), fill=(255, 214, 150), anchor="mm")
    return finish(img, "cc_mood_3_knight")


if __name__ == "__main__":
    previews = [hoodie(), forest(), knight()]
    sheet = Image.new("RGB", (SIZE * 3 + 80, SIZE + 40), (235, 235, 235))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "cc_mood_preview.jpg"), quality=90)

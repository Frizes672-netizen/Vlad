"""Тематические аватарки «Frizes» (приём матового стекла) с разными фонами:
Уолл-стрит и падение акций, мятая купюра, газетные вырезки, силуэт человека в шляпе.
1080×1080 PNG. Запуск из корня репозитория: python3 youtube_finance/make_frizes_themes.py
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "president", "fonts")
OUT, S = 1080, 2
W = OUT * S


def font(name, size, w):
    f = ImageFont.truetype(os.path.join(FONTS, name), int(size * S))
    f.set_variation_by_axes([w])
    return f


def ms(size, w=800):
    return font("Montserrat[wght].ttf", size, w)


def pf(size, w=800):
    return font("PlayfairDisplay[wght].ttf", size, w)


def noise(cells, seed):
    r = np.random.default_rng(seed)
    out = np.zeros((W, W), np.float32)
    amp = 1.0
    for c in cells:
        small = (r.random((c, c)) * 255).astype(np.uint8)
        out += np.asarray(Image.fromarray(small).resize((W, W), Image.BICUBIC), np.float32) / 255 * amp
        amp *= 0.5
    return out / out.max()


def P(pts):
    return [(x * S, y * S) for x, y in pts]


# ---------------------------------------------------------------- фоны
def bg_wallstreet():
    ys = np.linspace(0, 1, W)[:, None, None]
    a = np.array([34, 36, 42], np.float32) * (1 - ys) + np.array([14, 15, 18], np.float32) * ys
    img = Image.fromarray(np.repeat(a, W, 1).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    r = np.random.default_rng(4)
    x = 0
    while x < 1080:
        w = r.uniform(60, 130)
        h = r.uniform(260, 620)
        col = int(r.uniform(44, 62))
        d.rectangle(P([(x, 1080 - h), (x + w, 1080)]), fill=(col, col + 2, col + 6))
        if r.random() < 0.4:
            d.polygon(P([(x + w * 0.2, 1080 - h), (x + w * 0.5, 1080 - h - 70), (x + w * 0.8, 1080 - h)]),
                      fill=(col, col + 2, col + 6))
        for wy in range(int(1080 - h + 20), 1070, 30):
            for wx in range(int(x + 10), int(x + w - 10), 20):
                if r.random() < 0.3:
                    d.rectangle(P([(wx, wy), (wx + 8, wy + 12)]), fill=(90, 92, 98, 160))
        x += w + 4
    pts = [(40, 330), (170, 290), (260, 340), (380, 260), (480, 310), (600, 230), (700, 300), (790, 420),
           (880, 560), (960, 760), (1040, 880)]
    d.line(P(pts), fill=(210, 50, 50, 230), width=10 * S, joint="curve")
    return np.asarray(img, np.float32)


def bg_dollar():
    n = noise([5, 12, 30, 80], 7)
    base = np.array([96, 112, 92], np.float32)
    a = base * (0.55 + 0.6 * n[:, :, None])
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle(P([(60, 180), (1020, 900)]), outline=(40, 52, 40, 200), width=10 * S)
    d.rectangle(P([(90, 210), (990, 870)]), outline=(40, 52, 40, 140), width=3 * S)
    d.ellipse(P([(370, 280), (710, 800)]), outline=(40, 52, 40, 180), width=8 * S)
    for i in range(30):
        d.ellipse(P([(390 + i * 2, 300 + i * 3), (690 - i * 2, 780 - i * 3)]), outline=(50, 64, 50, 40), width=S)
    for x, y in ((150, 260), (930, 260), (150, 820), (930, 820)):
        d.text((x * S, y * S), "100", font=ms(64, 800), fill=(38, 50, 38, 220), anchor="mm")
    d.text((540 * S, 950 * S), "THE UNITED STATES OF AMERICA", font=pf(34, 700), fill=(38, 50, 38, 160), anchor="mm")
    a = np.asarray(img, np.float32)
    # изломы: светлые и тёмные полосы сгибов
    r = np.random.default_rng(3)
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
    for _ in range(9):
        x0, y0 = r.uniform(0, W), r.uniform(0, W)
        ang = r.uniform(0, math.pi)
        dist = (xx - x0) * math.sin(ang) - (yy - y0) * math.cos(ang)
        a *= (1 + 0.18 * np.exp(-(dist / (6 * S)) ** 2) - 0.14 * np.exp(-((dist - 10 * S) / (14 * S)) ** 2))[:, :, None]
    g = a.mean(axis=2, keepdims=True)
    a = (a * 0.5 + g * 0.5) * 0.55          # приглушить, чтобы надпись читалась
    return a


HEADLINES = [("FRAUD EXPOSED", "Billions vanished through shell firms"), ("CEO ARRESTED", "Investors lose everything"),
             ("$13M SETTLEMENT", "The lawsuit that went viral"), ("MARKET CRASH", "Wall Street in panic"),
             ("SCANDAL", "Secret deals revealed"), ("BANKRUPT", "Empire collapses overnight")]


def bg_newspaper():
    img = Image.new("RGB", (W, W), (24, 24, 26))
    r = np.random.default_rng(11)
    spots = [(250, 230, -9), (780, 210, 7), (180, 620, 5), (860, 640, -6), (330, 960, -4), (760, 980, 8)]
    for (x, y, rot), (h1, h2) in zip(spots, HEADLINES):
        cw, ch = 520, 300
        clip = Image.new("RGBA", (cw * S, ch * S), (214, 208, 194, 255))
        cd = ImageDraw.Draw(clip)
        cd.text((cw * S / 2, 70 * S), h1, font=pf(56, 900), fill=(28, 26, 26), anchor="mm")
        cd.line([(30 * S, 112 * S), ((cw - 30) * S, 112 * S)], fill=(60, 58, 56), width=2 * S)
        cd.text((cw * S / 2, 145 * S), h2, font=pf(26, 500), fill=(60, 58, 56), anchor="mm")
        for i in range(4):
            cd.line([(30 * S, (185 + i * 26) * S), ((cw - 30 - (i % 2) * 120) * S, (185 + i * 26) * S)],
                    fill=(150, 146, 138), width=5 * S)
        edge = Image.new("L", clip.size, 255)
        ed = ImageDraw.Draw(edge)
        for k in range(0, cw * S, 10 * S):
            ed.polygon([(k, 0), (k + 5 * S, r.uniform(4, 12) * S), (k + 10 * S, 0)], fill=0)
            ed.polygon([(k, ch * S), (k + 5 * S, ch * S - r.uniform(4, 12) * S), (k + 10 * S, ch * S)], fill=0)
        clip.putalpha(edge)
        clip = clip.rotate(rot, Image.BICUBIC, expand=True)
        sh = Image.new("RGBA", clip.size, (0, 0, 0, 0))
        sh.putalpha(clip.split()[3].point(lambda p: int(p * 0.6)))
        sh = sh.filter(ImageFilter.GaussianBlur(12))
        img.paste(sh, (int(x * S - clip.width / 2 + 12), int(y * S - clip.height / 2 + 16)), sh)
        img.paste(clip, (int(x * S - clip.width / 2), int(y * S - clip.height / 2)), clip)
    a = np.asarray(img, np.float32)
    g = a.mean(axis=2, keepdims=True)
    return g.repeat(3, 2) * 0.42 + 6


def bg_plain():
    ys = np.linspace(0, 1, W)[:, None, None]
    a = np.array([40, 42, 48], np.float32) * (1 - ys) + np.array([16, 17, 20], np.float32) * ys
    return np.repeat(a, W, 1)


def hat_man(img):
    """Минималистичная иконка: силуэт человека в шляпе в тени, контровой свет."""
    d = ImageDraw.Draw(img, "RGBA")
    cx, top = 540, 150
    for dx, col in ((5, (190, 190, 196, 255)), (0, (8, 8, 10, 255))):
        x = cx + dx
        d.polygon(P([(x - 170, 520), (x - 120, 430), (x - 60, 405), (x + 60, 405), (x + 120, 430), (x + 170, 520)]), fill=col)
        d.ellipse(P([(x - 62, top + 110), (x + 62, top + 270)]), fill=col)
        d.ellipse(P([(x - 150, top + 120), (x + 150, top + 160)]), fill=col)
        d.polygon(P([(x - 85, top + 135), (x - 70, top + 40), (x - 20, top + 25), (x + 20, top + 25),
                     (x + 70, top + 40), (x + 85, top + 135)]), fill=col)
        d.rectangle(P([(x - 30, top + 255), (x + 30, 420)]), fill=col)
    d.rectangle(P([(cx - 84, top + 110), (cx + 84, top + 126)]), fill=(150, 30, 30, 255))


# ---------------------------------------------------------------- стекло + слово
def compose(name, bg, ink=(245, 245, 247), tint=(70, 72, 80), cy=560, icon=None):
    a = bg.astype(np.float32)
    if icon:
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        icon(img)
        a = np.asarray(img, np.float32)
    word = Image.new("L", (W, W), 0)
    f = ms(186, 800)
    ImageDraw.Draw(word).text((W / 2 + 55 * S, cy * S), "Frizes", font=f, fill=255, anchor="mm")
    x0, y0, x1, y1 = word.getbbox()
    f_w = ImageDraw.Draw(word).textlength("F", font=f)
    gx0, gx1 = x0 - 120 * S, x0 + f_w * 0.55
    gy0, gy1 = (cy - 200) * S, (cy + 200) * S
    sharp = np.asarray(word, np.float32)[:, :, None] / 255
    soft = np.clip(np.asarray(word.filter(ImageFilter.GaussianBlur(26 * S)), np.float32)[:, :, None] / 255 * 1.3, 0, 0.9)
    g = Image.new("L", (W, W), 0)
    ImageDraw.Draw(g).rounded_rectangle([gx0, gy0, gx1, gy1], 34 * S, fill=255)
    g = np.asarray(g, np.float32)[:, :, None] / 255
    ink = np.array(ink, np.float32)
    # подложка за стеклом: фон заметно размывается, как настоящее матовое стекло
    blurred_bg = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(18 * S)), np.float32)
    outside = a * (1 - sharp) + ink * sharp
    inside = blurred_bg * (1 - soft) + ink * soft
    inside = inside * 0.6 + np.array(tint, np.float32) * 0.4
    sh = Image.new("L", (W, W), 0)
    ImageDraw.Draw(sh).rounded_rectangle([gx0 + 8 * S, gy0 + 20 * S, gx1 + 8 * S, gy1 + 24 * S], 34 * S, fill=110)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(22 * S)), np.float32)[:, :, None] / 255
    # мягкая тень под словом, чтобы оно читалось на пёстром фоне
    wsh = np.asarray(word.filter(ImageFilter.GaussianBlur(14 * S)), np.float32)[:, :, None] / 255
    outside = outside * (1 - wsh * 0.45 * (1 - sharp))
    a = (outside * (1 - sh * 0.5)) * (1 - g) + inside * g
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([gx0, gy0, gx1, gy1], 34 * S, outline=tuple(min(255, c + 40) for c in tint) + (150,), width=2 * S)
    img = img.resize((OUT, OUT), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (OUT, OUT), 0)
    ImageDraw.Draw(m).ellipse([0, 0, OUT, OUT], fill=255)
    prev = Image.new("RGB", (OUT, OUT), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


if __name__ == "__main__":
    prevs = [
        compose("frizes_theme_1_wallstreet", bg_wallstreet(), cy=600),
        compose("frizes_theme_2_dollar", bg_dollar(), tint=(80, 92, 78)),
        compose("frizes_theme_3_newspaper", bg_newspaper()),
        compose("frizes_theme_4_hat_man", bg_plain(), cy=720, icon=hat_man),
    ]
    sheet = Image.new("RGB", (OUT * 4 + 100, OUT + 40), (255, 255, 255))
    for i, p in enumerate(prevs):
        sheet.paste(p, (20 + i * (OUT + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(
        os.path.join(HERE, "frizes_themes_preview.jpg"), quality=90)

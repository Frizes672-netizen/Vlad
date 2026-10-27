"""Аватарки канала FRIZEcc (FRIZE — ник автора, cc — Capital Crimes).

Стиль Shorts про Red Bull: тёмный фон с размытыми огнями, жирный Montserrat
с чёрной обводкой, жёлтый и красный акценты. 1080×1080 PNG.
Запуск из корня репозитория: python3 youtube_finance/make_frizecc.py
"""
import math
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.join(HERE, "..", "redbull")
sys.path.insert(0, RB)
os.chdir(RB)  # движок ролика читает шрифты и разметку по относительным путям
import render as r  # noqa: E402

SIZE = 1080
YEL, RED, WHITE = r.YEL, (226, 46, 50), (250, 250, 250)


def canvas(seed, tint, glow=(170, 30, 30)):
    big = r.bokeh_bg(seed, tint)  # 1080×1920 ×S
    img = big.crop((0, 420 * r.S, SIZE * r.S, (420 + SIZE) * r.S))
    a = r.np.asarray(img, r.np.float32)
    a += r.radial(540, 540, 560, w=SIZE * r.S, h=SIZE * r.S)[:, :, None] * r.np.array(glow, r.np.float32) * 0.6
    img = r.to_img(a)
    return img, r.C(img)


def wordmark(d, cx, cy, size, stroke=None):
    """FRIZE белым + cc жёлтым, по центру."""
    f1, f2 = r.MS(size, 900), r.MS(int(size * 0.78), 900)
    stroke = stroke or max(4, size // 11)
    w1, w2 = d.textlength("FRIZE", font=f1), d.textlength("cc", font=f2)
    x = cx - (w1 + w2 + size * 0.04) / 2
    d.text((x, cy), "FRIZE", font=f1, fill=WHITE, anchor="lm", stroke_width=stroke, stroke_fill=(0, 0, 0))
    d.text((x + w1 + size * 0.04, cy + size * 0.07), "cc", font=f2, fill=YEL, anchor="lm",
           stroke_width=stroke, stroke_fill=(0, 0, 0))


def crash(d, pts, w):
    d.line(pts, fill=RED, width=w, joint="curve")
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    tip = (x1 + math.cos(a) * w * 2.6, y1 + math.sin(a) * w * 2.6)
    d.polygon([tip, (x1 + math.cos(a + 2.4) * w * 2.2, y1 + math.sin(a + 2.4) * w * 2.2),
               (x1 + math.cos(a - 2.4) * w * 2.2, y1 + math.sin(a - 2.4) * w * 2.2)], fill=RED)


def finish(img, name):
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(m).ellipse([0, 0, SIZE, SIZE], fill=255)
    prev = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


def v1():
    """Надпись FRIZEcc и красный обвал графика под ней."""
    img, c = canvas(21, (200, 60, 60))
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    crash(d, [(170, 700), (330, 640), (450, 680), (600, 560), (720, 610), (860, 820)], 16)
    wordmark(d, 540, 470, 190)
    return finish(img, "frizecc_avatar_1_wordmark")


def v2():
    """Молоток судьи и надпись FRIZEcc — разоблачения и суды."""
    img, c = canvas(22, (200, 170, 120), glow=(120, 80, 30))
    c.rect(250, 760, 830, 800, (60, 36, 22), r=10)
    c.ell(540, 760, 200, 30, (80, 50, 30))
    r.gavel(c, 210, 580, 0.05, 1.3)
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    wordmark(d, 540, 330, 170)
    return finish(img, "frizecc_avatar_2_gavel")


def v3():
    """Монограмма F$ в жёлтом кольце, снизу FRIZEcc."""
    img, c = canvas(23, (60, 200, 90), glow=(40, 120, 60))
    c.ell(540, 470, 330, 330, (10, 12, 18, 230))
    c.ell(540, 470, 330, 330, None, outline=YEL, w=18)
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    d.text((540, 470), "F", font=r.MS(380, 900), fill=WHITE, anchor="mm", stroke_width=14, stroke_fill=(0, 0, 0))
    d.text((660, 600), "$", font=r.MS(160, 900), fill=YEL, anchor="mm", stroke_width=10, stroke_fill=(0, 0, 0))
    wordmark(d, 540, 900, 120)
    return finish(img, "frizecc_avatar_3_monogram")


def v4():
    """Пачки денег, обвал и надпись — про богатство и крахи."""
    img, c = canvas(24, (60, 200, 90), glow=(30, 110, 50))
    for x, y, n in ((330, 900, 14), (540, 930, 20), (750, 900, 16)):
        r.money_stack(c, x, y, n, 0.95)
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    crash(d, [(150, 640), (320, 560), (470, 600), (640, 470), (780, 520), (900, 700)], 14)
    wordmark(d, 540, 330, 180)
    return finish(img, "frizecc_avatar_4_money")


if __name__ == "__main__":
    previews = [v1(), v2(), v3(), v4()]
    sheet = Image.new("RGB", (SIZE * 4 + 100, SIZE + 40), (255, 255, 255))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "frizecc_preview.jpg"), quality=90)

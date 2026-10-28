"""Аватарки «Frizes» в приёме шапки (матовое стекло, размытие), но с другим фоном
и без CC / Capital Crimes: стекло закрывает половину первой буквы — она размыта, остальное слово чёткое.
1080×1080 PNG. Запуск из корня репозитория: python3 youtube_finance/make_frizes_glass.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "president", "fonts", "Montserrat[wght].ttf")
OUT, S = 1080, 2
W = OUT * S


def ms(size, w):
    f = ImageFont.truetype(FONT, int(size * S))
    f.set_variation_by_axes([w])
    return f


def make(name, bg_top, bg_bottom, ink, glass_tint, by=False):
    ys = np.linspace(0, 1, W)[:, None, None]
    xs = np.linspace(-1, 1, W)[None, :, None]
    bg = np.array(bg_top, np.float32) * (1 - ys) + np.array(bg_bottom, np.float32) * ys
    bg = bg * (1 - xs ** 2 * 0.06) + np.random.default_rng(2).normal(0, 1.3, (W, W, 1))
    a = bg.astype(np.float32)

    word = Image.new("L", (W, W), 0)
    f = ms(186, 800)
    cy = 580 if by else 540
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
    outside = a * (1 - sharp) + ink * sharp
    inside = a * (1 - soft) + ink * soft
    inside = inside * 0.55 + np.array(glass_tint, np.float32) * 0.45
    sh = Image.new("L", (W, W), 0)
    ImageDraw.Draw(sh).rounded_rectangle([gx0 + 8 * S, gy0 + 20 * S, gx1 + 8 * S, gy1 + 24 * S], 34 * S, fill=90)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(22 * S)), np.float32)[:, :, None] / 255
    a = (outside * (1 - sh * 0.5)) * (1 - g) + inside * g
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    edge = tuple(min(255, c + 30) for c in glass_tint) + (140,)
    d.rounded_rectangle([gx0, gy0, gx1, gy1], 34 * S, outline=edge, width=2 * S)
    if by:
        d.text((W / 2 + 55 * S, (cy - 150) * S), "b y", font=ms(52, 500), fill=tuple(ink.astype(int)) + (230,), anchor="mm")
    img = img.resize((OUT, OUT), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (OUT, OUT), 0)
    ImageDraw.Draw(m).ellipse([0, 0, OUT, OUT], fill=255)
    prev = Image.new("RGB", (OUT, OUT), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


if __name__ == "__main__":
    prevs = [
        make("frizes_glass_dark", (40, 42, 48), (18, 19, 22), (245, 245, 247), (70, 72, 80)),
        make("frizes_glass_dark_by", (40, 42, 48), (18, 19, 22), (245, 245, 247), (70, 72, 80), by=True),
        make("frizes_glass_sand", (226, 214, 196), (204, 190, 170), (28, 24, 22), (240, 234, 224)),
    ]
    sheet = Image.new("RGB", (OUT * 3 + 80, OUT + 40), (255, 255, 255))
    for i, p in enumerate(prevs):
        sheet.paste(p, (20 + i * (OUT + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(
        os.path.join(HERE, "frizes_glass_preview.jpg"), quality=90)

"""Шапка канала 2560×1440 в стиле минималистичного постера: светлый фон,
матовое стекло, за которым размыты «CC», чёткие «CC» справа от края стекла,
«Capital Crimes.» и «BY | FRIZES». Всё в безопасной зоне 1546×423.
Запуск из корня репозитория: python3 youtube_finance/make_glass_banner.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "president", "fonts", "Montserrat[wght].ttf")
BW, BH = 2560, 1440
INK = (12, 12, 14)


def ms(size, w):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([w])
    return f


def main():
    ys = np.linspace(0, 1, BH)[:, None, None]
    xs = np.linspace(-1, 1, BW)[None, :, None]
    bg = np.array([226, 228, 231], np.float32) - ys * 14 - np.abs(xs) * 8
    bg = bg + np.random.default_rng(3).normal(0, 1.4, (BH, BW, 1))
    img = Image.fromarray(np.clip(bg.repeat(3, 2) if bg.shape[2] == 1 else bg, 0, 255).astype(np.uint8))

    # буквы CC
    cc = Image.new("L", (BW, BH), 0)
    f = ms(380, 900)
    cx, cy = 650, 720
    ImageDraw.Draw(cc).text((cx, cy), "CC", font=f, fill=255, anchor="lm")
    bbox = cc.getbbox()

    # стекло: закрывает первую C и половину второй
    gx0, gy0 = 560, 520
    gx1 = int(bbox[0] + (bbox[2] - bbox[0]) * 0.62)
    gy1 = 920
    glass = Image.new("L", (BW, BH), 0)
    ImageDraw.Draw(glass).rounded_rectangle([gx0, gy0, gx1, gy1], 34, fill=255)

    # размытые буквы под стеклом и чёткие снаружи
    blurred = cc.filter(ImageFilter.GaussianBlur(26)).point(lambda p: int(p * 0.85))
    a = np.asarray(img, np.float32)
    g = np.asarray(glass, np.float32)[:, :, None] / 255
    sharp = np.asarray(cc, np.float32)[:, :, None] / 255
    soft = np.asarray(blurred, np.float32)[:, :, None] / 255
    ink = np.array(INK, np.float32)
    outside = a * (1 - sharp) + ink * sharp
    inside = a * (1 - soft) + ink * soft
    inside = inside * 0.55 + np.array([244, 245, 247], np.float32) * 0.45      # матовая белизна стекла
    a = outside * (1 - g) + inside * g
    # мягкая тень под стеклом и тонкий блик по краю
    sh = Image.new("L", (BW, BH), 0)
    ImageDraw.Draw(sh).rounded_rectangle([gx0 + 6, gy0 + 18, gx1 + 6, gy1 + 22], 34, fill=60)
    sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(22)), np.float32)[:, :, None] / 255
    a = a * (1 - sh * (1 - g) * 0.5)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([gx0, gy0, gx1, gy1], 34, outline=(255, 255, 255, 150), width=2)

    # текст справа
    tx = bbox[2] + 110
    maxw = 2040 - tx
    size = 120
    while ms(size, 800).getlength("Capital Crimes.") > maxw:
        size -= 2
    d.text((tx, 650), "Capital Crimes.", font=ms(size, 800), fill=INK, anchor="lm")
    f2 = ms(34, 400)
    left, right = "B Y", "F R I Z E S"
    wl = f2.getlength(left)
    y2 = 800
    d.text((tx, y2), left, font=f2, fill=(40, 40, 44), anchor="lm")
    lx = tx + wl + 30
    d.line([(lx, y2 - 48), (lx, y2 + 48)], fill=(40, 40, 44), width=3)
    d.text((lx + 30, y2), right, font=f2, fill=(40, 40, 44), anchor="lm")

    img.save(os.path.join(HERE, "cc_glass_banner.png"))
    prev = img.resize((BW // 2, BH // 2), Image.LANCZOS)
    pd = ImageDraw.Draw(prev)
    sx, sy = (BW - 1546) / 4, (BH - 423) / 4
    pd.rectangle([sx, sy, sx + 1546 / 2, sy + 423 / 2], outline=(220, 60, 60), width=2)
    prev.save(os.path.join(HERE, "cc_glass_banner_safe_area.jpg"), quality=90)


if __name__ == "__main__":
    main()

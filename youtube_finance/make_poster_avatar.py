"""Аватарка в стиле шапки cc_poster_banner: блок «стекло + CC» (и вариант с BY | FRIZES)
из отредактированного постера (cc_poster.png). 1080×1080 PNG.
Запуск из корня репозитория (после make_poster_cc.py): python3 youtube_finance/make_poster_avatar.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = 1080
S = 2   # cc_poster.png — постер в двойном разрешении


def background(poster):
    p = np.asarray(poster, np.float32)
    col = p[:, :60 * S].mean(axis=(0, 1))
    ys = np.linspace(-1, 1, OUT)[:, None, None]
    xs = np.linspace(-1, 1, OUT)[None, :, None]
    bg = col[None, None, :] * (1 - (ys ** 2 + xs ** 2) * 0.025) - ys * 4
    bg = bg + np.random.default_rng(5).normal(0, 1.0, (OUT, OUT, 1))
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))


def paste_crop(canvas, poster, box, height, cx, cy, pad=26):
    crop = poster.crop([v * S for v in box])
    k = height / crop.height
    crop = crop.resize((int(crop.width * k), height), Image.LANCZOS)
    m = Image.new("L", crop.size, 0)
    ImageDraw.Draw(m).rectangle([pad, pad, crop.width - pad, crop.height - pad], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(pad / 2))
    canvas.paste(crop, (int(cx - crop.width / 2), int(cy - crop.height / 2)), m)


def circle_preview(img, name):
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).ellipse([0, 0, OUT, OUT], fill=255)
    prev = Image.new("RGB", img.size, (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


def square(poster, cx, cy, size):
    """Квадрат из постера; где не хватает ширины — фон постера продлевается краевыми пикселями."""
    p = np.asarray(poster)
    pad = max(0, int(size * S / 2 - min(cx * S, p.shape[1] - cx * S)) + 4)
    if pad:
        p = np.pad(p, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        p = np.asarray(Image.fromarray(p))
    x0 = int(cx * S + pad - size * S / 2)
    y0 = int(cy * S - size * S / 2)
    crop = Image.fromarray(p[y0:y0 + size * S, x0:x0 + size * S])
    return crop.resize((OUT, OUT), Image.LANCZOS)


def main():
    poster = Image.open(os.path.join(HERE, "cc_poster.png")).convert("RGB")
    a = square(poster, 330, 560, 780)                 # стекло + CC
    a.save(os.path.join(HERE, "cc_poster_avatar.png"))
    b = square(poster, 340, 742, 1110)                # стекло + CC + BY | FRIZES
    b.save(os.path.join(HERE, "cc_poster_avatar_by_frizes.png"))
    sheet = Image.new("RGB", (OUT * 2 + 60, OUT + 40), (255, 255, 255))
    for i, im in enumerate((a, b)):
        sheet.paste(circle_preview(im, i), (20 + i * (OUT + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(
        os.path.join(HERE, "cc_poster_avatar_preview.jpg"), quality=90)


if __name__ == "__main__":
    main()

"""Аватарки «CC» на фото автора (youtube_finance/photos/*.jpg). 1080×1080 PNG.

1) hoodie: надпись на худи заменяется на «CC / Capital Crimes / by Frizes»;
2) forest: светящиеся «CC» и «by Frizes» в лучах;
3) knight: золотые «CC» и «by Frizes» в закатном небе.
Запуск из корня репозитория: python3 youtube_finance/make_cc_photos.py
"""
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PH = os.path.join(HERE, "photos")
OUT = 1080
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
PLAYFAIR = os.path.join(HERE, "..", "president", "fonts", "PlayfairDisplay[wght].ttf")


def playfair(size, w=800):
    f = ImageFont.truetype(PLAYFAIR, size)
    f.set_variation_by_axes([w])
    return f


def square(img, top):
    w = img.width
    return img.crop((0, top, w, top + w))


def text_layer(size, items, angle=0.0):
    """items: (x, y, text, font, rgba). Возвращает RGBA-слой, повёрнутый на angle."""
    lay = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for x, y, txt, f, col in items:
        d.text((x, y), txt, font=f, fill=col, anchor="mm")
    return lay.rotate(angle, Image.BICUBIC, center=(items[0][0], items[0][1])) if angle else lay


def glow_text(img, items, blur, strength):
    lay = text_layer(img.size, items)
    glow = Image.new("RGB", img.size)
    glow.paste(lay.convert("RGB"), (0, 0), lay)
    glow = glow.filter(ImageFilter.GaussianBlur(blur))
    img = ImageChops.add(img, Image.fromarray(np.clip(np.asarray(glow, np.float32) * strength, 0, 255).astype(np.uint8)))
    img.paste(lay, (0, 0), lay)
    return img


def circle_preview(img):
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).ellipse([0, 0, img.width, img.height], fill=255)
    prev = Image.new("RGB", img.size, (235, 235, 235))
    prev.paste(img, (0, 0), m)
    return prev


def hoodie():
    img = Image.open(os.path.join(PH, "hoodie.jpg")).convert("RGB")
    a = np.asarray(img, np.float32)
    # 1) стираем «THE CREATOR / The Creator within you»
    x0, y0, x1, y1 = 305, 430, 548, 528
    lum = a.mean(axis=2)
    mask = np.zeros(lum.shape, bool)
    mask[y0:y1, x0:x1] = lum[y0:y1, x0:x1] > 55
    m = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    m = np.asarray(m, np.float32) / 255
    keep = 1 - m
    num = np.stack([np.asarray(Image.fromarray(np.clip(a[:, :, c] * keep, 0, 255).astype(np.uint8))
                               .filter(ImageFilter.GaussianBlur(9)), np.float32) for c in range(3)], 2)
    den = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)), np.float32)[:, :, None] / 255
    fill = num / np.maximum(den, 1e-3)
    fill += np.random.default_rng(2).normal(0, 2.2, fill.shape[:2])[:, :, None]
    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32)[:, :, None] / 255
    a = a * (1 - soft) + fill * soft
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    # 2) печатаем «CC / Capital Crimes» в том же стиле и с тем же наклоном
    items = [(428, 466, "CC", ImageFont.truetype(SERIF, 58), (178, 178, 182, 235)),
             (430, 505, "Capital Crimes", ImageFont.truetype(SERIF, 21), (168, 168, 172, 225)),
             (432, 531, "by Frizes", ImageFont.truetype(SERIF, 15), (150, 150, 156, 215))]
    lay = text_layer(img.size, items, angle=2.2).filter(ImageFilter.GaussianBlur(0.5))
    img.paste(lay, (0, 0), lay)
    img = square(img, 130).resize((OUT, OUT), Image.LANCZOS)
    img.save(os.path.join(HERE, "cc_photo_1_hoodie.png"))
    return img


def forest():
    img = Image.open(os.path.join(PH, "forest.jpg")).convert("RGB")
    img = square(img, 140)
    items = [(600, 560, "CC", playfair(300, 800), (255, 250, 236, 235)),
             (600, 775, "by Frizes", playfair(64, 500), (255, 250, 236, 225))]
    img = glow_text(img, items, 18, 0.35)
    img = img.resize((OUT, OUT), Image.LANCZOS)
    img.save(os.path.join(HERE, "cc_photo_2_forest.png"))
    return img


def knight():
    img = Image.open(os.path.join(PH, "knight.jpg")).convert("RGB")
    img = square(img, 110)
    items = [(560, 230, "CC", playfair(250, 800), (255, 214, 150, 250)),
             (560, 380, "by Frizes", playfair(54, 500), (255, 214, 150, 235))]
    lay = text_layer(img.size, items)
    glow = Image.new("RGB", img.size)
    glow.paste(Image.new("RGB", img.size, (230, 120, 40)), (0, 0), lay)
    glow = glow.filter(ImageFilter.GaussianBlur(24))
    img = ImageChops.add(img, Image.fromarray(np.clip(np.asarray(glow, np.float32) * 0.9, 0, 255).astype(np.uint8)))
    img.paste(lay, (0, 0), lay)
    img = img.resize((OUT, OUT), Image.LANCZOS)
    img.save(os.path.join(HERE, "cc_photo_3_knight.png"))
    return img


def forest_banner():
    """Шапка канала 2560×1440 из фото леса; надписи в безопасной зоне 1546×423."""
    BW, BH = 2560, 1440
    src = Image.open(os.path.join(PH, "forest.jpg")).convert("RGB")
    k = BW / src.width
    big = src.resize((BW, int(src.height * k)), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 60, 2))
    img = big.crop((0, 880, BW, 880 + BH))
    a = np.asarray(img, np.float32)
    ys = np.linspace(-1, 1, BH)[:, None]
    xs = np.linspace(-1, 1, BW)[None, :]
    a *= np.clip(1.15 - (xs ** 2 * 0.35 + ys ** 2 * 0.45), 0.55, 1)[:, :, None]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    items = [(BW // 2, 660, "CC", playfair(260, 800), (255, 250, 236, 240)),
             (BW // 2, 845, "by Frizes", playfair(64, 500), (255, 250, 236, 230))]
    img = glow_text(img, items, 18, 0.35)
    img.save(os.path.join(HERE, "cc_forest_banner.png"))
    prev = img.resize((BW // 2, BH // 2), Image.LANCZOS)
    d = ImageDraw.Draw(prev)
    sx, sy = (BW - 1546) / 4, (BH - 423) / 4
    d.rectangle([sx, sy, sx + 1546 / 2, sy + 423 / 2], outline=(255, 255, 255), width=2)
    prev.save(os.path.join(HERE, "cc_forest_banner_safe_area.jpg"), quality=90)


if __name__ == "__main__":
    prevs = [circle_preview(f()) for f in (hoodie, forest, knight)]
    sheet = Image.new("RGB", (OUT * 3 + 80, OUT + 40), (235, 235, 235))
    for i, p in enumerate(prevs):
        sheet.paste(p, (20 + i * (OUT + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "cc_photo_preview.jpg"), quality=90)
    forest_banner()

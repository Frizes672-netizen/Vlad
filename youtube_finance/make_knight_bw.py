"""Аватарка YouTube из ч/б фото рыцаря (photos/knight_bw.jpg) с надписью «by Frizes» сверху.
1080×1080 PNG. Запуск из корня репозитория: python3 youtube_finance/make_knight_bw.py
"""
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYFAIR = os.path.join(HERE, "..", "president", "fonts", "PlayfairDisplay[wght].ttf")
OUT = 1080


def main():
    src = Image.open(os.path.join(HERE, "photos", "knight_bw.jpg")).convert("RGB")
    w = src.width
    top = 290                                   # квадрат: небо сверху, рыцарь по центру
    img = src.crop((0, top, w, top + w)).resize((OUT, OUT), Image.LANCZOS)
    f = ImageFont.truetype(PLAYFAIR, 84)
    f.set_variation_by_axes([600])
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).text((OUT // 2, 150), "by Frizes", font=f, fill=(242, 242, 238, 245), anchor="mm")
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 200), (0, 0), lay.split()[3])
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    glow = Image.new("RGB", img.size)
    glow.paste((255, 255, 255), (0, 0), lay.split()[3])
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    img = ImageChops.add(img, Image.fromarray((np.asarray(glow, np.float32) * 0.25).astype(np.uint8)))
    img.paste(shadow, (0, 4), shadow)
    img.paste(lay, (0, 0), lay)
    img.save(os.path.join(HERE, "frizes_knight_avatar.png"))
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).ellipse([0, 0, OUT, OUT], fill=255)
    prev = Image.new("RGB", img.size, (235, 235, 235))
    prev.paste(img, (0, 0), m)
    prev.resize((540, 540), Image.LANCZOS).save(os.path.join(HERE, "frizes_knight_avatar_preview.jpg"), quality=90)


if __name__ == "__main__":
    main()

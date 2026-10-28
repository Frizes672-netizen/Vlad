"""Аватарка YouTube из ч/б фото рыцаря (photos/knight_bw.jpg) с надписью «by Frizes» сверху.
1080×1080 PNG. Запуск из корня репозитория: python3 youtube_finance/make_knight_bw.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CINZEL = os.path.join(HERE, "..", "president", "fonts", "CinzelDecorative-Bold.ttf")
OUT = 1080


def main():
    src = Image.open(os.path.join(HERE, "photos", "knight_bw.jpg")).convert("RGB")
    w = src.width
    top = 290                                   # квадрат: небо сверху, рыцарь по центру
    img = src.crop((0, top, w, top + w)).resize((OUT, OUT), Image.LANCZOS)
    f = ImageFont.truetype(CINZEL, 104)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).text((OUT // 2, 155), "by Frizes", font=f, fill=255, anchor="mm")
    m = np.asarray(mask, np.float32)[:, :, None] / 255
    a = np.asarray(img, np.float32)

    def blur(r):
        return np.asarray(mask.filter(ImageFilter.GaussianBlur(r)), np.float32)[:, :, None] / 255

    # свет: широкий мягкий ореол + яркое ближнее свечение
    a = a + blur(50) * 140 * np.array([1.0, 0.98, 0.92]) + blur(18) * 70 * np.array([1.0, 0.99, 0.95])
    # тень под буквами держит контур внутри ореола
    a = a * (1 - blur(4) * 0.55)
    # буквы: перелив от белого сверху к серебру снизу и блик по центру
    ys = np.linspace(0, 1, OUT)[:, None, None]
    band = np.clip(1 - np.abs(ys - 155 / OUT) * OUT / 60, 0, 1)
    letters = np.array([255, 253, 245], np.float32) * (0.82 + 0.18 * band)
    a = a * (1 - m) + letters * m
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    img.save(os.path.join(HERE, "frizes_knight_avatar.png"))
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).ellipse([0, 0, OUT, OUT], fill=255)
    prev = Image.new("RGB", img.size, (235, 235, 235))
    prev.paste(img, (0, 0), m)
    prev.resize((540, 540), Image.LANCZOS).save(os.path.join(HERE, "frizes_knight_avatar_preview.jpg"), quality=90)


if __name__ == "__main__":
    main()

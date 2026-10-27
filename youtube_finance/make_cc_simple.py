"""Простые аватарки «CC» (Capital Crimes): однотонный фон и крупная монограмма.

1080×1080 PNG. Запуск из корня репозитория: python3 youtube_finance/make_cc_simple.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "president", "fonts", "Montserrat[wght].ttf")
SIZE, SS = 1080, 2           # рисуем вдвое крупнее для гладких краёв

BLACK, WHITE, YEL, RED = (14, 14, 18), (250, 250, 250), (255, 210, 40), (226, 46, 50)


def font(size):
    f = ImageFont.truetype(FONT, size * SS)
    f.set_variation_by_axes([900])
    return f


def make(name, bg, c1, c2, arrow=None):
    img = Image.new("RGB", (SIZE * SS, SIZE * SS), bg)
    d = ImageDraw.Draw(img)
    f = font(520)
    w1 = d.textlength("C", font=f)
    gap = -40 * SS
    x = (SIZE * SS - (2 * w1 + gap)) / 2
    y = SIZE * SS / 2 - 10 * SS
    d.text((x, y), "C", font=f, fill=c1, anchor="lm")
    d.text((x + w1 + gap, y), "C", font=f, fill=c2, anchor="lm", stroke_width=16 * SS, stroke_fill=bg)
    if arrow:
        pts = [(x + w1 * 0.55, y + 40 * SS), (x + w1 * 0.95, y - 50 * SS), (x + w1 * 1.25, y + 10 * SS),
               (x + w1 * 1.6, y + 170 * SS)]
        d.line(pts, fill=arrow, width=30 * SS, joint="curve")
        import math
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        a = math.atan2(by - ay, bx - ax)
        L = 70 * SS
        d.polygon([(bx + math.cos(a) * L, by + math.sin(a) * L),
                   (bx + math.cos(a + 2.3) * L, by + math.sin(a + 2.3) * L),
                   (bx + math.cos(a - 2.3) * L, by + math.sin(a - 2.3) * L)], fill=arrow)
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    m = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(m).ellipse([0, 0, SIZE, SIZE], fill=255)
    prev = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    prev.paste(img, (0, 0), m)
    return prev


if __name__ == "__main__":
    previews = [
        make("cc_simple_1_black", BLACK, WHITE, YEL),
        make("cc_simple_2_yellow", YEL, BLACK, BLACK),
        make("cc_simple_3_red", RED, WHITE, WHITE),
        make("cc_simple_4_black_arrow", BLACK, WHITE, WHITE, arrow=RED),
    ]
    sheet = Image.new("RGB", (SIZE * 4 + 100, SIZE + 40), (235, 235, 235))
    for i, p in enumerate(previews):
        sheet.paste(p, (20 + i * (SIZE + 20), 20))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(HERE, "cc_simple_preview.jpg"), quality=90)

"""Постер-референс (photos/poster.jpg), в котором заменены только надписи:
$ -> CC (размыто под стеклом, чётко снаружи), «Make money.» -> «Capital Crimes.»,
«ALWAYS | FIRST» -> «BY | FRIZES». Фон, стекло и черта — с исходного фото.
Результат: cc_poster.png (как исходник, 2×) и cc_poster_banner.png (2560×1440).
Запуск из корня репозитория: python3 youtube_finance/make_poster_cc.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "president", "fonts", "Montserrat[wght].ttf")
GLASS = (88, 258, 355, 868)          # матовое стекло на исходнике (x0, y0, x1, y1)


def ms(size, w):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([w])
    return f


def fill_holes(a, mask, region, radius):
    """Заливает пиксели mask гладким продолжением соседних (только внутри region)."""
    keep = (~mask & region).astype(np.float32)
    out = a.copy()
    den = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255
    for c in range(3):
        num = np.asarray(Image.fromarray(np.clip(a[:, :, c] * keep, 0, 255).astype(np.uint8))
                         .filter(ImageFilter.GaussianBlur(radius)), np.float32)
        fill = num / np.maximum(den, 1e-4)
        out[:, :, c] = np.where(mask & region, fill, a[:, :, c])
    return out


def edit(scale=2):
    src = Image.open(os.path.join(HERE, "photos", "poster.jpg")).convert("RGB")
    a = np.asarray(src, np.float32)
    H, W = a.shape[:2]
    lum = a.mean(axis=2)
    yy, xx = np.mgrid[0:H, 0:W]
    gx0, gy0, gx1, gy1 = GLASS
    in_glass = (xx > gx0 + 3) & (xx < gx1 - 1) & (yy > gy0 + 3) & (yy < gy1 - 3)
    outside = ~((xx >= gx0) & (xx <= gx1 + 1) & (yy >= gy0) & (yy <= gy1))

    def dilate(m, k):
        return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k)), bool)

    # 1) убрать доллар: стекло заливаем по его чистому левому краю (построчно),
    #    фон справа от стекла — по чистому фону той же строки
    glass_col = a[:, gx0 + 8:gx0 + 16].mean(axis=1)          # цвет стекла по строкам
    bg_col = a[:, 560:600].mean(axis=1)                       # цвет фона по строкам
    for y in range(gy0 + 4, gy1 - 3):
        a[y, gx0 + 16:gx1 - 1] = glass_col[y]
    for y in range(300, 830):
        a[y, gx1 + 1:540] = bg_col[y]
    # тонкая светлая кромка стекла, как на исходнике
    a[gy0 + 4:gy1 - 3, gx1 - 1:gx1 + 1] = np.maximum(a[gy0 + 4:gy1 - 3, gx1 - 1:gx1 + 1], 234)
    # 2) убрать «Make money.» и слова ALWAYS / FIRST (черту оставляем)
    text1 = (lum < 170) & (yy > 940) & (yy < 1035) & (xx > 130) & (xx < 610)
    words = (lum < 170) & (yy > 1125) & (yy < 1178) & (((xx > 195) & (xx < 358)) | ((xx > 377) & (xx < 525)))
    a = fill_holes(a, dilate(text1 | words, 7), outside, 18)
    a += np.random.default_rng(4).normal(0, 1.0, a.shape[:2])[:, :, None]

    # всё дальнейшее — в 2× для чистых краёв букв
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((W * scale, H * scale), Image.LANCZOS)
    a = np.asarray(img, np.float32)
    S = scale
    # 3) CC: левая часть — размытая под стеклом, правая — чёткая
    cc = Image.new("L", img.size, 0)
    f = ms(300 * S, 900)
    cd = ImageDraw.Draw(cc)
    w = cd.textlength("CC", font=f)
    x_end = 568 * S
    cd.text((x_end - w, 560 * S), "CC", font=f, fill=255, anchor="lm")
    yy, xx = np.mgrid[0:img.height, 0:img.width]
    g_in = ((xx >= gx0 * S) & (xx <= gx1 * S) & (yy >= gy0 * S) & (yy <= gy1 * S))[:, :, None]
    sharp = np.asarray(cc, np.float32)[:, :, None] / 255
    soft = np.asarray(cc.filter(ImageFilter.GaussianBlur(30 * S)), np.float32)[:, :, None] / 255
    soft = np.clip(soft * 1.25, 0, 0.93)
    ink = np.array([8, 8, 10], np.float32)
    a = np.where(g_in, a * (1 - soft) + np.array([40, 42, 46], np.float32) * soft, a * (1 - sharp) + ink * sharp)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # 4) «Capital Crimes.» на месте «Make money.»
    size = 80
    while ms(size * S, 800).getlength("Capital Crimes.") > 520 * S:
        size -= 1
    d.text((368 * S, 990 * S), "Capital Crimes.", font=ms(size * S, 800), fill=(8, 8, 10), anchor="mm")
    # 5) «BY | FRIZES» с разрядкой, черта — родная
    f2 = ms(21 * S, 500)
    track = 9 * S

    def spaced_width(txt):
        return sum(f2.getlength(ch) for ch in txt) + track * (len(txt) - 1)

    def spaced(x, y, txt):
        for ch in txt:
            d.text((x, y), ch, font=f2, fill=(30, 30, 34), anchor="lm")
            x += f2.getlength(ch) + track
    line_x = 367 * S
    spaced(line_x - 22 * S - spaced_width("BY"), 1152 * S, "BY")
    spaced(line_x + 22 * S, 1152 * S, "FRIZES")
    img.save(os.path.join(HERE, "cc_poster.png"))
    return img


def banner(poster, S=2):
    """Шапка 2560×1440: фон и элементы взяты с отредактированного постера."""
    BW, BH = 2560, 1440
    p = np.asarray(poster, np.float32)
    # фон: средние цвета по строкам постера (вне элементов) растягиваем по ширине
    edge = np.concatenate([p[:, :40 * S], p[:, -40 * S:]], axis=1).mean(axis=1)      # (h, 3)
    rows = np.asarray(Image.fromarray(edge[None].astype(np.uint8)).resize((BH, 1), Image.BILINEAR), np.float32)[0]
    rows = np.clip(rows, 0, 255)
    xs = np.abs(np.linspace(-1, 1, BW))[None, :, None]
    bg = rows[:, None, :] * (1 - xs * 0.03)
    bg = bg + np.random.default_rng(5).normal(0, 1.0, (BH, BW))[:, :, None]
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))

    def paste_crop(box, height, x, y):
        crop = poster.crop([v * S for v in box])
        k = height / crop.height
        crop = crop.resize((int(crop.width * k), height), Image.LANCZOS)
        m = Image.new("L", crop.size, 0)
        pad = 24
        ImageDraw.Draw(m).rectangle([pad, pad, crop.width - pad, crop.height - pad], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(pad / 2))
        canvas.paste(crop, (x, y), m)
        return crop.size

    gw, gh = paste_crop((0, 190, 660, 930), 480, 470, 480)
    paste_crop((60, 920, 680, 1250), 320, 470 + gw - 20, 560)
    canvas.save(os.path.join(HERE, "cc_poster_banner.png"))
    prev = canvas.resize((BW // 2, BH // 2), Image.LANCZOS)
    pd = ImageDraw.Draw(prev)
    sx, sy = (BW - 1546) / 4, (BH - 423) / 4
    pd.rectangle([sx, sy, sx + 1546 / 2, sy + 423 / 2], outline=(220, 60, 60), width=2)
    prev.save(os.path.join(HERE, "cc_poster_banner_safe_area.jpg"), quality=90)


if __name__ == "__main__":
    banner(edit())

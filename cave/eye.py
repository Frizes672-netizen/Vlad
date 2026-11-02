"""Реалистичный глаз крупным планом: радужка с волокнами, зрачок, блики, сосуды, веки с ресницами."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine import fbm, to_img


@lru_cache(None)
def _iris_tex(size, seed=3):
    r = np.random.default_rng(seed)
    n = 720
    # волокна: шум по углу, растянутый по радиусу
    ang = r.random((n,)).astype(np.float32)
    ang = np.convolve(np.concatenate([ang, ang[:9]]), np.ones(9) / 9, "valid")[:n]
    fib = np.asarray(Image.fromarray((fbm(n, 64, [40, 120, 360], seed) * 255).astype(np.uint8)), np.float32) / 255
    return ang, fib


@lru_cache(None)
def _veins(size, seed=7):
    """Сосуды склеры: ветвящиеся линии от краёв к радужке."""
    lay = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(lay)
    r = np.random.default_rng(seed)
    c = size / 2

    def branch(x, y, ang, L, w, depth):
        pts = [(x, y)]
        for _ in range(int(L / 6)):
            ang += r.uniform(-0.35, 0.35)
            x += math.cos(ang) * 6
            y += math.sin(ang) * 6
            pts.append((x, y))
        d.line(pts, fill=255, width=max(1, int(w)))
        if depth > 0:
            for _ in range(2):
                i = r.integers(len(pts) // 3, len(pts))
                branch(pts[i][0], pts[i][1], ang + r.uniform(-1.0, 1.0), L * 0.55, w * 0.6, depth - 1)
    for k in range(26):
        a = r.uniform(0, 2 * math.pi)
        x, y = c + math.cos(a) * size * 0.49, c + math.sin(a) * size * 0.32
        branch(x, y, a + math.pi + r.uniform(-0.4, 0.4), size * r.uniform(0.18, 0.3), r.uniform(2, 4.5), 3)
    return np.asarray(lay.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255


def eye(size, t, red=0.0, open_k=1.0, pupil=0.36, look=(0.0, 0.0), skin=(196, 150, 124)):
    """Рендер глаза size×size. red 0..1 — сколько лопнувших сосудов, open_k — насколько открыт."""
    s = size
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx, cy = s / 2, s / 2
    # склера (белок) с лёгкой розовостью по краям
    rx, ry = s * 0.47, s * 0.30
    de = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    img = np.zeros((s, s, 3), np.float32)
    sclera = np.array([236, 230, 224], np.float32)
    img[:] = sclera
    img = img * (1 - 0.25 * np.clip(de, 0, 1)[:, :, None] ** 2) + np.clip(de, 0, 1)[:, :, None] ** 3 * np.array([30, -20, -20])
    veins = _veins(s)
    vk = 0.25 + 0.75 * red
    img = img * (1 - veins[:, :, None] * vk) + veins[:, :, None] * vk * np.array([170, 30, 34])
    img = img * (1 - red * 0.25 * np.clip(de, 0, 1)[:, :, None]) + red * 0.25 * np.clip(de, 0, 1)[:, :, None] * np.array([200, 80, 80])
    # радужка
    ix, iy = cx + look[0] * s * 0.12, cy + look[1] * s * 0.06
    R = s * 0.2
    dx, dy = xx - ix, yy - iy
    rr = np.sqrt(dx * dx + dy * dy) / R
    th = (np.arctan2(dy, dx) / (2 * math.pi) + 0.5)
    ang, fib = _iris_tex(s)
    ai = (th * (len(ang) - 1)).astype(int)
    stripes = ang[ai]
    fi = fib[np.clip((rr * 63).astype(int), 0, 63), (th * 719).astype(int)]
    base = np.array([70, 110, 130], np.float32)            # серо-голубая
    warm = np.array([150, 120, 70], np.float32)            # тёплый ободок у зрачка
    col = base[None, None] * (0.6 + 0.6 * stripes[:, :, None] * 0.7 + 0.4 * fi[:, :, None])
    collar = np.exp(-((rr - pupil * 1.45) / 0.12) ** 2)
    col = col * (1 - collar[:, :, None] * 0.6) + warm * collar[:, :, None] * 0.6
    limbal = np.clip((rr - 0.85) / 0.15, 0, 1)
    col = col * (1 - limbal[:, :, None] * 0.75)
    m_iris = np.clip((1 - rr) * R / 1.5, 0, 1)
    img = img * (1 - m_iris[:, :, None]) + col * m_iris[:, :, None]
    # зрачок
    m_p = np.clip((pupil - rr) * R / 1.5, 0, 1)
    img = img * (1 - m_p[:, :, None]) + np.array([6, 6, 8]) * m_p[:, :, None]
    # тень от верхнего века на глазное яблоко
    img = img * (1 - 0.35 * np.clip((cy - yy) / (s * 0.3), 0, 1)[:, :, None])
    im = to_img(img)
    d = ImageDraw.Draw(im, "RGBA")
    # блики: окно и маленькая точка
    bx, by = ix - R * 0.35, iy - R * 0.4
    d.rounded_rectangle([bx - R * 0.16, by - R * 0.12, bx + R * 0.16, by + R * 0.12], R * 0.04, fill=(255, 255, 255, 235))
    d.ellipse([ix + R * 0.3, iy + R * 0.25, ix + R * 0.38, iy + R * 0.33], fill=(255, 255, 255, 180))
    # веки: кожа сверху и снизу, щель зависит от open_k
    gap = ry * 1.05 * open_k
    upper = [(0, 0), (s, 0), (s, cy - gap * 0.2)] + [(x, cy - gap * math.sin(math.pi * x / s) - gap * 0.05) for x in np.linspace(s, 0, 40)]
    lower = [(0, s), (s, s), (s, cy + gap * 0.25)] + [(x, cy + gap * 0.82 * math.sin(math.pi * x / s) + gap * 0.05) for x in np.linspace(s, 0, 40)]
    lid = Image.new("RGB", (s, s))
    ln = fbm(s, s, [8, 40, 140], 21)
    la = np.array(skin, np.float32) * (0.8 + 0.3 * ln[:, :, None])
    shade = np.clip(1 - np.abs(yy - cy) / (s * 0.5), 0, 1)
    la = la * (0.75 + 0.35 * (1 - shade[:, :, None]))
    lid = to_img(la)
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    md.polygon(upper, fill=255)
    md.polygon(lower, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(s * 0.004))
    im.paste(lid, (0, 0), mask)
    d = ImageDraw.Draw(im, "RGBA")
    # тень и складка века, ресницы
    for k in range(6):
        pts = [(x, cy - gap * math.sin(math.pi * x / s) - gap * 0.05 - k * 3) for x in np.linspace(0, s, 40)]
        d.line(pts, fill=(60, 30, 26, 60 - k * 9), width=4)
    crease = [(x, cy - (gap + ry * 0.35) * math.sin(math.pi * x / s) - ry * 0.25) for x in np.linspace(s * 0.1, s * 0.9, 30)]
    d.line(crease, fill=(120, 80, 66, 120), width=max(2, s // 220))
    r = np.random.default_rng(4)
    for x in np.linspace(s * 0.12, s * 0.88, 70):
        y0 = cy - gap * math.sin(math.pi * x / s) - gap * 0.05
        L = s * 0.07 * (0.6 + 0.4 * math.sin(math.pi * x / s)) * r.uniform(0.8, 1.1)
        a = -math.pi / 2 + (x - s / 2) / s * 1.4 + r.uniform(-0.1, 0.1)
        d.line([(x, y0), (x + math.cos(a) * L * 0.6, y0 + math.sin(a) * L * 0.5), (x + math.cos(a) * L, y0 + math.sin(a) * L)],
               fill=(20, 14, 12, 230), width=max(1, s // 400))
    for x in np.linspace(s * 0.2, s * 0.8, 40):
        y0 = cy + gap * 0.82 * math.sin(math.pi * x / s) + gap * 0.05
        d.line([(x, y0), (x + (x - s / 2) / s * 20, y0 + s * 0.025)], fill=(30, 20, 18, 160), width=1)
    return im.filter(ImageFilter.GaussianBlur(s / 900))

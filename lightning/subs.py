"""Субтитры в стиле Shorts: группы по 2–4 слова, текущее слово подсвечивается."""
from functools import lru_cache

from PIL import ImageDraw

from engine import MS, eout, words
from timeline import v

FIX = {"поладка,": "палатка,", "вахуя.": "в ахуе.", "все,": "всё.", "подсчетам": "подсчётам"}


@lru_cache(None)
def tokens():
    """Слова из разметки с исправлениями распознавания и склейкой «62 -м» → «62-м»."""
    out = []
    for line in open("voice/words.tsv", encoding="utf-8"):
        s, e, w = line.rstrip("\n").split("\t")
        s, e = v(float(s)), v(float(e))
        lw = w.lower()
        w = FIX.get(lw, w)
        if w.startswith("-") and out:
            ps, pe, pw = out[-1]
            out[-1] = (ps, e, {"14-я,": "14-е,"}.get(pw + w, pw + w))
            continue
        out.append((s, e, w))
    return out


@lru_cache(None)
def groups():
    """Группы по 2–4 слова: разрыв на паузе, знаке препинания или длине."""
    res, cur = [], []
    toks = tokens()
    for i, (s, e, w) in enumerate(toks):
        cur.append((s, e, w))
        gap = toks[i + 1][0] - e if i + 1 < len(toks) else 9
        chars = sum(len(x[2]) for x in cur)
        if len(cur) >= 4 or chars > 18 or gap > 0.35 or w[-1] in ".?!,—" and len(cur) >= 2:
            res.append(cur)
            cur = []
    if cur:
        res.append(cur)
    return res


def draw_subs(im, t, y=1240, hide=()):
    for a, b in hide:
        if a <= t < b:
            return
    for g in groups():
        a, b = g[0][0] - 0.05, g[-1][1] + 0.25
        if a <= t < b:
            d = ImageDraw.Draw(im, "RGBA")
            f = MS(70, 800)
            text = [w.upper() for _, _, w in g]
            widths = [d.textlength(w, font=f) for w in text]
            space = d.textlength(" ", font=f)
            total = sum(widths) + space * (len(text) - 1)
            scale = min(1.0, 980 / total)
            if scale < 1:
                f = MS(int(70 * scale), 800)
                widths = [d.textlength(w, font=f) for w in text]
                space = d.textlength(" ", font=f)
                total = sum(widths) + space * (len(text) - 1)
            pop = 1.0 - 0.12 * (1 - eout((t - a) / 0.15))
            x = 540 - total / 2
            for (s, e, _), w, ww in zip(g, text, widths):
                active = s - 0.03 <= t < e + 0.05
                col = (255, 214, 60) if active else (255, 255, 255)
                d.text((x + ww / 2, y), w, font=f, fill=col, anchor="mm", stroke_width=7, stroke_fill=(0, 0, 0))
                x += ww + space
            return

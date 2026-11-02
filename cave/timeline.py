"""Таймлиния шортса «Два месяца в пещере без часов» (Мишель Сифр, 1962).

Озвучка: voice/voiceover.mp3 (ускорена в 1.1 раза из voiceover_orig.mp3), разметка: voice/words.tsv.
"""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
VOICE_LEN = 80.74
DURATION = round(VOICE_OFFSET + VOICE_LEN + 1.6, 2)


def v(x):
    """Время в озвучке -> время ролика."""
    return round(x + VOICE_OFFSET, 3)


SCENES = [
    (-0.3, "hook"), (8.5, "y1962"), (16.2, "descent"), (24.0, "rule"), (31.6, "week1"), (39.2, "confusion"),
    (52.3, "call"), (57.2, "dates"), (63.2, "lost"), (70.6, "teaser"), (77.8, "cta"),
]

QUOTES = [(53.8, 55.9, "СВЕРХУ", "Всё. Эксперимент окончен.")]

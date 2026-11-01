"""Таймлиния шортса «What happens if you don't sleep for 11 days» (англ. озвучка).

Озвучка: voice/voiceover.mp3, пословная разметка: voice/words.tsv.
"""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
VOICE_LEN = 108.36
DURATION = round(VOICE_OFFSET + VOICE_LEN + 1.8, 2)


def v(x):
    """Время в озвучке -> время ролика."""
    return round(x + VOICE_OFFSET, 3)


SCENES = [
    (-0.3, "hook"), (3.4, "y1964"), (14.9, "brain"), (17.6, "randy"), (28.8, "day1"), (33.6, "day2"),
    (41.7, "day3"), (49.0, "day4"), (60.0, "day5"), (69.3, "day11"), (84.1, "record"), (94.6, "later"),
    (99.3, "guinness"), (105.2, "question"),
]

QUOTES = []

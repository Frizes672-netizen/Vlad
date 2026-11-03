"""Таймлиния шортса «Struck by Lightning 7 Times» (Рой Салливан). Англ. озвучка voice/voiceover.mp3."""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
VOICE_LEN = 70.43
DURATION = round(VOICE_OFFSET + VOICE_LEN + 1.6, 2)


def v(x):
    return round(x + VOICE_OFFSET, 3)


SCENES = [
    (-0.3, "hook"), (6.3, "roy"), (10.9, "s1"), (19.6, "s2"), (24.9, "s3"), (30.2, "s4"), (36.6, "s5"),
    (41.8, "avoid"), (47.2, "s6"), (49.9, "s7"), (60.4, "odds"), (64.9, "hats"), (68.3, "question"),
]

QUOTES = []

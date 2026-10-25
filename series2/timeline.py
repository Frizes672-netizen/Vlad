"""Таймлиния серии 2 «Он знает моё имя».

Монтаж выровнен по озвучке voice/voiceover.mp3 (фразы распознаны по словам,
см. build/words.tsv). Все времена — секунды от начала ролика.
"""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
DURATION = 103.5

# крупные пословные субтитры в стиле референса (по умолчанию выключены)
CAPTIONS = False


def v(x):
    """Время из озвучки -> время ролика."""
    return round(x + VOICE_OFFSET, 3)


EV = {
    # ХИЖИНА
    "grab": v(3.4),
    "door_hits": tuple(v(x) for x in (0.4, 1.6, 2.9, 4.4, 6.4, 8.2)),
    "door_off": v(9.35),
    # ЛЮК
    "hatch_shot": v(11.0),
    "hatch_open": v(11.3),
    "jump": v(12.6),
    "below": v(16.5),
    "smile": v(17.8),
    "lid_close": v(22.55),
    "dark": v(22.75),
    # ТОННЕЛЬ
    "tunnel": v(24.9),
    "door_stop": v(28.3),
    "roar": v(35.3),
    "drop": v(37.9),
    # БУНКЕР
    "bunker": v(39.5),
    "slam": v(39.75),
    "board": v(41.2),
    "photo_zoom": v(42.6),
    "label": v(45.3),
    "talk_bunker": v(48.7),
    "look_away": v(52.4),
    # ЛАБОРАТОРИЯ
    "lab": v(57.3),
    "pc_on": v(58.0),
    "screen": v(59.3),
    "two_lab": v(67.4),
    "turn": v(67.8),
    "gun": v(70.0),
    # ТРЕВОГА
    "alert": v(72.6),
    "beep": v(72.85),
    "found": v(76.3),
    "protocol": v(79.6),
    "siren": v(81.1),
    # ЗОМБИ ЗА СТЕКЛОМ
    "zombies_shot": v(83.4),
    "zombies": v(83.7),
    "kneel": v(88.0),
    "drop_gun": v(91.0),
    "creator": v(96.5),
    "black": v(98.0),
    "final": v(98.6),
}

FINAL_TEXT = "ПРОДОЛЖЕНИЕ СЛЕДУЕТ…"

# (начало, локация) — красная плашка слева
LOCATIONS = [
    (0.0, "ЗАБРОШЕННАЯ ХИЖИНА"),
    (EV["below"], "ПОД ПОЛОМ"),
    (EV["tunnel"], "СТАРЫЙ ТОННЕЛЬ"),
    (EV["bunker"], "ПОДЗЕМНЫЙ БУНКЕР № 7"),
    (EV["lab"], "ЛАБОРАТОРИЯ | СЕКТОР С"),
]

# глубина (время, метры) — значения между точками плавно анимируются
DEPTH = [(0, 0), (EV["jump"], 0), (EV["jump"] + 0.6, -4), (EV["tunnel"], -4),
         (EV["door_stop"], -31), (EV["bunker"], -38), (EV["lab"], -38), (EV["lab"] + 1.0, -52)]

# сколько зомби рядом (время, число)
ZOMBIES_NEAR = [(0, 1), (EV["drop"], 2), (EV["zombies"], 2), (EV["zombies"] + 2.5, 40)]

DAY = "ДЕНЬ 1"
CLOCK_START = (23, 48)       # часы идут: 1 игровая минута за 9 секунд
CLOCK_STEP = 9.0

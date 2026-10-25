"""Таймлиния серии 3 «Кто из нас настоящий».

Монтаж выровнен по озвучке voice/voiceover.mp3 (пословная разметка
в voice/words.tsv). Все времена — секунды от начала ролика.
"""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
DURATION = 172.0

CAPTIONS = False


def v(x):
    """Время из озвучки -> время ролика."""
    return round(x + VOICE_OFFSET, 3)


EV = {
    # ЗОМБИ НА КОЛЕНЯХ
    "step": v(9.7),
    "heads_up": v(11.5),
    "voice": v(13.9),
    "pain": v(19.0),
    # ВОСПОМИНАНИЕ
    "flash": v(23.5),
    "scientists": v(25.5),
    "glass": v(26.6),
    "self": v(28.05),
    "door": v(29.5),
    "press": v(31.4),
    "island": v(32.8),
    "back": v(35.4),
    "virus": v(37.15),
    # СИРЕНА
    "siren": v(39.1),
    "countdown": v(45.95),
    "grab": v(48.85),
    "run": v(53.2),
    "pc": v(55.0),
    "old_video": v(56.3),
    # РАЗГОВОР
    "confront": v(68.4),
    "closer": v(72.96),
    "shout": v(74.24),
    "look_down": v(75.8),
    "step_back": v(83.04),
    # ПРОТОКОЛ
    "protocol_shot": v(90.9),
    "beep": v(91.1),
    "protocol": v(95.6),
    "auth": v(100.35),
    "pale": v(102.82),
    "press2": v(109.3),
    # ЗАПИСЬ
    "tape": v(110.4),
    "mask_in": v(122.1),
    "unmask": v(124.7),
    "tape_cut": v(128.0),
    "truth": v(128.9),
    "tear": v(133.7),
    # ПРОБУЖДЕНИЕ
    "boom": v(138.9),
    "stand_up": v(141.25),
    "awake": v(145.5),
    "leave": v(147.46),
    "door_open": v(154.74),
    "double": v(157.86),
    "smile": v(161.0),
    "black": v(167.6),
    "final": v(168.1),
}

FINAL_TEXT = "ПРОДОЛЖЕНИЕ СЛЕДУЕТ…"

LOCATIONS = [
    (0.0, "ЛАБОРАТОРИЯ | СЕКТОР С"),
    (EV["flash"], "ВОСПОМИНАНИЕ"),
    (EV["back"], "ЛАБОРАТОРИЯ | СЕКТОР С"),
]

DEPTH = [(0, -52), (DURATION, -52)]
ZOMBIES_NEAR = [(0, 40), (EV["stand_up"], 40), (EV["double"], 40), (EV["double"] + 1, 41)]

# часы: продолжают серию 2 и переходят через полночь в ДЕНЬ 2
CLOCK_START = (23, 59)
CLOCK_STEP = 14.0
DAY_START = 1

# обратный отсчёт до взрыва комплекса (секунды ролика -> 10:00 и вниз)
COUNTDOWN_FROM = EV["countdown"]
COUNTDOWN_SECONDS = 600

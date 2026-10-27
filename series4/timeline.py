"""Таймлиния серии 4 (финал) «Апокалипсис только начался».

Монтаж выровнен по озвучке voice/voiceover.mp3 (пословная разметка
в voice/words.tsv). Все времена — секунды от начала ролика.
"""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
DURATION = 181.0

CAPTIONS = False


def v(x):
    """Время из озвучки -> время ролика."""
    return round(x + VOICE_OFFSET, 3)


EV = {
    "backoff": v(7.2),
    "grab": v(7.4),
    "to_pc": v(14.8),
    "screen": v(17.9),
    "footage": v(26.9),
    "sci2": v(33.7),
    "who": v(40.7),
    "fight": v(48.1),
    "fall": v(50.0),
    "pipe": v(55.3),
    "shot": v(58.9),
    "run": v(62.4),
    "confess": v(66.4),
    "remote": v(68.9),
    "worked": v(73.3),
    "nod": v(78.7),
    "couldnt": v(84.3),
    "alarm": v(85.1),
    "sixty": v(90.3),
    "corridor": v(93.6),
    "no_smile": v(104.0),
    "warn": v(108.3),
    "button": v(113.5),
    "press": v(120.3),
    "doors": v(121.6),
    "boom": v(131.3),
    "silence": v(135.16),
    "shore": v(137.5),
    "help_up": v(144.26),
    "veins": v(148.1),
    "inner": v(158.2),
    "smile": v(168.8),
    "final": v(173.9),
    "black": v(177.6),
}

FINAL_TEXT = ("АПОКАЛИПСИС НЕ ЗАКОНЧИЛСЯ.", "ОН ТОЛЬКО НАЧАЛСЯ.")

LOCATIONS = [
    (0.0, "ЛАБОРАТОРИЯ | СЕКТОР С"),
    (EV["run"], "АВАРИЙНЫЙ КОРИДОР"),
    (EV["shore"], "БЕРЕГ ОСТРОВА"),
]

DEPTH = [(0, -52), (EV["boom"], -52), (EV["shore"], 0)]
ZOMBIES_NEAR = [(0, 41), (EV["corridor"], 41), (EV["corridor"] + 1.5, 500), (EV["boom"], 500),
                (EV["shore"], 0), (EV["veins"] + 3.0, 0), (EV["veins"] + 3.5, 1)]

# часы: продолжают серию 3
CLOCK_START = (0, 10)
CLOCK_STEP = 14.0
DAY_START = 2

# обратный отсчёт «до уничтожения острова»: 60 с -> 0 к моменту взрыва
COUNTDOWN_FROM = EV["sixty"]
COUNTDOWN_TO = EV["boom"]
COUNTDOWN_SECONDS = 60

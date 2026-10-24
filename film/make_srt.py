"""Печатает субтитры в формате SRT (для загрузки на YouTube/TikTok)."""
from timeline import SPEAKERS, SUBS


def ts(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


for i, (a, b, spk, text) in enumerate(SUBS, 1):
    print(f"{i}\n{ts(a)} --> {ts(b)}\n{SPEAKERS[spk][0]}: {text}\n")

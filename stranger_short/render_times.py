"""Времена начала сцен для звука (без импорта тяжёлого рендера)."""
from timeline import DURATION, SCENES, v

SHOT_START = {name: max(0.0, v(a)) for a, name in SCENES}
HOOK_N = 9
FLOOR_N = 20

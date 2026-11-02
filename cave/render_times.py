"""Времена начала сцен для звука (без импорта тяжёлого рендера)."""
from timeline import SCENES, v

SHOT_START = {name: max(0.0, v(a)) for a, name in SCENES}

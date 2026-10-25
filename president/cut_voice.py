"""Вырезает из озвучки зачитанные таймкоды («От 0,0 до 0,12» и т. п.).

Разрезы стоят в паузах вокруг этих фраз (найдены по пословной разметке
и silencedetect). На каждом стыке — короткие затухания и пауза.
Запуск: python3 cut_voice.py  ->  voice/voice_clean.wav
"""
import subprocess

SRC = "voice/voiceover.mp3"
OUT = "voice/voice_clean.wav"
CUTS = [(0.0, 2.30), (19.15, 22.40), (38.65, 41.60), (61.55, 64.55),
        (84.00, 87.05), (110.08, 113.10), (132.95, 135.90)]
END = 161.41
GAP = 0.25          # добавочная пауза на стыке блоков, с

keep, t = [], 0.0
for a, b in CUTS:
    if a > t:
        keep.append((t, a))
    t = b
keep.append((t, END))

parts, labels = [], []
for i, (a, b) in enumerate(keep):
    parts.append(f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,"
                 f"afade=t=out:st={b - a - 0.05:.3f}:d=0.05,apad=pad_dur={GAP}[p{i}]")
    labels.append(f"[p{i}]")
graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(keep)}:v=0:a=1[out]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SRC, "-filter_complex", graph,
                "-map", "[out]", "-ar", "44100", "-ac", "2", OUT], check=True)
print("сегменты:", keep)

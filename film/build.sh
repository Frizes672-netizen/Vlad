#!/usr/bin/env bash
# Полная сборка ролика: звук -> кадры -> субтитры -> итоговый mp4.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build output
echo "1/4 Звуковой дизайн..."
python3 audio.py build/sfx.wav
echo "2/4 Рендер кадров..."
python3 render.py build/video_noaudio.mp4
echo "3/4 Субтитры..."
python3 make_srt.py > output/subtitles.srt
echo "4/4 Сведение..."
ffmpeg -y -loglevel error -i build/video_noaudio.mp4 -i build/sfx.wav \
  -c:v libx264 -preset slow -crf 30 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 192k -shortest output/night_begins.mp4
echo "Готово: output/night_begins.mp4"

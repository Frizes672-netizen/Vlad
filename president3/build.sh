#!/usr/bin/env bash
# Полная сборка ролика: звук -> кадры -> субтитры -> сведение с озвучкой.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build output
VOICE=voice/voiceover.mp3
OFFSET=$(python3 -c "from timeline import VOICE_OFFSET; print(int(VOICE_OFFSET * 1000))")
DUR=$(python3 -c "from timeline import DURATION; print(DURATION)")
echo "1/4 Музыка и звуки..."
python3 audio.py build/sfx.wav
echo "2/4 Рендер кадров..."
python3 render.py build/video_noaudio.mp4
echo "3/4 Субтитры..."
python3 make_srt.py > output/subtitles.srt
echo "4/4 Сведение с озвучкой..."
# эффекты приглушаются, когда звучит голос (sidechain)
ffmpeg -y -loglevel error -i build/video_noaudio.mp4 -i build/sfx.wav -i "$VOICE" -filter_complex "\
[2:a]adelay=${OFFSET}|${OFFSET},aformat=sample_rates=44100:channel_layouts=stereo,volume=1.5,apad=whole_dur=${DUR},asplit=2[vo][key];\
[1:a][key]sidechaincompress=threshold=0.03:ratio=5:attack=20:release=350[sfx];\
[sfx][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]" \
  -map 0:v -map "[a]" -c:v libx264 -preset slow -crf 30 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 192k -shortest output/president_24h_part3.mp4
echo "Готово: output/president_24h_part3.mp4"

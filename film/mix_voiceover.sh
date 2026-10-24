#!/usr/bin/env bash
# Наложение вашей озвучки на готовый ролик.
#   ./mix_voiceover.sh путь/к/озвучке.mp3 [сдвиг_в_секундах]
# Звуковые эффекты автоматически приглушаются, когда звучит голос.
set -euo pipefail
cd "$(dirname "$0")"
VOICE="$1"
OFFSET="${2:-0}"
DELAY_MS=$(python3 -c "print(int(float('$OFFSET')*1000))")
ffmpeg -y -loglevel error -i output/night_begins.mp4 -i "$VOICE" -filter_complex "\
[1:a]adelay=${DELAY_MS}|${DELAY_MS},aformat=channel_layouts=stereo,volume=1.6,asplit=2[vo][key];\
[0:a][key]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[sfx];\
[sfx][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k output/night_begins_voice.mp4
echo "Готово: output/night_begins_voice.mp4"

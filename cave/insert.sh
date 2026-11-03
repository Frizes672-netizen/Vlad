#!/usr/bin/env bash
# Рекламный фрагмент (зелёный хромакей) поверх ролика на 45-й секунде: видео не останавливается,
# фон 7,2 с размыт и притемнён, наша озвучка приглушена, звук рекламы сверху; потом всё как было.
set -euo pipefail
cd "$(dirname "$0")"
BASE=output/peshera_bez_chasov.mp4
AD=insert/natria_vpn.mp4
T=45.6
ADDUR=7.2
END=$(python3 -c "print(${T} + ${ADDUR})")
ffmpeg -y -loglevel error -i "$BASE" -i "$AD" -filter_complex "\
[0:v]split=2[base][toblur];\
[toblur]boxblur=18:2,eq=brightness=-0.15[blurred];\
[base][blurred]overlay=enable='between(t,${T},${END})'[bg];\
[1:v]fps=24,chromakey=0x00FF00:0.22:0.08,scale=1020:-1,format=rgba,\
fade=t=in:st=0:d=0.25:alpha=1,fade=t=out:st=$(python3 -c "print(${ADDUR} - 0.3)"):d=0.3:alpha=1,setpts=PTS-STARTPTS+${T}/TB[ad];\
[bg][ad]overlay=(W-w)/2:(H-h)/2:eof_action=pass,format=yuv420p[v];\
[0:a]volume='if(between(t,${T},${END}),0.3,1)':eval=frame[orig];\
[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=1.6,adelay=$(python3 -c "print(int(${T}*1000))")|$(python3 -c "print(int(${T}*1000))")[adA];\
[orig][adA]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset medium -b:v 8000k -maxrate 10000k -bufsize 16M -pix_fmt yuv420p \
  -movflags +faststart -c:a aac -b:a 192k output/peshera_bez_chasov_vpn.mp4
echo "Готово: output/peshera_bez_chasov_vpn.mp4"

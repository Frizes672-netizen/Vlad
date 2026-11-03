#!/usr/bin/env bash
# Вставка рекламного фрагмента (зелёный хромакей) в готовый ролик: на паузе после «…намного больше»
# кадр замирает и размывается, поверх — карточка рекламы со своим звуком, затем ролик продолжается.
set -euo pipefail
cd "$(dirname "$0")"
BASE=output/peshera_bez_chasov.mp4
AD=insert/natria_vpn.mp4
T=45.6
ADDUR=7.2
ffmpeg -y -loglevel error -i "$BASE" -i "$AD" -filter_complex "\
[0:v]split=3[v0][v1][v2];\
[v0]trim=0:${T},setpts=PTS-STARTPTS[va];\
[v1]trim=start=${T}:duration=0.05,setpts=PTS-STARTPTS,loop=loop=-1:size=1:start=0,trim=0:${ADDUR},setpts=PTS-STARTPTS,boxblur=20:2,eq=brightness=-0.18[bg];\
[1:v]fps=24,chromakey=0x00FF00:0.22:0.08,scale=1020:-1,format=rgba,fade=t=in:st=0:d=0.25:alpha=1,fade=t=out:st=6.9:d=0.3:alpha=1[ad];\
[bg][ad]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p[vm];\
[v2]trim=start=${T},setpts=PTS-STARTPTS[vb];\
[0:a]asplit=2[a0][a2];\
[a0]atrim=0:${T},asetpts=PTS-STARTPTS[aa];\
[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=1.6,apad=whole_dur=${ADDUR},atrim=0:${ADDUR},asetpts=PTS-STARTPTS[am];\
[a2]atrim=start=${T},asetpts=PTS-STARTPTS[ab];\
[va][aa][vm][am][vb][ab]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset medium -b:v 7600k -maxrate 9500k -bufsize 15M -pix_fmt yuv420p \
  -movflags +faststart -c:a aac -b:a 192k output/peshera_bez_chasov_vpn.mp4
echo "Готово: output/peshera_bez_chasov_vpn.mp4"

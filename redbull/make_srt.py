"""Prints SRT subtitles from the aligned script words (voice/captions.tsv)."""
from timeline import VOICE_OFFSET


def ts(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


words = []
for line in open("voice/captions.tsv", encoding="utf-8"):
    s, e, w = line.rstrip("\n").split("\t")
    words.append((float(s) + VOICE_OFFSET, float(e) + VOICE_OFFSET, w))

n, cur = 1, []
for wd in words:
    cur.append(wd)
    if wd[2][-1:] in ".!?…”" or len(cur) >= 8:
        print(f"{n}\n{ts(cur[0][0])} --> {ts(cur[-1][1])}\n{' '.join(w for _, _, w in cur)}\n")
        n, cur = n + 1, []
if cur:
    print(f"{n}\n{ts(cur[0][0])} --> {ts(cur[-1][1])}\n{' '.join(w for _, _, w in cur)}\n")

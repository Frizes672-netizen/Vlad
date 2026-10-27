"""Привязывает слова сценария (voice/script.txt) к таймингам распознанной
озвучки (voice/words.tsv). Совпавшие слова берут время распознавания,
остальные распределяются между соседями. Результат: voice/captions.tsv."""
import difflib
import re


def norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9$']", "", w)
    return {"thirteen": "13", "$13": "13"}.get(w, w.replace("$", ""))


asr = [l.rstrip("\n").split("\t") for l in open("voice/words.tsv", encoding="utf-8")]
asr = [(float(s), float(e), w) for s, e, w in asr]
script = open("voice/script.txt", encoding="utf-8").read().split()

a_tok = [norm(w) for _, _, w in asr]
s_tok = [norm(w) for w in script]
times = [None] * len(script)
sm = difflib.SequenceMatcher(None, s_tok, a_tok, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        for k in range(i2 - i1):
            times[i1 + k] = (asr[j1 + k][0], asr[j1 + k][1])
    elif tag == "replace" and j2 > j1:
        t0, t1 = asr[j1][0], asr[j2 - 1][1]
        n = i2 - i1
        for k in range(n):
            times[i1 + k] = (t0 + (t1 - t0) * k / n, t0 + (t1 - t0) * (k + 1) / n)

# заполнить пропуски интерполяцией между соседями
for i in range(len(times)):
    if times[i] is None:
        prev_end = next((times[j][1] for j in range(i - 1, -1, -1) if times[j]), 0.0)
        nxt = next((j for j in range(i + 1, len(times)) if times[j]), None)
        next_start = times[nxt][0] if nxt is not None else prev_end + 0.4
        gap = nxt - i if nxt is not None else 1
        span = (next_start - prev_end) / (gap + 0)
        times[i] = (prev_end, prev_end + max(0.15, span))

with open("voice/captions.tsv", "w", encoding="utf-8") as f:
    for w, (s, e) in zip(script, times):
        f.write(f"{s:.2f}\t{e:.2f}\t{w}\n")
print(len(script), "слов; совпало:", sum(b.size for b in sm.get_matching_blocks()))

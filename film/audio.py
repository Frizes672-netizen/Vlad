"""Синтез звукового дизайна ролика (волны, огонь, рык, бег, двери, стуки,
сердцебиение, рация, стингеры) без внешних сэмплов — только numpy.

Запуск: python3 audio.py out.wav
"""
import sys
import wave

import numpy as np

from timeline import DURATION, EV

SR = 44100
rng = np.random.default_rng(1337)
N = int(DURATION * SR)
mix = np.zeros((N, 2), dtype=np.float64)


# ---------------------------------------------------------------- примитивы
def secs(d):
    return int(d * SR)


def tvec(d):
    return np.arange(secs(d)) / SR


def noise(d):
    return rng.standard_normal(secs(d))


def band(x, lo=None, hi=None, soft=0.15):
    """Полосовой фильтр через FFT с мягкими краями."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** (2 / soft * 0.3))
    if hi:
        g *= 1 / (1 + (f / hi) ** (2 / soft * 0.3))
    return np.fft.irfft(X * g, len(x))


def norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def decay(d, tau, attack=0.002):
    t = tvec(d)
    return np.minimum(t / attack, 1) * np.exp(-t / tau)


def bell(d):
    t = np.linspace(0, 1, secs(d))
    return np.sin(np.pi * t) ** 2


def fade(x, a=0.01, r=0.05):
    n = len(x)
    e = np.ones(n)
    na, nr = min(secs(a), n), min(secs(r), n)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return x * e


def reverb(x, tail=1.8, mix_=0.35):
    ir = rng.standard_normal(secs(tail)) * np.exp(-tvec(tail) / (tail / 5))
    ir = band(ir, 200, 6000)
    ir /= np.sqrt(np.sum(ir ** 2))
    n = len(x) + len(ir)
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)
    out = np.concatenate([x, np.zeros(len(ir))]) * (1 - mix_) + y * mix_
    return out


def place(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N:
        return
    s = sig[: N - i] * gain
    lg = np.cos((pan + 1) * np.pi / 4)
    rg = np.sin((pan + 1) * np.pi / 4)
    mix[i:i + len(s), 0] += s * lg * 1.41
    mix[i:i + len(s), 1] += s * rg * 1.41


def sweep_sine(f0, f1, d, curve=4.0):
    t = tvec(d)
    f = f1 + (f0 - f1) * np.exp(-t * curve)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


# ---------------------------------------------------------------- звуки
def thump(freq=60, d=0.5, drop=3.0, tau=0.15):
    s = sweep_sine(freq * drop, freq, d, 18) * decay(d, tau)
    s += band(noise(d), 40, 400) * decay(d, tau / 3) * 0.6
    return norm(s)


def wood_knock():
    d = 0.35
    s = band(noise(d), 150, 700) * decay(d, 0.03)
    s += np.sin(2 * np.pi * 190 * tvec(d)) * decay(d, 0.06) * 0.8
    s += np.sin(2 * np.pi * 410 * tvec(d)) * decay(d, 0.03) * 0.4
    return norm(reverb(s, 0.6, 0.25))


def door_slam():
    s = thump(55, 1.2, 2.5, 0.25)
    k = wood_knock()
    out = np.zeros(secs(1.6))
    out[: len(s)] += s
    out[: len(k)] += k * 0.8
    rat = band(noise(0.5), 600, 3000) * decay(0.5, 0.08) * \
        (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 23 * tvec(0.5))))
    out[secs(0.05): secs(0.05) + len(rat)] += rat * 0.3
    return norm(reverb(out, 1.2, 0.3))


def footstep(soft=False):
    d = 0.22
    s = thump(70, d, 1.6, 0.05) * 0.8
    crunch = band(noise(d), 900, 5000) * decay(d, 0.035)
    s += crunch * (0.25 if soft else 0.55)
    return s


def whoosh(d=0.45, lo=400, hi=3500):
    return norm(band(noise(d), lo, hi)) * bell(d)


def bone_crunch():
    d = 0.5
    out = np.zeros(secs(d))
    for k in range(7):
        i = secs(rng.uniform(0, 0.25))
        c = band(noise(0.04), 1200, 7000) * decay(0.04, 0.006)
        out[i:i + len(c)] += c * rng.uniform(0.5, 1)
    out += thump(90, d, 1.5, 0.05) * 0.4
    return norm(out)


def roar(d=2.4):
    t = tvec(d)
    f0 = 95 * np.exp(-t * 0.35) + 8 * np.sin(2 * np.pi * 6.5 * t)
    ph = np.cumsum(f0) / SR
    saw = 2 * (ph % 1) - 1
    gr = band(noise(d), 150, 1600) * (0.6 + 0.4 * np.sin(2 * np.pi * 31 * t))
    s = saw * 0.7 + gr * 0.8
    s = band(s, 60, 2200)
    env = np.minimum(t / 0.25, 1) * np.exp(-np.maximum(t - 1.2, 0) / 0.5)
    s = np.tanh(3 * s * env)
    return norm(reverb(s, 2.2, 0.4))


def growl(d=1.6):
    t = tvec(d)
    f0 = 70 + 10 * np.sin(2 * np.pi * 3 * t)
    saw = 2 * ((np.cumsum(f0) / SR) % 1) - 1
    s = band(saw + band(noise(d), 200, 900), 60, 1200)
    return norm(np.tanh(2 * s) * bell(d))


def stinger(d=3.0, low=True):
    t = tvec(d)
    out = np.zeros(secs(d))
    for f in (1046, 1108, 1174, 1244, 1318):
        out += np.sin(2 * np.pi * f * (1 + 0.003 * np.sin(2 * np.pi * 5.5 * t)) * t)
    out *= decay(d, 0.9, 0.01) * 0.25
    out += band(noise(d), 2000, 9000) * decay(d, 0.25) * 0.4
    if low:
        out += thump(40, d, 3.0, 0.6) * 1.2
    return norm(reverb(out, 2.5, 0.4))


def heartbeat(gain=1.0):
    d = 0.6
    out = np.zeros(secs(d))
    a = thump(48, 0.3, 1.8, 0.07)
    b = thump(44, 0.3, 1.8, 0.06) * 0.7
    out[: len(a)] += a
    out[secs(0.18): secs(0.18) + len(b)] += b
    return band(out, 20, 300) * gain


def scratch_stroke(d=0.45):
    t = tvec(d)
    s = band(noise(d), 1800, 7500)
    s *= 0.55 + 0.45 * np.sign(np.sin(2 * np.pi * (45 + 30 * t) * t))
    return norm(s) * fade(np.ones(secs(d)), 0.04, 0.12)


def table_drag(d=1.0):
    t = tvec(d)
    s = band(noise(d), 180, 1400)
    am = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 17 * t + 3 * np.sin(2 * np.pi * 2 * t)))
    return norm(s * am) * fade(np.ones(secs(d)), 0.05, 0.2)


def creak(d=1.5, f_lo=35, f_hi=80, pitch=1.0):
    """Скрип дерева: прерывистое трение (stick–slip) через полосовой фильтр."""
    t = tvec(d)
    rate = f_lo + (f_hi - f_lo) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + 1))
    ph = np.cumsum(rate) / SR
    imp = (np.diff(np.floor(ph), prepend=0) > 0).astype(float)
    s = band(imp, 250 * pitch, 2200 * pitch)
    s += np.sin(2 * np.pi * np.cumsum(380 * pitch + 60 * np.sin(2 * np.pi * 0.4 * t)) / SR) * 0.15
    return norm(s) * bell(d)


def knife_ring():
    d = 1.6
    t = tvec(d)
    s = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
            for f, tau in ((2950, 0.6), (4370, 0.4), (6120, 0.25), (8030, 0.15)))
    scrape = band(noise(0.3), 3000, 10000) * bell(0.3)
    s[: len(scrape)] += scrape * 1.5
    return norm(reverb(s, 1.0, 0.2))


def radio_static(d, voice=True):
    t = tvec(d)
    s = band(noise(d), 700, 3800)
    crack = (rng.random(secs(d)) > 0.9993).astype(float)
    crack = band(np.convolve(crack, np.ones(80), "same"), 500, 6000)
    s = s * 0.5 + crack * 2.0
    if voice:
        # неразборчивый «голос» сквозь помехи — ритм слогов
        syl = np.clip(np.sin(2 * np.pi * 4.3 * t) * np.sin(2 * np.pi * 0.9 * t + 1), 0, 1)
        f0 = 120 + 25 * np.sin(2 * np.pi * 1.3 * t)
        buzz = 2 * ((np.cumsum(f0) / SR) % 1) - 1
        s += band(buzz, 400, 2500) * syl * 1.2
    return norm(np.tanh(s * 1.5)) * fade(np.ones(secs(d)), 0.02, 0.05)


def squelch():
    d = 0.25
    s = np.sin(2 * np.pi * 1200 * tvec(d)) * (tvec(d) < 0.09)
    s += band(noise(d), 800, 5000) * (tvec(d) >= 0.09) * decay(d, 0.08) * 1.5
    return s * 0.7


def drip():
    d = 0.25
    return sweep_sine(700, 1700, d, 25)[: secs(d)] * decay(d, 0.04) * 0.5


def gasp():
    d = 0.9
    s = band(noise(d), 400, 3000) * np.minimum(tvec(d) / 0.15, 1) * np.exp(-tvec(d) / 0.35)
    return norm(s)


def breath_cycle(d=0.5):
    return norm(band(noise(d), 500, 2600)) * bell(d)


def drone(d, base=41.0):
    t = tvec(d)
    s = (np.sin(2 * np.pi * base * t) + np.sin(2 * np.pi * (base * 1.017) * t)
         + 0.6 * np.sin(2 * np.pi * base * 1.5 * t) + 0.3 * np.sin(2 * np.pi * base * 2.02 * t))
    s *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t)
    wind = band(noise(d), 80, 500) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + 2))
    return (s * 0.25 + wind * 0.4)


def riser(d, f=110):
    t = tvec(d)
    s = np.zeros(secs(d))
    for k, det in enumerate((1.0, 1.006, 0.994, 1.5, 1.509, 2.0, 2.37)):
        ff = f * det * (1 + 0.15 * t / d)
        s += 2 * ((np.cumsum(np.full(len(t), ff)) / SR + k * 0.13) % 1) - 1
    s = band(s, 80, 2500)
    return norm(s) * (t / d) ** 2.2


# ---------------------------------------------------------------- партитура
def score():
    E = EV
    # ПРОБУЖДЕНИЕ: вдох, волны, огонь, рык из джунглей
    place(gasp(), 0.25, 0.55)
    sea_d = 9.0
    t = tvec(sea_d)
    waves = band(noise(sea_d), 40, 700) * (0.35 + 0.65 * np.sin(np.pi * t / 3.3) ** 2)
    hiss = band(noise(sea_d), 2000, 8000) * (0.2 + 0.8 * np.sin(np.pi * (t - 0.6) / 3.3) ** 4)
    sea = (norm(waves) * 0.6 + norm(hiss) * 0.2) * fade(np.ones(secs(sea_d)), 1.2, 2.0)
    place(sea, 0, 0.75, -0.2)
    fire = band(noise(6.0), 40, 300) * 0.4
    for _ in range(170):
        i = rng.integers(0, len(fire) - 2000)
        c = band(noise(0.03), 1500, 8000) * decay(0.03, 0.004)
        fire[i:i + len(c)] += c * rng.uniform(0.2, 1.0)
    place(norm(fire) * fade(np.ones(len(fire)), 0.6, 0.8), 1.5, 0.35, 0.35)
    place(drone(DURATION, 41), 0, 0.22)
    place(roar(), E["roar"], 0.95)
    place(stinger(2.0, low=True) * 0.6, E["roar"], 0.5)

    # ОНИ РЯДОМ: бег, дыхание, ветки, хруст, преследователи
    tt, k = E["run"], 0
    while tt < E["slam"] - 0.2:
        place(footstep(), tt, 0.55, -0.15 if k % 2 else 0.15)
        tt += 0.27
        k += 1
    for tt in np.arange(E["run"] + 0.1, E["slam"] - 0.3, 0.55):
        place(breath_cycle(0.4), tt, 0.25)
    for tt in np.arange(E["run"] + 0.5, E["pursuers"], 0.75):
        place(whoosh(0.4), tt, 0.55, rng.uniform(-0.8, 0.8))
    for tt in (E["run"] + 1.2, E["run"] + 2.6, E["pursuers"] + 1.4):
        place(bone_crunch(), tt, 0.6, rng.uniform(-0.6, 0.6))
    place(stinger(2.5), E["pursuers"], 0.85)
    place(growl(1.8), E["pursuers"] + 0.1, 0.6, -0.3)
    place(growl(1.6), E["pursuers"] + 0.9, 0.5, 0.4)
    place(door_slam(), E["slam"], 1.0)
    place(table_drag(1.0), E["table"], 0.6)
    place(thump(70, 0.4, 1.5, 0.08), E["table"] + 1.0, 0.6)
    for tt in E["scratches"]:
        place(scratch_stroke(rng.uniform(0.35, 0.55)), tt, 0.5, rng.uniform(-0.3, 0.3))
    place(heartbeat(0.4), E["silence"] + 0.2, 1.0)
    place(heartbeat(0.45), E["silence"] + 0.85, 1.0)
    place(thump(45, 1.5, 2.0, 0.3), E["thud"], 1.0)
    for _ in range(25):
        place(band(noise(0.02), 2000, 8000) * decay(0.02, 0.004), E["thud"] + rng.uniform(0.05, 0.7), 0.25, rng.uniform(-1, 1))

    # ВТОРОЙ ВЫЖИВШИЙ: нож, бросок, стена «ОНИ УЧАТСЯ»
    place(knife_ring(), E["lunge_shot"] + 0.1, 0.45, 0.4)
    place(whoosh(0.5, 300, 2500), E["lunge"] - 0.25, 0.8, -0.4)
    place(stinger(2.0, low=True), E["lunge"], 0.8)
    place(thump(60, 0.5, 2, 0.1), E["lunge"] + 0.3, 0.6)
    for tt in np.arange(E["wall"] + 0.2, E["wall_text"] - 0.6, 0.4):
        place(scratch_stroke(0.18), tt, 0.25, rng.uniform(-0.4, 0.4))
    place(riser(E["wall_text"] - E["wall"] - 0.2, 98), E["wall"] + 0.2, 0.35)
    place(stinger(3.5), E["wall_text"], 0.9)
    for tt in (0.6, 1.1, 1.6):
        place(drip(), E["wall_text"] + tt, 0.5)

    # ЧТО-ТО НЕ ТАК: ровные шаги, три стука, сердцебиение
    for tt in E["steps"]:
        place(footstep(soft=True), tt, 0.75, 0.0)
    for tt in E["knocks"]:
        place(wood_knock(), tt, 1.0)
        place(thump(80, 0.3, 1.3, 0.05), tt, 0.5)
    for tt in np.arange(E["pale"], E["crack"], 0.95):
        place(heartbeat(0.55), tt, 1.0)

    # ПРАВДА: щель, спасатель, поворот головы
    place(creak(1.3, 30, 60), E["crack"], 0.45)
    for tt in np.arange(E["crack"] + 0.2, E["head_turn"], 0.82):
        place(heartbeat(0.45), tt, 1.0)
    place(creak(1.0, 40, 80, 1.3), E["grab"] - 0.1, 0.3, 0.3)
    place(creak(0.9, 20, 45, 0.7), E["head_turn"], 0.55)
    place(stinger(2.5), E["head_turn"] + 0.3, 0.8)
    for tt in np.arange(E["head_turn"] + 0.2, E["radio"], 0.55):
        place(heartbeat(1.0), tt, 1.1)

    # КЛИФФХЭНГЕР: рация, кровь, дверь прогибается
    for q in E["squelch"]:
        place(squelch(), q, 0.8, 0.3)
    a, b = E["radio_talk"]
    place(radio_static(b - a + 0.3), a - 0.1, 0.5, 0.3)
    place(stinger(2.0, low=False) * 0.6, E["turn_her"] + 2.1, 0.5)
    for tt in (1.4, 2.4, 3.4):
        place(drip(), E["blood"] + tt, 0.5, -0.2)
    bend = E["black"] - E["bend"]
    place(creak(bend, 15, 35, 0.5), E["bend"], 0.9)
    place(riser(bend, 82), E["bend"], 0.6)
    for d in (0.1, 0.9, 1.7, 2.4, 2.9):
        place(thump(50, 0.4, 2, 0.08), E["bend"] + d, 0.9)
        place(bone_crunch(), E["bend"] + d + 0.02, 0.5)

    # жёсткий обрыв звука на чёрном экране
    cut, end = secs(E["black"]), secs(E["final"])
    mix[cut:end] *= np.linspace(1, 0, end - cut)[:, None] ** 8

    # финальная надпись
    tail = DURATION - E["final"]
    place(thump(32, 4.0, 3.0, 1.2) * 1.3, E["final"], 1.0)
    place(stinger(4.0, low=False) * 0.5, E["final"], 0.6)
    place(drone(tail, 36) * fade(np.ones(secs(tail)), 0.3, 2.0), E["final"], 0.8)


def main(path):
    score()
    out = np.tanh(mix * 0.9)
    out = out / np.max(np.abs(out)) * 0.89
    # общий fade-in и fade-out
    out[: secs(0.3)] *= np.linspace(0, 1, secs(0.3))[:, None]
    out[-secs(1.2):] *= np.linspace(1, 0, secs(1.2))[:, None]
    pcm = (out * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sfx.wav")

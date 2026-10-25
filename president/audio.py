"""Музыка и звуковой дизайн «24 часа президентом»: пэды, пульс, тиканье,
печати, толпа, телефон, глитч — синтез на numpy, без сэмплов.

Запуск: python3 audio.py out.wav
"""
import sys
import wave

import numpy as np

from timeline import DURATION, v

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


# ---------------------------------------------------------------- музыка
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def hz(name, octave):
    return 440.0 * 2 ** ((NOTE[name] + 12 * (octave - 4) - 9) / 12)


CHORDS = {
    "Am": [("A", 2), ("E", 3), ("A", 3), ("C", 4), ("E", 4), ("B", 4)],
    "F": [("F", 2), ("C", 3), ("F", 3), ("A", 3), ("C", 4), ("G", 4)],
    "C": [("C", 2), ("G", 2), ("C", 3), ("E", 3), ("G", 3), ("D", 4)],
    "G": [("G", 2), ("D", 3), ("G", 3), ("B", 3), ("D", 4), ("A", 4)],
    "Dm": [("D", 2), ("A", 2), ("D", 3), ("F", 3), ("A", 3), ("E", 4)],
    "Bb": [("A#", 1), ("F", 2), ("A#", 2), ("D", 3), ("F", 3), ("C", 4)],
    "Gm": [("G", 1), ("D", 2), ("G", 2), ("A#", 2), ("D", 3), ("A", 3)],
}


def pad(chord, d, bright=1800):
    t = tvec(d)
    s = np.zeros(len(t))
    for i, (n, o) in enumerate(CHORDS[chord]):
        f = hz(n, o)
        for det in (-0.12, 0.0, 0.11):
            ph = rng.uniform(0, 1)
            s += (2 * ((f * (1 + det / 100) * t + ph) % 1) - 1) * (0.6 if i < 2 else 0.35)
    s = band(s, 40, bright)
    s *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.15 * t)
    return norm(s) * fade(np.ones(len(t)), 1.2, 1.5)


def progression(start, end, chords, seg, gain, bright=1800):
    t, i = start, 0
    while t < end - 0.5:
        d = min(seg + 1.5, end - t + 1.0)
        place(pad(chords[i % len(chords)], d, bright), t, gain)
        t += seg
        i += 1


def pulse(start, end, every, gain, freq=48):
    for tt in np.arange(start, end, every):
        place(thump(freq, 0.5, 2.2, 0.12), tt, gain)


def tick(gain=0.25, high=True):
    d = 0.05
    s = band(noise(d), 2500 if high else 1500, 9000) * decay(d, 0.006)
    return s * gain


def ticking(start, end, gain=0.4):
    for k, tt in enumerate(np.arange(start, end, 0.5)):
        place(tick(1.0, k % 2 == 0), tt, gain, 0.2 if k % 2 else -0.2)


def shimmer(d):
    t = tvec(d)
    s = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (1760, 2217, 2637, 3520))
    s *= (0.5 + 0.5 * np.sin(2 * np.pi * 6 * t)) * (t / d)
    return s * 0.2 * fade(np.ones(len(t)), 0.05, 0.3)


def impact(gain=1.0):
    return (thump(38, 3.0, 3.0, 0.9) * 1.2 + band(noise(3.0), 60, 800) * decay(3.0, 0.4) * 0.4) * gain


def stamp():
    s = thump(70, 0.5, 2.0, 0.07)
    s[: secs(0.06)] += band(noise(0.06), 600, 4000) * decay(0.06, 0.012) * 0.8
    return s


def paper_slide():
    return norm(band(noise(0.6), 1500, 7000)) * bell(0.6) * 0.5


def ping():
    t = tvec(0.4)
    return (np.sin(2 * np.pi * 1318 * t) + 0.5 * np.sin(2 * np.pi * 1975 * t)) * decay(0.4, 0.12) * 0.4


def crowd(d):
    t = tvec(d)
    s = band(noise(d), 300, 3500) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t) ** 2)
    claps = (rng.random(secs(d)) > 0.9985).astype(float)
    claps = band(np.convolve(claps, np.ones(60), "same"), 800, 6000)
    return norm(s + claps * 3) * fade(np.ones(len(t)), 0.8, 1.2)


def ring(d):
    t = tvec(d)
    s = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * (np.sin(2 * np.pi * 20 * t) > 0)
    on = ((t % 1.2) < 0.8).astype(float)
    return s * on * 0.3


def glitch(d):
    out = np.zeros(secs(d))
    for _ in range(int(d * 14)):
        i = rng.integers(0, len(out) - 3000)
        L = rng.integers(400, 3000)
        f = rng.uniform(80, 2000)
        nz = band(noise(L / SR + 0.01), 500, 8000)[:L]
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(len(nz)) / SR)) * 0.5 + nz
        out[i:i + len(seg)] += seg * rng.uniform(0.2, 0.6)
    return out


def news_sting():
    out = np.zeros(secs(1.2))
    for k, f in enumerate((880, 1174, 1568)):
        t = tvec(0.5)
        s = np.sin(2 * np.pi * f * t) * decay(0.5, 0.18)
        i = secs(k * 0.12)
        out[i:i + len(s)] += s * 0.4
    return out


# ---------------------------------------------------------------- партитура
def score():
    T = v
    # музыка: тёплая и загадочная → торжественная → тревожная
    progression(0.0, T(17.0), ["Am", "F", "C", "G"], 4.2, 0.30, 1400)
    progression(T(17.0), T(53.2), ["F", "C", "G", "Am"], 4.5, 0.30, 2200)
    pulse(T(33.5), T(53.0), 1.0, 0.25)
    progression(T(53.2), T(73.5), ["C", "G", "Am", "F"], 2.6, 0.34, 3000)
    pulse(T(53.2), T(70.0), 0.5, 0.3)
    progression(T(73.5), T(117.0), ["Dm", "Bb", "Dm", "Gm"], 5.0, 0.30, 1300)
    pulse(T(93.2), T(117.0), 1.0, 0.25, 42)
    progression(T(117.0), T(141.9), ["Dm", "Bb", "Gm", "Dm"], 4.0, 0.34, 1100)
    pulse(T(117.0), T(137.2), 0.5, 0.3, 40)
    ticking(T(117.0), T(141.9), 0.35)

    # сцены
    place(whoosh(1.4, 200, 1800) * 0.6, 0.1, 0.5)
    place(shimmer(2.0), T(1.0), 0.6)
    place(impact(0.7), T(2.5), 0.8)
    place(band(noise(6), 60, 500) * 0.3 * fade(np.ones(secs(6)), 1, 1), T(6.6), 0.6)
    place(impact(0.9), T(15.4), 0.9)
    ticking(T(12.6), T(17.0), 0.5)
    place(creak(1.0, 25, 50, 0.6), T(19.7), 0.4)
    place(whoosh(1.0, 100, 1200), T(19.7), 0.5)
    for tt in np.arange(T(23.8), T(26.0), 0.55):
        place(footstep(soft=True), tt, 0.35)
    place(impact(0.8), T(32.2), 0.8)
    for k, ts in enumerate((33.85, 40.74, 44.42)):
        base = T(ts) + (0.3 if k == 0 else 0)
        place(paper_slide(), base, 0.6, 0.3)
        place(stamp(), base + 1.6, 1.0)
    place(news_sting(), T(53.25), 0.8)
    place(crowd(4.6), T(56.8), 0.55)
    for i in range(6):
        place(thump(90, 0.3, 3, 0.04) + band(noise(0.3), 1000, 6000) * decay(0.3, 0.05) * 0.4,
              T(57.4) + i * 0.6 + rng.uniform(0, 0.2), 0.4, rng.uniform(-0.8, 0.8))
    for i in range(6):
        place(ping(), T(61.3) + i * 0.45, 0.6, rng.uniform(-0.5, 0.5))
    place(riser(5.0, 110), T(64.6), 0.25)
    place(shimmer(5.0), T(64.6), 0.5)
    place(thump(30, 3.0, 2.0, 1.0) + band(noise(3.0), 40, 900) * decay(3.0, 0.8), T(71.4), 0.9)
    place(door_slam(), T(73.6), 0.9)
    place(whoosh(0.4), T(76.5), 0.6)
    place(stamp() * 1.3, T(76.85), 1.0)
    place(sweep_sine(600, 120, 3.5, 0.6) * fade(np.ones(secs(3.5)), 0.2, 0.8) * 0.25, T(79.0), 0.6)
    place(door_slam() * 0.7, T(92.6), 0.7)
    place(ring(3.3), T(93.3), 0.7, 0.2)
    for tt in np.arange(T(96.9), T(100.5), 0.6):
        place(footstep(soft=True), tt, 0.35)
    place(np.sin(2 * np.pi * 1200 * tvec(0.1)) * 0.4, T(100.4), 0.6)
    place(heartbeat(0.6), T(110.0), 1.0)
    for tt in np.arange(T(110.6), T(117.0), 0.9):
        place(heartbeat(0.6), tt, 1.0)
    place(impact(0.7), T(117.0), 0.7)
    place(stamp() * 1.3, T(124.9), 1.0)
    for tt in np.arange(T(130.2), T(134.6), 0.75):
        place(heartbeat(0.8), tt, 1.0)
    for k in range(40):
        place(tick(0.8), T(134.6) + k * 0.035, 0.5)
    place(thump(60, 0.4, 2, 0.08), T(136.0), 0.8)
    place(riser(4.4, 90), T(137.2), 0.5)
    place(glitch(4.5), T(137.4), 0.5)
    place(impact(1.2), T(141.7), 1.0)
    # тишина на финальной карточке и мягкий удар
    cut, end = secs(T(142.4)), secs(T(142.9))
    mix[cut:end] *= np.linspace(1, 0.15, end - cut)[:, None]
    mix[end:] *= 0.15
    place(impact(0.6), T(143.0), 0.8)


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

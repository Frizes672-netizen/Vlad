"""Синтез звукового дизайна серии 2 (дверь, люк, тоннель, бункер, компьютер,
сирена, толпа зомби) без внешних сэмплов — только numpy.

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


# ---------------------------------------------------------------- новые звуки
def metal_clang(d=2.0):
    t = tvec(d)
    s = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / tau) for f, tau in
            ((180, 0.9), (412, 0.6), (737, 0.45), (1130, 0.3), (1690, 0.2)))
    s += thump(55, d, 2.0, 0.2) * 1.5
    return norm(reverb(s, 2.0, 0.4))


def beep(f=1400, d=0.12):
    return np.sin(2 * np.pi * f * tvec(d)) * fade(np.ones(secs(d)), 0.005, 0.02) * 0.6


def crt_on():
    d = 1.2
    t = tvec(d)
    s = band(noise(0.08), 1000, 8000) * decay(0.08, 0.02)
    out = np.zeros(secs(d))
    out[: len(s)] += s
    out += np.sin(2 * np.pi * 60 * t) * 0.25 * np.minimum(t / 0.2, 1)
    out += np.sin(2 * np.pi * (9000 + 3000 * np.exp(-t * 4)) * t) * 0.05
    return out * fade(np.ones(len(out)), 0.01, 0.4)


def siren(d):
    t = tvec(d)
    f = 650 + 350 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.55 * t))
    s = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.5 + np.sin(2 * np.pi * np.cumsum(f) / SR)
    return band(s, 300, 4000) * fade(np.ones(secs(d)), 0.3, 1.0)


def crowd_groan(d):
    out = np.zeros(secs(d))
    for _ in range(14):
        g = growl(rng.uniform(1.2, 2.5)) * rng.uniform(0.3, 0.8)
        i = rng.integers(0, max(1, len(out) - len(g)))
        out[i:i + len(g)] += g
    return band(out, 80, 1500)


def gun_cock():
    out = np.zeros(secs(0.4))
    for t0 in (0.0, 0.16):
        c = band(noise(0.04), 1500, 9000) * decay(0.04, 0.008)
        i = secs(t0)
        out[i:i + len(c)] += c
    return out


def clatter():
    out = np.zeros(secs(0.9))
    for t0, g in ((0.0, 1.0), (0.18, 0.6), (0.32, 0.4), (0.42, 0.25)):
        d = 0.25
        s = sum(np.sin(2 * np.pi * f * tvec(d)) for f in (1800, 2650, 3900)) * decay(d, 0.05)
        i = secs(t0)
        out[i:i + len(s)] += s * g
    return out * 0.4


def echo(x, delay=0.28, fb=0.45, n=4):
    out = np.concatenate([x, np.zeros(secs(delay * n))])
    for k in range(1, n + 1):
        i = secs(delay * k)
        out[i:i + len(x)] += x * fb ** k
    return out


# ---------------------------------------------------------------- партитура
def score():
    E = EV
    place(drone(DURATION, 41), 0, 0.2)
    # ХИЖИНА: удары в дверь, дверь вылетает
    for i, tt in enumerate(E["door_hits"]):
        place(door_slam() * 0.6, tt, 0.7 + i * 0.05)
        place(bone_crunch() * 0.4, tt + 0.05, 0.4)
    place(heartbeat(0.5), 2.0, 1.0)
    place(stinger(1.5, low=True) * 0.5, E["grab"], 0.4)
    place(door_slam(), E["door_off"], 1.1)
    place(whoosh(0.5, 200, 2000), E["door_off"] + 0.05, 0.7)
    place(thump(50, 0.6, 2, 0.15), E["door_off"] + 0.45, 0.9)
    place(stinger(2.0), E["door_off"], 0.6)
    # ЛЮК
    place(creak(0.7, 30, 60), E["hatch_open"], 0.6)
    for tt in np.arange(E["hatch_open"] + 0.3, E["jump"], 0.27):
        place(footstep(), tt, 0.5)
    place(whoosh(0.6, 150, 1500), E["jump"], 0.8)
    place(thump(55, 0.6, 2, 0.12), E["jump"] + 0.5, 1.0)
    place(thump(60, 0.6, 2, 0.12), E["jump"] + 0.75, 0.9)
    place(stinger(2.5), E["smile"], 0.7)
    place(thump(45, 0.8, 2, 0.15), E["lid_close"], 1.0)
    place(wood_knock(), E["lid_close"], 0.8)
    for tt in (E["dark"] + 0.3, E["dark"] + 1.1):
        place(heartbeat(0.5), tt, 1.0)
    # ТОННЕЛЬ: бег с эхом, капли, рёв, прыжок в люк
    tt, k = E["tunnel"], 0
    while tt < E["door_stop"]:
        place(echo(footstep()), tt, 0.45, -0.1 if k % 2 else 0.1)
        tt += 0.29
        k += 1
    for tt in np.arange(E["tunnel"] + 0.6, E["bunker"], 1.7):
        place(echo(drip(), 0.35, 0.5), tt, 0.4, rng.uniform(-0.8, 0.8))
    place(echo(roar(), 0.4, 0.5, 5), E["roar"], 1.0, -0.6)
    place(stinger(2.0), E["roar"], 0.5)
    place(echo(thump(50, 0.6, 2, 0.12), 0.3, 0.5), E["drop"], 0.9, -0.7)
    place(growl(1.5), E["drop"] + 0.4, 0.5, -0.7)
    # БУНКЕР: стальная дверь, фото
    place(metal_clang(), E["slam"], 1.0)
    for tt in np.arange(E["board"], E["talk_bunker"], 0.9):
        place(heartbeat(0.45), tt, 1.0)
    place(riser(E["label"] - E["photo_zoom"], 98), E["photo_zoom"], 0.35)
    place(stinger(3.0), E["label"], 0.9)
    # ЛАБОРАТОРИЯ: компьютер, запись, пистолет
    place(crt_on(), E["pc_on"], 0.8)
    place(crt_on() * 0.5, E["screen"], 0.6)
    place(band(noise(E["two_lab"] - E["screen"]), 2000, 6000) * 0.05, E["screen"], 1.0)
    place(gun_cock(), E["gun"], 0.9, 0.3)
    place(stinger(2.0, low=True) * 0.6, E["gun"], 0.6)
    # ТРЕВОГА
    for i in range(6):
        place(beep(1400 if i % 2 else 1700), E["beep"] + i * 0.45, 0.9)
    place(beep(900, 0.4), E["found"], 1.0)
    for i in range(4):
        place(beep(600, 0.25), E["protocol"] + i * 0.5, 1.0)
    place(siren(E["black"] - E["siren"]), E["siren"], 0.45)
    # ЗОМБИ ЗА СТЕКЛОМ
    place(crowd_groan(E["kneel"] - E["zombies"] + 1.5), E["zombies"], 0.7)
    for i in range(10):
        place(thump(70, 0.3, 1.5, 0.05), E["kneel"] + i * 0.13, 0.5, rng.uniform(-0.8, 0.8))
    place(stinger(3.0), E["kneel"], 0.6)
    place(clatter(), E["drop_gun"] + 0.3, 0.9, 0.3)
    for tt in np.arange(E["drop_gun"] + 1.0, E["black"], 0.85):
        place(heartbeat(0.7), tt, 1.0)
    place(stinger(3.0), E["creator"], 0.9)
    place(riser(E["black"] - E["creator"], 82), E["creator"], 0.5)
    # обрыв на чёрный и финал
    cut, end = secs(E["black"]), secs(E["final"])
    mix[cut:end] *= np.linspace(1, 0, end - cut)[:, None] ** 8
    tail = DURATION - E["final"]
    place(thump(32, 4.0, 3.0, 1.2) * 1.3, E["final"], 1.0)
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

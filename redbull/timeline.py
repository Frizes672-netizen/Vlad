"""Timeline for the Short "He Sued Red Bull for $13 Million!".

Voice: voice/voiceover.mp3. Script words with timings: voice/captions.tsv
(built by align.py from the ASR word timings in voice/words.tsv).
"""

W, H = 1080, 1920
FPS = 24
VOICE_OFFSET = 0.3
VOICE_LEN = 77.56
DURATION = round(VOICE_OFFSET + VOICE_LEN + 1.2, 2)


def v(x):
    """Voice time -> video time."""
    return round(x + VOICE_OFFSET, 3)


# scenes: (start in voice seconds, name); a scene ends where the next one starts
SCENES = [
    (-0.3, "hook"), (3.3, "fridge"), (7.2, "slogan"), (10.9, "man"), (16.9, "joke"),
    (19.5, "lawsuit"), (29.8, "classaction"), (35.3, "millions"), (42.4, "catch"),
    (49.1, "nowrong"), (54.7, "legend"), (58.4, "david"), (65.7, "promises"), (69.8, "cta"),
]

# keyword captions on screen (karaoke style) are always on; set False to hide
CAPTIONS = True

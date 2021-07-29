import glob
import sys
import tty
import termios
from pathlib import Path

import pydub
import pydub.playback
import pydub.effects
import pandas as pd
import numpy as np


# https://stackoverflow.com/a/510364
class _GetchUnix:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


TARGET = "story"
en_words_dir = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
N_CLIPS=100

def label():

    clips = glob.glob(str(en_words_dir / TARGET / "*.wav"))
    clips.sort()
    rng = np.random.RandomState(123)
    rng.shuffle(clips)

    getch = _GetchUnix()

    results = []

    for c in clips[:N_CLIPS]:
        print(Path(c).name)
        while True:
            print("-")
            wav = pydub.AudioSegment.from_file(c)
            wav = pydub.effects.normalize(wav)
            pydub.playback.play(wav)

            choice = getch()
            if choice == "g":
                usable = True
                break
            elif choice == "b":
                usable = False
                break
            elif choice == "q":
                sys.exit()
        
        result = "good" if usable else "bad"
        row = f"{Path(c).name},{result}"
        print(row)
        results.append(row)
    
    csv = Path.cwd() / "tmp" / f"{TARGET}.csv"
    with open(csv, 'w') as fh:
        fh.write("\n".join(results))


if __name__ == "__main__":
    label()

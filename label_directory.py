import glob
import sys
import tty
import termios
from pathlib import Path
import csv
import os

import pydub
import pydub.playback
import pydub.effects


class _GetchUnix:
    """https://stackoverflow.com/a/510364"""

    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class Modes:
    CLOSEST = "closest"
    FARTHEST = "farthest"


######## change these:
CLOSEST_FARTHEST_DIR = (
    Path.home() / "tinyspeech_harvard/distance_sorting/closest_farthest/"
)
WORD = "reading"
MODE = Modes.FARTHEST
###### //////////////


def label():

    in_csv = CLOSEST_FARTHEST_DIR / WORD / MODE / f"{WORD}_{MODE}_50_input.csv"
    assert os.path.isfile(in_csv), f"{in_csv} not found"
    clips = []
    with open(in_csv, "r") as fh:
        reader = csv.reader(fh)
        for r in reader:
            clips.append([r[0], float(r[1])])

    getch = _GetchUnix()

    results = []

    for ix,(clip, dist) in enumerate(clips):
        print(f":::::: CLIP # {ix} :::", clip)
        fpath = str(CLOSEST_FARTHEST_DIR / WORD / MODE / clip)
        while True:
            print("-")
            wav = pydub.AudioSegment.from_file(fpath)
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
        row = [clip, dist, result]
        print(row)
        results.append(row)

    out_csv = CLOSEST_FARTHEST_DIR / f"{WORD}_{MODE}_50_results.csv"
    with open(out_csv, "w") as fh:
        writer = csv.writer(fh)
        writer.writerows(results)
    # summary
    print("\n\n:::::::::: SUMMARY  ")
    print("num good:", len([g for g in results if g[2] == "good"]))
    print("num bad:", len([g for g in results if g[2] == "bad"]))
    print(f"results written to {out_csv}")


if __name__ == "__main__":
    label()

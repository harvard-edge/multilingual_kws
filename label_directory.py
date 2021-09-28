from ast import parse
import glob
import argparse
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


def label(args):
    datadir = Path(args.datadir)
    print("datadir:", datadir)
    if args.closest:
        MODE = Modes.CLOSEST
    else:
        MODE = Modes.FARTHEST

    print("mode:", MODE)

    in_csv = datadir / args.word / MODE / f"{args.word}_{MODE}_50_input.csv"
    assert os.path.isfile(in_csv), f"{in_csv} not found"
    clips = []
    with open(in_csv, "r") as fh:
        reader = csv.reader(fh)
        for r in reader:
            clips.append([r[0], float(r[1])])

    getch = _GetchUnix()

    results = []

    for ix, (clip, dist) in enumerate(clips):
        print(f"\n:::::: CLIP # {ix} :::", clip)
        fpath = str(datadir / args.word / MODE / clip)
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

    if not args.dryrun:
        out_csv = datadir / f"{args.word}_{MODE}_50_results.csv"
        with open(out_csv, "w") as fh:
            writer = csv.writer(fh)
            writer.writerows(results)
    # summary
    print("\n\n\n\n:::::::::: SUMMARY  ")
    print("mode:", MODE)
    print("num good:", len([g for g in results if g[2] == "good"]))
    print("num bad:", len([g for g in results if g[2] == "bad"]))
    if not args.dryrun:
        print(f">>>>>> results written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="labeler for listening data")
    parser.add_argument("datadir", help="directory of closest/farthest data")
    parser.add_argument("word", help="word to analyze")
    parser.add_argument('--dryrun', action="store_true", help="do not write to output csv file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--closest", action="store_true", help="Mode: CLOSEST")
    group.add_argument("-f", "--farthest", action="store_true", help="Mode: FARTHEST")
    label(parser.parse_args())

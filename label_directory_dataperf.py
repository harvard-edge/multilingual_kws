import fire

import subprocess
import argparse
import sys
import tty
import termios
from pathlib import Path
import csv
import os

import code

# import pydub
# import pydub.playback
# import pydub.effects


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


def label_hf(datadir: os.PathLike, keyword: str, csv_outdir: os.PathLike):
    datadir = Path(datadir)
    print("datadir:", datadir)
    print(f"keyword: {keyword}")
    val_kws = (datadir / "validation").glob(f"{keyword}_*.wav")
    test_kws = (datadir / "test").glob(f"{keyword}_*.wav")
    wav_list = list(val_kws) + list(test_kws)
    outfile = Path(csv_outdir) / f"{keyword}_validated.csv"
    assert not outfile.exists(), f"outfile already exists: {outfile}"
    assert len(wav_list) > 0, f"no files found for keyword: {keyword}"

    getch = _GetchUnix()

    results = []

    for ix, clip in enumerate(wav_list):
        print(
            f"\n:::::: CLIP # {ix:03d} / {len(wav_list)} ::: {ix / len(wav_list):0.2f}",
            clip.name,
        )
        fpath = str(datadir / clip)
        while True:
            # for linux:
            # wav = pydub.AudioSegment.from_wav(fpath)
            # wav = pydub.effects.normalize(wav)
            # pydub.playback.play(wav)
            # for mac (note: skips normalization)
            # assumes ffmpeg (loudnorm or speechnorm) was used already to normalize
            # https://superuser.com/questions/323119/how-can-i-normalize-audio-using-ffmpeg
            subprocess.call(["afplay", fpath])

            print("rating? (g)ood, (b)ad, listen (a)gain, (q)uit")
            choice = getch()
            if choice == "g":
                usable = True
                break
            elif choice == "b":
                usable = False
                break
            elif choice == "q":
                print("not writing output")
                sys.exit()

        result = "good" if usable else "bad"
        # convert keyword_common_voice_id_1234.wav to keyword/common_voice_id_1234.wav
        dataperf_clip_id = keyword + "/" + clip.name[len(keyword) + 1 :]
        row = [dataperf_clip_id, result]
        print(f"{ix:03d} {row}")
        results.append(row)

    with open(outfile, "a") as fh:  # append to exisiting file
        writer = csv.writer(fh)
        writer.writerows(results)
    # summary
    print("\n\n\n\n:::::::::: SUMMARY  ")
    print("num good:", len([g for g in results if g[1] == "good"]))
    print("num bad:", len([g for g in results if g[1] == "bad"]))
    print(f">>>>>> results written to {outfile}")


def label(args):

    already_listened = []
    if os.path.isfile(args.out_csv):
        previous_file = Path(args.out_csv)
        print("datadir:", previous_file)
        with open(previous_file, "r") as fh:
            reader = csv.reader(fh)  # file, good?
            for row in reader:
                already_listened.append(row[0])

    already_listened = set(already_listened)

    datadir = Path(args.datadir)
    print("datadir:", datadir)

    splitdir = Path(args.splitdir)
    print("splitdir:", splitdir)

    wav_list = []  # list of wav file names in the dev and test splits

    dev_splits = splitdir / "en_dev.csv"
    with open(dev_splits, "r") as fh:
        reader = csv.reader(fh)  # SET,LINK,WORD,VALID,SPEAKER,GENDER
        for row in reader:
            if row[1] == args.word:
                opus = row[0]  # aachen/common_voice_en_18833718.opus
                wav = opus.replace("opus", "wav")
                if wav in already_listened:
                    continue
                wav_list.append(wav)

    test_splits = splitdir / "en_test.csv"
    with open(test_splits, "r") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if row[1] == args.word:
                opus = row[0]
                wav = opus.replace("opus", "wav")
                if wav in already_listened:
                    continue
                wav_list.append(wav)

    print("len(wav_list):", len(wav_list))

    getch = _GetchUnix()

    results = []

    for ix, clip in enumerate(wav_list):
        print(f"\n:::::: CLIP # {ix} :::", clip)
        fpath = str(datadir / clip)
        stop_and_save = False
        while True:
            print("-")
            # for linux:
            # wav = pydub.AudioSegment.from_wav(fpath)
            # wav = pydub.effects.normalize(wav)
            # pydub.playback.play(wav)
            # for mac (note: skips normalization)
            # assumes ffmpeg (loudnorm or speechnorm) was used already to normalize
            # https://superuser.com/questions/323119/how-can-i-normalize-audio-using-ffmpeg
            subprocess.call(["afplay", fpath])

            choice = getch()
            if choice == "g":
                usable = True
                break
            elif choice == "b":
                usable = False
                break
            elif choice == "s":
                stop_and_save = True
                break
            elif choice == "q":
                sys.exit()

        result = "good" if usable else "bad"
        row = [clip, result]
        print(row)
        results.append(row)
        if stop_and_save:
            break  # allows stopping and saving part of the way through a word

    if not args.dryrun:
        out_csv = args.out_csv
        with open(out_csv, "a") as fh:  # append to exisiting file
            writer = csv.writer(fh)
            writer.writerows(results)
    # summary
    print("\n\n\n\n:::::::::: SUMMARY  ")
    print("num good:", len([g for g in results if g[1] == "good"]))
    print("num bad:", len([g for g in results if g[1] == "bad"]))
    if not args.dryrun:
        print(f">>>>>> results written to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="labeler for listening data")
    parser.add_argument("datadir", help="directory of wavs")
    parser.add_argument("splitdir", help="directory of split csvs")
    parser.add_argument("word", help="word to analyze")
    parser.add_argument(
        "out_csv", help="output filepath. location of checkpoint if previously saved"
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="do not write to output csv file"
    )
    label(parser.parse_args())


def test():
    getch = _GetchUnix()
    while True:
        choice = getch()
        print(choice)
        if choice == "q":
            break


if __name__ == "__main__":
    fire.Fire(label_hf)

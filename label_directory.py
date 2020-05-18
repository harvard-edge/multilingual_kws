import glob
import pydub
from pydub.playback import play
import pandas as pd
import sys
import tty
import termios

WAVS = "./micro_dataset/extractions_deepspeech/down/*.wav"
CSVS = "./counts/*.csv"

# https://stackoverflow.com/a/510364
class _GetchUnix:
    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def label():
    outfile_name = "_".join(WAVS.split('/')[-3:-1])
    wavs = pd.Series(glob.glob(WAVS))

    df = pd.DataFrame(wavs, columns=["clips"])
    df["usable"] = False
    getch = _GetchUnix()

    usage = "good: g, bad: b, play again: a"

    for ix, r in df.iterrows():
        path = r.clips
        clip = pydub.AudioSegment.from_wav(path)
        while True:
            print()
            print(ix, path)
            play(clip)
            print(usage)
            #choice = sys.stdin.read(1)
            choice = getch()
            if choice == 'g':
                usable = True
                break
            elif choice == 'b':
                usable = False
                break
            elif choice == 'q':
                sys.exit()
        df.at[ix, "usable"] = usable
        print(usable)

    df.to_csv("counts/" + outfile_name + ".csv", index=False)

def count():
    csvs = glob.glob(CSVS)
    for c in csvs:
        print(c)
        df = pd.read_csv(c)
        good = df.usable.value_counts().loc[True]
        bad = df.usable.value_counts().loc[False]
        print("Good:", good, "Bad:", bad)


if __name__ == "__main__":
    count()
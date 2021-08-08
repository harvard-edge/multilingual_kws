# %%
import os
import glob
import shutil
import subprocess
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import numpy as np
import pydub
from pydub.playback import play

# %%
l_data = Path("/media/mark/hyperion/makerere/uliza-clips")
l_csv = l_data / "transcripts.csv"
l_test = Path("/media/mark/hyperion/makerere/alignment/cs288")
l_dest = l_test / "cs288_clips"

# %%
# keyword = "akawuka"
keyword_set = set(["corona", "senyiga", "masiki", "mask", "okugema"])
# loosely follows https://git.io/JOG7h
# but treats each file as a separate speaker
# see: https://montreal-forced-aligner.readthedocs.io/en/latest/data_prep.html
written = 0
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header [wav_filename,wav_filesize,transcript]
    for ix, row in enumerate(reader):
        if ix % 3000 == 0:
            print(ix)
        wav_filename = Path(row[0])
        transcript = row[2]
        has_kw = any([w in keyword_set for w in transcript.split()])
        if has_kw:
            assert os.path.isfile(l_data / wav_filename)
            lab_name = f"{wav_filename.stem}.lab"
            # fake that each wavfile is a separate speaker
            speaker_dir = os.makedirs(l_dest / wav_filename.stem)
            with open(
                l_dest / wav_filename.stem / lab_name, "w", encoding="utf8"
            ) as fh:
                fh.write(transcript)
            shutil.copy2(l_data / wav_filename, l_dest / wav_filename.stem)
            written += 1
            print(f"written: {written}")

# %%
# mini experiment
N_EXAMPLES = 100
os.makedirs(l_test / "data")
os.makedirs(l_test / "alignments")

labs = glob.glob(str(l_dest / "*.lab"))
sample = np.random.choice(labs, N_EXAMPLES, replace=False)
for s in sample:
    basename = Path(s).stem
    wav_name = f"{basename}.wav"

    shutil.copy2(s, l_test / "data")
    shutil.copy2(l_data / wav_name, l_test / "data")


# %%
# make lexicon (word followed by spelling)
# word w o r d
transcriptions = glob.glob(str(l_dest / "**/*.lab"))
words = set()
for t in transcriptions:
    with open(t, "r") as fh:
        l = fh.read()
    for w in l.split(" "):
        words.add(w)
print("Num words", len(words))
with open(l_test / "lexicon.txt", "w") as fh:
    for w in words:
        # filter apostrophes
        # https://montreal-forced-aligner.readthedocs.io/en/latest/data_format.html
        # spelling = " ".join(filter(lambda c: c != "'", w))
        spelling = " ".join(w)
        wl = f"{w} {spelling}\n"
        fh.write(wl)

# %%
# %%
# https://montreal-forced-aligner.readthedocs.io/en/latest/aligning.html#align-using-only-the-data-set

# docker run --rm -v $(pwd):/context -w /context -it montreal /bin/bash
# mfa train --clean akawuka_clips/ lexicon.txt alignments/

# %%
# ensure all audacity extractions conform to model settings
checks = ["16000 samples", "75 CDDA sectors"]
train_wavs = glob.glob("/home/mark/tinyspeech_harvard/luganda/cs288_training/**/*.wav")
for w in train_wavs:
    res = subprocess.run(f"soxi {w}", shell=True, capture_output=True)
    o = res.stdout.decode("utf8")
    print(o)
    for check in checks:
        if not check in o:
            raise ValueError(w)

# %%
# find duplicates

# https://stackoverflow.com/a/32558749
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


keyword = "senyiga"
#data = Path("/media/mark/hyperion/makerere/alignment/cs288")
data = Path("/media/mark/hyperion/makerere/alignment/covid")
alignments = data / "alignments"
#clips = data / "cs288_clips"
clips = data / "covid_clips"
alignment_speakers = [
    d for d in os.listdir(alignments) if os.path.isdir(alignments / d)
]  # skip unaligned.txt
tlist = []
for ix, speaker in enumerate(alignment_speakers):
    lab = clips / speaker / f"{speaker}.lab"
    wav = clips / speaker / f"{speaker}.wav"
    tgfile = alignments / speaker / f"{speaker}.TextGrid"
    with open(lab, "r") as fh:
        transcript = fh.read()
    has_kw = any([keyword == w for w in transcript.split()])
    if has_kw:
        tgdir = alignments / speaker
        tlist.append((transcript, tgdir))
print(len(tlist))

# %%
DISTANCE_THRESHOLD = 30
original = []
paths_for_deletion = []
for ix in range(len(tlist)):
    # if ix % int(len(tlist) / 20) == 0:
    #     print("completed----", ix, len(tlist))
    transcript1, tgdir1 = tlist[ix]
    for jx in range(ix + 1, len(tlist)):
        transcript2, tgdir2 = tlist[jx]
        if tgdir2 in paths_for_deletion:
            break
        dist = levenshteinDistance(transcript1, transcript2)
        if dist < DISTANCE_THRESHOLD:
            print(dist)
            print(transcript1)
            print(transcript2)
            original.append(tgdir1)
            paths_for_deletion.append(tgdir2)
print("total", len(paths_for_deletion))

# %%
# listen to duplicates
for (o, d) in zip(original, paths_for_deletion):
    w1 = clips / f"{o.name}/{o.name}.wav"
    w2 = clips / f"{d.name}/{d.name}.wav"
    print(w1)
    play(pydub.AudioSegment.from_file(w1))
    print(w2)
    play(pydub.AudioSegment.from_file(w2))
    print("----")


# %%
for d in paths_for_deletion:
    cmd = f"rm -rf {d}"
    print(cmd)
    #subprocess.run(cmd, shell=True)

# %%
# repackage unknown files

with open("/home/mark/tinyspeech_harvard/multilingual_embedding_wc/unknown_files.txt", 'r') as fh:
    unknown_files = fh.read().splitlines()


# %%
fwbase = Path("/media/mark/hyperion/kws_data/frequent_words/")
uwbase = Path("/media/mark/hyperion/kws_data/unknown_words/")
unknown_list = []
for f in unknown_files:
    p = Path(f[45:])
    source = fwbase / p
    dest = uwbase / p.parent
    # os.makedirs(dest, exist_ok=True)
    # shutil.copy2(source, dest)
    unknown = Path("unknown_words") / p
    unknown_list.append(unknown)

f = "/media/mark/hyperion/kws_data/unknown_files.txt"
assert not os.path.exists(f), "already present"
with open(f, 'w') as fh:
    for w in unknown_list:
        fh.write(str(w) + "\n")

# %%
with open("/media/mark/hyperion/kws_data/unknown_files.txt", 'r') as fh:
    unknown_files = fh.read().splitlines()
for f in unknown_files:
    if not os.path.exists("/media/mark/hyperion/kws_data/" + f):
        raise ValueError(f)

# %%

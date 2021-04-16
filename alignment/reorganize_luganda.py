# %%
import os
import glob
import shutil
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import numpy as np

# %%
l_data = Path("/media/mark/hyperion/makerere/uliza-clips")
l_csv = l_data / "transcripts.csv"
l_dest = Path("/media/mark/hyperion/makerere/alignment/covid_clips")
l_test = Path("/media/mark/hyperion/makerere/test")

# %%
keyword = "covid"
# loosely follows https://git.io/JOG7h
# but does not separate by speaker
# see: https://montreal-forced-aligner.readthedocs.io/en/latest/data_prep.html
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader) # skip header [wav_filename,wav_filesize,transcript]
    for ix, row in enumerate(reader):
        if ix % 3000 == 0:
            print(ix)
        wav_filename = Path(row[0])
        transcript = row[2]
        has_kw = any([w == keyword for w in transcript.split()])
        if has_kw:
            assert os.path.isfile(l_data / wav_filename)
            lab_name = f"{wav_filename.stem}.lab"
            # fake that each wavfile is a separate speaker
            speaker_dir = os.makedirs(l_dest / wav_filename.stem)
            with open(l_dest / wav_filename.stem / lab_name, 'w', encoding='utf8') as fh:
                fh.write(transcript)
            shutil.copy2(l_data / wav_filename, l_dest / wav_filename.stem)

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
transcriptions = glob.glob("/media/mark/hyperion/makerere/alignment/covid_clips/**/*.lab")
words = set()
for t in transcriptions:
    with open(t, 'r') as fh:
        l = fh.read()
    for w in l.split(' '):
        words.add(w)
print("Num words", len(words))
with open(l_test / "lexicon.txt", 'w') as fh:
    for w in words:
        # filter apostrophes
        # https://montreal-forced-aligner.readthedocs.io/en/latest/data_format.html
        #spelling = " ".join(filter(lambda c: c != "'", w))
        spelling = " ".join(w)
        wl = f"{w} {spelling}\n"
        fh.write(wl)

# %%
# %%
# https://montreal-forced-aligner.readthedocs.io/en/latest/aligning.html#align-using-only-the-data-set

# docker run --rm -v $(pwd):/context -w /context -it montreal /bin/bash
# mfa train --clean covid_clips/ lexicon.txt alignments/ 

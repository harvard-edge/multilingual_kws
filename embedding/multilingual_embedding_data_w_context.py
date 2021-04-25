import numpy as np
import os
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict
import functools
from collections import Counter, OrderedDict
import csv
import pathlib
import textgrid
import sox
from pathlib import Path
import pickle
import multiprocessing

import word_extraction

source_data = Path("/home/mark/tinyspeech_harvard/multilang_embedding/")

with open(source_data / "train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open(source_data / "val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()
with open(source_data / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()
print("SOURCE DATA")
print(len(train_files), train_files[0])
print(len(val_files), val_files[0])
print(len(commands))
print("-----")

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")


# build a nested map of common_voice_id to start and end times
# lang_isocode -> word -> commonvoice_id -> (start_s, end_s)
timing_dict = {}
for lang_isocode in ["ca", "de", "en", "es", "fa", "fr", "it", "nl", "rw"]:
    print(lang_isocode)
    timing_dict[lang_isocode] = {}
    for word_csv in glob.glob(str(frequent_words / lang_isocode / "timings" / "*.csv")):
        word_csv = Path(word_csv)
        word = word_csv.stem
        d = {}

        with open(word_csv, "r") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                common_voice_id = row[0]
                start_s = row[1]
                end_s = row[2]

                if common_voice_id in d:
                    continue
                d[common_voice_id] = (start_s, end_s)
        timing_dict[lang_isocode][word] = d

# how many extracted words were taken from common voice clips that have multiple occurences of the word
#    frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")
#    multi_extraction = 0
#
#    for ix, t in enumerate(train_files):
#        if ix % int(len(train_files) / 5) == 0:  # 20%
#            print(ix)
#        t = Path(t)
#        wav_noext = t.stem
#
#        if wav_noext.count("_") > 3:
#            multi_extraction += 1
#            continue
#
#    print(multi_extraction, multi_Apr 28, 2021extraction / len(train_files))
#    # around 3%

# build sister dataset with context surrounding extractions
dest_base = Path("/media/mark/hyperion/frequent_words_w_context")


def extract_context(training_sample):
    t = Path(training_sample)
    lang_isocode = t.parts[5]
    word = t.parts[7]
    wav = t.parts[8]
    wav_noext = t.stem

    if wav_noext.count("_") > 3:
        return

    start_s, end_s = timing_dict[lang_isocode][word][wav_noext]
    start_s = float(start_s)
    end_s = float(end_s)

    if lang_isocode == "es":
        cv_clipsdir = Path(
            "/media/mark/hyperion/common_voice/cv-corpus-5.1-2020-06-22/es/clips"
        )
    else:
        cv_clipsdir = Path(
            f"/media/mark/hyperion/common_voice/cv-corpus-6.1-2020-12-11/{lang_isocode}/clips"
        )

    dest_dir = dest_base / lang_isocode / "clips" / word
    os.makedirs(dest_dir, exist_ok=True)

    dest_file = dest_dir / wav
    if os.path.exists(dest_file):
        return # already generated from a previous run
    
    source_mp3 = cv_clipsdir / (wav_noext + ".mp3")
    if not os.path.exists(source_mp3):
        print("warning: source mp3 not found", source_mp3)
        return


    word_extraction.extract_shot_from_mp3(
        mp3name_no_ext=wav_noext,
        start_s=start_s,
        end_s=end_s,
        dest_dir=dest_dir,
        include_context=True,
        cv_clipsdir=cv_clipsdir,
    )


pool = multiprocessing.Pool()
num_processed = 0
for _ in pool.imap_unordered(extract_context, train_files + val_files, chunksize=4000):
    num_processed += 1
print(num_processed)

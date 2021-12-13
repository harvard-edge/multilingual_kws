#%%
import argparse
from collections import Counter
from dataclasses import dataclass
import shutil
import logging
import os
import sys
import pprint
import pickle
import glob
import subprocess
import pathlib
from pathlib import Path
from typing import List
import sox

import numpy as np
import tensorflow as tf

#%%


#%%
train100 = "/media/mark/hyperion/librispeech/LibriSpeech/train-clean-100/"

speakers = set(os.listdir(train100))
print(speakers)

transcription_lines = []

w = os.walk(train100)
ix = 0
for dirpath, dirnames, filenames in w:
    # print('here')
    # print(dirpath, dirnames, filenames)
    if dirnames == []:
        # in a flac folder
        for f in filenames:
            fname, fext = os.path.splitext(f)
            if fext == ".txt":
                fullpath = f"{dirpath}/{f}"
                # print(fullpath)
                with open(fullpath, "r") as fh:
                    ls = fh.read().splitlines()
                    transcription_lines.extend(ls)

#%%
len(transcription_lines)

#%%

speaker2text = {}

for l in transcription_lines:
    s_b_f = l.split(" ")[0]
    speaker, book, sentence_id = s_b_f.split("-")
    transcription = l[len(s_b_f) + 1 :]
    if not speaker in speaker2text:
        speaker2text[speaker] = []
    speaker2text[speaker].append((book, sentence_id, transcription))

#%%
speaker2counts = {}
for speaker, sentences in speaker2text.items():
    if not speaker in speaker2counts:
        speaker2counts[speaker] = Counter()
    for book, sentence_id, transcription in sentences:
        words = transcription.split()
        for w in words:
            speaker2counts[speaker][w] += 1

#%%
for speaker, counts in speaker2counts.items():
    print(speaker, "\t", counts.most_common(12)[5:12])
# 1553 	 [('IT', 54), ('IN', 52), ('REBECCA', 52)

#%%
speaker_id = "1553"
lines = speaker2text[speaker_id]
print(len(lines))

target = "REBECCA"
targets = [l for l in lines if target in l[2]]
# note: might have multiple targets in the same sentence
print(len(targets))

#%%
books = [l[0] for l in lines]
print(set(books))
#%%
# find candidate targets
rand_ixs = np.random.choice(len(targets), 3, replace=False)
print(rand_ixs)
for r in rand_ixs:
    print(targets[r])
#%%
# target sources to excerpt from
sources = [
    (
        "140048",
        "0012",
        "THE VERY PUREST CORROBORATED REBECCA NO ACID IN IT NOT A TRACE AND YET A CHILD COULD DO THE MONDAY WASHING WITH IT AND USE NO FORCE A BABE CORRECTED REBECCA",
    ),
    (
        "140048",
        "0001",
        "AND INTERVIEWED ANY ONE WHO SEEMED OF A COMING ON DISPOSITION EMMA JANE HAD DISPOSED OF THREE SINGLE CAKES REBECCA OF THREE SMALL BOXES FOR A DIFFERENCE IN THEIR ABILITY TO PERSUADE THE PUBLIC WAS CLEARLY DEFINED AT THE START",
    ),
]
source_books = [s[0] for s in sources]
source_fileids =  [s[1] for s in sources]

DEST_DIR = "/home/mark/tinyspeech_harvard/source_flacs/"
# for book_id, sentence_id, _ in sources:
#     fn = f"{speaker_id}-{book}-{sentence_id}.flac"
#     flac = Path(train100) / "1553" / book / fn
#     shutil.copy2(flac, DEST_DIR)

#%%

#%%
# write excerpted wav files with padding to destination directory
for excerpt_id in range(1,4):
    wav = DEST_DIR + f"rebecca{excerpt_id}_excerpt.wav"
    dur_s = sox.file_info.duration(wav)
    print(dur_s)
    if dur_s >= 1.0:
        raise ValueError(wav)
    pad_s = (1 - dur_s) / 2

    dest_target = DEST_DIR + f"targets/rebecca{excerpt_id}.wav"
    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.pad(start_duration=pad_s, end_duration=pad_s)
    transformer.build(wav, dest_target)

#%%
# calculate total duration of wavfile
duration = 0
flacs_to_combine = []
for book_id, sentence_id, transcript in lines:
    if book_id in source_books and sentence_id in source_fileids:
        print("skiping", book_id, sentence_id)
        continue
    fn = f"{speaker_id}-{book_id}-{sentence_id}.flac"
    flac = Path(train100) / speaker_id / book_id / fn
    dur_s = sox.file_info.duration(flac)
    duration += dur_s
    flacs_to_combine.append(str(flac))
print(duration, duration / 60)

#%%

#%%
combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(flacs_to_combine, DEST_DIR + "stream.wav", "concatenate")
#%%
sox.file_info.duration(DEST_DIR + "stream.wav")

#%%
base_dir = Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/rebecca/")
found_file = base_dir / f"found_words_w_confidences_{target}.pkl"
with open(found_file, 'rb') as fh:
    found_w_confidences = pickle.load(fh)

#%%
#%%
# excerpt all found words from streaming wav
result_wavs = base_dir / "result_wavs"

for found_word, time_ms, confidence in found_w_confidences:
    if found_word != target:
        continue
    dest_wav = str(result_wavs / f"{target}_{time_ms}.wav")
    print(dest_wav)
    time_s = time_ms / 1000.

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(time_s, time_s + 1)
    transformer.build(DEST_DIR + "stream.wav", dest_wav)

#%%
# how many found
len(os.listdir(result_wavs))
#%%
# false positives
print(256160 / 1000 / 60)
print(262200 / 1000 / 60)
print(633140 / 1000 / 60)


#%%
total_words = 0
for t in targets:
    words = t[2].split()
    total_words += len(words)
total_words
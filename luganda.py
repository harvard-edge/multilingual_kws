#%%
from input_data import UNKNOWN_WORD_INDEX
import numpy as np
import os
import glob
import shutil
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict, Set
import functools
from collections import Counter
import csv
import textgrid
import sox
import pickle
from scipy.io import wavfile

# import tensorflow as tf
# import input_data
from pathlib import Path
import pydub
from pydub.playback import play
import time

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
import input_data


# %%
# Luganda

l_data = Path("/media/mark/hyperion/makerere/luganda/luganda/")
l_csv = l_data / "data.csv"
counts = word_extraction.wordcounts(l_csv, skip_header=False, transcript_column=1)

# %%
# find keywords
N_WORDS_TO_SAMPLE = 10
MIN_CHAR_LEN = 4
SKIP_FIRST_N = 5

counts.most_common(SKIP_FIRST_N)
# %%
non_stopwords = counts.copy()
# get rid of words that are too short
to_expunge = counts.most_common(SKIP_FIRST_N)
for k, _ in to_expunge:
    del non_stopwords[k]

longer_words = [kv for kv in non_stopwords.most_common() if len(kv[0]) >= MIN_CHAR_LEN]

print("num words to be extracted", len(longer_words[:N_WORDS_TO_SAMPLE]))
print("counts for last word", longer_words[N_WORDS_TO_SAMPLE - 1])
print("words:\n", " ".join([l[0] for l in longer_words[:N_WORDS_TO_SAMPLE]]))


# %%
# visualize frequencies of top words
fig, ax = plt.subplots()
topn = longer_words[:N_WORDS_TO_SAMPLE]
ax.bar([c[0] for c in topn], [c[1] for c in topn])
ax.set_xticklabels([c[0] for c in topn], rotation=70)
# ax.set_ylim([0, 3000])
# fig.set_size_inches(40, 10)

# %%
keyword = "covid"
wavs = []
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    for ix, row in enumerate(reader):
        words = row[1].split()
        for w in words:
            if w == keyword:
                wavs.append(row[0])
# %%
# listen to random samples
ix = np.random.randint(len(wavs))
w = l_data / "clips" / wavs[ix]
print(ix, w)
play(pydub.AudioSegment.from_file(w))

# %%
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
silence_padded = workdir / "silence_padded"

# %%
# extract covid from first 5 wavs using audacity
fiveshot_dest = workdir / "originals"
os.makedirs(fiveshot_dest)
for ix in range(5):
    w = l_data / "clips" / wavs[ix]
    shutil.copy2(w, fiveshot_dest)

# %%
# pad with silence out to 1 second

unpadded = workdir / "unpadded"
# os.makedirs(silence_padded)

for f in os.listdir(unpadded):
    src = str(unpadded / f)
    print(src)
    duration_s = sox.file_info.duration(src)
    if duration_s < 1:
        pad_amt_s = (1.0 - duration_s) / 2.0
    else:
        raise ValueError("utterance longer than 1s", src)

    dest = silence_padded / f
    # words can appear multiple times in a sentence: above should have filtered these
    if os.path.exists(dest):
        raise ValueError("already exists:", dest)

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
    transformer.build(src, str(dest))


# %%
# select random wavs without the keyword to intersperse stream with
non_targets = []
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    for ix, row in enumerate(reader):
        words = row[1].split()
        has_keyword = False
        for w in words:
            if w == keyword:
                has_keyword = True
        if not has_keyword:
            non_targets.append(row[0])
# %%
n_stream_wavs = len(wavs[5:])
print(n_stream_wavs)
selected_nontargets = np.random.choice(non_targets, n_stream_wavs, replace=False)
# %%
# make streaming wav
intermediate_wavdir = workdir / "intermediate_wavs"
os.makedirs(intermediate_wavdir)

stream_wavs = []
for ix, (target_wav, nontarget_wav) in enumerate(zip(wavs, selected_nontargets)):
    tw = l_data / "clips" / target_wav
    nw = l_data / "clips" / nontarget_wav

    # convert all to same samplerate
    for w in [tw, nw]:
        dest = str(intermediate_wavdir / w.name)
        transformer = sox.Transformer()
        transformer.convert(samplerate=16000)  # from 48K mp3s
        transformer.build(str(w), dest)
        stream_wavs.append(dest)

stream_wavfile = str(workdir / "covid_stream.wav")

combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(stream_wavs, stream_wavfile, "concatenate")

print(sox.file_info.duration(stream_wavfile), "seconds in length")

# %%

# load embedding model
traindir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding")

# SELECT MODEL
base_model_path = (
    traindir / "models" / "multilang_resume40_resume05_resume20_resume22.007-0.7981/"
)

model_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis_ooe/")
unknown_collection_path = model_dir / "unknown_collection.pkl"
with open(unknown_collection_path, "rb") as fh:
    unknown_collection = pickle.load(fh)
# unknown_lang_words = unknown_collection["unknown_lang_words"]
unknown_files = unknown_collection["unknown_files"]
# oov_lang_words = unknown_collection["oov_lang_words"]
# commands = unknown_collection["commands"]
# unknown_words = set([lw[1] for lw in unknown_lang_words])

# %%

target_n_shots = os.listdir(silence_padded)

train_files = [str(silence_padded / w) for w in target_n_shots]
# reuse train for val
val_files = [str(silence_padded / w) for w in target_n_shots]
print(train_files)

# %%

model_dest_dir = workdir / "model"
model_settings = input_data.standard_microspeech_model_settings(3)
name, model, details = transfer_learning.transfer_learn(
    target=keyword,
    train_files=train_files,
    val_files=val_files,
    unknown_files=unknown_files,
    num_epochs=4,  # 9
    num_batches=1,  # 3
    batch_size=64,
    model_settings=model_settings,
    base_model_path=base_model_path,
    base_model_output="dense_2",
    csvlog_dest=model_dest_dir / "log.csv",
)
print("saving", name)
model.save(model_dest_dir / name)

# %%
# sanity check model outputs
specs = [input_data.file2spec(model_settings, f) for f in val_files]
specs = np.expand_dims(specs, -1)
print(specs.shape)
preds = model.predict(specs)
amx = np.argmax(preds, axis=1)
print(amx)
print("VAL ACCURACY", amx[amx == 2].shape[0] / preds.shape[0])
print("--")

with np.printoptions(precision=3, suppress=True):
    print(preds)
# %%
# run inference
modelpath = model_dest_dir / name
streamwav = workdir / "covid_stream.wav"
empty_gt = workdir / "empty.txt"
dest_pkl = workdir / "results" / "streaming_results.pkl"
dest_inf = workdir / "results" / "inferences.npy"
streamtarget = sa.StreamTarget("lu", "covid", modelpath, streamwav, empty_gt, dest_pkl, dest_inf )

sa.eval_stream_test(streamtarget)

# %%
# DONE 0.05
# No ground truth yet, 129false positives
# DONE 0.1
# No ground truth yet, 350false positives
# DONE 0.15
# No ground truth yet, 484false positives
# DONE 0.2
# No ground truth yet, 552false positives
# DONE 0.25
# No ground truth yet, 530false positives
# DONE 0.3
# No ground truth yet, 463false positives
# DONE 0.35
# No ground truth yet, 390false positives
# DONE 0.39999999999999997
# No ground truth yet, 321false positives
# DONE 0.44999999999999996
# No ground truth yet, 255false positives
# DONE 0.49999999999999994
# No ground truth yet, 175false positives
# DONE 0.5499999999999999
# No ground truth yet, 122false positives
# DONE 0.6
# No ground truth yet, 93false positives
# DONE 0.65
# No ground truth yet, 69false positives
# DONE 0.7
# No ground truth yet, 41false positives
# DONE 0.75
# No ground truth yet, 29false positives
# DONE 0.7999999999999999
# No ground truth yet, 15false positives
# DONE 0.85
# No ground truth yet, 8false positives
# DONE 0.9
# No ground truth yet, 4false positives
# DONE 0.95
# No ground truth yet, 0false positives
# DONE 1.0
# No ground truth yet, 0false positives

# %%
with open(streamtarget.destination_result_pkl, "rb") as fh:
    results = pickle.load(fh)
# %%
operating_point = 0.65
for thresh, (_, found_words, all_found_w_confidences) in results[keyword].items():
    if np.isclose(thresh, operating_point):
        break
print(len(found_words), "targets found")

# %%
stream = pydub.AudioSegment.from_file(streamtarget.stream_wav)

# %%
ix = np.random.randint(len(found_words))
time_ms = found_words[ix][1]
time_s = time_ms / 1000
print(ix, time_s)
context_ms = 1000
play(stream[time_ms - context_ms: time_ms + context_ms])
# %%
# save detections from stream
extractions = workdir / "extractions"
os.makedirs(extractions)
for ix, (_, time_ms) in enumerate(found_words):
    print(time_ms)

    dest_wav = str(extractions / f"{ix:03d}_{keyword}_detection_thresh_{operating_point}_{time_ms}ms.wav")
    print(dest_wav)
    time_s = time_ms / 1000.

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(time_s - 1, time_s + 1)
    transformer.build(str(streamtarget.stream_wav), dest_wav)

# %%

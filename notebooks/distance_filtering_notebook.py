# %%
import os
from pathlib import Path
import pickle
import logging
import glob
import csv
import time
import shutil
import pprint

import numpy as np
import tensorflow as tf
import sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns

import pydub
import pydub.playback
import pydub.effects

import sys
sys.path.insert(0, str(Path.cwd().parents[0]))
import embedding.input_data as input_data
import embedding.distance_filtering as distance_filtering

sns.set()
sns.set_style("white")
sns.set_palette("bright")

# %%
embedding_model_dir = Path.home() / "tinyspeech_harvard/multilingual_embedding_wc"
with open(embedding_model_dir / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()

em = distance_filtering.embedding_model()
# %%
#words_dir = Path.home() / "tinyspeech_harvard/distance_sorting/cv7_extractions/listening_data"
words_dir = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
word = "doing"
clips = glob.glob(str(words_dir / word / "*.wav"))
clips.sort()
print(len(clips))

# %%

results = distance_filtering.cluster_and_sort(clips, em)
print(len(results["sorted_clips"])) # will be 50 less than above


# %%
en_words_dir = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
en_words = os.listdir(en_words_dir)
print("all en words", len(en_words))

en_non_embedding_words = set(en_words).difference(commands)
print("non-embedding words", len(en_non_embedding_words))
# %%
print(en_non_embedding_words)

# %%

# %%
distances = results["distances"]

fig, ax = plt.subplots(ncols=2, dpi=150)
ax[0].plot(np.arange(distances.shape[0]), distances)
ax[0].set_xlabel("sorted index of training sample")
ax[0].set_ylabel(f"L2 distance to nearest cluster centroids (K=5)")
ax[0].set_title(f"sorted distances for {word}")
ax[1].hist(distances)
ax[1].set_title("distances histogram")
fig.set_size_inches(8, 4)


# %%
# "worst" clips
sorted_clips = results["sorted_clips"]
for ix, (f,dist) in enumerate(reversed(list(zip(sorted_clips, distances)))):
    if ix > 10:
        break
    print(Path(f).name, dist)
    wav = pydub.AudioSegment.from_file(f)
    wav = pydub.effects.normalize(wav)
    pydub.playback.play(wav)

# %%
# "best" clips (closest)
for ix, (f,dist) in enumerate(list(zip(sorted_clips, distances))):
    if ix > 5:
        break
    print(Path(f).name, dist)
    wav = pydub.AudioSegment.from_file(f)
    wav = pydub.effects.normalize(wav)
    pydub.playback.play(wav)

# %%
dfdir = Path.home() / "tinyspeech_harvard/distance_sorting"
csvs = glob.glob(str(dfdir / "*.csv"))
print(csvs)
# %%

bad_clips = []
good_clips = []
with open(csvs[1], "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header
    for ix, line in enumerate(reader):
        if line[3] == "bad":
            bad_clips.append(Path(line[2]).name)
        else:
            good_clips.append(Path(line[2]).name)
# %%
len(bad_clips)

# %%

# %%
for ix, c in enumerate(bad_clips):
    clip = en_words_dir / "doing" / c
    wav = pydub.AudioSegment.from_file(clip)
    wav = pydub.effects.normalize(wav)
    print(ix, c)
    pydub.playback.play(wav)
    time.sleep(0.5)
    # time.sleep(1)
    # if ix > 9:
    #     break

# %%

for ix, c in enumerate(good_clips):
    # if ix != 2:
    #     continue
    clip = en_words_dir / "doing" / c
    wav = pydub.AudioSegment.from_file(clip)
    wav = pydub.effects.normalize(wav)
    pydub.playback.play(wav)
    print(ix, clip)
    # time.sleep(1)
    # if ix > 9:
    #     break

# %%
# c = "common_voice_en_18471371.wav"
c = good_clips[37]
clip = en_words_dir / "doing" / c
wav = pydub.AudioSegment.from_file(clip)
wav = pydub.effects.normalize(wav)
pydub.playback.play(wav)
print(c)
# %%

# %%
scsv = Path.home() / "tinyspeech_harvard/tinyspeech/tmp/story.csv"
bad_clips= []
with open(scsv, "r") as fh:
    reader = csv.reader(fh)
    for ix, line in enumerate(reader):
        if line[1] == "bad":
            bad_clips.append(line[0])

for ix, c in enumerate(bad_clips):
    clip = en_words_dir / "story" / c
    wav = pydub.AudioSegment.from_file(clip)
    wav = pydub.effects.normalize(wav)
    print(ix, c)
    pydub.playback.play(wav)

# %%
def play(c):
    clip = en_words_dir / "story" / c
    wav = pydub.AudioSegment.from_file(clip)
    wav = pydub.effects.normalize(wav)
    print(c)
    pydub.playback.play(wav)
# %%
play(bad_clips[8])

# %%
a2 = Path.home() / "tinyspeech_harvard/distance_sorting/aug2_csvs"
csvs = glob.glob(str(a2/"*.csv"))
csvs.sort()

# %%
_ = [print(ix, c) for ix,c in enumerate(csvs)]

# %%
story_bad_clips = bad_clips
story_bad_clips
# %%
csvs[3]

# %%
scsv = csvs[0]
print(scsv)
all_clips = []
with open(scsv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)
    for ix, line in enumerate(reader):
        clip = Path(line[-2]).name
        all_clips.append(clip)
len(all_clips)

# %%
scsv = csvs[0]
print(scsv)
bad_clips = []
with open(scsv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)
    for ix, line in enumerate(reader):
        if line[-1] == "bad" or line[-1] == "unsure":
            clip = Path(line[-2]).name
            bad_clips.append(clip)
bad_clips

# %%
set(story_bad_clips).intersection(bad_clips)
# %%
set(all_clips).intersection(bad_clips)

# %%
set(all_clips_f).intersection(all_clips)

# %%
all_clips_f= all_clips
# %%
all_clips_f
# %%
play(bad_clips[3])
# all 

# %%
def get_bad(fpath):
    bad_clips = []
    with open(fpath, 'r') as fh:
        reader = csv.reader(fh)
        next(reader) # skip header
        for ix,line in enumerate(reader):
            if line[-1] == "bad" or line[-1] == "unsure":
            # if line[-1] == "bad":
                clip = Path(line[-2]).name
                bad_clips.append(clip)
    return bad_clips, ix + 1
# %%
rollup = {}
for fpath in csvs:
    # print(fpath)
    components = Path(fpath).stem.split("_")
    word = components[1]
    sampling = components[2]
    bad_clips, n_lines = get_bad(fpath)
    print(word, sampling, len(bad_clips), n_lines)
    if word not in rollup:
        rollup[word] = {}
    bad_pct = len(bad_clips) / n_lines * 100
    rollup[word][sampling] = f"{bad_pct:0.1f}%"


# %%
del rollup['along']
# %%
pprint.pprint(rollup)

# %%
bad_clips, n = get_bad(csvs[6])
print(len(bad_clips))
# %%

en_words_dir = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
clips = os.listdir(en_words_dir)
clips.sort()
len(clips)


# %%
counts = {}
for c in clips:
    wavs = glob.glob(str(en_words_dir / c / "*.wav"))
    counts[c] = len(wavs)

# %%
counts['film']

# %%
counts

# %%

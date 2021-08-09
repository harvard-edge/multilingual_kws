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

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data

sns.set()
sns.set_style("white")
sns.set_palette("bright")

# %%
embedding_model_dir = Path("/home/mark/tinyspeech_harvard/multilingual_embedding_wc")

with open(embedding_model_dir / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()
model_settings = input_data.standard_microspeech_model_settings(label_count=761)

# %%
base_model_path = embedding_model_dir / "models" / "multilingual_context_73_0.8011"
tf.get_logger().setLevel(logging.ERROR)
base_model = tf.keras.models.load_model(base_model_path)
tf.get_logger().setLevel(logging.INFO)

base_model_output = "dense_2"
embedding = tf.keras.models.Model(
    name="TransferLearnedModel",
    inputs=base_model.inputs,
    outputs=base_model.get_layer(name=base_model_output).output,
)
embedding.trainable = False

# %%
en_words_dir = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
en_words = os.listdir(en_words_dir)
print("all en words", len(en_words))

en_non_embedding_words = set(en_words).difference(commands)
print("non-embedding words", len(en_non_embedding_words))
# %%
print(en_non_embedding_words)

# %%
query_word = "doing"
query_clips = glob.glob(str(en_words_dir / query_word / "*.wav"))
query_clips.sort()
np.random.seed(123)
np.random.shuffle(query_clips)
print("num extractions", len(query_clips))
print("\n".join(query_clips[:3]))

N_TRAIN = 40
N_CLUSTERS = 3
train_clips = query_clips[:N_TRAIN]
dev_clips = query_clips[N_TRAIN:]

# %%
# cluster training and compare against dev
train_spectrograms = np.array(
    [input_data.file2spec(model_settings, fp) for fp in train_clips]
)
feature_vectors = embedding.predict(train_spectrograms)

kmeans = sklearn.cluster.KMeans(n_clusters=N_CLUSTERS, random_state=123).fit(
    feature_vectors
)

dev_spectrograms = np.array(
    [input_data.file2spec(model_settings, fp) for fp in dev_clips]
)
dev_vectors = embedding.predict(dev_spectrograms)

l2_distances = np.linalg.norm(
    kmeans.cluster_centers_[:, np.newaxis] - dev_vectors[np.newaxis], axis=-1
)
max_l2 = np.max(l2_distances, axis=0)

print(np.argmax(max_l2))
furthest = np.argsort(max_l2)[::-1]
print(furthest[:5])

fig, ax = plt.subplots(ncols=2, dpi=150)
ax[0].plot(np.arange(max_l2.shape[0]), np.sort(max_l2)[::-1])
ax[0].set_xlabel("sorted index of training sample")
ax[0].set_ylabel(f"max L2 distance to cluster centroids (K={N_CLUSTERS})")
ax[0].set_title(f"sorted max(L2) distances for {query_word}")
ax[1].hist(max_l2)
ax[1].set_title("max(L2) distances histogram")
fig.set_size_inches(8, 4)


# %%
dest = Path.home() / "tinyspeech_harvard/distance_sorting" / "worst" / query_word
os.makedirs(dest, exist_ok=True)
# "worst" clips
for f in furthest[:5]:
    c = dev_clips[f]
    print(max_l2[f], c)
    wav = pydub.AudioSegment.from_file(c)
    wav = pydub.effects.normalize(wav)
    pydub.playback.play(wav)
    shutil.copy2(c, dest)
# %%
# "best" clips (closest)
dest = Path.home() / "tinyspeech_harvard/distance_sorting" / "best" / query_word
os.makedirs(dest, exist_ok=True)
for f in furthest[::-1][:5]:
    c = dev_clips[f]
    print(max_l2[f], c)
    wav = pydub.AudioSegment.from_file(c)
    wav = pydub.effects.normalize(wav)
    pydub.playback.play(wav)
    shutil.copy2(c, dest)
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

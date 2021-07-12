# %%
import os
from pathlib import Path
import pickle
import logging
import glob
import shutil

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
query_word = "along"
query_clips = glob.glob(str(en_words_dir / query_word / "*.wav"))
query_clips.sort()
np.random.seed(123)
np.random.shuffle(query_clips)
print("num extractions", len(query_clips))
print("\n".join(query_clips[:3]))

N_TRAIN = 40
N_CLUSTERS = 10
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

fig,ax = plt.subplots(ncols=2,dpi=150)
ax[0].plot(np.arange(max_l2.shape[0]), np.sort(max_l2)[::-1])
ax[0].set_xlabel("sorted index of training sample")
ax[0].set_ylabel(f"max L2 distance to cluster centroids (K={N_CLUSTERS})")
ax[0].set_title(f"sorted max(L2) distances for {query_word}")
ax[1].hist(max_l2)
ax[1].set_title("max(L2) distances histogram")
fig.set_size_inches(8,4)


# %%
dest = Path.home() / "tinyspeech_harvard/distance_sorting" / "worst" / query_word
os.makedirs(dest,exist_ok=True)
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
os.makedirs(dest,exist_ok=True)
for f in furthest[::-1][:5]:
    c = dev_clips[f]
    print(max_l2[f], c)
    wav = pydub.AudioSegment.from_file(c)
    wav = pydub.effects.normalize(wav)
    pydub.playback.play(wav)
    shutil.copy2(c, dest)
# %%

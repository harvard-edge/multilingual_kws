#%%
import numpy as np
import os
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns

# import nltk
from typing import Set, List, Dict
import functools
from collections import Counter, OrderedDict
import csv
import pathlib

# import stopwordsiso as stopwords
import textgrid
import sox
from pathlib import Path
import pickle

import word_extraction

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1.6)

#%%
# how large is each language?
langs = {}
alignments = Path("/home/mark/tinyspeech_harvard/common-voice-forced-alignments")
for lang in os.listdir(alignments):
    if not os.path.isdir(alignments / lang):
        continue
    validated = alignments / lang / "validated.csv"
    if not os.path.isfile(validated):
        continue
    with open(validated, "r") as fh:
        langs[lang] = len(fh.readlines())
langs = OrderedDict(sorted(langs.items(), key=lambda kv: kv[1], reverse=True))
# too many results in english
del langs["en"]

fig, ax = plt.subplots()
ax.bar(langs.keys(), langs.values())
ax.set_xticklabels(langs.keys(), rotation=90)
fig.set_size_inches(20, 5)

# %%
#  # https://matplotlib.org/stable/gallery/ticks_and_spines/tick-formatters.html
#  from matplotlib import ticker
#  # https://stackoverflow.com/a/579376
#  def human_format(num):
#      magnitude = 0
#      while abs(num) >= 1000:
#          magnitude += 1
#          num /= 1000.0
#      # add more suffixes if you need them
#      return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
#  @ticker.FuncFormatter
#  def major_formatter(x, pos):
#      return human_format(x)
#  flangs = {k:langs[k] for k in ["de", "rw", "es", "it", "nl"]}
#
#  fig, ax = plt.subplots()
#  ax.bar(flangs.keys(), flangs.values())
#  ax.set_xticklabels(flangs.keys(), rotation=90)
#  # ax.set_xlabel("language")
#  # ax.set_ylabel("")
#  ax.yaxis.set_major_formatter(major_formatter)
#  fig.set_size_inches(3,10)

# %%
iso2lang = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ky": "Kyrgyz",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "ta": "Tamil",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukranian",
}

# %%
# total
# ['ar', 'ca', 'cs', 'cy', 'de', 'en', 'es', 'et', 'eu', 'fa', 'fr', 'id', 'it', 'ky', 'nl', 'pl', 'pt', 'ru', 'rw', 'ta', 'tr', 'tt', 'uk']
# non-embedding classification:
# done: cs cy eu
# ['ar', 'cs', 'cy', 'et', 'eu', 'id', 'ky', 'pl', 'pt', 'ru', 'ta', 'tr', 'tt', 'uk']
LANG_ISOCODE = "uk"
print("Generating", LANG_ISOCODE, iso2lang[LANG_ISOCODE])

frequent_words_dir = f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}"
timings_dir = f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/timings"
errors_dir = f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/errors"
clips_dir = f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips"

for dir in [frequent_words_dir, timings_dir, errors_dir, clips_dir]:
    os.makedirs(dir)
for dir in [frequent_words_dir, timings_dir, errors_dir, clips_dir]:
    if not os.path.isdir(dir):
        raise ValueError("need to create", dir)

# we are just using the speech commands dir directly, ignore this for now
# background_noise_dir=f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/_background_noise_"
# BASE_BACKGROUND_NOISE="/home/mark/tinyspeech_harvard/speech_commands/_background_noise_"
# shutil.copytree(BASE_BACKGROUND_NOISE, background_noise_dir)
# if not os.path.isdir(background_noise_dir):
#     raise ValueError("need to copy bg data", background_noise_dir)

# %%
# generate most frequent words and their counts
alignments = Path("/home/mark/tinyspeech_harvard/common-voice-forced-alignments")
counts = word_extraction.wordcounts(alignments / LANG_ISOCODE / "validated.csv")

# %%
# look for stopwords that are too short
counts.most_common(15)

# %%

# %%
N_WORDS_TO_SAMPLE = 50
MIN_CHAR_LEN = 5
SKIP_FIRST_N = 20

non_stopwords = counts.copy()
# get rid of words that are too short
to_expunge = counts.most_common(SKIP_FIRST_N)
for k, _ in to_expunge:
    del non_stopwords[k]

already_exists = os.listdir(clips_dir)
skipped = 0
for k in already_exists:
    if k in non_stopwords.keys():
        print("already exists", k)
        skipped += 1
        del non_stopwords[k]
print("skipped", skipped)
longer_words = [kv for kv in non_stopwords.most_common() if len(kv[0]) >= MIN_CHAR_LEN]

print("num words to be extracted", len(longer_words[:N_WORDS_TO_SAMPLE]))
print("counts for last word", longer_words[N_WORDS_TO_SAMPLE - 1])
print("words:\n", " ".join([l[0] for l in longer_words[:N_WORDS_TO_SAMPLE]]))

new_words_file = (
    f"/home/mark/tinyspeech_harvard/paper_data/new_words/new_words_{LANG_ISOCODE}.txt"
)
with open(new_words_file, "w") as fh:
    fh.write(LANG_ISOCODE + "\n")
    fh.write(",".join([l[0] for l in longer_words[:N_WORDS_TO_SAMPLE]]))
    fh.write("\n")

# %%
# visualize frequencies of top words
fig, ax = plt.subplots()
topn = longer_words[:N_WORDS_TO_SAMPLE]
ax.bar([c[0] for c in topn], [c[1] for c in topn])
ax.set_xticklabels([c[0] for c in topn], rotation=70)
ax.set_ylim([0, 3000])
fig.set_size_inches(40, 10)

# %%
# fetch timings for all words of interest
timings_count = 0
dest_pkl = f"{frequent_words_dir}/all_timings_{timings_count}.pkl"
while os.path.isfile(dest_pkl):
    print("already exists", dest_pkl)
    timings_count += 1
    dest_pkl = f"{frequent_words_dir}/all_timings_{timings_count}.pkl"
words = set([w[0] for w in longer_words[:N_WORDS_TO_SAMPLE]])
tgs = word_extraction.generate_filemap(
    lang_isocode=LANG_ISOCODE, alignment_basedir=alignments
)
print("extracting timings")
timings, notfound = word_extraction.generate_wordtimings(
    words_to_search_for=words,
    mp3_to_textgrid=tgs,
    lang_isocode=LANG_ISOCODE,
    alignment_basedir=alignments,
)
print("errors", len(notfound))
print("saving timings to", dest_pkl)
with open(dest_pkl, "wb") as fh:
    pickle.dump(timings, fh)

# %%

# %%
# with open(f"{frequent_words_dir}/all_timings_0.pkl", "rb") as fh:
#    timings = pickle.load(fh)

# %%
# write timings to csvs per word
MAX_NUM_UTTERANCES_TO_SAMPLE = 300
df_dest = pathlib.Path(timings_dir)
for word, times in timings.items():
    df = pd.DataFrame(times, columns=["mp3_filename", "start_time_s", "end_time_s"])
    if df.shape[0] > MAX_NUM_UTTERANCES_TO_SAMPLE:
        print(word, "SUBSAMPLING")
        df = df.sample(n=MAX_NUM_UTTERANCES_TO_SAMPLE, replace=False)
    print(df_dest / (word + ".csv"))
    df.to_csv(df_dest / (word + ".csv"), quoting=csv.QUOTE_MINIMAL, index=False)

# %%

################################################################
# now, run extract_frequent_words.py
# below, select splits between 165 commands and 85 other words
################################################################

# %%
# select commands and other words

embedding_model_dir = f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/"
save_model_dir = f"{embedding_model_dir}/models"
print("creating embedding model dir", embedding_model_dir)
if os.path.isdir(embedding_model_dir):
    raise ValueError("Embedding model already exists!", embedding_model_dir)
os.makedirs(save_model_dir)

# %%
data_dir = Path(clips_dir)
os.chdir(embedding_model_dir)

NUM_COMMANDS = 165

# choose 165 most frequent words
most_frequent = []
words = os.listdir(data_dir)
for w in words:
    utterances = os.listdir(data_dir / w)
    most_frequent.append((w, len(utterances)))

most_frequent.sort(key=lambda word_count: word_count[1], reverse=True)
commands = [word_count[0] for word_count in most_frequent[:NUM_COMMANDS]]

other_words = set(words).difference(set(commands))

print("num commands", len(commands))
print("num other words", len(other_words))

# %%
# with open("commands.txt", 'w') as fh:
#     for w in commands:
#         fh.write(f"{w}\n")
# with open("other_words.txt", 'w') as fh:
#     for w in other_words:
#         fh.write(f"{w}\n")

# %%
with open("commands.txt", "r") as fh:
    commands = fh.read().splitlines()
with open("other_words.txt", "r") as fh:
    other_words = fh.read().splitlines()

# %%
print(len(commands), len(other_words), len(set(commands).intersection(other_words)))

# %%
train_val_test_data = {}

VALIDATION_FRAC = 0.1
TEST_FRAC = 0.1

for c in commands:
    utterances = glob.glob(str(data_dir / c / "*.wav"))
    np.random.shuffle(utterances)

    n_val = int(VALIDATION_FRAC * len(utterances))
    n_test = int(TEST_FRAC * len(utterances))

    val_utterances = utterances[:n_val]
    test_utterances = utterances[n_val : n_val + n_test]
    train_utterances = utterances[n_val + n_test :]

    print(
        "val\t",
        len(val_utterances),
        "\ttest\t",
        len(test_utterances),
        "\ttrain\t",
        len(train_utterances),
    )
    train_val_test_data[c] = dict(
        train=train_utterances, val=val_utterances, test=test_utterances
    )


# %%
# with open("train_val_test_data.pkl", 'wb') as fh:
#     pickle.dump(train_val_test_data, fh)

# %%

with open("train_val_test_data.pkl", "rb") as fh:
    train_val_test_data = pickle.load(fh)

# %%
train_val_test_counts = [
    (w, len(d["train"]), len(d["val"]), len(d["test"]))
    for w, d in train_val_test_data.items()
]
train_val_test_counts = sorted(train_val_test_counts, key=lambda c: c[1], reverse=True)
train_val_test_counts[:3]

# %%
fig, ax = plt.subplots()
btrain = ax.bar(
    [c[0] for c in train_val_test_counts], [c[1] for c in train_val_test_counts]
)
bval = ax.bar(
    [c[0] for c in train_val_test_counts],
    [c[2] for c in train_val_test_counts],
    bottom=[c[1] for c in train_val_test_counts],
)
btest = ax.bar(
    [c[0] for c in train_val_test_counts],
    [c[3] for c in train_val_test_counts],
    bottom=[c[1] + c[2] for c in train_val_test_counts],
)
ax.set_xticklabels([c[0] for c in train_val_test_counts], rotation=70)
plt.legend((btrain[0], bval[0], btest[0]), ("train", "val", "test"))
fig.set_size_inches(40, 20)

# %%
train_files = []
val_files = []
test_files = []
for w, d in train_val_test_data.items():
    train_files.extend(d["train"])
    val_files.extend(d["val"])
    test_files.extend(d["test"])
np.random.shuffle(train_files)

# %%
# for fname, data in zip(["train_files.txt", "val_files.txt", "test_files.txt"], [train_files, val_files, test_files]):
#    print(fname)
#    with open(fname, 'w') as fh:
#        for utterance_path in data:
#            fh.write(f"{utterance_path}\n")

# %%
with open("train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open("val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()
with open("test_files.txt", "r") as fh:
    test_files = fh.read().splitlines()
print(len(train_files), train_files[0])
print(len(val_files), val_files[0])
print(len(test_files), test_files[0])
assert (
    set(train_files).intersection(set(val_files)) == set()
), "error: overlap between train and val data"
assert (
    set(train_files).intersection(set(test_files)) == set()
), "error: overlap between train and test data"

# %%
# copy background data

# %%

##############################
# Extract a single word      #
##############################
target_word = "stop"
LANG_ISOCODE = "en"
tgs = word_extraction.generate_filemap(
    lang_isocode=LANG_ISOCODE, alignment_basedir=alignments
)
timings, notfound = word_extraction.generate_wordtimings(
    words_to_search_for={target_word},
    mp3_to_textgrid=tgs,
    lang_isocode=LANG_ISOCODE,
    alignment_basedir=alignments,
)
# %%
target_word = "stop"
print("num occurences", len(timings[target_word]))
# %%
timings_dir = "/home/mark/tinyspeech_harvard/tmp/en_stop/"
assert os.path.isdir(timings_dir) and os.listdir(timings_dir) == [], "make empty dir"
df_dest = pathlib.Path(timings_dir)
for word, times in timings.items():
    df = pd.DataFrame(times, columns=["mp3_filename", "start_time_s", "end_time_s"])
    # if df.shape[0] > MAX_NUM_UTTERANCES_TO_SAMPLE:
    #     print(word, "SUBSAMPLING")
    #     df = df.sample(n=MAX_NUM_UTTERANCES_TO_SAMPLE, replace=False)
    print(df_dest / (word + ".csv"))
    df.to_csv(df_dest / (word + ".csv"), quoting=csv.QUOTE_MINIMAL, index=False)
# %%
t = 0
# d = Path("/home/mark/tinyspeech_harvard/frequent_words/en/clips/")
d = Path("/home/mark/tinyspeech_harvard/speech_commands/")
for w in ["stop", "left", "right", "up", "down"]:
    c = len(os.listdir(d / w))
    t += c
    print(w, c)
print(t)
# %%

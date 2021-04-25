# %%
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

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1.6)

# %%

NUM_WAVS = 2200

per_lang = {}

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words/")
for lang in os.listdir(frequent_words):
    per_lang[lang] = []
    clips = frequent_words / lang / "clips"
    words = os.listdir(clips)
    raise ValueError(
        "do we need to make words a set? what caused the bug with duplicate words in commands??"
    )
    for word in words:
        wavs = glob.glob(str(clips / word / "*.wav"))
        if len(wavs) > NUM_WAVS:
            per_lang[lang].append((word, clips / word))


# %%
print("num commands", sum([len(v) for v in per_lang.values()]))
print(per_lang["en"][0])
plt.bar(per_lang.keys(), [len(v) for v in per_lang.values()])
print([(kv[0], len(kv[1])) for kv in per_lang.items()])

# %%
# raise ValueError("DUPLICATE WORD ISSUE")
for lang, words in per_lang.items():
    print(lang, len(words))
    commands = [w[0] for w in words]
    commands_set = set(commands)
    if not len(commands) == len(commands_set):
        print("error", lang, len(commands))
print("--------------")
dup_word = "entre"
commands = []
for lang, commands_worddirs in per_lang.items():
    print(lang)
    for (command, worddir) in commands_worddirs:
        if command == dup_word:
            print("---------------------*****", lang, dup_word)
        if command in commands:
            print("adding dup", lang, command)
        commands.append(command)
print("total", len(commands), len(set(commands)))
# %%
print("num commands: list", sum([len(v) for v in per_lang.values()]))
print("num commands: set", len())

# %%

data_dest = Path("/home/mark/tinyspeech_harvard/multilang_embedding/")
save_model_dir = data_dest / "models"
for d in [data_dest, save_model_dir]:
    if not os.path.isdir(d):
        raise ValueError("missing dir", d)
assert not os.path.isfile(data_dest / "per_lang.pkl"), "file exists"
with open(data_dest / "per_lang.pkl", "wb") as fh:
    pickle.dump(per_lang, fh)
os.chdir(data_dest)

# %%
commands = []
for lang, commands_worddirs in per_lang.items():
    for (command, worddir) in commands_worddirs:
        commands.append(command)

print("num commands", len(commands))
with open("commands.txt", "w") as fh:
    for w in commands:
        fh.write(f"{w}\n")
# %%
train_val_data = {}

VALIDATION_FRAC = 0.1

for lang, commands_worddirs in per_lang.items():
    for (command, worddir) in commands_worddirs:
        all_wavs = glob.glob(str(worddir / "*.wav"))
        utterances = np.random.choice(all_wavs, size=NUM_WAVS, replace=False)

        n_val = int(VALIDATION_FRAC * len(utterances))

        val_utterances = utterances[:n_val]
        train_utterances = utterances[n_val:]

        print("val\t", len(val_utterances), "\ttrain\t", len(train_utterances))
        train_val_data[command] = dict(train=train_utterances, val=val_utterances)

# %%
# inspect
c, d = list(train_val_data.items())[0]
print(c)
print(d["train"][0])

# %%
if os.path.isfile("train_val_data.pkl"):
    raise ValueError("data exists")
with open("train_val_data.pkl", "wb") as fh:
    pickle.dump(train_val_data, fh)

# %%

with open("train_val_test_data.pkl", "rb") as fh:
    train_val_test_data = pickle.load(fh)

# %%
train_val_counts = [
    (w, len(d["train"]), len(d["val"])) for w, d in train_val_data.items()
]
train_val_counts = sorted(train_val_counts, key=lambda c: c[1], reverse=True)
train_val_counts[:3]

# %%
fig, ax = plt.subplots()
btrain = ax.bar([c[0] for c in train_val_counts], [c[1] for c in train_val_counts])
bval = ax.bar(
    [c[0] for c in train_val_counts],
    [c[2] for c in train_val_counts],
    bottom=[c[1] for c in train_val_counts],
)
ax.set_xticklabels([c[0] for c in train_val_counts], rotation=70)
plt.legend((btrain[0], bval[0]), ("train", "val"))
fig.set_size_inches(40, 20)

# %%
train_files = []
val_files = []
for w, d in train_val_data.items():
    train_files.extend(d["train"])
    val_files.extend(d["val"])
np.random.shuffle(train_files)

# %%

for fname, data in zip(["train_files.txt", "val_files.txt"], [train_files, val_files]):
    print(fname)
    if os.path.isfile(fname):
        raise ValueError("exists", fname)
    with open(fname, "w") as fh:
        for utterance_path in data:
            fh.write(f"{utterance_path}\n")
# %%

data_dest = Path("/home/mark/tinyspeech_harvard/multilang_embedding/")

with open(data_dest / "train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open(data_dest / "val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()
with open(data_dest / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()
print(len(train_files), train_files[0])
print(len(val_files), val_files[0])
assert (
    set(train_files).intersection(set(val_files)) == set()
), "error: overlap between train and val data"
print(len(commands))

# %%
# train / val dataset sizes on disk
t_sz_bytes = 0
for f in train_files:
    t_sz_bytes += Path(f).stat().st_size
v_sz_bytes = 0
for f in val_files:
    v_sz_bytes += Path(f).stat().st_size
print("train gb", t_sz_bytes / 1024 ** 3, "val gb", v_sz_bytes / 1024 ** 3)

# %%
# create dataset using hyperion with silence padding + context padding
def find_and_combine(training_sample):
    result = []

    fw = Path(
        "/media/mark/hyperion/mercury_backup_april_13_2021/tinyspeech_harvard/frequent_words"
    )
    fwwc = Path(
        "/media/mark/hyperion/multilingual_embedding_data_w_context/frequent_words_w_context"
    )

    t = Path(training_sample)
    lang_isocode = t.parts[5]
    word = t.parts[7]
    wav = t.parts[8]

    sw = fw / lang_isocode / "clips" / word / wav
    swwc = fwwc / lang_isocode / "clips" / word / wav

    if os.path.isfile(sw):
        result.append(str(sw))
    if os.path.isfile(swwc):
        result.append(str(swwc))

    return result


train_w_context_nested = map(find_and_combine, train_files)
val_w_context_nested = map(find_and_combine, val_files)

train_w_context = [f for l in train_w_context_nested for f in l]
val_w_context = [f for l in val_w_context_nested for f in l]

# %%
print(len(train_w_context))
print(train_w_context[0])
# %%
np.random.shuffle(train_w_context)

# %%
dest_dir = Path("/media/mark/hyperion/multilingual_embedding_data_w_context")
# with open(dest_dir / "train_files.txt", "w") as fh:
#     fh.write("\n".join(train_w_context))
# with open(dest_dir / "val_files.txt", "w") as fh:
#     fh.write("\n".join(val_w_context))


# %%

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

import word_extraction

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1.6)

# %%

NUM_WAVS=2200

per_lang = {}

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words/")
for lang in os.listdir(frequent_words):
    per_lang[lang] = []
    clips = frequent_words / lang / "clips"
    words = os.listdir(clips)
    for word in words:
        wavs = glob.glob(str(clips / word / "*.wav"))
        if len(wavs) > NUM_WAVS:
            per_lang[lang].append((word, clips/word))


# %%
print("num commands", sum([len(v) for v in per_lang.values()]))
print(per_lang['en'][0])
plt.bar(per_lang.keys(), [len(v) for v in per_lang.values()])

# %%

data_dest = Path("/home/mark/tinyspeech_harvard/multilang_embedding/")
save_model_dir= data_dest / "models"
for d in [data_dest, save_model_dir]:
    if not os.path.isdir(d):
        raise ValueError("missing dir", d)
with open(data_dest / "per_lang.pkl", 'wb') as fh:
    pickle.dump(per_lang, fh)
os.chdir(data_dest)

# %%
commands = []
for lang,commands_worddirs in per_lang.items():
    for (command, worddir) in commands_worddirs:
        commands.append(command)

print("num commands", len(commands))
with open("commands.txt", 'w') as fh:
    for w in commands:
        fh.write(f"{w}\n")
# %%
train_val_data = {}

VALIDATION_FRAC = 0.1

for lang,commands_worddirs in per_lang.items():
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
c,d = list(train_val_data.items())[0]
print(c)
print(d["train"][0])

# %%
if os.path.isfile("train_val_data.pkl"): 
    raise ValueError("data exists")
with open("train_val_data.pkl", 'wb') as fh:
    pickle.dump(train_val_data, fh)

# %%

with open("train_val_test_data.pkl", 'rb') as fh:
    train_val_test_data = pickle.load(fh)

# %%
train_val_counts = [(w, len(d["train"]), len(d["val"])) for w,d in train_val_data.items()]
train_val_counts = sorted(train_val_counts, key=lambda c: c[1], reverse=True)
train_val_counts[:3]

# %%
fig,ax = plt.subplots()
btrain = ax.bar([c[0] for c in train_val_counts], [c[1] for c in train_val_counts])
bval   = ax.bar([c[0] for c in train_val_counts], [c[2] for c in train_val_counts], bottom=[c[1] for c in train_val_counts])
ax.set_xticklabels([c[0] for c in train_val_counts], rotation=70);
plt.legend((btrain[0], bval[0]), ('train', 'val'))
fig.set_size_inches(40,20)

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
   with open(fname, 'w') as fh:
       for utterance_path in data:
           fh.write(f"{utterance_path}\n")
# %%

with open("train_files.txt", 'r') as fh:
    train_files = fh.read().splitlines()
with open("val_files.txt", 'r') as fh:
    val_files = fh.read().splitlines()
print(len(train_files), train_files[0])
print(len(val_files), val_files[0])
assert set(train_files).intersection(set(val_files)) == set(), "error: overlap between train and val data"
print(len(commands))
# %%

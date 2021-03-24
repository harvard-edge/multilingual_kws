
#%%
import numpy as np
import os
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict, Set
import functools
from collections import Counter
import csv
import pathlib
import textgrid
import sox
import pickle
from scipy.io import wavfile
from pathlib import Path

# %%
preamble = """
#!/usr/bin/env bash
set -e

TF=/home/mark/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/
BKGD=/home/mark/tinyspeech_harvard/speech_commands/_background_noise_
# this script is only for single-target-keyword models

#  super long test w large gaps between words:
#  --word_gap_ms=40000 \
"""

def make_script(target_word, target_lang, dest):
    script=f"""

echo +++++++++++++++++++++++GENERATING {target_word}
DATA=/home/mark/tinyspeech_harvard/frequent_words/{target_lang}/clips/
WORD="{target_word}"
python $TF/generate_streaming_test_wav.py \
  --data_dir=$DATA \
  --wanted_words=$WORD \
  --background_dir=$BKGD \
  --test_duration_seconds=1200 \
  --unknown_percentage=50 \
  --output_audio_file={dest}/streaming_test.wav \
  --output_labels_file={dest}/streaming_labels.txt
"""
    return script
#%%

base_dir = Path("/home/mark/tinyspeech_harvard/streaming_batch_perword")
frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")

if os.path.isfile(base_dir / "gen.sh"):
    raise ValueError("already exists")

#TODO(mmaz): copy in models from paper_data instead of retraining

N_SHOTS = 5
N_VAL = 30
script_commands = []
for ix in range(5):
    langs = os.listdir(frequent_words)
    langs = ["cy", "eu", "cs", "it", "nl", "fr"]
    target_lang = np.random.choice(langs)
    words = os.listdir(frequent_words / target_lang / "clips")
    target_word = np.random.choice(words)
    print("target_word", target_word, target_lang)
    dest = base_dir / target_word
    if os.path.isdir(dest):
        print("EXISTS", dest)
        continue
    os.makedirs(dest)
    os.makedirs(dest / "n_shots")
    os.makedirs(dest / "val")
    utterances = glob.glob(str(frequent_words / target_lang / "clips" / target_word / "*.wav"))
    if len(utterances) < N_SHOTS + N_VAL:
        print("NOT ENOUGH DATA", target_word)
        continue
    selected = np.random.choice(utterances, N_SHOTS + N_VAL, replace=False)
    selected_shots = selected[:N_SHOTS]
    selected_val = selected[N_SHOTS:]
    for u in selected_shots:
        shutil.copy2(u, dest / "n_shots")
    for u in selected_val:
        shutil.copy2(u, dest / "val")
    script_commands.append(make_script(target_word, target_lang, str(dest)))
print("----------------------")
with open(base_dir / "gen.sh", 'w') as fh:
    fh.write(preamble + "\n".join(script_commands))
# %%

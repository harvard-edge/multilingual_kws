
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
import shlex
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

def make_script(idx, total, target_word, target_lang, dest):
    script=f"""

echo +++++++++++++++++++++++GENERATING {target_word} -  {idx} of {total}
DATA=/home/mark/tinyspeech_harvard/frequent_words/{target_lang}/clips/
python $TF/generate_streaming_test_wav.py \
  --data_dir=$DATA \
  --wanted_words={target_word} \
  --background_dir=$BKGD \
  --test_duration_seconds=600 \
  --unknown_percentage=50 \
  --output_audio_file={dest}/streaming_test.wav \
  --output_labels_file={dest}/streaming_labels.txt
date

"""
    return script
#%%

# fmt: off
paper_data = Path("/home/mark/tinyspeech_harvard/paper_data")
# in_embedding_mlc_pkl = paper_data / "multilang_classification_in_embedding_all_lang_targets.pkl"
# with open(in_embedding_mlc_pkl, 'rb') as fh:
#     in_embedding_mlc = pickle.load(fh)
# for target_data in in_embedding_mlc:
#     print(target_data.keys())
#     break
#data_dir = Path("/home/mark/tinyspeech_harvard/frequent_words")
target_data = []
target_word_counts = {}
#multilang_results_dir = paper_data / "multilang_classification"
multilang_results_dir = paper_data / "ooe_multilang_classification"
for multiclass_lang in os.listdir(multilang_results_dir):
    lang_isocode = multiclass_lang.split("_")[-1]
    print("lang_isocode", lang_isocode)
    for result_file in os.listdir(multilang_results_dir / multiclass_lang / "results"):
        target_word = os.path.splitext(result_file.split("_")[-1])[0]
        # find model path for this target
        model_file = None
        for m in os.listdir(multilang_results_dir / multiclass_lang / "models"):
            m_target = m.split("_")[-1]
            if m_target == target_word:
                model_file = multilang_results_dir / multiclass_lang / "models" / m
                break
        if not model_file:
            raise ValueError
        print(lang_isocode, target_word)
        # wav_dir = data_dir / lang_isocode / "clips" / target_word
        # num_wavs = len(glob.glob(str(wav_dir / "*.wav")))
        # target_word_counts[f"{lang_isocode}_{target_word}"] = num_wavs
        d = (lang_isocode, target_word, model_file)
        target_data.append(d)
#print(len(target_word_counts.keys()))
# fmt: on

#%%
#perword_pkl  = paper_data / "data_streaming_perword.pkl"
perword_pkl  = paper_data / "data_ooe_streaming_perword.pkl"
assert not os.path.exists(perword_pkl), "already there"
with open(perword_pkl, 'wb') as fh:
    pickle.dump(target_data, fh)
#%%

# for ix, (lang_isocode, target_word, model_file) in enumerate(target_data):
#     print(ix)
#     print(shlex.quote(target_word))

#%%
# script to generate wavs for existing models
n_words = len(target_data)
#base_dir = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_perword")
base_dir = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_streaming_batch_perword")
#frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")
script_commands = []
for ix, (lang_isocode, target_word, model_file) in enumerate(target_data):
    # where to write the wavs
    dest_dir = base_dir / f"streaming_{lang_isocode}" / f"streaming_{target_word}"
    os.makedirs(dest_dir, exist_ok=True)

    model_dir = dest_dir / "model"
    os.makedirs(model_dir, exist_ok=True)
    # copy model file

    model_name = os.path.split(model_file)[-1]
    shutil.copytree(model_file, model_dir / model_name)

    # need to escape words with apostrophes in them, and their directories too:
    script_commands.append(make_script(ix, n_words, shlex.quote(target_word), lang_isocode, shlex.quote(str(dest_dir))))

#script_genfile = base_dir / "data_ine_streaming_perword.sh"
script_genfile = base_dir / "data_ooe_streaming_perword.sh"
assert not os.path.exists(script_genfile), "already exists"
with open(script_genfile, 'w') as fh:
    fh.write(preamble + "\n".join(script_commands))

#%%

#%%
######################
############
# generate new data
############
######################
"""
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
"""
# %%

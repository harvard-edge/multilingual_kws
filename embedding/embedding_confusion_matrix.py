# %%
import os
from pathlib import Path
import pandas as pd

import glob
from typing import Dict, List
import numpy as np
import logging

import tensorflow as tf
import pickle

import sys

import input_data

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set()

# %%
# embedding_model_dir = f"/home/mark/tinyspeech_harvard/multilang_embedding/"
# save_models_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding/models/")
embedding_model_dir = Path(f"/home/mark/tinyspeech_harvard/multilingual_embedding_wc")
save_models_dir = embedding_model_dir / "models"
os.chdir(embedding_model_dir)

with open("commands.txt", "r") as fh:
    commands = fh.read().splitlines()
with open("train_files.txt", "r") as fh:
    train_files = fh.read().splitlines()
with open("val_files.txt", "r") as fh:
    val_files = fh.read().splitlines()

# %%
print(len(set(commands)), len(commands))
# %%
dup_commands = []
sc = set()
for ix, c in enumerate(commands):
    if c in sc:
        print(ix, c)
        dup_commands.append(c)
    sc.add(c)
print(len(commands) - len(sc))
dup_commands = set(dup_commands)

# %%
for ix,c in enumerate(commands):
    if c == 'will':
        print(ix,c)
# %%
num_dups = 0
list_dups = set()
for f in train_files:
    lang = f[45:47]
    word = f.split("/")[-2]
    if word in dup_commands:
        num_dups += 1
        list_dups.add((lang, word))
print(num_dups)
for d in list_dups:
    print(d)

# %%
dls = set()
for l,w in list_dups:
    if w == "dos":
        dls.add(l)
print(dls)


# %%
langs=set()
for f in  val_files:
    lang = f[45:47]
    langs.add(lang)
print(langs)

# %%
base_model_path = save_models_dir  / "multilingual_context_73_0.8011"
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(base_model_path)
tf.get_logger().setLevel(logging.INFO)

# %%

iso2lang = {"en": "English", "fr": "French", "ca" : "Catalan", "rw" : "Kinyarwanda", "de" : "German", "it" : "Italian", "nl":"Dutch", "fa" : "Persian", "es":"Spanish"}
print(len(iso2lang.keys()))

# %%
for ix,f in enumerate(train_files):
    #lang = f[45:47] # for silence padded
    offset=84
    print(f[offset:offset+2], f)
    break

# %%
iso2count_utts_train = { k : 0 for k in iso2lang.keys()}
iso2count_words_train = { k : set() for k in iso2lang.keys()}
for ix,f in enumerate(train_files):
    offset=84 # for multilingual w context
    lang = f[offset:offset+2]
    word = f.split("/")[-2]
    #print(lang, word)
    iso2count_utts_train[lang] += 1
    iso2count_words_train[lang].add(word)
print(iso2count_utts_train)

iso2count_utts_val = { k : 0 for k in iso2lang.keys()}
iso2count_words_val = { k : set() for k in iso2lang.keys()}
for ix,f in enumerate(val_files):
    offset=84 # for multilingual w context
    lang = f[offset:offset+2]
    word = f.split("/")[-2]
    #print(lang, word)
    iso2count_utts_val[lang] += 1
    iso2count_words_val[lang].add(word)
print(iso2count_utts_val)

#%%



# %%
len(commands)

# %%

# %%
for ix, c in enumerate(commands):
    found = False
    for lang, ws in iso2count_words_train.items():
        if c in ws:
            found = True
            break
    if not found:
        print(c)
print(ix)

# %%
for ix, c in enumerate(commands):
    found = False
    for lang, ws in iso2count_words_train.items():
        if c in ws and found == False:
            found = True
            continue
        if c in ws and found == True:
            print(ix, lang, c)
    if not found:
        print(c)
print(ix)
# %%
# dataframe of counts and valacc
df = pd.DataFrame(columns=["Language", "# words", "# train", "# val", "val acc"])
iso2counts_train_sorted = sorted(list(iso2count_utts_train.items()), key=lambda kv : kv[1], reverse=True)
for ix, (k,v) in enumerate(iso2counts_train_sorted):
    print(k,v)
    df.loc[ix] = [iso2lang[k], len(iso2count_words_val[k]), v, iso2count_utts_val[k], iso2valacc[k] * 100]

total_train = sum([kv[1] for kv in iso2count_utts_train.items()])
total_val = sum([kv[1] for kv in iso2count_utts_val.items()])
total_words = sum(len(kv[1]) for kv in iso2count_words_train.items())
#total_wordsv = sum(len(kv[1]) for kv in iso2count_words_val.items())
total_words = 760
df.loc[ix + 1] = ["Total", "760", total_train, total_val, 80.11]
df

# %%
print(df.to_latex(header=True, index=False, float_format="%.2f", label="tab:embacc",))
# %%
iso2val_files = {k : [] for k in iso2lang.keys()}
for ix,f in enumerate(val_files):
    offset=84 # for multilingual w context
    lang = f[offset:offset+2]
    #word = f.split("/")[-2]
    iso2val_files[lang].append(f)
# %%
# calculate val accuracy per language
iso2valacc = {k : 0 for k in iso2lang.keys()}
model_settings = input_data.standard_microspeech_model_settings(label_count=len(commands) + 1) # add silence
for lang_isocode, vfs in iso2val_files.items():
    val_audio = []
    val_labels = []

    print(len(vfs))
    for ix,f in enumerate(vfs):
        val_audio.append(input_data.file2spec(model_settings, f))

        word = f.split("/")[-2]
        val_labels.append(commands.index(word) + 1) # add silence 
        if ix % 2000 == 0:
            print(ix)
    val_audio = np.array(val_audio)
    val_labels = np.array(val_labels)

    y_pred = np.argmax(model.predict(val_audio), axis=1)
    y_true = val_labels

    val_acc = sum(y_pred == y_true) / len(y_true)
    print(f'{lang_isocode} accuracy: {val_acc:.0%}')
    iso2valacc[lang_isocode] = val_acc

# %%
test_audio = np.array(test_audio)
test_labels = np.array(test_labels)
test_audio = []
test_labels = []

# add silence
model_settings = input_data.standard_microspeech_model_settings(label_count=len(commands) + 1)

#for ix,(audio, label) in enumerate(test_ds):
print(len(val_files))
for ix,f in enumerate(val_files[:40]):
    test_audio.append(input_data.file2spec(model_settings, f))

    word = f.split("/")[-2]
    # add silence 
    test_labels.append(commands.index(word) + 1)
    if ix % 2000 == 0:
        print(ix)
    #if ix > 8000:
    #    break

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

# %%
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

# %%
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
# %%
# #cmds = [c.numpy().decode('utf-8') for c in a.commands]
# cmds = ["_silence_"] + commands
# #plt.figure(figsize=(10, 8))
# # the problem is that this will have 760 rows and 760 columns, too large to visualize
# ax = sns.heatmap(confusion_mtx,
#                   # xticklabels=cmds, yticklabels=cmds, 
#                    annot=False, cbar=False)
# %%

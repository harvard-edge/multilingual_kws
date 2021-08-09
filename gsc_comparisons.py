#%%
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

import hashlib
import re
from tensorflow.python.util import compat

import pydub
import pydub.playback
import pydub.effects

import sys

import input_data
import embedding.transfer_learning as tl

sns.set()
sns.set_style("white")
sns.set_palette("bright")

# %%
def which_set(filename, validation_percentage, testing_percentage):
    """from  https://git.io/JRRKW"""
    base_name = os.path.basename(filename)
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (
        100.0 / MAX_NUM_WAVS_PER_CLASS
    )
    if percentage_hash < validation_percentage:
        result = "validation"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "testing"
    else:
        result = "training"
    return result


# %%
keyword = "right"
VAL_PCT = 10
TEST_PCT = 10


# keywords from GSC
gsc_data = Path.home() / "tinyspeech_harvard/speech_commands"
gsc_kw_data = glob.glob(str(gsc_data / keyword / "*.wav"))
gsc_kw_data.sort()
print("GSC:", len(gsc_kw_data))
gsc_train = [kw for kw in gsc_kw_data if which_set(kw, VAL_PCT, TEST_PCT) == "training"]
gsc_val = [kw for kw in gsc_kw_data if which_set(kw, VAL_PCT, TEST_PCT) == "validation"]
gsc_test = [kw for kw in gsc_kw_data if which_set(kw, VAL_PCT, TEST_PCT) == "testing"]
print("GSC splits", len(gsc_train), len(gsc_val), len(gsc_test))

# unknown words from GSC
gsc_other = [
    d
    for d in os.listdir(gsc_data)
    if os.path.isdir(gsc_data / d) and d != keyword and d != "_background_noise_"
]
print(gsc_other)
gsc_unknown = [
    sample for kw in gsc_other for sample in glob.glob(str(gsc_data / kw / "*.wav"))
]
gsc_unknown.sort()
print("GSC unknown", len(gsc_unknown))

# keywords from Common Voice extractions [interspeech21 paper]
cv_data = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
cv_keyword_data = glob.glob(str(cv_data / keyword / "*.wav"))
cv_keyword_data.sort()
print("CV:", len(cv_keyword_data))

#%%
def train(train_files, val_files, model_settings):
    embedding_model_dir = Path.home() / "tinyspeech_harvard/multilingual_embedding_wc"
    unknown_files = []
    unknown_files_dir = Path.home() / "tinyspeech_harvard/unknown_files"
    with open(unknown_files_dir / "unknown_files.txt", "r") as fh:
        for w in fh.read().splitlines():
            unknown_files.append(str(unknown_files_dir / w))
    base_model_path = embedding_model_dir / "models" / "multilingual_context_73_0.8011"

    name, model, details = tl.transfer_learn(
        target=keyword,
        train_files=train_files,
        val_files=val_files,
        unknown_files=unknown_files,
        num_epochs=4,
        num_batches=2,
        batch_size=32,
        primary_lr=0.001,
        backprop_into_embedding=False,
        embedding_lr=0,
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
        csvlog_dest=None,
    )
    # modelpath = model_dest_dir / name
    # model.save(modelpath)
    return model


# %%
def cross_compare(
    train_dataset,
    test_dataset=None,
    cross_testset=None,
    N_TRAIN=5,
    N_VAL=20,
    unknown_test=None,
):
    rng = np.random.RandomState(123)
    dataset = rng.permutation(train_dataset)
    train_files = dataset[:N_TRAIN]
    # only for visualization of training performance
    val_files = dataset[N_TRAIN : N_TRAIN + N_VAL]
    model_settings = input_data.standard_microspeech_model_settings(3)
    model = train(train_files, val_files, model_settings)

    if test_dataset is None:
        test_files = dataset[N_TRAIN+N_VAL:]
    else:
        test_files = test_dataset
    specs = [input_data.file2spec(model_settings, f) for f in test_files]
    specs = np.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    print(f"Test accuracy on training dataset: {accuracy:0.2f}")

    if cross_testset is None:
        return
    specs = [input_data.file2spec(model_settings, f) for f in cross_testset]
    specs = np.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    print(f"Test accuracy on cross-testset: {accuracy:0.2f}")

    # TODO(mmaz) https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    # or F1
    if unknown_test is None:
        return
    specs = [input_data.file2spec(model_settings, f) for f in unknown_test]
    specs = np.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    accuracy = amx[amx == 1].shape[0] / preds.shape[0]
    print(f"Test accuracy on unknown: {accuracy:0.2f}")


# %%
# filter CV extractions by embedding distance clusters

# %%
# estimate unknown accuracy
unknown_files = np.random.RandomState(123).choice(gsc_unknown, 1000, replace=False)

print("Train on CV, cross-compare on GSC:")
cross_compare(
    cv_keyword_data,
    test_dataset=None,
    cross_testset=gsc_test,
    unknown_test=unknown_files,
)
print("Train on GSC, cross-compare on CV:")
cross_compare(
    gsc_train, test_dataset=gsc_test, cross_testset=cv_keyword_data, unknown_test=unknown_files
)


# %%

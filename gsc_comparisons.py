#%%
from dataclasses import dataclass
from typing import List
import multiprocessing
import os
import json
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
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

import hashlib
import re
from tensorflow.python.util import compat

import pydub
import pydub.playback
import pydub.effects

import embedding.input_data as input_data
import embedding.transfer_learning as tl
import embedding.distance_filtering as ef

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


# keywords from Common Voice extractions [interspeech21 paper]
# cv_data = Path.home() / "tinyspeech_harvard/frequent_words/silence_padded/en/clips"
# cv_keyword_data = glob.glob(str(cv_data / keyword / "*.wav"))
# cv_keyword_data.sort()
# print("CV:", len(cv_keyword_data))

#%%
def train(train_files, val_files, model_settings, verbose=0):
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
        verbose=verbose,
    )
    return model


# %%
def cross_compare_piecewise(
    keyword,
    train_files,
    val_files,
    test_files,
    cross_testset=None,
    unknown_test=None,
    verbose=0,
):
    """Note: does not evaluate silence"""
    model_settings = input_data.standard_microspeech_model_settings(3)
    model = train(train_files, val_files, model_settings, verbose)

    specs = [input_data.file2spec(model_settings, f) for f in test_files]
    specs = tf.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    print(f"Test accuracy on training dataset: {accuracy:0.2f}")

    if cross_testset is None:
        return
    specs = [input_data.file2spec(model_settings, f) for f in cross_testset]
    specs = tf.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    print(f"Test accuracy on cross-testset: {accuracy:0.2f}")

    # TODO(mmaz) https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    # or F1
    if unknown_test is None:
        return
    specs = [input_data.file2spec(model_settings, f) for f in unknown_test]
    specs = tf.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    accuracy = amx[amx == 1].shape[0] / preds.shape[0]
    print(f"Test accuracy on unknown: {accuracy:0.2f}")

    # testing evaluate mode
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    bg_datadir = Path.home() / "tinyspeech_harvard/speech_commands/_background_noise_"
    audio_dataset = input_data.AudioDataset(
        model_settings=model_settings,
        commands=[keyword],
        background_data_dir=bg_datadir,
        unknown_files=unknown_test,
        unknown_percentage=50,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
    )
    test_ds = audio_dataset.eval_with_silence_unknown(
        AUTOTUNE, test_files, label_from_parent_dir=False
    ).batch(32)
    results = model.evaluate(test_ds)
    print("evaluate results", results)


@dataclass(frozen=True)
class CrossCompare:
    keyword: str
    train_files: List[str]
    val_files: List[str]
    test_files: List[str]
    cross_testset: List[str]
    unknown_test: List[str]
    unknown_cross: List[str]
    verbose: int = 0


def cross_compare(c: CrossCompare, q):
    model_settings = input_data.standard_microspeech_model_settings(3)
    model = train(c.train_files, c.val_files, model_settings, c.verbose)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    bg_datadir = Path.home() / "tinyspeech_harvard/speech_commands/_background_noise_"
    audio_dataset_test = input_data.AudioDataset(
        model_settings=model_settings,
        commands=[keyword],
        background_data_dir=bg_datadir,
        unknown_files=c.unknown_test,
        unknown_percentage=50,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
        seed=0,
    )
    test_ds = audio_dataset_test.eval_with_silence_unknown(
        AUTOTUNE, c.test_files, label_from_parent_dir=False
    ).batch(32)
    test_results = model.evaluate(test_ds)
    print(test_results)

    audio_dataset_cross = input_data.AudioDataset(
        model_settings=model_settings,
        commands=[keyword],
        background_data_dir=bg_datadir,
        unknown_files=c.unknown_cross,
        unknown_percentage=50,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
        seed=0,
    )
    cross_ds = audio_dataset_cross.eval_with_silence_unknown(
        AUTOTUNE, c.cross_testset, label_from_parent_dir=False
    ).batch(32)
    cross_results = model.evaluate(cross_ds)
    print(cross_results)

    testacc_crossacc = (test_results[1], cross_results[1])
    q.put(testacc_crossacc)


# %%
def load_gsc_data(keyword, VAL_PCT=10, TEST_PCT=10):
    # keywords from GSC
    gsc_data = Path.home() / "tinyspeech_harvard/speech_commands"
    gsc_kw_data = glob.glob(str(gsc_data / keyword / "*.wav"))
    gsc_kw_data.sort()
    # print("GSC:", len(gsc_kw_data))
    gsc_train = [
        kw for kw in gsc_kw_data if which_set(kw, VAL_PCT, TEST_PCT) == "training"
    ]
    gsc_val = [
        kw for kw in gsc_kw_data if which_set(kw, VAL_PCT, TEST_PCT) == "validation"
    ]
    gsc_test = [
        kw for kw in gsc_kw_data if which_set(kw, VAL_PCT, TEST_PCT) == "testing"
    ]
    gsc_train.sort()  # ensure random sampling below is stable
    # print("GSC splits", len(gsc_train), len(gsc_val), len(gsc_test))

    # unknown words from GSC
    gsc_other = [
        d
        for d in os.listdir(gsc_data)
        if os.path.isdir(gsc_data / d) and d != keyword and d != "_background_noise_"
    ]
    # print(gsc_other)
    # str not posixpath for tf:
    gsc_unknown = [
        str(sample)
        for kw in gsc_other
        for sample in glob.glob(str(gsc_data / kw / "*.wav"))
    ]
    gsc_unknown.sort()
    # print("GSC unknown", len(gsc_unknown))
    return gsc_train, gsc_val, gsc_test, gsc_unknown


# %%
paper_data = {}
embedding_results = (
    Path.home() / "tinyspeech_harvard/distance_sorting/embedding_results.json"
)
assert not os.path.exists(embedding_results)
q = multiprocessing.Queue()

for keyword in ["left", "right", "off", "down", "yes"]:
    print("::::::::::::::::::", keyword, "::::::::::::::::")
    # load sorted distances for MSWC target keywords
    TOP_PCT = 0.8
    gsc_msc_dir = Path.home() / "tinyspeech_harvard/distance_sorting/gsc_msc/"
    distances_dir = gsc_msc_dir / "distances_k_3"
    sorted_clips = []
    with open(distances_dir / f"{keyword}_distances.csv", "r") as fh:
        reader = csv.reader(fh)
        for row in reader:
            p = gsc_msc_dir / keyword / row[0]
            sorted_clips.append((p, float(row[1])))
    n_sorted = int(TOP_PCT * len(sorted_clips))
    print("loading sorted", n_sorted, "/", len(sorted_clips))
    cv_good_clips = [str(p) for (p, d) in sorted_clips[:n_sorted]]

    cv_other_dir = (
        Path.home() / "tinyspeech_harvard/distance_sorting/msc_other_files/test/"
    )
    cv_other = sorted(list(cv_other_dir.rglob("*.wav")))
    cv_other = [str(p) for p in cv_other]  # TF wants strings not posixpaths
    # print(len(cv_other))

    N_TRAIN = 5
    N_VAL = 20
    paper_data[keyword] = {}
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.RandomState(seed)

        # train on random splits of the dataset
        cv_dataset = rng.permutation(cv_good_clips)
        cv_train_files = cv_dataset[:N_TRAIN]
        # here, no distinction between val and test, we are not optimizing with val
        # we only keep val small for a fast model.fit
        cv_val_files = cv_dataset[N_TRAIN : N_TRAIN + N_VAL]
        cv_test_files = cv_dataset[N_TRAIN:]

        gsc_train, gsc_val, gsc_test, gsc_unknown = load_gsc_data(keyword)
        gsc_train_files = np.random.RandomState(seed).choice(
            gsc_train, N_TRAIN, replace=False
        )
        gsc_val = gsc_val[:N_VAL]  # dont care, speed up fit

        print("seed:", seed)
        verbose = 0
        print("___Train on CV, cross-compare on GSC___")
        cc_cv2gsc = CrossCompare(
            keyword=keyword,
            train_files=cv_train_files,
            val_files=cv_val_files,
            test_files=cv_test_files,
            cross_testset=gsc_test,
            unknown_test=cv_other,
            unknown_cross=gsc_unknown,
            verbose=verbose,
        )
        p = multiprocessing.Process(target=cross_compare, args=(cc_cv2gsc, q,))
        p.start()
        p.join()
        cv2cv_test_acc, cv2gsc_cross_acc = q.get()

        print("___Train on GSC, cross-compare on CV___")
        cc_gsc2cv = CrossCompare(
            keyword=keyword,
            train_files=gsc_train_files,
            val_files=gsc_val,
            test_files=gsc_test,
            cross_testset=cv_good_clips,
            unknown_test=gsc_unknown,
            unknown_cross=cv_other,
            verbose=verbose,
        )
        p = multiprocessing.Process(target=cross_compare, args=(cc_gsc2cv, q,))
        p.start()
        p.join()
        gsc2gsc_test_acc, gsc2cv_cross_acc = q.get()
        print(f"keyword {keyword}")
        print(
            f"{keyword} / MSWC->MSWC {cv2cv_test_acc:0.2f} GSC->MSWC {gsc2cv_cross_acc:0.2f}"
        )
        print(
            f"{keyword} / MSWC->GSC {cv2gsc_cross_acc:0.2f} GSC->GSC {gsc2gsc_test_acc:0.2f}"
        )
        paper_data[keyword][seed] = dict(
            cv2cv=cv2cv_test_acc,
            gsc2cv=gsc2cv_cross_acc,
            cv2gsc=cv2gsc_cross_acc,
            gsc2gsc=gsc2gsc_test_acc,
        )
print("done")
with open(embedding_results, "w") as fh:
    json.dump(paper_data, fh)


# %%
embedding_results = (
    Path.home() / "tinyspeech_harvard/distance_sorting/embedding_results.json"
)
with open(embedding_results) as fh:
    results = json.load(fh)

tests = []
for kw, seeds in results.items():
    print(kw)
    for seed, data in seeds.items():
        print(seed)
        cv2cv = data["cv2cv"]
        gsc2cv = data["gsc2cv"]
        cv2gsc = data["cv2gsc"]
        gsc2gsc = data["gsc2gsc"]
        test = np.array([[cv2cv, gsc2cv], [cv2gsc, gsc2gsc]])
        tests.append(test)
tests = np.stack(tests)
print(np.mean(tests, axis=0))

# [[0.94753037 0.87601529]
#  [0.85248329 0.89008864]]
print(np.std(tests, axis=0))
# [[0.0392504  0.07447217]
#  [0.07388507 0.03652708]]

# %%
# filter CV extractions by embedding distance clusters

# embedding = ef.embedding_model()
# sorting_results = ef.cluster_and_sort(cv_keyword_data, embedding)
# cv_sorted = sorting_results["sorted_clips"]

# cv_best = cv_sorted[: int(len(cv_sorted) * 0.5)]
# print("best 90% of evaluated clips", len(cv_best), f"from {len(cv_keyword_data)}")

# # estimate unknown accuracy (w fixed seed for now)
# unknown_files = np.random.RandomState(123).choice(gsc_unknown, 1000, replace=False)


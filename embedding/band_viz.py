# %%
import os
from os.path import split
from pathlib import Path

import glob
from typing import Dict, List
import numpy as np

# import tensorflow as tf
import pickle


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# verified np.isclose(rtol=0.01) for average=binary
# from sklearn.metrics import f1_score

sns.set()
sns.set_style("whitegrid")
sns.set_palette("bright")
# sns.set(font_scale=1.6)

from viz_colors import iso2lang, iso2color, iso2line

# %%


# %%
def roc_single_target(target_results, unknown_results, f1_at_threshold=None):
    # _TARGET_ is class 2, _UNKNOWN_ is class 1

    # positive label: target keywords are classified as _TARGET_ if above threshold
    # false negatives -> target kws incorrectly classified as _UNKNOWN_ if below threshold:
    # true negatives -> unknown classified as unknown if below threshold
    # false positives: _UNKNOWN_ keywords incorrectly (falsely) classified as _TARGET_ (positive) if above threshold

    # positives

    total_positives = target_results.shape[0]

    # negatives

    unknown_total = unknown_results.shape[0]

    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)

    # F1  = TP / (TP + 1/2 (FP + FN))
    # FNR = FN / P == FN / (TP + FN)

    tprs, fprs = [], []
    error_rate_to_f1_scores = []

    # threshs = np.arange(0, 1.01, 0.01)
    threshs = np.arange(0.01, 0.99, 0.01)
    for threshold in threshs:
        # fmt: off
        false_negatives = target_results[target_results < threshold].shape[0]
        true_positives = target_results[target_results > threshold].shape[0]
        false_positives = unknown_results[unknown_results > threshold].shape[0]

        tpr = true_positives / total_positives
        tprs.append(tpr)
        fpr = false_positives / unknown_total
        fprs.append(fpr)

        fnr = false_negatives / total_positives

        f1_score = true_positives / (true_positives + 0.5 *( false_positives + false_negatives ))
        error_rate = np.abs(fnr - fpr)
        if f1_at_threshold is None:
            error_rate_to_f1_scores.append([error_rate, threshold, f1_score, fpr, tpr])
        else:
            if np.isclose(threshold, f1_at_threshold):
                error_rate_to_f1_scores.append([error_rate, threshold, f1_score, fpr, tpr])

        # fmt: on

    error_rate_to_f1_scores = np.array(error_rate_to_f1_scores)
    if f1_at_threshold is None:
        # find EER https://stackoverflow.com/a/46026962
        equal_error_rate = np.nanargmin(error_rate_to_f1_scores[:, 0])
        print(error_rate_to_f1_scores[equal_error_rate])
        error_rate_info = error_rate_to_f1_scores[equal_error_rate]
    else:
        assert error_rate_to_f1_scores.shape[0] == 1
        error_rate_info = error_rate_to_f1_scores[0]
    return tprs, fprs, threshs, error_rate_info


def roc_curve_multiclass(target_resuts, unknown_results):
    # _TARGET_ is class 2, _UNKNOWN_ is class 0

    # positive label: target keywords classified as _TARGET_
    # true positives
    target_correct = np.array(target_resuts["correct"])
    # false negatives -> target kws incorrectly classified as _UNKNOWN_:
    target_incorrect = np.array(target_resuts["incorrect"])
    total_positives = target_correct.shape[0] + target_incorrect.shape[0]

    # negative labels

    # true negatives -> unknown classified as unknown
    unknown_correct = np.array(unknown_results["correct"])
    # false positives: _UNKNOWN_ keywords incorrectly (falsely) classified as _TARGET_ (positive)
    unknown_incorrect = np.array(unknown_results["incorrect"])
    unknown_total = unknown_correct.shape[0] + unknown_incorrect.shape[0]

    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)

    tprs, fprs = [], []

    threshs = np.arange(0, 1.01, 0.01)
    for threshold in threshs:
        tpr = target_correct[target_correct > threshold].shape[0] / total_positives
        tprs.append(tpr)
        fpr = unknown_incorrect[unknown_incorrect > threshold].shape[0] / unknown_total
        fprs.append(fpr)
    return tprs, fprs, threshs


# %%

# one language roc curves without bands
"""
LANG_ISOCODE = "it"

data_dir = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips/")
traindir = Path(f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/")
model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/sweep_{LANG_ISOCODE}")
results_dir = model_dest_dir / "results"
results = []

for pkl_file in os.listdir(model_dest_dir / "results"):
    filename = model_dest_dir / "results" / pkl_file
    print(filename)
    with open(filename, "rb") as fh:
        result = pickle.load(fh)
        results.append(result)
print("N words", len(results))

def make_roc(results: List[Dict]):
    fig, ax = plt.subplots()
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        ne = res["details"]["num_epochs"]
        nb = res["details"]["num_batches"]
        target = res["target"]
        curve_label = f"{target} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels = roc_curve(target_results, unknown_results)

        ax.plot(fprs, tprs, label=curve_label)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(-0.01, 1)

    # v = res["val_acc"]
    # wl = ", ".join(res["words"]) + f" (val acc {v})"
    # ax.set_title(wl)
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.legend(loc="lower right")
    return fig, ax

fig,ax= make_roc(results)
fig.set_size_inches(20,10)
"""

# %%
# one language ROC curves with bands
"""
LANG_ISOCODE = "it"

data_dir = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips/")
traindir = Path(f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/")
model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/sweep_{LANG_ISOCODE}")
results_dir = model_dest_dir / "results"
results = []

for pkl_file in os.listdir(model_dest_dir / "results"):
    filename = model_dest_dir / "results" / pkl_file
    print(filename)
    with open(filename, "rb") as fh:
        result = pickle.load(fh)
        results.append(result)
print("N words", len(results))

# def make_roc(results: List[Dict]):
fig, ax = plt.subplots()
all_tprs, all_fprs = [], []
for ix, res in enumerate(results):
    target_results = res["target_results"]
    unknown_results = res["unknown_results"]
    ne = res["details"]["num_epochs"]
    nb = res["details"]["num_batches"]
    target = res["target"]
    curve_label = f"{target} (e:{ne},b:{nb})"
    # curve_label=target
    tprs, fprs, thresh_labels = roc_curve(target_results, unknown_results)
    all_tprs.append(tprs)
    all_fprs.append(fprs)
    ax.plot(fprs, tprs, label=curve_label)
all_tprs = np.array(all_tprs)
all_fprs = np.array(all_fprs)

# make sure all tprs and fprs are monotonically increasing
for ix in range(all_fprs.shape[0]):
    # https://stackoverflow.com/a/47004533
    if not np.all(np.diff(np.flip(all_fprs[ix,:])) >=0):
        raise ValueError("fprs not in sorted order")
    if not np.all(np.diff(np.flip(all_tprs[ix,:])) >=0):
        raise ValueError("tprs not in sorted order")

# https://stackoverflow.com/a/43035301
x_all = np.unique(all_fprs.ravel())
y_all = np.empty((x_all.shape[0], all_tprs.shape[0]))
for ix in range(all_fprs.shape[0]):
    y_all[:, ix] = np.interp(x_all, np.flip(all_fprs[ix,:]), np.flip(all_tprs[ix,:]))
ymin = y_all.min(axis=1)
ymax = y_all.max(axis=1)
ax.fill_between(x_all, ymin, ymax, alpha=0.3)
ax.set_xlim(-0.01, 1)
ax.set_ylim(-0.01, 1)

# min_tpr = np.min(all_tprs, axis=0)
# max_tpr = np.max(all_tprs, axis=0)
# min_fpr = np.min(all_fprs, axis=0)
# max_fpr = np.max(all_fprs, axis=0)

# plt.plot(range(x_all.shape[0]), x_all)
# plt.plot(range(y_all[0, :].shape[0]), y_all[0, :])
# plt.plot(range(y_all[1, :].shape[0]), y_all[1, :])
# plt.plot(range(ymin.shape[0]), ymin)
# plt.plot(np.arange(0, min_fpr.shape[0]), min_fpr)
# plt.plot(np.arange(0, min_tpr.shape[0]), min_tpr)
"""

# %%
# %%
## Per-Language Embedding Model

# f1_thresh = None
f1_thresh = 0.8
# unweighted f1 @ f1_thresh = 0.58

fig, ax = plt.subplots()
paper_results = Path("/home/mark/tinyspeech_harvard/paper_data/perlang/")
# for ix, LANG_ISOCODE in enumerate(["de", "rw", "es", "it", "nl"]):
all_f1_scores = []
for i, langdir in enumerate(os.listdir(paper_results)):
    lang_isocode = langdir.split("_")[-1]
    # color = sns.color_palette("bright")[i % len(sns.color_palette("bright"))]

    # data_dir = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips/")
    # traindir = Path(f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/")
    # model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/sweep_{LANG_ISOCODE}")
    results_dir = paper_results / langdir / "results"
    results = []

    for pkl_file in os.listdir(results_dir):
        filename = results_dir / pkl_file
        print(filename)
        with open(filename, "rb") as fh:
            result = pickle.load(fh)
            results.append(result)
    print("N words", len(results))

    # def make_roc(results: List[Dict]):
    all_tprs, all_fprs, lang_f1_scores = [], [], []
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        target = res["target"]
        # ne = res["details"]["num_epochs"]
        # nb = res["details"]["num_batches"]
        # curve_label = f"{target} (e:{ne},b:{nb})"
        curve_label = target
        tprs, fprs, thresh_labels, er_info = roc_single_target(
            target_results, unknown_results, f1_at_threshold=f1_thresh
        )
        all_tprs.append(tprs)
        all_fprs.append(fprs)
        # plot just the line
        ax.plot(fprs, tprs, color=iso2color(lang_isocode), alpha=0.05)
        # add the label:
        # ax.plot(fprs, tprs, label=curve_label)

        # eer / f1
        # ax.plot(er_info[3], er_info[4], marker='o', markersize=2, color='red')
        lang_f1_scores.append(er_info[2])

    all_tprs = np.array(all_tprs)
    all_fprs = np.array(all_fprs)

    lang_f1_scores = np.array(lang_f1_scores)
    all_f1_scores.append(np.mean(lang_f1_scores))

    # make sure all tprs and fprs are monotonically increasing
    for ix in range(all_fprs.shape[0]):
        # https://stackoverflow.com/a/47004533
        if not np.all(np.diff(np.flip(all_fprs[ix, :])) >= 0):
            raise ValueError("fprs not in sorted order")
        if not np.all(np.diff(np.flip(all_tprs[ix, :])) >= 0):
            raise ValueError("tprs not in sorted order")

    # # https://stackoverflow.com/a/43035301
    x_all = np.unique(all_fprs.ravel())
    y_all = np.empty((x_all.shape[0], all_tprs.shape[0]))
    for ix in range(all_fprs.shape[0]):
        y_all[:, ix] = np.interp(
            x_all, np.flip(all_fprs[ix, :]), np.flip(all_tprs[ix, :])
        )

    # draw bands over min and max:
    # ymin = y_all.min(axis=1)
    # ymax = y_all.max(axis=1)
    # ax.fill_between(x_all, ymin, ymax, alpha=0.1, label=f"{iso2lang[lang_isocode]}")

    ymean = y_all.mean(axis=1)
    # draw mean
    ax.plot(x_all, ymean, alpha=0.7, linewidth=6, color=iso2color(lang_isocode), label=f"{iso2lang[lang_isocode]}")
    # draw bands over stdev
    ystdev = y_all.std(axis=1)
    ax.fill_between(x_all, ymean - ystdev, ymean + ystdev, color=iso2color(lang_isocode), alpha=0.1)

AX_LIM = 0.75
ax.set_xlim(0, 1 - AX_LIM)
ax.set_ylim(AX_LIM, 1)
ax.legend(loc="lower right")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_legend().get_texts()
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(40)

fig.set_size_inches(14, 14)
fig.tight_layout()
figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/individual_language_embedding_models.png"
fig.savefig(figdest)
print(figdest)

all_f1_scores = np.array(all_f1_scores)
print("AVG F1", np.mean(all_f1_scores))

# %%

# %%
# multilang embedding model results

# openai viz: https://github.com/openai/baselines/blob/master/docs/viz/viz.ipynb

results = []
# emb_langs = Path("/home/mark/tinyspeech_harvard/multilang_analysis")
# non_emb_langs = Path("/home/mark/tinyspeech_harvard/multilang_analysis_ooe")
# all_langs = [emb_langs, non_emb_langs]
# non_emb_langs = Path("/home/mark/tinyspeech_harvard/multilang_analysis_ooe_v2")
# all_langs = [non_emb_langs]
# for model_dest_dir in all_langs:

# f1_thresh=None # EER
f1_thresh = 0.8

# fmt: off

# avg unweighted f1 @ f1_thresh - 0.75
# base_dir = Path("/home/mark/tinyspeech_harvard/paper_data/multilang_classification/")
# figdest="/home/mark/tinyspeech_harvard/tinyspeech_images/multilang_classification.png"

# avg unweighted f1 @ f1_thresh - 0.65
base_dir = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_multilang_classification/")
figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/ooe_multilang_classification.png"

# fmt: on

for model_dest_dir in os.listdir(base_dir):
    ix = 0
    for pkl_file in os.listdir(base_dir / model_dest_dir / "results"):
        filename = base_dir / model_dest_dir / "results" / pkl_file
        print(filename)
        with open(filename, "rb") as fh:
            result = pickle.load(fh)
            results.append(result)
        ix += 1
    print("------ n results for language:", ix, model_dest_dir)
print("N words", len(results))

lang2results = {}
for ix, res in enumerate(results):
    target_lang = res["target_lang"]
    if not target_lang in lang2results:
        lang2results[target_lang] = []
    lang2results[target_lang].append(res)

all_f1_scores = []
fig, ax = plt.subplots()
for ix, (lang, results) in enumerate(lang2results.items()):
    # color = sns.color_palette("bright")[ix % len(sns.color_palette("bright"))]

    all_tprs, all_fprs, lang_f1_scores = [], [], []
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        # ne = res["details"]["num_epochs"]
        # nb = res["details"]["num_batches"]
        target_word = res["target_word"]
        target_lang = res["target_lang"]
        # curve_label = f"{target} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels, er_info = roc_single_target(
            target_results, unknown_results, f1_at_threshold=f1_thresh
        )
        # print("target results mean", np.mean(target_results))
        # print("unknown results mean", np.mean(unknown_results))
        all_tprs.append(tprs)
        all_fprs.append(fprs)
        ax.plot(fprs, tprs, color=iso2color(lang), alpha=0.05)
        # ax.plot(fprs, tprs, label=curve_label)

        # eer / f1
        # ax.plot(er_info[3], er_info[4], marker='o', markersize=2, color='red')
        lang_f1_scores.append(er_info[2])
    all_tprs = np.array(all_tprs)
    all_fprs = np.array(all_fprs)

    lang_f1_scores = np.array(lang_f1_scores)
    all_f1_scores.append(np.mean(lang_f1_scores))

    # make sure all tprs and fprs are monotonically increasing
    for ix in range(all_fprs.shape[0]):
        # https://stackoverflow.com/a/47004533
        if not np.all(np.diff(np.flip(all_fprs[ix, :])) >= 0):
            raise ValueError("fprs not in sorted order")
        if not np.all(np.diff(np.flip(all_tprs[ix, :])) >= 0):
            raise ValueError("tprs not in sorted order")

    # https://stackoverflow.com/a/43035301
    x_all = np.unique(all_fprs.ravel())
    y_all = np.empty((x_all.shape[0], all_tprs.shape[0]))
    for ix in range(all_fprs.shape[0]):
        y_all[:, ix] = np.interp(
            x_all, np.flip(all_fprs[ix, :]), np.flip(all_tprs[ix, :])
        )

    # draw bands over min and max:
    # ymin = y_all.min(axis=1)
    # ymax = y_all.max(axis=1)
    # ax.fill_between(x_all, ymin, ymax, alpha=0.1, label=f"{iso2lang[lang_isocode]}")

    ymean = y_all.mean(axis=1)
    # draw mean
    ax.plot(x_all, ymean, alpha=0.7, color=iso2color(lang), linewidth=6, label=f"{iso2lang[lang]}")
    # draw bands over stdev
    ystdev = y_all.std(axis=1)
    ax.fill_between(x_all, ymean - ystdev, ymean + ystdev, color=iso2color(lang), alpha=0.1)

AX_LIM = 0.75
ax.set_xlim(0, 1 - AX_LIM)
ax.set_ylim(AX_LIM, 1)
ax.legend(loc="lower right")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_legend().get_texts()
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(40)
fig.set_size_inches(14, 14)
fig.tight_layout()
fig.savefig(figdest)
print(figdest)

all_f1_scores = np.array(all_f1_scores)
print("AVG F1", np.mean(all_f1_scores))
# %%

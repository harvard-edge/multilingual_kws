# %%
import os
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

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1.6)


# %%

def roc_sc(target_resuts, unknown_results):
    # _TARGET_ is class 1, _UNKNOWN_ is class 0

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
        tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)

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

# %%



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
    tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
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

# %%
# %%

## Per-Language Embedding Model 
fig, ax = plt.subplots()
for ix, LANG_ISOCODE in enumerate(["de", "rw", "es", "it", "nl"]):
# for ix, LANG_ISOCODE in enumerate(["nl"]):
    color = sns.color_palette("bright")[ix % len(sns.color_palette("bright"))]

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
    all_tprs, all_fprs = [], []
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        ne = res["details"]["num_epochs"]
        nb = res["details"]["num_batches"]
        target = res["target"]
        curve_label = f"{target} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
        all_tprs.append(tprs)
        all_fprs.append(fprs)
        ax.plot(fprs, tprs, color=color, alpha=0.1)
        # ax.plot(fprs, tprs, label=curve_label)
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
    ax.fill_between(x_all, ymin, ymax, alpha=0.2, label=f"{LANG_ISOCODE}")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.6, 1)
    ax.legend(loc="lower right")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts() +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    
    fig.set_size_inches(14,14)

# %%
# %%



# %%
# def sc_roc_plotly(results: List[Dict]):
#     fig = go.Figure()
#     for ix, res in enumerate(results):
#         target_results = res["target_results"]
#         unknown_results = res["unknown_results"]
#         ne = res["details"]["num_epochs"]
#         nb = res["details"]["num_batches"]
#         target = res["target"]
#         curve_label = f"{target} (e:{ne},b:{nb})"
#         # curve_label=target
#         tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
#         fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=curve_label))

#     fig.update_layout(
#         xaxis_title="FPR",
#         yaxis_title="TPR",
#         title=f"{LANG_ISOCODE} 5-shot classification accuracy",
#     )
#     fig.update_xaxes(range=[0, 1])
#     fig.update_yaxes(range=[0, 1])
#     return fig


# fig = sc_roc_plotly(results)
# dest_plot = str(model_dest_dir / f"5shot_classification_roc_{LANG_ISOCODE}.html")
# print("saving to", dest_plot)
# fig.write_html(dest_plot)
# fig



# def make_roc(results: List[Dict], nrows: int, ncols: int):
#     assert nrows * ncols == len(results), "fewer results than requested plots"
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
#     for ix, (res, ax) in enumerate(zip(results, axes.flatten())):
#         tprs, fprs, threshs = roc_sc(res)

#         ax.plot(fprs, tprs)
#         ax.set_xlim(-0.01, 1)
#         ax.set_ylim(-0.01, 1)

#         v = res["val_acc"]
#         wl = ", ".join(res["words"]) + f" (val acc {v})"
#         ax.set_title(wl)
#         ax.set_xlabel("fpr")
#         ax.set_ylabel("tpr")
#         # ax.legend(loc="lower right")
#     return fig, axes

# %%

# multilang embedding model results
results = []
emb_langs = Path("/home/mark/tinyspeech_harvard/multilang_analysis")
non_emb_langs = Path("/home/mark/tinyspeech_harvard/multilang_analysis_ooe")
for model_dest_dir in [emb_langs, non_emb_langs]:
    for pkl_file in os.listdir(model_dest_dir / "results"):
        filename = model_dest_dir / "results" / pkl_file
        print(filename)
        with open(filename, "rb") as fh:
            result = pickle.load(fh)
            results.append(result)
print("N words", len(results))

lang2results = {}
for ix, res in enumerate(results):
    target_lang = res["target_lang"]
    if not target_lang in lang2results:
        lang2results[target_lang] = []
    lang2results[target_lang].append(res)

fig, ax = plt.subplots()
for ix, (lang, results) in enumerate(lang2results.items()):
    color = sns.color_palette("bright")[ix % len(sns.color_palette("bright"))]

    # def make_roc(results: List[Dict]):
    all_tprs, all_fprs = [], []
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        ne = res["details"]["num_epochs"]
        nb = res["details"]["num_batches"]
        target_word = res["target_word"]
        target_lang = res["target_lang"]
        # curve_label = f"{target} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
        all_tprs.append(tprs)
        all_fprs.append(fprs)
        ax.plot(fprs, tprs, color=color, alpha=0.1)
        # ax.plot(fprs, tprs, label=curve_label)
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
    ax.fill_between(x_all, ymin, ymax, alpha=0.2, label=f"{lang}")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.6, 1)
    ax.legend(loc="lower right")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts() +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.set_size_inches(14,14)
# %%

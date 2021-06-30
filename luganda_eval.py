#%%
import os
import glob
import shutil
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import sox
import pydub
from pydub.playback import play

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
from embedding.tpr_fpr import tpr_fpr
import input_data
import textgrid
from luganda_train import SweepData
from luganda_info import WavTranscript

import seaborn as sns

sns.set()
# sns.set_style("darkgrid")
sns.set_style("whitegrid")
sns.set_palette("bright")

def count_nontarget_words(keyword, groundtruth):
    num_nontarget_words = 0
    for wav in groundtruth["stream_data"]:
        transcript = wav["transcript"]
        num_nontarget_words += sum([w != keyword for w in transcript.split()])
    return num_nontarget_words


# graph hyperparam sweep
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
evaldir = workdir / "cs288_eval"
hpsweep = workdir / "hp_sweep"
#hpsweep = workdir / "export"
# evaldir = workdir / "cs288_test"
# hpsweep = workdir / "tt_sweep"

eval_data = {}
for kw in os.listdir(evaldir):
    with open(evaldir / kw / f"{kw}_groundtruth.pkl", "rb") as fh:
        groundtruth = pickle.load(fh)

    num_nt = count_nontarget_words(keyword=kw, groundtruth=groundtruth)
    print(kw, "num nontarget words", num_nt)
    duration_s = sum([d["duration_s"] for d in groundtruth["stream_data"]])
    print("Duration (m)", duration_s / 60)
    eval_data[kw] = dict(
        times=groundtruth["groundtruth_target_times_ms"],
        num_nt=num_nt,
        duration_s=duration_s,
    )

#%%
eval_data["mask"]["times"]
#%%

lugandamap = {
    "corona": "kolona",
    "okugema": "okugema",
    "covid": "covid",
    "mask": "masiki",
    "akawuka": "akawuka",
}

use_mpl = True

if use_mpl:
    fig, ax = plt.subplots()
else:
    fig = go.Figure()

for exp in os.listdir(hpsweep):
    # if int(exp[4:]) < 11:
    #     continue
    for trial in os.listdir(hpsweep / exp):
        rp = hpsweep / exp / trial / "result.pkl"
        if not os.path.isfile(rp):
            continue  # still calculating
        with open(rp, "rb") as fh:
            result = pickle.load(fh)
        with open(hpsweep / exp / trial / "sweep_data.pkl", "rb") as fh:
            sweep_data = pickle.load(fh)
            sweep_info = sweep_data["sweep_datas"]

        keyword = sweep_info[0].target

        all_tprs = []
        all_fprs = []
        all_threshs = []
        for post_processing_settings, results_per_thresh in result[keyword]:
            for thresh, (found_words, _) in results_per_thresh.items():
                if thresh < 0.3:
                    continue
                #print(thresh)
                analysis = tpr_fpr(
                    keyword,
                    thresh,
                    found_words,
                    eval_data[keyword]["times"],
                    duration_s=eval_data[keyword]["duration_s"],
                    time_tolerance_ms=post_processing_settings.time_tolerance_ms,
                    num_nontarget_words=eval_data[keyword]["num_nt"],
                )
                tpr = analysis["tpr"]
                # fpr = analysis["fpr"]
                fpr = analysis["false_accepts_per_hour"]
                all_tprs.append(tpr)
                all_fprs.append(fpr)
                all_threshs.append(thresh)
                if np.isclose(thresh, 0.90):
                    pprint.pprint(analysis)

            sd = sweep_info[0]
            if sd.backprop_into_embedding:
                lrinfo = f"lr1: {sd.primary_lr} lr2: {sd.embedding_lr}"
            else:
                lrinfo = f"lr1: {sd.primary_lr}"

            if sd.with_context:
                wc = "t"
            else:
                wc = "f"

            num_train = len(sd.train_files)

            # label = f"{exp} s: {num_train:02d} c: {wc} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            # label = f"{keyword} s: {num_train:02d} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            label = f"{lugandamap[keyword]} ({num_train}-shot)"
            if use_mpl:
                ax.plot(all_fprs, all_tprs, label=label, linewidth=3)
            else:
                fig.add_trace(
                    go.Scatter(x=all_fprs, y=all_tprs, text=all_threshs, name=label)
                )

AX_LIM = 0.0
if not use_mpl:
    SIZE = 700
    fig.update_layout(
        #xaxis_title="FPR",
        xaxis_title="False accepts/hour",
        yaxis_title="TPR",
        title=f"streaming accuracy",
        width=SIZE,
        height=SIZE,
    )
    #fig.update_xaxes(range=[0, 1 - AX_LIM])
    fig.update_xaxes(range=[0, 400])
    fig.update_yaxes(range=[AX_LIM, 1])
    fig.show()
    #fig.write_html("/home/mark/tinyspeech_harvard/tinyspeech_images/mask_search.html")
else:
    ax.axvline(
        x=50, label=f"nominal cutoff for false accepts", linestyle="--", color="black",
    )

    # ax.set_xlim(0, 1 - AX_LIM)
    ax.set_xlim(0, 200)
    # ax.set_ylim(AX_LIM, 1.01)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    # ax.set_xlabel("False Positive Rate")
    ax.set_xlabel("False Accepts per Hour")
    ax.set_ylabel("True Positive Rate")
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_legend().get_texts()
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(25)
    fig.set_size_inches(14, 14)
    fig.tight_layout()
    figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/luganda_5_keywords.png"
    fig.savefig(figdest)
    print(figdest)

# %%
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
with open(workdir / "stream_info.pkl", "rb") as fh:
    stream_info = pickle.load(fh)
timing_csv = workdir / "covid_19_timing.csv"
covid_timings = {}
keyword = "covid"
with open(timing_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header
    for ix, row in enumerate(reader):
        wav = row[0]
        transcript = row[1]
        start_time_s = row[3]
        contains_keyword = any([w == keyword for w in transcript.split()])
        if contains_keyword:
            covid_timings[wav] = (start_time_s, transcript)
print(covid_timings)

# %%
# generate groundtruth timings
gt_target_times_ms = []
keyword = "covid"
cur_time_s = 0.0
for segment in stream_info:
    transcript = segment["transcript"]
    contains_keyword = any([w == keyword for w in transcript.split()])
    if contains_keyword:
        wav = segment["wav"]
        offset_s = float(covid_timings[wav][0])
        keyword_start_s = cur_time_s + offset_s
        gt_target_times_ms.append(keyword_start_s * 1000)
    cur_time_s += segment["duration_s"]
num_nontarget_words = 0
for segment in stream_info:
    non_target_words = [w != keyword for w in transcript.split()]
    num_nontarget_words += len(non_target_words)
print("nontarget words", num_nontarget_words)
# %%

fig, ax = plt.subplots()
for graphing_context in [True, False]:
    hpsweep = workdir / "sweep_silence_v_context"
    all_tprs = []
    all_fprs = []
    for exp in os.listdir(hpsweep):
        for trial in os.listdir(hpsweep / exp):
            rp = hpsweep / exp / trial / "result.pkl"
            with open(rp, "rb") as fh:
                result = pickle.load(fh)
            with open(hpsweep / exp / trial / "sweep_data.pkl", "rb") as fh:
                sweep_data = pickle.load(fh)
                sweep_info = sweep_data["sweep_datas"]

            keyword = sweep_info[0].target

            sd = sweep_info[0]
            try:
                if sd.with_context:
                    wc = "t"
                else:
                    wc = "f"
            except AttributeError:
                wc = "f"

            if graphing_context and wc == "f":
                # print("SKIPPING")
                continue
            elif not graphing_context and wc == "t":
                # print("SKIPPING")
                continue

            tprs = []
            fprs = []
            threshs = []
            for thresh, (found_words, _) in result[keyword].items():
                if thresh < 0.3:
                    continue
                # print(thresh)

                analysis = tpr_fpr(
                    keyword,
                    thresh,
                    found_words,
                    gt_target_times_ms,
                    time_tolerance_ms=post_processing_settings.time_tolerance_ms,
                    num_nontarget_words=num_nontarget_words,
                )
                tpr = analysis["tpr"]
                fpr = analysis["fpr"]
                tprs.append(tpr)
                fprs.append(fpr)
                threshs.append(thresh)
                # pprint.pprint(analysis)

            if sd.backprop_into_embedding:
                lrinfo = f"lr1: {sd.primary_lr} lr2: {sd.embedding_lr}"
            else:
                lrinfo = f"lr1: {sd.primary_lr}"

            num_train = len(sd.train_files)

            # label = f"{exp} t: {num_train:02d} c: {wc} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            label = f"{keyword} t: {num_train:02d} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            # ax.plot(fprs,tprs, label=label, linewidth=1)
            # print(len(tprs))
            all_tprs.append(np.array(tprs))
            all_fprs.append(np.array(fprs))

    all_tprs = np.array(all_tprs)
    all_fprs = np.array(all_fprs)
    print(all_tprs.shape, graphing_context)

    # lang_f1_scores = np.array(lang_f1_scores)
    # all_f1_scores.append(np.mean(lang_f1_scores))

    # make sure all tprs and fprs are monotonically increasing
    # for ix in range(all_fprs.shape[0]):
    #     # https://stackoverflow.com/a/47004533
    #     if not np.all(np.diff(np.flip(all_fprs[ix, :])) >= 0):
    #         raise ValueError("fprs not in sorted order")
    #     if not np.all(np.diff(np.flip(all_tprs[ix, :])) >= 0):
    #         raise ValueError("tprs not in sorted order")

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
    # ax.plot(x_all, ymean, alpha=0.7, linewidth=6, color=iso2color[lang_isocode], label=f"{iso2lang[lang_isocode]}")
    if graphing_context:
        gl = "context-padded and\nsilence-padded"
    else:
        gl = "silence-padded only"
    ax.plot(x_all, ymean, alpha=0.7, linewidth=6, label=gl)
    # draw bands over stdev
    ystdev = y_all.std(axis=1)
    # ax.fill_between(x_all, ymean - ystdev, ymean + ystdev, color=iso2color[lang_isocode], alpha=0.1)
    ax.fill_between(x_all, ymean - ystdev, ymean + ystdev, alpha=0.4)

AX_LIM = 0.6
ax.set_xlim(0, 1 - AX_LIM)
ax.set_ylim(AX_LIM, 1.01)
ax.legend(loc="lower right")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_legend().get_texts()
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(35)
fig.set_size_inches(14, 14)
fig.tight_layout()
figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/context_v_silence.png"
fig.savefig(figdest)
print(figdest)
# %%

# %%

fig, ax = plt.subplots()

kws = [
    "0 targets [mask]",
    "91 targets [mask]",
    "0 targets [corona]",
    "83 targets [corona]",
]
ax.bar(
    kws,
    [0, 16 + 59, 0, 30 + 48],
    label="true positives",
    color=sns.color_palette("bright")[2],
)
ax.bar(
    kws, [11, 16, 33, 30], label="false positives", color=sns.color_palette("bright")[3]
)
ax.set_xticklabels(kws, rotation=50)
ax.legend(loc="lower right")
ax.set_ylabel("detections")


for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_legend().get_texts()
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(25)
fig.set_size_inches(10, 10)
fig.tight_layout()
figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/testing_data.png"
fig.savefig(figdest)
print(figdest)
# %%

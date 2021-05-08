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
import input_data
import textgrid
from luganda_train import SweepData
from luganda_info import WavTranscript

import seaborn as sns

sns.set()
sns.set_style("darkgrid")
sns.set_palette("bright")


def tpr_fpr(
    keyword,
    thresh,
    found_words,
    gt_target_times_ms,
    num_nontarget_words,
    time_tolerance_ms,
):
    found_target_times = [t for f, t in found_words if f == keyword]

    # find false negatives
    false_negatives = 0
    for time_ms in gt_target_times_ms:
        latest_time = time_ms + time_tolerance_ms
        earliest_time = time_ms - time_tolerance_ms
        potential_match = False
        for found_time in found_target_times:
            if found_time > latest_time:
                break
            if found_time < earliest_time:
                continue
            potential_match = True
        if not potential_match:
            false_negatives += 1

    # find true/false positives
    false_positives = 0  # no groundtruth match for model-found word
    true_positives = 0
    for word, time in found_words:
        if word == keyword:
            # highlight spurious words
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for gt_time in gt_target_times_ms:
                if gt_time > latest_time:
                    break
                if gt_time < earliest_time:
                    continue
                potential_match = True
            if not potential_match:
                false_positives += 1
            else:
                true_positives += 1
    if true_positives > len(gt_target_times_ms):
        print("WARNING: weird timing issue")
        true_positives = len(gt_target_times_ms)
        # if thresh is low, mult dets map to single gt (above suppression_ms)
        # single_target_recognize_commands already uses suppression_ms
        # raise suppression value?

    tpr = true_positives / len(gt_target_times_ms)
    false_rejections_per_instance = false_negatives / len(gt_target_times_ms)
    false_positives = len(found_target_times) - true_positives
    fpr = false_positives / num_nontarget_words
    pp = pprint.PrettyPrinter()
    result = dict(
        tpr=tpr,
        fpr=fpr,
        thresh=thresh,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        false_rejections_per_instance=false_rejections_per_instance,
    )
    # pp.pprint(result)
    # print("thresh", thresh, false_rejections_per_instance)
    # print("thresh", thresh, "true positives ", true_positives, "TPR:", tpr)
    # TODO(MMAZ) is there a beter way to calculate false positive rate?
    # fpr = false_positives / (false_positives + true_negatives)
    # fpr = false_positives / negatives
    # print(
    #     "false positives (model detection when no groundtruth target is present)",
    #     false_positives,
    # )
    # fpr = false_positives / num_nontarget_words
    # false_accepts_per_seconds = false_positives / (duration_s / (3600))
    return result


def count_nontarget_words(keyword, groundtruth):
    num_nontarget_words = 0
    for wav in groundtruth["stream_data"]:
        transcript = wav["transcript"]
        num_nontarget_words += sum([w != keyword for w in transcript.split()])
    return num_nontarget_words


# graph hyperparam sweep
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
evaldir = workdir / "cs288_eval"

eval_data = {}
for kw in os.listdir(evaldir):
    with open(evaldir / kw / f"{kw}_groundtruth.pkl", "rb") as fh:
        groundtruth = pickle.load(fh)

    num_nt = count_nontarget_words(keyword=kw, groundtruth=groundtruth)
    print(kw, "num nontarget words", num_nt)
    eval_data[kw] = dict(
        times=groundtruth["groundtruth_target_times_ms"], num_nt=num_nt
    )

#%%

use_mpl = True

if use_mpl:
    fig, ax = plt.subplots()
else:
    fig = go.Figure()

hpsweep = workdir / "hp_sweep"
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
                print(thresh)
                analysis = tpr_fpr(
                    keyword,
                    thresh,
                    found_words,
                    eval_data[keyword]["times"],
                    num_nontarget_words=eval_data[keyword]["num_nt"],
                    time_tolerance_ms=post_processing_settings.time_tolerance_ms,
                )
                tpr = analysis["tpr"]
                fpr = analysis["fpr"]
                all_tprs.append(tpr)
                all_fprs.append(fpr)
                all_threshs.append(thresh)
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

            # label = f"{exp} t: {num_train:02d} c: {wc} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            label = f"{keyword} t: {num_train:02d} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            if use_mpl:
                ax.plot(all_fprs, all_tprs, label=label, linewidth=3)
            else:
                fig.add_trace(
                    go.Scatter(x=all_fprs, y=all_tprs, text=all_threshs, name=label)
                )

if not use_mpl:
    fig.update_layout(
        xaxis_title="FPR", yaxis_title="TPR", title=f"streaming accuracy",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.show()
else:
    AX_LIM = 0.7
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
        item.set_fontsize(25)
    fig.set_size_inches(14, 14)
    fig.tight_layout()
    figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/lu_sweep_wc.png"
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
                    num_nontarget_words=num_nontarget_words,
                    time_tolerance_ms=post_processing_settings.time_tolerance_ms,
                )
                tpr = analysis["tpr"]
                fpr = analysis["fpr"]
                tprs.append(tpr)
                fprs.append(fpr)
                threshs.append(thresh)
                #pprint.pprint(analysis)

            if sd.backprop_into_embedding:
                lrinfo = f"lr1: {sd.primary_lr} lr2: {sd.embedding_lr}"
            else:
                lrinfo = f"lr1: {sd.primary_lr}"


            num_train = len(sd.train_files)

            # label = f"{exp} t: {num_train:02d} c: {wc} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            label = (
                f"{keyword} t: {num_train:02d} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            )
            #ax.plot(fprs,tprs, label=label, linewidth=1)
            #print(len(tprs))
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
    #ax.plot(x_all, ymean, alpha=0.7, linewidth=6, color=iso2color[lang_isocode], label=f"{iso2lang[lang_isocode]}")
    if graphing_context:
        gl = "context-padded"
    else:
        gl = "silence-padded"
    ax.plot(x_all, ymean, alpha=0.7, linewidth=6, label=gl)
    # draw bands over stdev
    ystdev = y_all.std(axis=1)
    #ax.fill_between(x_all, ymean - ystdev, ymean + ystdev, color=iso2color[lang_isocode], alpha=0.1)
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
    item.set_fontsize(25)
fig.set_size_inches(14, 14)
fig.tight_layout()
figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/context_v_silence.png"
fig.savefig(figdest)
print(figdest)
# %%

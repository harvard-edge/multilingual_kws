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
import numpy as np
import sox
import pydub
from pydub.playback import play

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
import input_data
import textgrid
from luganda_sweep import SweepData

import seaborn as sns

sns.set()
sns.set_style("darkgrid")
sns.set_palette("bright")


def tpr_fpr(
    keyword, found_words, gt_target_times_ms, num_nontarget_words, time_tolerance_ms,
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


# graph hyperparam sweep
workdir = Path("/home/mark/tinyspeech_harvard/luganda")

keyword = "akawuka"

with open(workdir / "akawuka_groundtruth.pkl", "rb") as fh:
    groundtruth = pickle.load(fh)

gt_target_times_ms = groundtruth["groundtruth_target_times_ms"]


def count_nontarget_words(keyword, groundtruth):
    num_nontarget_words = 0
    for wav in groundtruth["stream_data"]:
        transcript = wav["transcript"]
        num_nontarget_words += sum([w != keyword for w in transcript.split()])
    return num_nontarget_words


num_nontarget_words = count_nontarget_words(keyword=keyword, groundtruth=groundtruth)

#%%
fig, ax = plt.subplots()
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
        all_tprs = []
        all_fprs = []
        for post_processing_settings, results_per_thresh in result[keyword]:
            for thresh, (found_words, _) in results_per_thresh.items():
                if thresh < 0.3:
                    continue
                print(thresh)
                analysis = tpr_fpr(
                    keyword,
                    found_words,
                    gt_target_times_ms,
                    num_nontarget_words=num_nontarget_words,
                    time_tolerance_ms=post_processing_settings.time_tolerance_ms,
                )
                tpr = analysis["tpr"]
                fpr = analysis["fpr"]
                all_tprs.append(tpr)
                all_fprs.append(fpr)
                # if exp == "exp_10":
                #     pprint.pprint(analysis)

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

            label = f"{exp} t: {num_train:02d} c: {wc} e: {sd.n_epochs} b: {sd.n_batches} {lrinfo}"
            ax.plot(all_fprs, all_tprs, label=label, linewidth=3)

AX_LIM = 0.0
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
# figdest = "/home/mark/tinyspeech_harvard/tinyspeech_images/lu_sweep_wc.png"
# fig.savefig(figdest)
# print(figdest)
# %%

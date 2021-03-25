#%%
from dataclasses import dataclass
import logging
import sox
import datetime
import os
import sys
import pprint
import pickle
import glob
import subprocess
import pathlib
from pathlib import Path
from typing import List

import numpy as np

# import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sox.file_info import duration

sns.set()
sns.set_palette("bright")

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
# import input_data
from accuracy_utils import StreamingAccuracyStats
from single_target_recognize_commands import (
    SingleTargetRecognizeCommands,
    RecognizeResult,
)

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/embedding/")
import batch_streaming_analysis as sa
from batch_streaming_analysis import StreamTarget


#%%

iso2lang = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ky": "Kyrgyz",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "ta": "Tamil",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukranian",
}
#%%

def multi_streaming_FRR_FAR_curve(
    lang2results, is_full_sentence, time_tolerance_ms=1500,
):
    fig, ax = plt.subplots()
    # fmt:off

    for target_lang, all_targets in lang2results.items():
        # aggregate bars for target_lang
        y_all_false_rejection_rates = []
        x_all_false_accepts_secs = []
        x_all_false_accept_rates = []

        for (results_for_target, target, num_nontarget_words, duration_s) in all_targets:
            false_rejection_rates, false_accepts_secs, false_accept_rates, threshs = [], [], [], []
            for ix, (thresh, (stats, found_words, all_found_w_confidences)) in enumerate(results_for_target.items()):
                if thresh < 0.3 and is_full_sentence:
                    # at a low threshold, everything triggers a detection
                    # so ROC curves will loop back on themselves
                    # (we dont really have traditional pos/neg ROC examples since this is timeseries data)
                    continue
                groundtruth = stats._gt_occurrence
                gt_target_times = [t for g, t in groundtruth if g == target]
                # print("gt target occurences", len(gt_target_times))
                found_target_times = [t for f, t in found_words if f == target]
                # print("num found targets", len(found_target_times))

                # find false negatives
                false_negatives = 0
                for word, time in groundtruth:
                    if word == target:
                        latest_time = time + time_tolerance_ms
                        earliest_time = time - time_tolerance_ms
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
                    if word == target:
                        # highlight spurious words
                        latest_time = time + time_tolerance_ms
                        earliest_time = time - time_tolerance_ms
                        potential_match = False
                        for gt_time in gt_target_times:
                            if gt_time > latest_time:
                                break
                            if gt_time < earliest_time:
                                continue
                            potential_match = True
                        if not potential_match:
                            false_positives += 1
                        else:
                            true_positives += 1
                if true_positives > len(gt_target_times):
                    true_positives = len(gt_target_times)  # if thresh < 0.2:
                #     continue
                tpr = true_positives / len(gt_target_times)
                false_rejections_per_instance = false_negatives / len(gt_target_times)
                print("thresh", thresh, false_rejections_per_instance)
                # print("thresh", thresh, "true positives ", true_positives, "TPR:", tpr)
                # TODO(MMAZ) is there a beter way to calculate false positive rate?
                # fpr = false_positives / (false_positives + true_negatives)
                # fpr = false_positives / negatives
                # print(
                #     "false positives (model detection when no groundtruth target is present)",
                #     false_positives,
                # )
                fpr = false_positives / num_nontarget_words
                false_accepts_per_seconds = false_positives / (duration_s / (3600))
                # print("thresh", thresh,
                #     "FPR",
                #     f"{fpr:0.2f} {false_positives} false positives/{num_nontarget_words} nontarget words",
                # )
                # fpr_s = f"{fpr:0.2f}"
                threshs.append(thresh)
                false_accepts_secs.append(false_accepts_per_seconds)
                false_accept_rates.append(fpr)
                false_rejection_rates.append(false_rejections_per_instance)
            # draw a roc curve per keyword
            #ax.plot(false_accepts_secs, false_rejection_rates, label=f"{target} ({target_lang})")

            # collect data for whole language
            x_all_false_accept_rates.append(false_accept_rates)
            x_all_false_accepts_secs.append(false_accepts_secs)
            y_all_false_rejection_rates.append(false_rejection_rates)

        x_all_false_accept_rates = np.array(x_all_false_accept_rates)
        x_all_false_accepts_secs = np.array(x_all_false_accepts_secs)
        y_all_false_rejection_rates = np.array(y_all_false_rejection_rates)



        # make sure all tprs and fprs are monotonically increasing
        # for ix in range(x_all_false_accept_rates.shape[0]):
        #     # https://stackoverflow.com/a/47004533
        #     if not np.all(np.diff(np.flip(x_all_false_accepts_secs[ix,:])) >=0):
        #         raise ValueError("fprs (time) not in sorted order")
        #     if not np.all(np.diff(np.flip(y_all_false_rejection_rates[ix,:])) >=0):
        #         raise ValueError("fprs (rate) not in sorted order")
        #     if not np.all(np.diff(np.flip(x_all_false_accept_rates[ix,:])) >=0):
        #         raise ValueError("tprs not in sorted order")
        

        # for false accepts per hour
        # https://stackoverflow.com/a/43035301
        # x_all = np.unique(x_all_false_accepts_secs.ravel())
        # y_all = np.empty((x_all.shape[0], y_all_false_rejection_rates.shape[0]))
        # for ix in range(x_all_false_accepts_secs.shape[0]):
        #     y_all[:, ix] = np.interp(
        #         x_all, np.flip(x_all_false_accepts_secs[ix, :]), np.flip(y_all_false_rejection_rates[ix, :])
        #     )

        # for false accepts rate
        # https://stackoverflow.com/a/43035301
        x_all = np.unique(x_all_false_accept_rates.ravel())
        y_all = np.empty((x_all.shape[0], y_all_false_rejection_rates.shape[0]))
        for ix in range(x_all_false_accept_rates.shape[0]):
            y_all[:, ix] = np.interp(
                x_all, np.flip(x_all_false_accept_rates[ix, :]), np.flip(y_all_false_rejection_rates[ix, :])
            )


        # draw bands over min and max:
        # ymin = y_all.min(axis=1)
        # ymax = y_all.max(axis=1)
        # ax.fill_between(x_all, ymin, ymax, alpha=0.1, label=f"{iso2lang[lang_isocode]}")

        ymean = y_all.mean(axis=1)
        # draw mean
        ax.plot(x_all, ymean, alpha=0.7, linewidth=6, label=f"{iso2lang[target_lang]}")
        # draw bands over stdev
        ystdev = y_all.std(axis=1)
        ax.fill_between(x_all, ymean - ystdev, ymean + ystdev, alpha=0.1)
    # fmt:on

    #ax.set_xlim(left=0, right=1000)
    ax.set_xlim(left=0, right=0.14)
    ax.set_ylim([0,1])
    # ax.set_xlim(left=-5, right=200)
    # ax.set_ylim([-0.001, 0.4])
    ax.legend(loc="upper right")
    #ax.set_xlabel("False Accepts/Hour")
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("False Rejection Rate")
    # ax.set_xlim([175000,200000])
    fig.set_size_inches(15, 15)
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_legend().get_texts()
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)
    # ax.set_title(
    #     f"{target} -> threshold: {threshold:0.2f}, TPR: {tpr}, FPR: {fpr_s}, false positives: {false_positives}, false_negatives: {false_negatives}, groundtruth positives: {len(gt_target_times)}"
    # )
    # fig.savefig(f"/home/mark/tinyspeech_harvard/tmp/analysis/{target}/{target}_{threshold:0.2f}.png",dpi=300)
    return fig, ax

# %%

# fmt: off
in_embedding_sentence_data_dir = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_sentences")
with open("/home/mark/tinyspeech_harvard/paper_data/data_streaming_batch_sentences.pkl", 'rb') as fh:
    in_embedding_data = pickle.load(fh)
# fmt: on

print("n in-embedding examples", len(in_embedding_data))
available_results = []
for dat in in_embedding_data:
    if os.path.isfile(dat.destination_result_pkl):
        available_results.append((True, dat))

multi_results = []
for ix, (is_in_emedding, r) in enumerate(available_results):
    print(ix, "/", len(available_results))
    # this is a full sentence streaming example, use the transcript

    # fmt: off
    sdata_pkl = in_embedding_sentence_data_dir / f"streaming_{r.target_lang}" / f"streaming_{r.target_word}" / "streaming_test_data.pkl"
    # fmt: on

    # dict_keys(['target_word', 'target_lang', 'mp3_to_textgrid', 'timings', 'sample_data', 'num_target_words_in_stream', 'num_non_target_words_in_stream'])
    with open(sdata_pkl, "rb") as fh:
        sdata = pickle.load(fh)
    n_nontargets = sdata["num_non_target_words_in_stream"]
    duration_s = sox.file_info.duration(r.stream_wav)

    with open(r.destination_result_pkl, "rb") as fh:
        results = pickle.load(fh)
    
    # assumes groundtruth is the same for every threshold
    # stats_for_first_thresh = list(results[r.target_word].items())[0][1][0]
    # groundtruth = stats_for_first_thresh._gt_occurrence

    multi_results.append(
        (results[r.target_word], r.target_word, r.target_lang, n_nontargets, duration_s)
    )
lang2results = { l : [] for l in set([r[1].target_lang for r in available_results])}
for (all_target_results, target_word, target_lang, n_nontargets, duration_s) in multi_results:
    lang2results[target_lang].append((all_target_results, target_word, n_nontargets, duration_s))

# %%
multi_streaming_FRR_FAR_curve(lang2results, is_full_sentence=True)

# %%


# %%
"""
in_embedding_results_dir = Path(
    "/home/mark/tinyspeech_harvard/paper_data/results_streaming_batch_sentences/"
)


multi_results = []
for target in os.listdir(sse):
    if not os.path.isdir(sse / target):
        continue
    print("::::::: target", target)
    res = sse / target / "stream_results.pkl"
    if not os.path.isfile(res):
        print(res, "missing")
        continue
    with open(res, "rb") as fh:
        results = pickle.load(fh)
    sdata_pkl = sse / target / "streaming_test_data.pkl"

    if not os.path.isfile(sdata_pkl):
        # this is a per-word streaming example, must manually count unknowns
        for lang in os.listdir(frequent_words):
            for word in os.listdir(frequent_words / lang / "clips"):
                if word == target:
                    target_lang = lang
                    break
        with open(sse / target / "streaming_labels.txt") as fh:
            ls = fh.readlines()
        n_nontargets = [l.split(",")[0] for l in ls].count("_unknown_")
        duration_s = sox.file_info.duration(sse / target / "streaming_test.wav")
    else:
        # this is a full sentence streaming example, use the transcript
        with open(sdata_pkl, "rb") as fh:
            sdata = pickle.load(fh)
        print(sdata.keys())
        n_nontargets = sdata["num_non_target_words_in_stream"]
        target_lang = sdata["target_lang"]
        duration_s = sox.file_info.duration(sse / target / "stream.wav")
    multi_results.append(
        (results[target], target, target_lang, n_nontargets, duration_s)
    )
multi_streaming_FRR_FAR_curve(multi_results)
"""
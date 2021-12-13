# based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/test_streaming_accuracy.py


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
import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set()
sns.set_palette("bright")

import input_data
from accuracy_utils import StreamingAccuracyStats
from single_target_recognize_commands import SingleTargetRecognizeCommands, RecognizeResult

# sys.path.insert(
#     0, "/home/mark/tinyspeech_harvard/tensorflow/tensorflow/examples/speech_commands/"
# )
import batch_streaming_analysis as sa


#%%

tf.config.list_physical_devices("GPU")



# %%
############################################################################
#    full sentence streaming test
############################################################################

target = "ychydig"

model_settings = input_data.standard_microspeech_model_settings(label_count=3)
sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
#sse = pathlib.Path("/home/mark/tinyspeech_harvard/multilingual_streaming_sentence_experiments/")
base_dir = sse / target
model_dir = sse / target / "model"
model_name = os.listdir(model_dir)[0]
assert len(os.listdir(model_dir)) == 1, "multiple models?"
print(model_name)
model_path = model_dir / model_name
print(model_path)
wav_path = base_dir / "stream.wav"
ground_truth_path = base_dir / "labels.txt"
#wav_path = base_dir / "per_word" / "iaith2" / "streaming_test.wav"
#ground_truth_path = base_dir / "per_word" / "iaith2" / "streaming_labels.txt"

DESTINATION_RESULTS_PKL = base_dir / "stream_results.pkl"
DESTINATION_INFERENCES = base_dir / "raw_inferences.npy"
#DESTINATION_RESULTS_PKL = base_dir / "per_word" / "iaith2" / "stream_results.pkl"
#DESTINATION_INFERENCES = base_dir / "per_word" / "iaith2" / "raw_inferences.npy"
print("SAVING results TO\n", DESTINATION_RESULTS_PKL)
print("SAVING inferences TO\n", DESTINATION_INFERENCES)

# %%

assert not os.path.isfile(DESTINATION_RESULTS_PKL), "results already present"
assert not os.path.isfile(DESTINATION_INFERENCES), "inferences already present"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

# t_min = 0.4
# t_max = 0.95
# t_steps = int(np.ceil((t_max - t_min) / 0.05)) + 1
# threshs = np.linspace(t_min, t_max, t_steps).tolist()
flags = FlagTest(
    wav=str(wav_path),
    ground_truth=str(ground_truth_path),
    target_keyword=target,
    detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    #detection_thresholds=threshs,  # step threshold 0.05
    #detection_thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.8],  # step threshold 0.05
    #detection_thresholds=[0.55,0.6, 0.65, 0.7],  # step threshold 0.05
)
results = {}
results[target], inferences = calculate_streaming_accuracy(model, model_settings, flags)

with open(DESTINATION_RESULTS_PKL, "wb") as fh:
    pickle.dump(results, fh)
np.save(DESTINATION_INFERENCES, inferences)


# %%
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

# %%
# reuse existing saved inferences
# existing_inferences = np.load(base_dir / "raw_inferences.npy")
existing_inferences = np.load(DESTINATION_INFERENCES)

flags = FlagTest(
    wav=str(wav_path),
    ground_truth=str(ground_truth_path),
    target_keyword=target,
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    #detection_thresholds=threshs,  # step threshold 0.05
    detection_thresholds=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  # step threshold 0.05
    # detection_thresholds=[0.55,0.6, 0.65, 0.7],  # step threshold 0.05
)
results = {}
results[target], _ = calculate_streaming_accuracy(model, model_settings, flags, existing_inferences=existing_inferences)
print(len(results[target].keys()))


#%%
assert not os.path.isfile(DESTINATION_RESULTS_PKL), "results already present"
print("saving to", DESTINATION_RESULTS_PKL)
with open(DESTINATION_RESULTS_PKL, "wb") as fh:
    pickle.dump(results, fh)



#%%
############################################################################
#    batch streaming tests: [full sentence, individual-word]
############################################################################

#sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_batch_sentences/")
sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_batch_perword/")
for ix,target in enumerate(os.listdir(sse)):
    if not os.path.isdir(sse / target):
        continue
    print(f":::::::::::::::: {ix} ::::::::: - {target}")
    base_dir = sse / target
    model_dir = sse / target / "model"
    if not os.path.isdir(model_dir):
        print("skipping", target, "may be a word used for the embedding or unknown words")
        continue
    model_name = os.listdir(model_dir)[0]
    assert len(os.listdir(model_dir)) == 1, "multiple models?"
    print(model_name)
    model_path = model_dir / model_name
    print(model_path)

    #wav_path = base_dir / "stream.wav"
    #ground_truth_path = base_dir / "labels.txt"
    wav_path = base_dir / "streaming_test.wav"
    ground_truth_path = base_dir / "streaming_labels.txt"

    DESTINATION_RESULTS_PKL = base_dir / "stream_results.pkl"
    DESTINATION_INFERENCES = base_dir / "raw_inferences.npy"

    if os.path.isfile(DESTINATION_RESULTS_PKL):
        print("results already present", DESTINATION_RESULTS_PKL)
        continue
    print("SAVING results TO\n", DESTINATION_RESULTS_PKL)
    inferences_exist = False
    if os.path.isfile(DESTINATION_INFERENCES):
        print("inferences already present")
        loaded_inferences = np.load(DESTINATION_INFERENCES)
        inferences_exist=True
    else:
        print("SAVING inferences TO\n", DESTINATION_INFERENCES)

    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(model_path)
    tf.get_logger().setLevel(logging.INFO)
    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    flags = FlagTest(
        wav=str(wav_path),
        ground_truth=str(ground_truth_path),
        target_keyword=target,
        detection_thresholds=np.linspace(0.05, 1, 20).tolist(),  # step threshold 0.05
    )
    start = datetime.datetime.now()
    results = {}
    if inferences_exist:
        results[target], _ = calculate_streaming_accuracy(model, model_settings, flags, loaded_inferences)
    else:
        results[target], inferences = calculate_streaming_accuracy(model, model_settings, flags)
    end = datetime.datetime.now()
    print("elapsed time:", end-start)

    with open(DESTINATION_RESULTS_PKL, "wb") as fh:
        pickle.dump(results, fh)
    if not inferences_exist:
        np.save(DESTINATION_INFERENCES, inferences)
##############################################################################3

#%%
# undo batch (dangerous)
# sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_batch_sentences/")
# for ix,target in enumerate(os.listdir(sse)):
#     base_dir = sse / target
#     DESTINATION_RESULTS_PKL = base_dir / "stream_results.pkl"
#     if os.path.isfile(DESTINATION_RESULTS_PKL):
#         print("REMOVING", DESTINATION_RESULTS_PKL)
#         #os.remove(DESTINATION_RESULTS_PKL)

#%%
def multi_streaming_FRR_FAR_curve(
    multi_results,
    time_tolerance_ms=1500,
):
    fig, ax = plt.subplots()
    for results_for_target, target, target_lang, num_nontarget_words, duration_s in multi_results:
        false_rejection_rates, false_accepts_secs, threshs= [], [], []
        for ix, (thresh, (stats, found_words, all_found_w_confidences)) in enumerate(results_for_target.items()):
            if thresh < 0.15:
                continue
            groundtruth = stats._gt_occurrence
            gt_target_times = [t for g, t in groundtruth if g == target]
            #print("gt target occurences", len(gt_target_times))
            found_target_times = [t for f, t in found_words if f == target]
            #print("num found targets", len(found_target_times))

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
                true_positives = len(gt_target_times)            # if thresh < 0.2:
            #     continue
            tpr = true_positives / len(gt_target_times)
            false_rejections_per_instance = false_negatives / len(gt_target_times)
            print("thresh", thresh,false_rejections_per_instance)
            #print("thresh", thresh, "true positives ", true_positives, "TPR:", tpr)
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
            false_rejection_rates.append(false_rejections_per_instance)
        ax.plot(false_accepts_secs, false_rejection_rates, label=f"{target} ({target_lang})")

    # ax.set_xlim(left=-5)
    # ax.set_ylim([0,1.0])
    ax.set_xlim(left=-5, right=200)
    ax.set_ylim([-0.001,0.4])
    ax.legend(loc="upper right")
    ax.set_xlabel("False accepts / hour")
    ax.set_ylabel("False rejections / instance")
    # ax.set_xlim([175000,200000])
    fig.set_size_inches(15, 15)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_legend().get_texts() +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    # ax.set_title(
    #     f"{target} -> threshold: {threshold:0.2f}, TPR: {tpr}, FPR: {fpr_s}, false positives: {false_positives}, false_negatives: {false_negatives}, groundtruth positives: {len(gt_target_times)}"
    # )
    # fig.savefig(f"/home/mark/tinyspeech_harvard/tmp/analysis/{target}/{target}_{threshold:0.2f}.png",dpi=300)
    return fig, ax

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words/")
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
        for lang in os.listdir(frequent_words):
            for word in os.listdir(frequent_words / lang / "clips"):
                if word == target:
                    target_lang = lang
                    break
        with open(sse /target/ "streaming_labels.txt") as fh:
            ls = fh.readlines()
        n_nontargets = [l.split(",")[0] for l in ls].count("_unknown_")
        duration_s = sox.file_info.duration(sse / target / "streaming_test.wav")
    else:
        with open(sdata_pkl, "rb") as fh:
            sdata = pickle.load(fh)
        print(sdata.keys())
        n_nontargets = sdata["num_non_target_words_in_stream"]
        target_lang = sdata["target_lang"]
        duration_s = sox.file_info.duration(sse / target / "stream.wav")
    multi_results.append((results[target], target, target_lang, n_nontargets, duration_s))
multi_streaming_FRR_FAR_curve(multi_results)

#############################################################################

#%%

#%%
found_words_w_confidences = results[target][0.65][2]
with open(base_dir / f"found_words_w_confidences_{target}.py", "wb") as fh:
    pickle.dump(found_words_w_confidences, fh)


#%%
target = "merchant"

sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
base_dir = sse / target
model_dir = sse / target / "model"
model_name = os.listdir(model_dir)[0]
print(model_name)
model_path = model_dir / model_name
print(model_path)

DESTINATION = base_dir / "stream_results.pkl"
print("SAVING TO", DESTINATION)

#%%
#%%
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

# t_min = 0.4
# t_max = 0.95
# t_steps = int(np.ceil((t_max - t_min) / 0.05)) + 1
# threshs = np.linspace(t_min, t_max, t_steps).tolist()
flags = FlagTest(
    wav=str(base_dir / "stream.wav"),
    ground_truth=str(base_dir / "labels.txt"),
    target_keyword=target,
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    #detection_thresholds=threshs,  # step threshold 0.05
    detection_thresholds=[0.65],  # step threshold 0.05
)
results = {}
results[target], inferences = calculate_streaming_accuracy(model, model_settings, flags)

with open(DESTINATION, "wb") as fh:
    pickle.dump(results, fh)
np.save(base_dir / "raw_inferences.npy", inferences)

#%%
merchant_video = Path("/home/mark/tinyspeech_harvard/merchant_video/")
merchant_wav = merchant_video / "stream.wav"
# t_min = 0.4
# t_max = 0.95
# t_steps = int(np.ceil((t_max - t_min) / 0.05)) + 1
# threshs = np.linspace(t_min, t_max, t_steps).tolist()
flags = FlagTest(
    wav=str(merchant_wav),
    ground_truth=str(merchant_video / "labels.txt"),
    target_keyword="merchant",
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    detection_thresholds=[0.4],  # step threshold 0.05
)
model_settings = input_data.standard_microspeech_model_settings(3)
res = generate_inferences_offline(None, model_settings, flags)
res.shape
#%%
model_settings

#%%

# threshold=0.6
# flags = FlagTest(
#     wav=str(base_dir / "stream.wav"),
#     ground_truth=str(base_dir / "labels.txt"),
#     target_keyword=target,
#     detection_thresholds=[threshold]
# )
# inferences = np.load(base_dir / "inferences/inferences.npy")
# results={}
# results[target] = calculate_inferences_offline(inferences, flags)

# DESTINATION = base_dir / "results" / (f"stream_results_{threshold}.pkl")
# with open(DESTINATION, "wb") as fh:
#     pickle.dump(results, fh)


#%%
##############################
### VISUALIZE STREAM TIMELINE
##############################


def viz_stream_timeline(
    groundtruth,
    found_words,
    target,
    threshold,
    time_tolerance_ms=1500,
    num_nontarget_words=None,
):
    fig, ax = plt.subplots()
    gt_target_times = [t for g, t in groundtruth if g == target]
    print("gt target occurences", len(gt_target_times))
    found_target_times = [t for f, t in found_words if f == target]
    print("num found targets", len(found_target_times))

    # find false negatives
    false_negatives = 0
    for word, time in groundtruth:
        if word == target:
            ax.axvline(x=time, linestyle="--", linewidth=2, color=(0, 0.4, 0, 0.5))

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
                ax.axvline(
                    x=time,
                    linestyle="-",
                    linewidth=4,
                    color=(0.8, 0.3, 0, 0.5),
                    ymax=0.9,
                )

    # find true/false positives
    false_positives = 0  # no groundtruth match for model-found word
    true_positives = 0
    for word, time in found_words:
        if word == target:
            # q = False
            # if time == 189540:
            #    q=True
            #    ax.axvline(x=time, linestyle="-", linewidth=3, color=(1,0,1,0.4), ymax=0.5)
            # else:
            ax.axvline(
                x=time, linestyle="-", linewidth=3, color=(0, 0, 1, 0.4), ymax=0.6
            )
            # highlight spurious words
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for gt_time in gt_target_times:
                if gt_time > latest_time:
                    break
                if gt_time < earliest_time:
                    continue
                # if q:
                #    print(gt_time, np.abs(gt_time - time))
                #    print("b")
                potential_match = True
            if not potential_match:
                false_positives += 1
                ax.axvline(
                    x=time, linestyle="-", linewidth=5, color=(1, 0, 0, 0.4), ymax=0.4
                )
            else:
                true_positives += 1
    tpr = true_positives / len(gt_target_times)
    print("true positives ", true_positives, "TPR:", tpr)
    # TODO(MMAZ) is there a beter way to calculate false positive rate?
    # fpr = false_positives / (false_positives + true_negatives)
    # fpr = false_positives / negatives
    print(
        "false positives (model detection when no groundtruth target is present)",
        false_positives,
    )
    if num_nontarget_words is not None:
        fpr = false_positives / num_nontarget_words
        print(
            "FPR",
            f"{fpr:0.2f} {false_positives} false positives/{num_nontarget_words} nontarget words",
        )
        fpr_s = f"{fpr:0.2f}"
    else:
        fpr_s = "[not enough info]"

    max_x = groundtruth[-1][1] + 1000
    ax.set_xlim([0, max_x])
    # ax.set_xlim([175000,200000])
    fig.set_size_inches(40, 5)
    ax.set_title(
        f"{target} -> threshold: {threshold:0.2f}, TPR: {tpr}, FPR: {fpr_s}, false positives: {false_positives}, false_negatives: {false_negatives}, groundtruth positives: {len(gt_target_times)}"
    )
    # fig.savefig(f"/home/mark/tinyspeech_harvard/tmp/analysis/{target}/{target}_{threshold:0.2f}.png",dpi=300)
    return fig, ax


#%%

#sse = "/home/mark/tinyspeech_harvard/streaming_sentence_experiments/"
#res = sse + "old_merchant_5_shot/stream_results.pkl"
#target = "ychydig"
target = "ddechrau"
res = sse / target / "stream_results.pkl"
# res = sse / target / "per_word" / "iaith2" / "stream_results.pkl"
with open(res, "rb") as fh:
    results = pickle.load(fh)
for ix, thresh in enumerate(results[target].keys()):
    print(ix, thresh)

thresh_ix = 9
thresh, (stats, all_found_words, all_found_w_confidences) = list(results[target].items())[thresh_ix]
print("THRESH", thresh)
fig, ax = viz_stream_timeline(
    stats._gt_occurrence, all_found_words, target, thresh, num_nontarget_words=None
)

#%%

def roc_curve_streaming(
    results_for_target,
    target,
    num_nontarget_words,
    duration_s,
    time_tolerance_ms=1500,
):
    fig, ax = plt.subplots()
    fprs, tprs, threshs= [], [], []
    for ix, (thresh, (stats, found_words, all_found_w_confidences)) in enumerate(results_for_target.items()):
        if ix == 0:
            continue
        groundtruth = stats._gt_occurrence
        gt_target_times = [t for g, t in groundtruth if g == target]
        #print("gt target occurences", len(gt_target_times))
        found_target_times = [t for f, t in found_words if f == target]
        #print("num found targets", len(found_target_times))

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
            true_positives = len(gt_target_times)
        tpr = true_positives / len(gt_target_times)
        print("thresh", thresh, "true positives ", true_positives, "TPR:", tpr)
        # TODO(MMAZ) is there a beter way to calculate false positive rate?
        # fpr = false_positives / (false_positives + true_negatives)
        # fpr = false_positives / negatives
        # print(
        #     "false positives (model detection when no groundtruth target is present)",
        #     false_positives,
        # )
        fpr = false_positives / num_nontarget_words
        print("thresh", thresh, 
            "FPR",
            f"{fpr:0.2f} {false_positives} false positives/{num_nontarget_words} nontarget words",
        )
        # fpr_s = f"{fpr:0.2f}"
        threshs.append(thresh)
        fprs.append(fpr)
        tprs.append(tpr)

    ax.set_xlim([0,1.01])
    ax.set_ylim([0,1.01])
    ax.plot(fprs, tprs)
    # ax.set_xlim([175000,200000])
    fig.set_size_inches(15, 15)
    # ax.set_title(
    #     f"{target} -> threshold: {threshold:0.2f}, TPR: {tpr}, FPR: {fpr_s}, false positives: {false_positives}, false_negatives: {false_negatives}, groundtruth positives: {len(gt_target_times)}"
    # )
    # fig.savefig(f"/home/mark/tinyspeech_harvard/tmp/analysis/{target}/{target}_{threshold:0.2f}.png",dpi=300)
    return fig, ax

target = "ychydig"
res = sse / target / "stream_results.pkl"
with open(res, "rb") as fh:
    results = pickle.load(fh)
sdata_pkl = sse / target / "streaming_test_data.pkl"
with open(sdata_pkl, "rb") as fh:
    sdata = pickle.load(fh)
print(sdata.keys())
n_nontargets = sdata["num_non_target_words_in_stream"]
duration_s = sox.file_info.duration(sse / target / "stream.wav")
roc_curve_streaming(results[target], target, n_nontargets, duration_s)

#%%
def streaming_FRR_FAR_curve(
    results_for_target,
    target,
    num_nontarget_words,
    duration_s,
    time_tolerance_ms=1500,
):
    fig, ax = plt.subplots()
    false_rejection_rates, false_accepts_secs, threshs= [], [], []
    for ix, (thresh, (stats, found_words, all_found_w_confidences)) in enumerate(results_for_target.items()):
        if thresh < 0.2:
            continue
        groundtruth = stats._gt_occurrence
        gt_target_times = [t for g, t in groundtruth if g == target]
        #print("gt target occurences", len(gt_target_times))
        found_target_times = [t for f, t in found_words if f == target]
        #print("num found targets", len(found_target_times))

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
            true_positives = len(gt_target_times)
        tpr = true_positives / len(gt_target_times)
        false_rejections_per_instance = false_negatives / len(gt_target_times)
        print("thresh", thresh,false_rejections_per_instance)
        #print("thresh", thresh, "true positives ", true_positives, "TPR:", tpr)
        # TODO(MMAZ) is there a beter way to calculate false positive rate?
        # fpr = false_positives / (false_positives + true_negatives)
        # fpr = false_positives / negatives
        # print(
        #     "false positives (model detection when no groundtruth target is present)",
        #     false_positives,
        # )
        fpr = false_positives / num_nontarget_words
        false_accepts_per_seconds = false_positives / duration_s
        # print("thresh", thresh, 
        #     "FPR",
        #     f"{fpr:0.2f} {false_positives} false positives/{num_nontarget_words} nontarget words",
        # )
        # fpr_s = f"{fpr:0.2f}"
        threshs.append(thresh)
        false_accepts_secs.append(false_accepts_per_seconds)
        false_rejection_rates.append(false_rejections_per_instance)
    ax.plot(false_accepts_secs, false_rejection_rates)

    ax.set_xlim([0,1.01])
    ax.set_ylim([0,1.01])
    ax.set_xlabel("False accepts / sec")
    ax.set_ylabel("False rejections / instance")
    # ax.set_xlim([175000,200000])
    fig.set_size_inches(15, 15)
    # ax.set_title(
    #     f"{target} -> threshold: {threshold:0.2f}, TPR: {tpr}, FPR: {fpr_s}, false positives: {false_positives}, false_negatives: {false_negatives}, groundtruth positives: {len(gt_target_times)}"
    # )
    # fig.savefig(f"/home/mark/tinyspeech_harvard/tmp/analysis/{target}/{target}_{threshold:0.2f}.png",dpi=300)
    return fig, ax

target = "ychydig"
res = sse / target / "stream_results.pkl"
with open(res, "rb") as fh:
    results = pickle.load(fh)
sdata_pkl = sse / target / "streaming_test_data.pkl"
with open(sdata_pkl, "rb") as fh:
    sdata = pickle.load(fh)
print(sdata.keys())
n_nontargets = sdata["num_non_target_words_in_stream"]
duration_s = sox.file_info.duration(sse / target / "stream.wav")
streaming_FRR_FAR_curve(results[target], target, n_nontargets, duration_s)

#%%

#%%
############################################################################
#    Speech Commands
############################################################################
scanalysisdir = "/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
scmodeldir = scanalysisdir + "models/"
DESTINATION = scanalysisdir + "streaming_results_w_words.pkl"

scmodels = os.listdir(scmodeldir)
results = {}
for modelname in scmodels:
    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(scmodeldir + modelname)
    tf.get_logger().setLevel(logging.INFO)

    # xfer_5_shot_6_epochs_marvin_val_acc_0.95
    prefix = "xfer_5_shot_6_epochs_"
    target = modelname[len(prefix) :].split("_")[0]
    print(target)

    flags = FlagTest(
        wav=f"{scanalysisdir}/streaming/{target}/streaming_test.wav",
        ground_truth=f"{scanalysisdir}/streaming/{target}/streaming_labels.txt",
        target_keyword=target,
        detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    )
    results[target] = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)
# with open(DESTINATION, "wb") as fh:
#     pickle.dump(results, fh)

#%%
scanalysisdir = "/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
with open(scanalysisdir + "streaming_results_w_words.pkl", "rb") as fh:
    results = pickle.load(fh)

#%%
os.listdir(scmodeldir)

#%% analyze short tree
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(
    scmodeldir + "xfer_5_shot_6_epochs_tree_val_acc_0.93"
)
tf.get_logger().setLevel(logging.INFO)

target = "tree"
flags = FlagTest(
    wav=f"{scanalysisdir}/small_streaming/small_{target}/small.wav",
    ground_truth=f"{scanalysisdir}/small_streaming/small_{target}/small.txt",
    target_keyword=target,
    detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
)
results = {}
results[target] = calculate_streaming_accuracy(model, model_settings, FLAGS=flags)

#%%
print("---")

#%%


#%%
# Why do the curves drop here? is Y-axis not the TPR? 
# is it possible this is happening instead?
# (number of ALL model positives, both true and false ones)/(# of groundtruth positives)
# seems to not be the case, i am using n_correct / n_targets
# need to dig in further to the stats behavior

fig = go.Figure()
for target, results_per_threshold in results.items():
    print(target)
    target_match_over_gt_positives = []
    false_positives_over_silence_or_unknown = []
    thresh_labels = []
    for thresh, (stats, found_words) in results_per_threshold.items():
        infomsg, statdict = stats.print_accuracy_stats()
        n_targets = statdict["num_groundtruth_target"]
        n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
        n_correct = statdict["matched"][target]
        n_false_positives = (
            statdict["wrong"][input_data.SILENCE_LABEL]
            + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
        )
        target_match_over_gt_positives.append(n_correct / n_targets)
        false_positives_over_silence_or_unknown.append(
            n_false_positives / n_silence_unknown
        )
        thresh_labels.append(f"thresh: {thresh}")
        print(
            ":::", thresh, n_correct / n_targets, n_false_positives / n_silence_unknown
        )
    fig.add_trace(
        go.Scatter(
            x=false_positives_over_silence_or_unknown,
            y=target_match_over_gt_positives,
            text=thresh_labels,
            name=target,
        )
    )
fig.update_layout(
    xaxis_title="(false positive target matches)/(# GT silence, unknown)",
    yaxis_title="(target matches)/(# GT targets)",
)
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])
# fig.write_html(scanalysisdir + "speech_commands_streaming_roc.html")
# fig.write_html(scanalysisdir + "short_tree.html")

fig


#%%  long pauses between also
############################################################################
modelname = "xfer_5_shot_6_epochs_also_val_acc_0.89"
model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

flags = FlagTest(
    wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_test.wav",
    ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also2/streaming_labels.txt",
    target_keyword="also",
    detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    # detection_thresholds=[0.6],
)
results = {}
results["also"] = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)


#%%


##############################################################################

#%% COLLECT AGGREGATE STATS

DESTINATION = "/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming_stats_fast.pkl"

models = [
    ("also", "also", "xfer_5_shot_6_epochs_also_val_acc_0.89/"),
    ("always", "always", "xfer_5_shot_6_epochs_always_val_acc_0.98/"),
    ("area", "area", "xfer_5_shot_6_epochs_area_val_acc_0.91/"),
    ("between_1", "between", "xfer_5_shot_6_epochs_between_val_acc_0.91/"),
    ("between_2", "between", "xfer_5_shot_6_epochs_between_val_acc_0.94/"),
    ("last", "last", "xfer_5_shot_6_epochs_last_val_acc_0.97/"),
    ("long", "long", "xfer_5_shot_6_epochs_long_val_acc_0.95/"),
    ("thing", "thing", "xfer_5_shot_6_epochs_thing_val_acc_0.88/"),
    ("will", "will", "xfer_5_shot_6_epochs_will_val_acc_0.87/"),
]

full_results = {}
for resultname, target_keyword, modelname in models:
    model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(model_path)
    tf.get_logger().setLevel(logging.INFO)

    full_results[resultname] = {}

    flags = FlagTest(
        wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/{target_keyword}/streaming_test.wav",
        ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/{target_keyword}/streaming_labels.txt",
        target_keyword=target_keyword,
        detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    )

    print("-:::::::::::::-", resultname)
    results = calculate_streaming_accuracy(model, audio_dataset, FLAGS=flags)

    full_results[resultname] = results

with open(DESTINATION, "wb") as fh:
    pickle.dump(full_results, fh)
print("FULL RESULTS DONE")

#%%
for resultname, results_per_threshold in full_results.items():
    print(resultname)
    target_match_over_gt_positives = []
    false_positives_over_silence_or_unknown = []
    thresh_labels = []
    for (thresh, stats) in results_per_threshold.items():
        target = stats[0]
        statdict = stats[2]
        n_targets = statdict["num_groundtruth_target"]
        n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
        n_correct = statdict["matched"][target]
        n_false_positives = (
            statdict["wrong"][input_data.SILENCE_LABEL]
            + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
        )
        print(target, n_targets, n_silence_unknown, n_correct, n_false_positives)
        break
    break


#%%
fig = go.Figure()
for resultname, results_per_threshold in full_results.items():
    print(resultname)
    target_match_over_gt_positives = []
    false_positives_over_silence_or_unknown = []
    thresh_labels = []
    for (thresh, stats) in results_per_threshold.items():
        target = stats[0]
        statdict = stats[2]
        n_targets = statdict["num_groundtruth_target"]
        n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
        n_correct = statdict["matched"][target]
        n_false_positives = (
            statdict["wrong"][input_data.SILENCE_LABEL]
            + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
        )
        target_match_over_gt_positives.append(n_correct / n_targets)
        false_positives_over_silence_or_unknown.append(
            n_false_positives / n_silence_unknown
        )
        thresh_labels.append(f"thresh: {thresh}")
    fig.add_trace(
        go.Scatter(
            x=false_positives_over_silence_or_unknown,
            y=target_match_over_gt_positives,
            text=thresh_labels,
            name=resultname,
        )
    )
fig.update_layout(
    xaxis_title="(false positive target matches)/(# GT silence, unknown)",
    yaxis_title="(target matches)/(# GT targets)",
)
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])
fig

#%%
fig.write_html(
    "/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming_performance_aggregate.html"
)


#%%
with np.printoptions(precision=3, suppress=True):
    pprint.pprint(full_results)
#%%
def markstats(FLAGS, stats):
    cliplength_s = 600.0  # seconds
    false_accepts_per_second = stats._how_many_fp / cliplength_s
    false_accepts_per_hour = false_accepts_per_second * 3600.0
    false_rejects_per_instance = stats._how_many_fn / stats._how_many_gt
    correct_match_percentage = stats._how_many_c / stats._how_many_gt
    #   with open("./markstats/stats.csv", 'a') as fh:
    #     print(f"\n\n\n  {FLAGS.detection_threshold},{false_accepts_per_hour},{false_rejects_per_instance},{correct_match_percentage}  \n\n\n")
    #     fh.write(f"{FLAGS.detection_threshold},{false_accepts_per_hour},{false_rejects_per_instance},{correct_match_percentage}\n")
    report = stats.print_accuracy_stats()


#   with open("./markstats/report.txt", 'a') as fh:
#     fh.write("\n" + str(FLAGS.detection_threshold) + "\n" + report + "\n")
#     break


######################################################################


#%% internals of calculating stats
modelname = "xfer_5_shot_6_epochs_also_val_acc_0.89"
model_path = f"/home/mark/tinyspeech_harvard/xfer_efnet_5/models/{modelname}"

tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel(logging.INFO)

FLAGS = FlagTest(
    wav=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also3/streaming_test.wav",
    ground_truth=f"/home/mark/tinyspeech_harvard/xfer_efnet_5/streaming/also3/streaming_labels.txt",
    target_keyword="also",
    # detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
    detection_thresholds=[0.6],
)

# scanalysisdir="/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
# scmodeldir = scanalysisdir + "models/"
# tf.get_logger().setLevel(logging.ERROR)
# model = tf.keras.models.load_model(scmodeldir + 'xfer_5_shot_6_epochs_tree_val_acc_0.93')
# tf.get_logger().setLevel(logging.INFO)

# target="tree"
# FLAGS = FlagTest(
#     wav=f"{scanalysisdir}/small_streaming/small_{target}/small.wav",
#     ground_truth=f"{scanalysisdir}/small_streaming/small_{target}/small.txt",
#     target_keyword=target,
#     detection_thresholds=np.linspace(0, 1, 21).tolist(),  # step threshold 0.05
# )

wav_loader = tf.io.read_file(FLAGS.wav)
(audio, sample_rate) = tf.audio.decode_wav(wav_loader, desired_channels=1)
sample_rate = sample_rate.numpy()
audio = audio.numpy().flatten()

# Init instance of StreamingAccuracyStats and load ground truth.
data_samples = audio.shape[0]
clip_duration_samples = int(FLAGS.clip_duration_ms * sample_rate / 1000)
clip_stride_samples = int(FLAGS.clip_stride_ms * sample_rate / 1000)
audio_data_end = data_samples - clip_duration_samples

spectrograms = np.zeros(
    (
        int(np.ceil(audio_data_end / clip_stride_samples)),
        model_settings["spectrogram_length"],
        model_settings["fingerprint_width"],
    )
)
print("building spectrograms")
# Inference along audio stream.
for ix, audio_data_offset in enumerate(range(0, audio_data_end, clip_stride_samples)):
    input_start = audio_data_offset
    input_end = audio_data_offset + clip_duration_samples
    spectrograms[ix] = audio_dataset.to_micro_spectrogram(audio[input_start:input_end])

inferences = model.predict(spectrograms[:, :, :, np.newaxis])
print("inferences complete")

#%%

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import accuracy_utils as mau

#%%
import importlib

importlib.reload(mau)


#%%

# for q in stats._gt_occurrence:
#     if q[1] > 1165640:
#         print(q)

FLAGS.time_tolerance_ms

#%%


threshold = 0.15
target = "also"
stats = mau.StreamingAccuracyStats(target_keyword=FLAGS.target_keyword)
stats.read_ground_truth_file(FLAGS.ground_truth)
recognize_element = RecognizeResult()
recognize_commands = RecognizeCommands(
    labels=FLAGS.labels(),
    average_window_duration_ms=FLAGS.average_window_duration_ms,
    detection_threshold=threshold,
    suppression_ms=FLAGS.suppression_ms,
    minimum_count=4,
)
all_found_words = []
# calculate statistics using inferences
for ix, audio_data_offset in enumerate(range(0, audio_data_end, clip_stride_samples)):
    output_softmax = inferences[ix]
    current_time_ms = int(audio_data_offset * 1000 / sample_rate)
    recognize_commands.process_latest_result(
        output_softmax, current_time_ms, recognize_element
    )
    if (
        recognize_element.is_new_command
        and recognize_element.founded_command != "_silence_"
    ):
        # print(current_time_ms/1000, output_softmax, recognize_element.founded_command, recognize_element.score)
        # print(current_time_ms/1000, recognize_element.founded_command)#, recognize_element.score)
        all_found_words.append([recognize_element.founded_command, current_time_ms])
        stats.calculate_accuracy_stats(
            all_found_words, current_time_ms, FLAGS.time_tolerance_ms
        )
        recognition_state = stats.delta()
        # print(
        #     "{}ms {}:{}{}".format(
        #         current_time_ms,
        #         recognize_element.founded_command,
        #         recognize_element.score,
        #         recognition_state,
        #     )
        # )
        stats.print_accuracy_stats()
print("DONE", threshold)
print("LENGTH", len(all_found_words))
# calculate final stats for full wav file:
stats.calculate_accuracy_stats(all_found_words, -1, FLAGS.time_tolerance_ms)
msg, info = stats.print_accuracy_stats()
statdict = info
n_targets = statdict["num_groundtruth_target"]
n_silence_unknown = statdict["num_groundtruth_unknown_or_silence"]
n_correct = statdict["matched"][target]
n_false_positives = (
    statdict["wrong"][input_data.SILENCE_LABEL]
    + statdict["wrong"][input_data.UNKNOWN_WORD_LABEL]
)
# target_match_over_gt_positives.append(n_correct / n_targets)
tpr = n_correct / n_targets
# false_positives_over_silence_or_unknown.append(
#     n_false_positives / n_silence_unknown
# )
fpr = n_false_positives / n_silence_unknown
# thresh_labels.append(f"thresh: {threshold}")
print(info)
print("threshold", threshold, "tpr", tpr, "fpr", fpr)
#%%
print(len(stats._gt_occurrence))
print(len(all_found_words))
num_unknowns = len([w for w in all_found_words if w[0] == "_unknown_"])
print(num_unknowns, num_unknowns / len(all_found_words))

#%%
print(
    {
        "correct_match_percentage": 88.7218045112782,
        "wrong_match_percentage": 7.518796992481203,
        "howmanyfp": 0,
        "howmanyfn": 7,
        "wrong": {"_silence_": 0, "_unknown_": 2, "also": 0},
        "matched": {"_silence_": 0, "_unknown_": 56, "also": 62},
        "num_groundtruth_target": 71,
        "num_groundtruth_unknown_or_silence": 62,
    }
)

#%%
afw = [["_unknown_", 2820], ["also", 16580], ["_unknown_", 24980], ["also", 25500]]
stats._gt_occurrence[:5]

#%%
len(stats._gt_occurrence)
#%%
gt_u, gt_t = [], []
for l, t in stats._gt_occurrence:
    if l == "_unknown_":
        gt_u.append([t, 1])
        gt_t.append([t, 0])
    else:
        gt_u.append([t, 0])
        gt_t.append([t, 1])
gt_u = np.array(gt_u)
gt_t = np.array(gt_t)

#%%
f_u, f_t = [], []
for l, t in all_found_words:
    if l == "_unknown_":
        f_u.append([t, 0.5])
        f_t.append([t, 0])
    else:
        f_u.append([t, 0])
        f_t.append([t, 0.5])
f_u = np.array(f_u)
f_t = np.array(f_t)

#%%
fig, ax = plt.subplots()
ax.plot(gt_u[:, 0], gt_u[:, 1], label="groundtruth unknown", color="blue")
ax.fill_between(gt_u[:, 0], gt_u[:, 1], color=(0, 0, 0.8, 0.3))
ax.plot(gt_t[:, 0], gt_t[:, 1], label="groundtruth target", color="green")
ax.fill_between(gt_t[:, 0], gt_t[:, 1], color=(0.2, 0.8, 0, 0.3))

ax.plot(f_u[:, 0], f_u[:, 1], label="found unknown", color="orange")
ax.fill_between(f_u[:, 0], f_u[:, 1], color="orange")
ax.plot(f_t[:, 0], f_t[:, 1], label="found target", color="purple")
ax.fill_between(f_t[:, 0], f_t[:, 1], color="purple")
ax.legend(loc="upper right")
fig.set_size_inches(30, 5)


#%%
import os
import logging
import re
from typing import Dict, List
import datetime

import glob
import numpy as np
import tensorflow as tf
import pickle

import sys

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1)

#%%

tf.config.list_physical_devices("GPU")


#%% analysis




def analyze_model(
    model_path: os.PathLike,
    model_commands: List[str],
    val_acc: float,
    data_dir: os.PathLike,
    audio_dataset: input_data.AudioDataset,
    unknown_training_words: List[str],
    oov_words: List[str],
    embedding_commands: List[str],
    num_samples_command=1500,
    n_words_oov_unknown=50,
    n_examples_oov_unknown=200,
):
    print("loading", model_path)
    tf.get_logger().setLevel(logging.ERROR)
    xfer = tf.keras.models.load_model(model_path)
    tf.get_logger().setLevel(logging.INFO)

    i = 0
    assert len(model_commands) == 1, "need to refactor for multiple commands"
    # label_id = i+1 # skip [silence]
    label_id = i + 2  # skip [silence, unknown]

    target_results = evaluate(
        model_commands, label_id, data_dir, num_samples_command, xfer, audio_dataset
    )

    oov_testing = set(oov_words).difference(set(model_commands))

    ots = np.random.choice(list(oov_testing), n_words_oov_unknown, replace=False)
    oov_results = evaluate(
        ots,
        input_data.UNKNOWN_WORD_INDEX,
        data_dir,
        n_examples_oov_unknown,
        xfer,
        audio_dataset,
    )

    # words used to training the _UNKNOWN_ category in the fine-tuned model
    if len(unknown_training_words) > n_words_oov_unknown:
        uts = np.random.choice(
            unknown_training_words, n_words_oov_unknown, replace=False
        )
    else:
        uts = unknown_training_words
    unknown_training_results = evaluate(
        uts,
        input_data.UNKNOWN_WORD_INDEX,
        data_dir,
        n_examples_oov_unknown,
        xfer,
        audio_dataset,
    )

    # now-unknown original words (words used to train the embedding model))
    uws = np.random.choice(embedding_commands, n_words_oov_unknown, replace=False)
    original_embedding_results = evaluate(
        uws,
        input_data.UNKNOWN_WORD_INDEX,
        data_dir,
        n_examples_oov_unknown,
        xfer,
        audio_dataset,
    )

    results = {
        "oov_testing": oov_testing,
        "unknown_training_words": uts,
        "original_embedding_words": uws,
        "oov": oov_results,
        "original_embedding": original_embedding_results,
        "target_keywords": target_results,
        "unknown_training": unknown_training_results,
        "words": model_commands,
        "val_acc": val_acc,
    }
    return results


# def make_roc(results: List[Dict], nrows: int, ncols: int):
#     assert nrows * ncols == len(results), "fewer results than requested plots"
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
#     for ix, (res, ax) in enumerate(zip(results, axes.flatten())):
#         ccs = np.array(res["target_keywords"]["correct"])
#         ics = np.array(res["target_keywords"]["incorrect"])
#         num_samples = ccs.shape[0] + ics.shape[0]

#         # TODO(mmaz): there are like four different styles of expressing the same calculations here, which one is clearest?

#         # total false positives
#         tkcs = np.array(res["target_keywords"]["incorrect"])
#         owcs = np.array(res["oov"]["incorrect"])
#         utcs = np.array(res["unknown_training"]["incorrect"])
#         oecs = np.array(res["original_embedding"]["incorrect"])
#         all_incorrect = np.concatenate([tkcs, owcs, utcs, oecs])
#         all_correct = np.array(
#             res["target_keywords"]["correct"]
#             + res["oov"]["correct"]
#             + res["unknown_training"]["correct"]
#             + res["original_embedding"]["correct"]
#         )
#         total_predictions = all_incorrect.shape[0] + all_correct.shape[0]

#         # total_unknown = len(correct_unknown_confidences + incorrect_unknown_confidences + original_words_correct_unknown_confidences + original_words_incorrect_unknown_confidences)
#         total_unknown = sum(
#             [
#                 len(res[k]["correct"]) + len(res[k]["incorrect"])
#                 for k in ["oov", "unknown_training", "original_embedding"]
#             ]
#         )

#         total_tprs, total_fprs = [], []
#         target_tprs, unknown_fprs = [], []
#         threshs = np.arange(0, 1, 0.01)
#         for threshold in threshs:
#             total_tpr = all_correct[all_correct > threshold].shape[0] / total_predictions
#             total_tprs.append(total_tpr)

#             target_tpr = ccs[ccs > threshold].shape[0] / num_samples
#             target_tprs.append(target_tpr)
#             total_fpr = (
#                 all_incorrect[all_incorrect > threshold].shape[0] / total_predictions
#             )
#             total_fprs.append(total_fpr)
#             fpr_unknown = (
#                 np.where(np.concatenate([owcs, utcs, oecs]) > threshold)[0].shape[0]
#                 / total_unknown
#             )
#             unknown_fprs.append(fpr_unknown)

#         ax.plot(total_fprs, total_tprs, label="total accuracy")
#         ax.plot(unknown_fprs, target_tprs,label="target TPR vs unknown FPR")
#         ax.set_xlim(-0.01, 1)
#         ax.set_ylim(-0.01, 1)

#         v = res["val_acc"]
#         wl = ", ".join(res["words"]) + f" (val acc {v})"
#         ax.set_title(wl)
#         ax.set_xlabel("fpr")
#         ax.set_ylabel("tpr")
#         ax.legend(loc="lower right")
#     return fig, axes


def calc_roc(res):
    # _TARGET_ is class 1, _UNKNOWN_ is class 0

    # positive label: target keywords classified as _TARGET_
    target_correct = np.array(res["target_keywords"]["correct"])
    # false negatives - target kws incorrectly classified as _UNKNOWN_:
    target_incorrect = np.array(res["target_keywords"]["incorrect"])
    total_positives = target_correct.shape[0] + target_incorrect.shape[0]

    # negative labels
    oov_correct = np.array(res["oov"]["correct"])
    oov_incorrect = np.array(res["oov"]["incorrect"])
    oov_total = oov_correct.shape[0] + oov_incorrect.shape[0]

    unknown_correct = np.array(res["unknown_training"]["correct"])
    unknown_incorrect = np.array(res["unknown_training"]["incorrect"])
    unknown_total = unknown_correct.shape[0] + unknown_incorrect.shape[0]

    original_correct = np.array(res["original_embedding"]["correct"])
    original_incorrect = np.array(res["original_embedding"]["incorrect"])
    original_total = original_correct.shape[0] + original_incorrect.shape[0]

    total_negatives = oov_total + unknown_total + original_total

    # false positives: _UNKNOWN_ keywords incorrectly classified as _TARGET_
    false_positives = np.concatenate(
        [oov_incorrect, unknown_incorrect, original_incorrect]
    )

    # target_tprs, target_fprs = [], []
    # oov_tprs, oov_fprs = [],[]
    # unknown_tprs, unknown_fprs = [],[]
    # original_tprs, original_fprs = [], []
    tprs, fprs = [], []

    threshs = np.arange(0, 1.01, 0.01)
    for threshold in threshs:
        tpr = target_correct[target_correct > threshold].shape[0] / total_positives
        tprs.append(tpr)
        fpr = false_positives[false_positives > threshold].shape[0] / total_negatives
        fprs.append(fpr)
    return tprs, fprs


def make_roc_plotly(results: List[Dict]):
    fig = go.Figure()
    for ix, res in enumerate(results):
        tprs, fprs = calc_roc(res)

        v = res["val_acc"]
        title = ", ".join(res["words"]) + f" (val acc {v})"

        labels = np.arange(0, 1, 0.01)
        fig.add_trace(go.Scatter(x=fprs, y=tprs, text=labels, name=title))

    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def make_roc(results: List[Dict], nrows: int, ncols: int):
    assert nrows * ncols == len(results), "fewer results than requested plots"
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for ix, (res, ax) in enumerate(zip(results, axes.flatten())):
        tprs, fprs = calc_roc(res)

        ax.plot(fprs, tprs)
        ax.set_xlim(-0.01, 1)
        ax.set_ylim(-0.01, 1)

        v = res["val_acc"]
        wl = ", ".join(res["words"]) + f" (val acc {v})"
        ax.set_title(wl)
        ax.set_xlabel("fpr")
        ax.set_ylabel("tpr")
        # ax.legend(loc="lower right")
    return fig, axes


def make_viz(results: List[Dict], threshold: float, nrows: int, ncols: int):
    assert nrows * ncols == len(results), "fewer results than requested plots"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    for ix, (res, ax) in enumerate(zip(results, axes.flatten())):
        # row = ix // ncols
        # col = ix % ncols
        # ax = axes[row][col]

        # fmt: off
        # correct
        # k=res["target_keywords"]["correct"]
        # ax.hist(k, bins=50, alpha=0.3, label=f'correct classifications (n={len(k)})', color=sns.color_palette("bright")[0],)
        # k=res["oov"]["correct"]
        # ax.hist(k, bins=50, alpha=0.3, label=f"oov words correct (n={len(k)})", color=sns.color_palette("bright")[8],)
        # k=res["original_embedding"]["correct"]
        # ax.hist(k, bins=50, alpha=0.3, label=f"original embedding words correct (n={len(k)})", color=sns.color_palette("bright")[9],)
        # k=res["unknown_training"]["correct"]
        # ax.hist(k, bins=50, alpha=0.3, label=f"unknown training words correct (n={len(k)})", color=sns.color_palette("bright")[9],)


        k=res["target_keywords"]["incorrect"]
        ax.hist(k, bins=50, alpha=0.3, label=f'incorrect classifications (n={len(k)})', color="orange",)
        k=res["oov"]["incorrect"]
        ax.hist(k, bins=50, alpha=0.3, label=f"oov words incorrect (n={len(k)})", color="red")
        k=res["original_embedding"]["incorrect"]
        ax.hist(k, bins=50, alpha=0.3, label=f"original embedding words incorrect (n={len(k)})", color="darkred")
        k=res["unknown_training"]["incorrect"]
        ax.hist(k, bins=50, alpha=0.3, label=f"unknown training words incorrect (n={len(k)})", color="pink")

        # fmt: on
        ccs = np.array(res["target_keywords"]["correct"])
        ics = np.array(res["target_keywords"]["incorrect"])
        # correct_share_above_thresh = ccs[ccs>threshold].shape[0]/ccs.shape[0]
        num_samples = ccs.shape[0] + ics.shape[0]
        correct_thresh_all = ccs[ccs > threshold].shape[0] / (num_samples)

        # total false positives
        tkcs = np.array(res["target_keywords"]["incorrect"])
        owcs = np.array(res["oov"]["incorrect"])
        utcs = np.array(res["unknown_training"]["incorrect"])
        oecs = np.array(res["original_embedding"]["incorrect"])
        all_incorrect = np.concatenate([tkcs, owcs, utcs, oecs])
        all_correct = np.array(
            res["target_keywords"]["correct"]
            + res["oov"]["correct"]
            + res["unknown_training"]["correct"]
            + res["original_embedding"]["correct"]
        )
        total_predictions = all_incorrect.shape[0] + all_correct.shape[0]

        # total_unknown = len(correct_unknown_confidences + incorrect_unknown_confidences + original_words_correct_unknown_confidences + original_words_incorrect_unknown_confidences)
        total_fpr = (
            all_incorrect[all_incorrect > threshold].shape[0] / total_predictions
        )
        total_unknown = sum(
            [
                len(res[k]["correct"]) + len(res[k]["incorrect"])
                for k in ["oov", "unknown_training", "original_embedding"]
            ]
        )
        fpr_unknown = (
            np.where(np.concatenate([owcs, utcs, oecs]) > threshold)[0].shape[0]
            / total_unknown
        )

        ax.axvline(
            x=threshold,
            # label=f"share {correct_share_above_thresh:0.2f}, correct {correct_thresh_all:0.2f}, other {frac_other:0.2f}",
            label=f"tpr: {correct_thresh_all:0.2f}, fpr_unknown: {fpr_unknown:0.2f}, total_fpr: {total_fpr:0.2f}",
            linestyle="--",
            color=sns.color_palette("bright")[1],
        )

        ax.legend(loc="upper left")
        v = res["val_acc"]
        wl = ", ".join(res["words"]) + f" (val acc {v})"
        ax.set_title(wl)
        ax.set_xlabel("confidence (softmax)")
        ax.set_ylabel("count")
    return fig, axes


#%% LOAD DATA
data_dir = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/"
model_dir = "/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/"
with open(model_dir + "unknown_words.pkl", "rb") as fh:
    unknown_words = pickle.load(fh)
with open(model_dir + "oov_words.pkl", "rb") as fh:
    oov_words = pickle.load(fh)
with open(model_dir + "unknown_files.pkl", "rb") as fh:
    unknown_files = pickle.load(fh)

with open(
    "/home/mark/tinyspeech_harvard/train_100_augment/" + "wordlist.txt", "r"
) as fh:
    commands = fh.read().splitlines()

print(
    len(commands), len(unknown_words), len(oov_words),
)

other_words = [
    w for w in os.listdir(data_dir) if w != "_background_noise_" and w not in commands
]
other_words.sort()
print(len(other_words))
assert len(set(other_words).intersection(commands)) == 0

#%%  TRAINING
#  destdir = "/home/mark/tinyspeech_harvard/xfer_efnet_10/"
#  basemodelpath = "/home/mark/tinyspeech_harvard/train_100_augment/hundredword_efficientnet_1600_selu_specaug80.0146-0.8736"
#  random_sample_transfer_models(
#      NUM_MODELS=1,
#      N_SHOTS=10,
#      VAL_UTTERANCES=400,
#      oov_words=oov_words,
#      dest_dir=destdir,
#      unknown_files=unknown_files,
#      base_model_path=basemodelpath,
#      EPOCHS=6,
#      data_dir="/home/mark/tinyspeech_harvard/frequent_words/en/clips/",
#  )


#%%
#
# epochs = "_epochs_"
# start = m.find(epochs) + len(epochs)
# valacc_str = "_val_acc_"
# last_word = m.find(valacc_str)
# n_word_xfer = m[start:last_word].split("_")
# print(n_word_xfer)
# valacc_idx = last_word + len(valacc_str)
# valacc = float(m[valacc_idx:])
# print(valacc)
#
# fig, axes = make_viz(rc.values(), threshold=0.55, nrows=3, ncols=3)
# fig.savefig(destdir + "5shot.png", dpi=200, tight_layout=True)
#


#%% LISTEN
###############################################
##               LISTEN
###############################################
# import pydub
# from pydub.playback import play
# import time
#
# audiofiles = glob.glob(destdir + "*.txt")
# # audiofiles = glob.glob(scmodeldir + "*.txt")
# for ix, f in enumerate(audiofiles):
#     print(ix, f)
# ix = 3
# with open(audiofiles[ix], "rb") as fh:
#     utterances = fh.read().decode("utf8").splitlines()
#
# for u in utterances:
#     print(u)
#     play(pydub.AudioSegment.from_wav(u))
#     time.sleep(0.5)
#

#%%
###############################################
##               SPEECH COMMANDS
###############################################

speech_commands = "/home/mark/tinyspeech_harvard/speech_commands/"
dnames = os.listdir(speech_commands)
speech_commands_words = [
    w
    for w in dnames
    if os.path.isdir(speech_commands + w) and w != "_background_noise_"
]
# print(speech_commands_words)
# words which have been sampled as part of unknown_files and should not be used in the OOV SC set
unknown_words_in_speech_commands = set(unknown_words).intersection(
    set(speech_commands_words)
)
# speech commands words that are oov for the embedding model (they do not show up in the 100 word commands list)
non_embedding_speech_commands = list(
    set(speech_commands_words).difference(set(commands))
    - unknown_words_in_speech_commands
)
print(non_embedding_speech_commands)
#
# #%%
#
# #%%  TRAINING AGAINST SPEECH COMMANDS
# destdir = "/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
# basemodelpath = "/home/mark/tinyspeech_harvard/train_100_augment/hundredword_efficientnet_1600_selu_specaug80.0146-0.8736"
# random_sample_transfer_learn(
#     NUM_MODELS=9,
#     N_SHOTS=5,
#     VAL_UTTERANCES=400,
#     oov_words=non_embedding_speech_commands,
#     dest_dir=destdir,
#     unknown_files=unknown_files,
#     base_model_path=basemodelpath,
#     EPOCHS=6,
#     data_dir=speech_commands,
# )
#
# #%% eval speech commands
#
# scmodeldir = "/home/mark/tinyspeech_harvard/xfer_speechcommands_5/"
# scmodels = [
#     ("xfer_5_shot_6_epochs_marvin_val_acc_0.95", "marvin"),
#     ("xfer_5_shot_6_epochs_left_val_acc_0.87", "left"),
#     ("xfer_5_shot_6_epochs_nine_val_acc_0.94", "nine"),
#     ("xfer_5_shot_6_epochs_zero_val_acc_0.94", "zero"),
#     ("xfer_5_shot_6_epochs_tree_val_acc_0.93", "tree"),
#     ("xfer_5_shot_6_epochs_two_val_acc_0.81", "two"),
#     ("xfer_5_shot_6_epochs_bird_val_acc_0.89", "bird"),
#     ("xfer_5_shot_6_epochs_wow_val_acc_0.90", "wow"),
#     ("xfer_5_shot_6_epochs_six_val_acc_0.95", "six"),
# ]
#
# modelname, target = scmodels[4]
# print(target)
#
# tf.get_logger().setLevel(logging.ERROR)
# scmodel = tf.keras.models.load_model(scmodeldir + modelname)
# tf.get_logger().setLevel(logging.INFO)
#
# target_results = evaluate([target], 2, speech_commands, 1500, scmodel, audio_dataset)
#
# non_target_words = list(set(non_embedding_speech_commands) - {target})
# unknown_sample = np.random.choice(non_target_words, 10, replace=False).tolist()
#
# unknown_results = evaluate(
#     unknown_sample, 1, speech_commands, 1500, scmodel, audio_dataset
# )
#
# #%%
#
#
# #%%
# for results in [target_results, unknown_results]:
#     c = len(results["correct"])
#     w = len(results["incorrect"])
#
#     print(c, w, c / (c + w))
#
# #%%


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


#%%
# fig = go.Figure()
# tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
# fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=target))
#
# fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
# fig.update_xaxes(range=[0, 1])
# fig.update_yaxes(range=[0, 1])
# fig
#
# #%%
# results = []
# for modelname, target in scmodels:
#     print(target)
#     tf.get_logger().setLevel(logging.ERROR)
#     scmodel = tf.keras.models.load_model(scmodeldir + modelname)
#     tf.get_logger().setLevel(logging.INFO)
#
#     target_results = evaluate(
#         [target], 2, speech_commands, 1500, scmodel, audio_dataset
#     )
#
#     non_target_words = list(set(non_embedding_speech_commands) - {target})
#     unknown_sample = np.random.choice(non_target_words, 10, replace=False).tolist()
#
#     unknown_results = evaluate(
#         unknown_sample, 1, speech_commands, 1500, scmodel, audio_dataset
#     )
#     results.append(
#         {
#             "target_results": target_results,
#             "unknown_results": unknown_results,
#             "target": target,
#         }
#     )
#
# #%%
#
#
# def sc_roc_plotly(results: List[Dict]):
#     fig = go.Figure()
#     for ix, res in enumerate(results):
#         target_results = res["target_results"]
#         unknown_results = res["unknown_results"]
#         target = res["target"]
#         tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
#         fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=target))
#
#     fig.update_layout(
#         xaxis_title="FPR",
#         yaxis_title="TPR",
#         title="speech commands classification accuracy",
#     )
#     fig.update_xaxes(range=[0, 1])
#     fig.update_yaxes(range=[0, 1])
#     return fig
#
#
# fig = sc_roc_plotly(results)
# fig.write_html(
#     scmodeldir + "5shot_classification_roc_vs_10_unknown_sampled_from_SC.html"
# )
# fig
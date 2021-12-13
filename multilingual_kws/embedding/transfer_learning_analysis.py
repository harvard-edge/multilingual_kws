#%%
import os
import logging
from typing import Dict, List
import datetime
from pathlib import Path

import glob
import numpy as np
import tensorflow as tf
import pickle

import sys

import input_data

import transfer_learning

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


def roc_single_target(target_results, unknown_results):
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

    tprs, fprs = [], []

    threshs = np.arange(0, 1.01, 0.01)
    for threshold in threshs:
        tpr = target_results[target_results > threshold].shape[0] / total_positives
        tprs.append(tpr)
        fpr = unknown_results[unknown_results > threshold].shape[0] / unknown_total
        fprs.append(fpr)
    return tprs, fprs, threshs


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


#%%
# how many utterances total in dataset
n_utterances = 0
for w in os.listdir(data_dir):
    wavs = glob.glob(str(data_dir / w / "*.wav"))
    n_utterances += len(wavs)
print(n_utterances, len(os.listdir(data_dir)))


#%%
###############################################
##   per language embedding model
###############################################

isocode2model = dict(
    #en="/home/mark/tinyspeech_harvard/train_100_augment/hundredword_efficientnet_1600_selu_specaug80.0150-0.8727",
    #rw="rw_165commands_efficientnet_selu_specaug80_resume93.008-0.7895",
    # nl="nl_165commands_efficientnet_selu_specaug80_resume62_resume08.037-0.7960",
    #es="es_165commands_efficientnet_selu_specaug80_resume34_10_.009-0.8620",
    # de="de_165commands_efficientnet_selu_specaug80_resume93.127-0.8515",
    it="it_165commands_efficientnet_selu_specaug80_resume53.018-0.8208",
)

LANG_ISOCODE = "it"

data_dir = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips/")

# SELECT MODEL
if LANG_ISOCODE == "en":
    traindir = Path("/home/mark/tinyspeech_harvard/train_100_augment/")
    base_model_path = isocode2model[LANG_ISOCODE]
    with open(traindir / "wordlist.txt", "r") as fh:
        commands = fh.read().splitlines()
    words = os.listdir(data_dir)
    other_words = list(set(words).difference(set(commands)))

else:
    traindir = Path(f"/home/mark/tinyspeech_harvard/train_{LANG_ISOCODE}_165/")
    base_model_path = traindir / "models" / isocode2model[LANG_ISOCODE]
    with open(traindir / "commands.txt", "r") as fh:
        commands = fh.read().splitlines()
    with open(traindir / "other_words.txt", "r") as fh:
        other_words = fh.read().splitlines()

model_dest_dir = Path(
    f"/home/mark/tinyspeech_harvard/paper_data/perlang/perlang_{LANG_ISOCODE}"
)


bg_datadir = "/home/mark/tinyspeech_harvard/speech_commands/_background_noise_/"
if not os.path.isdir(bg_datadir):
    raise ValueError("no bg data", bg_datadir)

results_dir = model_dest_dir / "results"

# comment these out if you want to rerun existing models
if os.path.isdir(model_dest_dir):
    raise ValueError("model dir exists", model_dest_dir)
os.makedirs(model_dest_dir)
os.makedirs(model_dest_dir / "models")
os.makedirs(results_dir)
#%%


#%%
# find layer name (optional)
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(base_model_path)
tf.get_logger().setLevel(logging.INFO)
model.summary()

#%%
N_UNKNOWN_WORDS = 50
N_SAMPLES_FROM_UNKNOWN = 70
N_SAMPLES_FROM_COMMANDS = 20

unknown_words = np.random.choice(other_words, N_UNKNOWN_WORDS, replace=False)
oov_words = list(set(other_words).difference(set(unknown_words)))
print("OOV:", oov_words, len(oov_words))

unknown_files = []
for w in unknown_words:
    wavs = glob.glob(str(data_dir / w / "*.wav"))
    if len(wavs) > N_SAMPLES_FROM_UNKNOWN:
        sample = np.random.choice(wavs, N_SAMPLES_FROM_UNKNOWN, replace=False)
    else:
        sample = wavs
    unknown_files.extend(sample)
np.random.shuffle(unknown_files)

print("before including original embeddings", len(unknown_files))

for w in commands:
    wavs = glob.glob(str(data_dir / w / "*.wav"))
    if len(wavs) > N_SAMPLES_FROM_COMMANDS:
        sample = np.random.choice(wavs, N_SAMPLES_FROM_COMMANDS, replace=False)
    else:
        sample = wavs
    unknown_files.extend(sample)
np.random.shuffle(unknown_files)

print("after including original embeddings", len(unknown_files))

# TODO(mmaz): we only sample like 600 [= 128 steps * 9 epochs / 2] of these :(

# %%
# run on existing models
N_SHOTS = 5
for ix, model_file in enumerate(os.listdir(model_dest_dir / "models")):
    raise ValueError("caution - overwrites results")

    start_word = datetime.datetime.now()
    print(f"::::::::::::::::::::{ix} --- {LANG_ISOCODE}::::::::::::::::::::::::::")

    model_path = model_dest_dir / "models" / model_file
    if not os.path.isdir(model_path):
        continue  # probably a training log

    target = model_file.split("_")[-1]
    target_wavs = glob.glob(str(data_dir / target / "*.wav"))
    np.random.shuffle(target_wavs)
    train_files = target_wavs[:N_SHOTS]
    val_files = target_wavs[N_SHOTS:]
    print("\n".join(train_files))

    num_targets = len(os.listdir(data_dir / target))
    print("n targets", num_targets)

    some_commands = np.random.choice(commands, 30, replace=False)
    some_unknown = np.random.choice(unknown_words, 30, replace=False)
    some_oov = np.random.choice(oov_words, 30, replace=False)
    flat = set([w for l in [some_commands, some_unknown, some_oov] for w in l])
    unknown_sample = list(flat.difference({target}))
    print(unknown_sample, len(unknown_sample))

    # load model
    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(model_path)
    tf.get_logger().setLevel(logging.INFO)

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    target_results, all_preds_target = transfer_learning.evaluate_fast_single_target(
        [target], 2, str(data_dir) + "/", num_targets, model, model_settings
    )
    # note: passing in the _TARGET_ category ID (2) for negative examples too:
    # we ignore other categories altogether
    unknown_results, all_preds_unknown = transfer_learning.evaluate_fast_single_target(
        unknown_sample, 2, str(data_dir) + "/", 300, model, model_settings
    )

    # tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
    results = dict(
        target_results=target_results,
        unknown_results=unknown_results,
        all_predictions_targets=all_preds_target,
        all_predictions_unknown=all_preds_unknown,
        # details=details,
        target=target,
        train_files=train_files,
        unknown_words=unknown_words,
        oov_words=oov_words,
        unknown_sample=unknown_sample,
    )
    with open(
        model_dest_dir / "results" / f"result_{LANG_ISOCODE}_{target}.pkl", "wb",
    ) as fh:
        pickle.dump(results, fh)

    tf.keras.backend.clear_session()  # https://keras.io/api/utils/backend_utils/
    end_word = datetime.datetime.now()
    print("time elapsed", end_word - start_word)


#%%
### SELECT TARGET


def results_exist(target, results_dir):
    results = [
        os.path.splitext(r)[0]
        for r in os.listdir(results_dir)
        if os.path.splitext(r)[1] == ".pkl"
    ]
    for r in results:
        result_target = r.split("_")[-1]
        if result_target == target:
            return True
    return False


for ix in range(20):
    start_word = datetime.datetime.now()
    print(f":::::::::::::::::::::::::{ix} --- {LANG_ISOCODE}:::::::::::::::::::::::")
    N_SHOTS = 5
    MAX_TRIES = 100

    has_results = True
    n_tries = 0
    # skip existing results
    while has_results and n_tries < MAX_TRIES:
        target = np.random.choice(oov_words)
        has_results = results_exist(target, results_dir)
        n_tries += 1
    if n_tries >= MAX_TRIES:
        raise ValueError("not enough oov words left to choose from")
    print("TARGET:", target)
    target_wavs = glob.glob(str(data_dir / target / "*.wav"))
    np.random.shuffle(target_wavs)
    train_files = target_wavs[:N_SHOTS]
    val_files = target_wavs[N_SHOTS:]
    print("\n".join(train_files))

    num_targets = len(os.listdir(data_dir / target))
    print("n targets", num_targets)

    ###############
    ## FINETUNING
    ###############

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    csvlog_dest = str(model_dest_dir / "models" / f"{target}_trainlog.csv")
    assert not os.path.isfile(csvlog_dest), f"{csvlog_dest} csvlog already exists"

    name, model, details = transfer_learning.transfer_learn(
        target=target,
        train_files=train_files,
        val_files=val_files,
        unknown_files=unknown_files,
        num_epochs=4,
        num_batches=1,
        batch_size=64,
        model_settings=model_settings,
        csvlog_dest=csvlog_dest,
        base_model_path=base_model_path,
        base_model_output="dense_2",
    )
    print("saving", name)
    model.save(model_dest_dir / "models" / name)

    some_commands = np.random.choice(commands, 30, replace=False)
    some_unknown = np.random.choice(unknown_words, 30, replace=False)
    some_oov = np.random.choice(oov_words, 30, replace=False)
    flat = set([w for l in [some_commands, some_unknown, some_oov] for w in l])
    unknown_sample = list(flat.difference({target}))
    print(unknown_sample, len(unknown_sample))

    target_results, all_preds_target = transfer_learning.evaluate_fast_single_target(
        [target], 2, str(data_dir) + "/", num_targets, model, model_settings
    )
    # note: passing in the _TARGET_ category ID (2) for negative examples too:
    # we ignore other categories altogether
    unknown_results, all_preds_unknown = transfer_learning.evaluate_fast_single_target(
        unknown_sample, 2, str(data_dir) + "/", 300, model, model_settings
    )

    # tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
    results = dict(
        target_results=target_results,
        unknown_results=unknown_results,
        all_predictions_targets=all_preds_target,
        all_predictions_unknown=all_preds_unknown,
        details=details,
        target=target,
        train_files=train_files,
        unknown_words=unknown_words,
        oov_words=oov_words,
    )
    result_file = model_dest_dir / "results" / f"result_{LANG_ISOCODE}_{target}.pkl"
    assert not os.path.isfile(result_file), f"{result_file} already exists"
    with open(result_file, "wb",) as fh:
        pickle.dump(results, fh)

    tf.keras.backend.clear_session()  # https://keras.io/api/utils/backend_utils/
    end_word = datetime.datetime.now()
    print("time elapsed", end_word - start_word)

#%%


def sc_roc_plotly(results: List[Dict]):
    fig = go.Figure()
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        ne = res["details"]["num_epochs"]
        nb = res["details"]["num_batches"]
        target = res["target"]
        curve_label = f"{target} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
        fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=curve_label))

    fig.update_layout(
        xaxis_title="FPR",
        yaxis_title="TPR",
        title=f"{LANG_ISOCODE} 5-shot classification accuracy",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


results = []
for pkl_file in os.listdir(model_dest_dir / "results"):
    filename = model_dest_dir / "results" / pkl_file
    print(filename)
    with open(filename, "rb") as fh:
        result = pickle.load(fh)
        results.append(result)
print("N words", len(results))
fig = sc_roc_plotly(results)
dest_plot = str(model_dest_dir / f"5shot_classification_roc_{LANG_ISOCODE}.html")
print("saving to", dest_plot)
fig.write_html(dest_plot)
fig


#%%
# how many utterances total in dataset
n_utterances = 0
for w in os.listdir(data_dir):
    wavs = glob.glob(str(data_dir / w / "*.wav"))
    n_utterances += len(wavs)
print(n_utterances, len(os.listdir(data_dir)))

# %%
# how many oov words are saved:
uc = "/home/mark/tinyspeech_harvard/multilang_analysis_ooe_v2/unknown_collection.pkl"
with open(uc, "rb") as fh:
    unknown_collection = pickle.load(fh)
# %%
oov_lang_words = unknown_collection["oov_lang_words"]
iso2lang = {
    "en": "English",
    "fr": "French",
    "ca": "Catalan",
    "rw": "Kinyarwanda",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "fa": "Persian",
    "es": "Spanish",
}
counts = {l: 0 for l in iso2lang.keys()}
for l, w in oov_lang_words:
    if l not in counts.keys():
        continue
    counts[l] += 1
print(counts)


#%%
###############################################
##               MULTILANG
###############################################

traindir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding")

# SELECT MODEL
base_model_path = (
    traindir / "models" / "multilang_resume40_resume05_resume20.022-0.7969"
)

LANG_ISOCODE = "de"
model_dest_dir = Path(
    f"/home/mark/tinyspeech_harvard/paper_data/multilang_class/multilang_{LANG_ISOCODE}"
)
if os.path.isdir(model_dest_dir):
    raise ValueError("model dir exists", model_dest_dir)
os.makedirs(model_dest_dir)
os.makedirs(model_dest_dir / "models")
results_dir = model_dest_dir / "results"
os.makedirs(results_dir)

with open(traindir / "commands.txt", "r") as fh:
    commands = fh.read().splitlines()

bg_datadir = "/home/mark/tinyspeech_harvard/speech_commands/_background_noise_/"
if not os.path.isdir(bg_datadir):
    raise ValueError("no bg data", bg_datadir)

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words/")
#%%
# find layer name (optional)
tf.get_logger().setLevel(logging.ERROR)
model = tf.keras.models.load_model(base_model_path)
tf.get_logger().setLevel(logging.INFO)
model.summary()

#%%
# select unknown words for finetuning
N_UNKNOWN_WORDS = 300
N_UNKNOWN_UTTERANCES_PER_WORD = 10
N_ORIGINAL_EMBEDDING_AS_UNKNOWN = 200

unknown_collection_path = model_dest_dir / "unknown_collection.pkl"
if os.path.isfile(unknown_collection_path):
    raise ValueError("already exists", unknown_collection_path)

# only sample unkonwn words from languages already in multilingual embedding model
# otherwise eval would be unfair (finetuned model might see more than just
# the 5shot examples, i.e., unknown words in target language )
language_isocodes = os.listdir(frequent_words)
lang_isocodes_for_unknown_words = [
    li
    for li in language_isocodes
    if li in ["fa", "de", "ca", "it", "en", "fr", "es", "nl", "rw"]
]

lang2words = {}
for lang in lang_isocodes_for_unknown_words:
    clips = frequent_words / lang / "clips"
    words = os.listdir(clips)
    lang2words[lang] = words

print("total words", sum([len(v) for v in lang2words.values()]))
print("total commands", len(commands))

command_set = set(commands)
unknown_lang_words = []
while len(unknown_lang_words) < N_UNKNOWN_WORDS:
    lang = np.random.choice(list(lang2words.keys()))
    word = np.random.choice(os.listdir(frequent_words / lang / "clips"))
    if word not in command_set:
        unknown_lang_words.append((lang, word))
print(len(unknown_lang_words))
unknown_word_set = set([w for l, w in unknown_lang_words])

unknown_files = []
for lang, word in unknown_lang_words:
    wavs = glob.glob(str(frequent_words / lang / "clips" / word / "*.wav"))
    if len(wavs) > N_UNKNOWN_UTTERANCES_PER_WORD:
        utterances = np.random.choice(
            wavs, N_UNKNOWN_UTTERANCES_PER_WORD, replace=False
        )
        unknown_files.extend(utterances)
    else:
        unknown_files.extend(wavs)
print("unknown files before original embeddings added", len(unknown_files))
print(unknown_files[0])

# include from original embeddings
original_as_unknown = np.random.choice(
    commands, N_ORIGINAL_EMBEDDING_AS_UNKNOWN, replace=False
)
for word in original_as_unknown:
    # which language?
    for lang in lang_isocodes_for_unknown_words:
        word_dir = frequent_words / lang / "clips" / word
        if os.path.isdir(word_dir):
            break
    wavs = glob.glob(str(word_dir / "*.wav"))
    if len(wavs) > N_UNKNOWN_UTTERANCES_PER_WORD:
        utterances = np.random.choice(
            wavs, N_UNKNOWN_UTTERANCES_PER_WORD, replace=False
        )
        unknown_files.extend(utterances)
    else:
        unknown_files.extend(wavs)

np.random.shuffle(unknown_files)
print("unknown files after original embeddings added", len(unknown_files))
print(unknown_files[0])
# TODO(mmaz): we only sample like 600 [= 128 steps * 9 epochs / 2] of these :(

# search through all languages (both embedding langs and non-embedding langs) for oov words
oov_lang_words = []
for lang in language_isocodes:
    clips = frequent_words / lang / "clips"
    words = os.listdir(clips)
    for word in words:
        if word in command_set or word in unknown_word_set:
            continue
        oov_lang_words.append((lang, word))
print("oov lang words", len(oov_lang_words))

unknown_collection = dict(
    unknown_lang_words=unknown_lang_words,
    unknown_files=unknown_files,
    oov_lang_words=oov_lang_words,
    commands=commands,
    lang_isocodes_for_unknown_words=lang_isocodes_for_unknown_words,
)
assert not os.path.isfile(unknown_collection_path)
with open(unknown_collection_path, "wb") as fh:
    pickle.dump(unknown_collection, fh)

#%%

unknown_collection_path = model_dest_dir / "unknown_collection.pkl"
with open(unknown_collection_path, "rb") as fh:
    unknown_collection = pickle.load(fh)
unknown_lang_words = unknown_collection["unknown_lang_words"]
unknown_files = unknown_collection["unknown_files"]
oov_lang_words = unknown_collection["oov_lang_words"]

#%%
# select unknown/oov/command words for model evaluation (not finetuning)


def generate_random_non_target_files_for_eval(
    target_word,
    unknown_lang_words,
    oov_lang_words,  # these should prob be in the language being evaluated
    commands,
    frequent_words,
    n_non_targets_per_category=100,
    n_utterances_per_non_target=80,
):
    # select unknown/oov/command words for model evaluation (**not for finetuning**)

    # 3 categories: unknown, oov, command
    some_unknown_ixs = np.random.choice(
        range(len(unknown_lang_words)), n_non_targets_per_category, replace=False
    )
    some_lang_unknown = np.array(unknown_lang_words)[some_unknown_ixs]
    some_oov_ixs = np.random.choice(
        range(len(oov_lang_words)), n_non_targets_per_category, replace=False
    )
    some_lang_oov = np.array(oov_lang_words)[some_oov_ixs]
    some_commands = np.random.choice(
        commands, n_non_targets_per_category, replace=False
    )
    some_lang_commands = []
    for c in some_commands:
        for lang in os.listdir(frequent_words):
            if os.path.isdir(frequent_words / lang / "clips" / c):
                some_lang_commands.append((lang, c))
                break

    # filter out target word if present
    flat = [
        (lang, word)
        for lw in [some_lang_unknown, some_lang_oov, some_lang_commands]
        for lang, word in lw
        if word != target_word
    ]
    non_target_files = []
    for lang, word in flat:
        word_dir = frequent_words / lang / "clips" / word
        wavs = glob.glob(str(word_dir / "*.wav"))
        if len(wavs) > n_utterances_per_non_target:
            sample = np.random.choice(wavs, n_utterances_per_non_target, replace=False)
            non_target_files.extend(sample)
        else:
            non_target_files.extend(wavs)
    return non_target_files


# %%

# %%
# run on existing models

for model_file in os.listdir(model_dest_dir / "models"):
    raise ValueError("caution - overwrites results")

    target_word = model_file.split("_")[-1]
    target_lang = None
    for lang, word in oov_lang_words:
        if word == target_word:
            target_lang = lang
            break
    print(model_file, "\t target word:", target_word, "\t target lang:", target_lang)

    target_wavs = glob.glob(
        str(frequent_words / target_lang / "clips" / target_word / "*.wav")
    )
    if len(target_wavs) == 0:
        print(str(frequent_words / target_lang / "clips" / target_word / "*.wav"))
        raise ValueError("bug in word search due to unicode issues in", target_word)

    # load model
    model_path = model_dest_dir / "models" / model_file

    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(model_path)
    tf.get_logger().setLevel(logging.INFO)

    # get non-target (negative) examples
    unknown_sample = generate_random_non_target_files(
        target_word=target_word,
        unknown_lang_words=unknown_lang_words,
        oov_lang_words=oov_lang_words,
        commands=commands,
    )
    print("num unknown sample", len(unknown_sample))

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    target_results = transfer_learning.evaluate_files_single_target(
        target_wavs, 2, model, model_settings
    )
    # note: passing in the _TARGET_ category ID (2) for negative examples too:
    # we ignore other categories altogether
    unknown_results = transfer_learning.evaluate_files_single_target(
        unknown_sample, 2, model, model_settings
    )

    # tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
    results = dict(
        target_results=target_results,
        unknown_results=unknown_results,
        # details=details,
        target_word=target_word,
        target_lang=target_lang,
        # train_files=train_files,
        unknown_words=unknown_lang_words,
        oov_words=oov_lang_words,
        commands=commands,
    )
    with open(
        model_dest_dir / "results" / f"hpsweep_{target_lang}_{target_word}.pkl", "wb",
    ) as fh:
        pickle.dump(results, fh)

    tf.keras.backend.clear_session()  # https://keras.io/api/utils/backend_utils/

#%%
### SELECT TARGET


def results_exist(target, results_dir):
    results = [
        os.path.splitext(r)[0]
        for r in os.listdir(results_dir)
        if os.path.splitext(r)[1] == ".pkl"
    ]
    for r in results:
        result_target = r.split("_")[-1]
        if result_target == target:
            return True
    return False


for ix in range(20):
    start_word = datetime.datetime.now()
    print(f":::::::::::::::::::::::::{ix} - {LANG_ISOCODE}:::::::::::::::::::::::::")

    has_results = True
    # skip existing results
    while has_results:
        rand_ix = np.random.randint(len(oov_lang_words))
        target_lang, target_word = oov_lang_words[rand_ix]
        # if target_lang not in ["cs", "cy", "eu"]:  # generate for a specific language
        if target_lang not in [LANG_ISOCODE]:  # generate for a specific language
            continue
        has_results = results_exist(target_word, results_dir)
    print("TARGET:", target_lang, target_word)
    target_wavs = glob.glob(
        str(frequent_words / target_lang / "clips" / target_word / "*.wav")
    )
    if len(target_wavs) == 0:
        print(str(frequent_words / target_lang / "clips" / target_word / "*.wav"))
        raise ValueError("bug in word search due to unicode issues in", target_word)
    np.random.shuffle(target_wavs)
    train_files = target_wavs[:N_SHOTS]
    val_files = target_wavs[N_SHOTS:]
    print("\n".join(train_files))

    num_targets = len(target_wavs)
    print("n targets", num_targets)

    ###############
    ## FINETUNING
    ###############

    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    csvlog_dest = str(
        model_dest_dir / "models" / f"{target_word}_{target_lang}_trainlog.csv"
    )
    if os.path.isfile(csvlog_dest):
        raise ValueError("overwriting existing file")

    # TODO(mmaz): use keras class weights?
    name, model, details = transfer_learning.transfer_learn(
        target=target_word,
        train_files=train_files,
        val_files=val_files,
        unknown_files=unknown_files,
        num_epochs=4,
        num_batches=1,
        batch_size=64,
        model_settings=model_settings,
        csvlog_dest=csvlog_dest,
        base_model_path=base_model_path,
        base_model_output="dense_2",
    )
    print("saving", name)
    model.save(model_dest_dir / "models" / name)

    unknown_sample = generate_random_non_target_files_for_eval(
        target_word=target_word,
        unknown_lang_words=unknown_lang_words,
        oov_lang_words=oov_lang_words,
        commands=commands,
        frequent_words=frequent_words,
    )
    print("num unknown sample", len(unknown_sample))

    target_results, all_preds_target = transfer_learning.evaluate_files_single_target(
        target_wavs, 2, model, model_settings
    )
    # note: passing in the _TARGET_ category ID (2) for negative examples too:
    # we ignore other categories altogether
    unknown_results, all_preds_unknown = transfer_learning.evaluate_files_single_target(
        unknown_sample, 2, model, model_settings
    )

    # tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
    results = dict(
        target_results=target_results,
        unknown_results=unknown_results,
        all_predictions_targets=all_preds_target,
        all_predictions_unknown=all_preds_unknown,
        details=details,
        target_word=target_word,
        target_lang=target_lang,
        train_files=train_files,
        unknown_words=unknown_lang_words,
        oov_words=oov_lang_words,
        commands=commands,
    )
    with open(
        model_dest_dir / "results" / f"hpsweep_{target_lang}_{target_word}.pkl", "wb",
    ) as fh:
        pickle.dump(results, fh)

    tf.keras.backend.clear_session()  # https://keras.io/api/utils/backend_utils/
    end_word = datetime.datetime.now()
    print("elapsed time", end_word - start_word)

#%%

# model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis/")
model_dest_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis_ooe/")
if not os.path.isdir(model_dest_dir):
    raise ValueError("no model dir", model_dest_dir)
results_dir = model_dest_dir / "results"
if not os.path.isdir(results_dir):
    raise ValueError("no results dir", results_dir)


def sc_roc_plotly(results: List[Dict]):
    fig = go.Figure()
    for ix, res in enumerate(results):
        target_results = res["target_results"]
        unknown_results = res["unknown_results"]
        ne = res["details"]["num_epochs"]
        nb = res["details"]["num_batches"]
        target_word = res["target_word"]
        target_lang = res["target_lang"]
        curve_label = f"{target_lang} {target_word} (e:{ne},b:{nb})"
        # curve_label=target
        tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
        fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=curve_label))

    fig.update_layout(
        xaxis_title="FPR", yaxis_title="TPR", title=f"5-shot classification accuracy",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


results = []
for pkl_file in os.listdir(model_dest_dir / "results"):
    filename = model_dest_dir / "results" / pkl_file
    print(filename)
    with open(filename, "rb") as fh:
        result = pickle.load(fh)
        results.append(result)
print("N words", len(results))
fig = sc_roc_plotly(results)
dest_plot = str(model_dest_dir / f"5shot_classification_roc.html")
print("saving to", dest_plot)
fig.write_html(dest_plot)
fig


# %%

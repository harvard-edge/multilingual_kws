#%%
import os
import logging
from typing import Dict, List

import glob
import numpy as np
import tensorflow as tf
import pickle

import sys

sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
import input_data


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1)


def evaluate(
    words_to_evaluate: List[str],
    target_id: int,
    data_dir: os.PathLike,
    utterances_per_word: int,
    model: tf.keras.Model,
    audio_dataset: input_data.AudioDataset,
):
    correct_confidences = []
    incorrect_confidences = []
    track_correct = {}
    track_incorrect = {}

    for word in words_to_evaluate:
        fs = np.random.choice(
            glob.glob(data_dir + word + "/*.wav"), utterances_per_word, replace=False
        )

        track_correct[word] = []
        track_incorrect[word] = []

        specs = np.array([audio_dataset.file2spec(f) for f in fs])
        preds = model.predict(np.expand_dims(specs, -1))

        # softmaxes = np.max(preds,axis=1)
        # unknown_other_words_confidences.extend(softmaxes.tolist())
        cols = np.argmax(preds, axis=1)
        # figure out how to fancy-index this later
        for row, col in enumerate(cols):
            confidence = preds[row][col]
            if col == target_id:
                correct_confidences.append(confidence)
                track_correct[word].append(confidence)
            else:
                incorrect_confidences.append(confidence)
                track_incorrect[word].append(confidence)
    return {
        "correct": correct_confidences,
        "incorrect": incorrect_confidences,
        "track_correct": track_correct,
        "track_incorrect": track_incorrect,
    }


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
    assert (len(model_commands) == 1, "need to refactor for multiple commands")
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
    false_positives = np.concatenate([oov_incorrect, unknown_incorrect, original_incorrect])

    # target_tprs, target_fprs = [], []
    # oov_tprs, oov_fprs = [],[]
    # unknown_tprs, unknown_fprs = [],[]
    # original_tprs, original_fprs = [], []
    tprs, fprs = [], []

    threshs = np.arange(0, 1, 0.01)
    for threshold in threshs:
        tpr = target_correct[target_correct > threshold].shape[0] / total_positives
        tprs.append(tpr)
        fpr = false_positives[false_positives > threshold].shape[0] / total_negatives
        fprs.append(fpr)
    return tprs,fprs

def make_roc_plotly(results: List[Dict]):
    fig = go.Figure()
    for ix, res in enumerate(results):
        tprs, fprs = calc_roc(res)

        v = res["val_acc"]
        title = ", ".join(res["words"]) + f" (val acc {v})"

        labels=np.arange(0, 1, 0.01)
        fig.add_trace(go.Scatter(x=fprs, y=tprs, text=labels, name=title))

    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
    fig.update_xaxes(range=[0,1])
    fig.update_yaxes(range=[0,1])
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
        #ax.legend(loc="lower right")
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


#%%
res = list(rc.values())[0]
threshold = 0.5
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
total_fpr = all_incorrect[all_incorrect > threshold].shape[0] / total_predictions
total_unknown = sum(
    [
        len(res[k]["correct"]) + len(res[k]["incorrect"])
        for k in ["oov", "unknown_training", "original_embedding"]
    ]
)
fpr_unknown = (
    np.where(np.concatenate([owcs, utcs, oecs]) > threshold)[0].shape[0] / total_unknown
)
print(total_fpr, fpr_unknown)
print(np.concatenate([owcs, utcs, oecs]).shape[0])


#%%
ms = os.listdir("/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/")
ms = list(filter(lambda f: not f.endswith(".pkl"), ms))

#%%
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

#%%
other_words = [
    w for w in os.listdir(data_dir) if w != "_background_noise_" and w not in commands
]
other_words.sort()
print(len(other_words))
assert len(set(other_words).intersection(commands)) == 0

#%%
model_settings = input_data.prepare_model_settings(
    label_count=100,
    sample_rate=16000,
    clip_duration_ms=1000,
    window_size_ms=30,
    window_stride_ms=20,
    feature_bin_count=40,
    preprocess="micro",
)
bg_datadir = "/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/"
audio_dataset = input_data.AudioDataset(
    model_settings,
    ["NONE"],
    bg_datadir,
    unknown_files=[],
    unknown_percentage=0,
    spec_aug_params=input_data.SpecAugParams(percentage=0),
)
#%%

rc = {}
print(len(ms))
for m in ms:
    epochs = "_epochs_"
    start = m.find(epochs) + len(epochs)
    valacc_str = "_val_acc_"
    last_word = m.find(valacc_str)
    n_word_xfer = m[start:last_word].split("_")
    print(n_word_xfer)
    valacc_idx = last_word + len(valacc_str)
    valacc = float(m[valacc_idx:])
    print(valacc)

    mp = "/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/" + m
    print(mp)

    rc[m] = analyze_model(
        mp,
        n_word_xfer,
        valacc,
        data_dir,
        audio_dataset,
        unknown_words,
        oov_words,
        commands,
    )
#%%
# with open("/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/results.pkl", 'wb') as fh:
#    pickle.dump(rc, fh)
with open(
    "/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/results.pkl", "rb"
) as fh:
    rc = pickle.load(fh)

#%%
fig, axes = make_viz(rc.values(), threshold=0.5, nrows=3, ncols=3)
fig.set_size_inches(25, 25)

#%%
fig, axes = make_roc(rc.values(), nrows=3, ncols=3)
fig.set_size_inches(25, 25)

#%%
fig=make_roc_plotly(rc.values())
fig
#%%
fig.write_html("/home/mark/tinyspeech_harvard/xfer_oov_efficientnet_binary/roc.html")
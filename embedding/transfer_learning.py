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
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1)

#%%

tf.config.list_physical_devices("GPU")

#%% training


def transfer_learn(
    dest_dir,
    target,
    train_files,
    val_files,
    unknown_files,
    EPOCHS,
    base_model_path: os.PathLike,
    base_model_output="dense_2",
    UNKNOWN_PERCENTAGE=50.0,
    NUM_BATCHES=1,
    batch_size=64,
    name_prefix="",
    bg_datadir="/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/",
):
    assert os.path.isdir(dest_dir), f"dest dir {dest_dir} not found"

    tf.get_logger().setLevel(logging.ERROR)
    base_model = tf.keras.models.load_model(base_model_path)
    tf.get_logger().setLevel(logging.INFO)
    xfer = tf.keras.models.Model(
        name="TransferLearnedModel",
        inputs=base_model.inputs,
        outputs=base_model.get_layer(name=base_model_output).output,
    )
    xfer.trainable = False

    # dont use softmax unless losses from_logits=False
    CATEGORIES = 3  # silence + unknown + target_keyword
    xfer = tf.keras.models.Sequential(
        [xfer, tf.keras.layers.Dense(units=CATEGORIES, activation="softmax")]
    )

    xfer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    model_settings = input_data.prepare_model_settings(
        label_count=100,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=20,
        feature_bin_count=40,
        preprocess="micro",
    )

    a = input_data.AudioDataset(
        model_settings=model_settings,
        commands=[target],
        background_data_dir=bg_datadir,
        unknown_files=unknown_files,
        unknown_percentage=UNKNOWN_PERCENTAGE,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = a.init(AUTOTUNE, train_files, is_training=True)
    val_ds = a.init(AUTOTUNE, val_files, is_training=False)
    # test_ds = a.init(AUTOTUNE, test_files, is_training=False)
    train_ds = train_ds.shuffle(buffer_size=1000).repeat().batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    history = xfer.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=batch_size * NUM_BATCHES,
        epochs=EPOCHS,
    )

    va = history.history["val_accuracy"][-1]
    if name_prefix != "":
        name_prefix = name_prefix + "_"
    name = f"{name_prefix}xfer_epochs_{EPOCHS}_bs_{batch_size}_nbs_{NUM_BATCHES}_val_acc_{va:0.2f}_target_{target}"
    print("saving model", name)
    xfer.save(dest_dir + name)
    details = {
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "num_batches": NUM_BATCHES,
        "val accuracy": va,
        "target": target,
    }
    return name, xfer, details


def random_sample_transfer_models(
    NUM_MODELS,
    N_SHOTS,
    VAL_UTTERANCES,
    oov_words,
    dest_dir,
    unknown_files,
    EPOCHS,
    data_dir,
    base_model_path: os.PathLike,
    base_model_output="dense_2",
    UNKNOWN_PERCENTAGE=50.0,
    NUM_BATCHES=1,
    bg_datadir="/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/",
):
    assert os.path.isdir(dest_dir), f"dest dir {dest_dir} not found"
    models = np.random.choice(oov_words, NUM_MODELS, replace=False)

    for target in models:
        wavs = glob.glob(data_dir + target + "/*.wav")
        selected = np.random.choice(wavs, N_SHOTS + VAL_UTTERANCES, replace=False)

        train_files = selected[:N_SHOTS]
        np.random.shuffle(train_files)
        val_files = selected[N_SHOTS:]

        print(len(train_files), "shot:", target)

        utterances_fn = target + "_utterances.txt"
        utterances = dest_dir + utterances_fn
        print("saving", utterances)
        with open(utterances, "w") as fh:
            fh.write("\n".join(train_files))

        transfer_learn(
            dest_dir=dest_dir,
            target=target,
            train_files=train_files,
            val_files=val_files,
            unknown_files=unknown_files,
            EPOCHS=EPOCHS,
            base_model_path=base_model_path,
            base_model_output=base_model_output,
            UNKNOWN_PERCENTAGE=UNKNOWN_PERCENTAGE,
            NUM_BATCHES=NUM_BATCHES,
            bg_datadir=bg_datadir,
        )


#%% analysis


def evaluate_fast(
    words_to_evaluate: List[str],
    target_id: int,
    data_dir: os.PathLike,
    utterances_per_word: int,
    model: tf.keras.Model,
    audio_dataset: input_data.AudioDataset,
):
    correct_confidences = []
    incorrect_confidences = []

    specs = []
    for word in words_to_evaluate:
        fs = np.random.choice(
            glob.glob(data_dir + word + "/*.wav"), utterances_per_word, replace=False
        )
        specs.extend([audio_dataset.file2spec(f) for f in fs])
    specs = np.array(specs)
    preds = model.predict(np.expand_dims(specs, -1))

    # softmaxes = np.max(preds,axis=1)
    # unknown_other_words_confidences.extend(softmaxes.tolist())
    cols = np.argmax(preds, axis=1)
    # figure out how to fancy-index this later
    for row, col in enumerate(cols):
        confidence = preds[row][col]
        if col == target_id:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    return {
        "correct": correct_confidences,
        "incorrect": incorrect_confidences,
    }


def evaluate_and_track(
    words_to_evaluate: List[str],
    target_id: int,
    data_dir: os.PathLike,
    utterances_per_word: int,
    model: tf.keras.Model,
    audio_dataset: input_data.AudioDataset,
):
    #TODO(mmaz) rewrite and combine with evaluate_fast

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


#%%
####################################################
##   HYPERPARAMETER OPTIMIZATION ON SPEECH COMMANDS
####################################################

#%%  generate dataset

# bird left marvin nine six tree two wow zero
# destdir = "/home/mark/tinyspeech_harvard/hyperparam_analysis/"
# speech_commands = "/home/mark/tinyspeech_harvard/speech_commands/"

# target = "two"
# wavs = glob.glob(speech_commands + target + "/*.wav")
# N_SHOTS = 5
# VAL_UTTERANCES = 400
# selected = np.random.choice(wavs, N_SHOTS + VAL_UTTERANCES, replace=False)

# train_files = selected[:N_SHOTS]
# np.random.shuffle(train_files)
# val_files = selected[N_SHOTS:]
# print(train_files)


#%% audio dataset

def create_empty_audio_dataset():
    # TODO(mmaz): this is usually just used for file2spec, unnecessary codebloat
    model_settings = input_data.prepare_model_settings(
        label_count=100,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=20,
        feature_bin_count=40,
        preprocess="micro",
    )
    bg_datadir = (
        "/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/"
    )
    audio_dataset = input_data.AudioDataset(
        model_settings,
        ["NONE"],
        bg_datadir,
        unknown_files=[],
        unknown_percentage=0,
        spec_aug_params=input_data.SpecAugParams(percentage=0),
    )
    return audio_dataset


#%%
target = "two"
two_utterances = [
    "/home/mark/tinyspeech_harvard/speech_commands/two/b83c1acf_nohash_2.wav",
    "/home/mark/tinyspeech_harvard/speech_commands/two/a55105d0_nohash_2.wav",
    "/home/mark/tinyspeech_harvard/speech_commands/two/067f61e2_nohash_3.wav",
    "/home/mark/tinyspeech_harvard/speech_commands/two/ce7a8e92_nohash_1.wav",
    "/home/mark/tinyspeech_harvard/speech_commands/two/e4be0cf6_nohash_4.wav",
]
all_twos = glob.glob(speech_commands + "two" + "/*.wav")
print(len(all_twos))
non_utterances = list(set(all_twos) - set(two_utterances))
print(len(non_utterances))
val_twos = np.random.choice(non_utterances, 400, replace=False)
print(len(val_twos))

non_target_words = list(set(non_embedding_speech_commands) - {target})
print(non_target_words)
unknown_sample = np.random.choice(non_target_words, 24, replace=False).tolist()
print("UNKNOWN_SAMPLE:")
print(unknown_sample)


#%% train
speech_commands = "/home/mark/tinyspeech_harvard/speech_commands/"
destdir = "/home/mark/tinyspeech_harvard/hyperparam_analysis/"
modeldestdir = destdir + "models/"
basemodelpath = "/home/mark/tinyspeech_harvard/train_100_augment/hundredword_efficientnet_1600_selu_specaug80.0146-0.8736"

with open(destdir + "unknown_words.txt", "w") as fh:
    fh.write("\n".join(unknown_sample))

train_files = two_utterances
val_files = val_twos

n_models = 0
for epochs in range(1, 10):
    for n_batches in range(1, 4):
        for batch_size in [32, 64]:
            for trial in range(1, 4):
                n_models += 1
print("N MODELS", n_models)

results = {}
ix = 0
for epochs in range(1, 10):
    for n_batches in range(1, 4):
        for batch_size in [32, 64]:
            for trial in range(1, 4):
                ix += 1
                # TODO(mmaz): workaround for memory leak (yikes)
                if ix <= 144:
                    continue
                print(
                    f"::::: {ix}: epochs {epochs} nb {n_batches} bs {batch_size} t {trial}"
                )

                # https://keras.io/api/utils/backend_utils/
                tf.keras.backend.clear_session()

                start = datetime.datetime.now()
                modelname, model, details = transfer_learn(
                    dest_dir=modeldestdir,
                    target=target,
                    train_files=train_files,
                    val_files=val_files,
                    unknown_files=unknown_files,
                    EPOCHS=epochs,
                    base_model_path=basemodelpath,
                    NUM_BATCHES=n_batches,
                    batch_size=batch_size,
                    name_prefix=f"trial{trial}",
                )

                audio_dataset = create_empty_audio_dataset()
                target_results = evaluate_fast(
                    [target], 2, speech_commands, 1500, model, audio_dataset
                )
                unknown_results = evaluate_fast(
                    unknown_sample, 1, speech_commands, 600, model, audio_dataset
                )

                # tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
                results[modelname] = {
                    "ix": ix,
                    "target": target,
                    "epochs": epochs,
                    "n_batches": n_batches,
                    "trial": trial,
                    "target_results": target_results,
                    "unknown_results": unknown_results,
                    "details": details,
                }
                with open(destdir + f"results/hpsweep_{ix:03d}.pkl", "wb") as fh:
                    pickle.dump(results, fh)

                end = datetime.datetime.now()
                nicedelta = str(end-start)[:-5]
                print(f":::::::::: DONE {ix}/{n_models} --- {nicedelta}")


#%%

# with open(destdir + "hyperparamsweep.pkl", "rb") as fh:
#     results = pickle.load(fh)

#%%

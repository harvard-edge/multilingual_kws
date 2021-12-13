#%%
import logging
import os
import re
from typing import Dict, List
import datetime
from dataclasses import asdict, dataclass

import glob
import numpy as np
import pickle
import multiprocessing

import sys

import input_data

import transfer_learning

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns

sns.set()
sns.set_palette("bright")

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

print(
    len(commands), len(unknown_words), len(oov_words),
)

other_words = [
    w for w in os.listdir(data_dir) if w != "_background_noise_" and w not in commands
]
other_words.sort()
print(len(other_words))
assert len(set(other_words).intersection(commands)) == 0

####################################################
##   HYPERPARAMETER OPTIMIZATION ON SPEECH COMMANDS
####################################################
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

target = "two"
# two_utterances = [
#     "/home/mark/tinyspeech_harvard/speech_commands/two/b83c1acf_nohash_2.wav",
#     "/home/mark/tinyspeech_harvard/speech_commands/two/a55105d0_nohash_2.wav",
#     "/home/mark/tinyspeech_harvard/speech_commands/two/067f61e2_nohash_3.wav",
#     "/home/mark/tinyspeech_harvard/speech_commands/two/ce7a8e92_nohash_1.wav",
#     "/home/mark/tinyspeech_harvard/speech_commands/two/e4be0cf6_nohash_4.wav",
# ]
# all_twos = glob.glob(speech_commands + "two" + "/*.wav")
# print(len(all_twos))
# non_utterances = list(set(all_twos) - set(two_utterances))
# print(len(non_utterances))
# val_twos = np.random.choice(non_utterances, 400, replace=False)
# print(len(val_twos))
non_target_words = list(set(non_embedding_speech_commands) - {target})
print(non_target_words)
unknown_sample = np.random.choice(non_target_words, 24, replace=False).tolist()
print("UNKNOWN_SAMPLE:")
print(unknown_sample)

#%%
model_settings = input_data.standard_microspeech_model_settings(3)

#%%


@dataclass
class RunTransferLearning:
    dest_dir: os.PathLike
    ix: int
    trial: int
    target_set: int
    target: str
    train_files: List[str]
    val_files: List[str]
    # words to sample from the data_dir as unknown when evaluating
    unknown_sample: List[str]
    # paths of utterances to use as unknown when training
    unknown_files: List[os.PathLike]
    num_epochs: int
    num_batches: int
    batch_size: int
    model_settings: Dict
    data_dir: os.PathLike
    base_model_path: os.PathLike
    base_model_output: str


def run_transfer_learning(rtl: RunTransferLearning):
    import tensorflow as tf

    name, model, details = transfer_learning.transfer_learn(
        target=rtl.target,
        train_files=rtl.train_files,
        val_files=rtl.val_files,
        unknown_files=rtl.unknown_files,
        num_epochs=rtl.num_epochs,
        num_batches=rtl.num_batches,
        batch_size=rtl.batch_size,
        model_settings=rtl.model_settings,
        base_model_path=rtl.base_model_path,
        base_model_output=rtl.base_model_output,
    )

    save_dest = (
        rtl.dest_dir
        + os.path.sep
        + "models"
        + os.path.sep
        + f"targetset{rtl.target_set}_trial{rtl.trial}__{name}"
    )
    print("SAVING", save_dest)
    model.save(save_dest)

    target_results = transfer_learning.evaluate_fast(
        [rtl.target], 2, rtl.data_dir, 1500, model, rtl.model_settings
    )
    unknown_results = transfer_learning.evaluate_fast(
        rtl.unknown_sample, 1, rtl.data_dir, 600, model, rtl.model_settings
    )

    # tprs, fprs, thresh_labels = roc_sc(target_results, unknown_results)
    results = dict(
        target_results=target_results,
        unknown_results=unknown_results,
        details=details,
        rtl=asdict(rtl),
    )
    with open(
        rtl.dest_dir
        + os.path.sep
        + "results"
        + os.path.sep
        + f"hpsweep_{rtl.ix:03d}.pkl",
        "wb",
    ) as fh:
        pickle.dump(results, fh)

    # https://keras.io/api/utils/backend_utils/
    tf.keras.backend.clear_session()

    return

#%%
model_settings = input_data.standard_microspeech_model_settings(3)

#%%
@dataclass
class SamplePoint:
    num_epochs: int
    num_batches: int
    batch_size: int


sample_points = [
    # SamplePoint(num_epochs=3, num_batches=3, batch_size=64),
    # SamplePoint(num_epochs=4, num_batches=2, batch_size=64),
    # SamplePoint(num_epochs=7, num_batches=2, batch_size=32),
    # SamplePoint(num_epochs=8, num_batches=1, batch_size=64),
    #SamplePoint(num_epochs=9, num_batches=2, batch_size=32),
    SamplePoint(num_epochs=9, num_batches=1, batch_size=64),
]
n_trials = 1
n_target_sets = 1
ix = 0
for sample_point in sample_points:
    for target_set in range(1, n_target_sets + 1):
        for trial in range(1, n_trials + 1):
            ix += 1
print("num runs", ix)

#%%
N_SHOTS = 5
VAL_UTTERANCES = 400
dest_dir = "/home/mark/tinyspeech_harvard/utterance_sweep_3/"
trial_info = {}

data_dir = "/home/mark/tinyspeech_harvard/speech_commands/"
target = "forward"
all_utterances = set(glob.glob(data_dir + target + "/*.wav"))
used_utterances = set()

ix = 0
for sample_point in sample_points:
    for target_set in range(1, n_target_sets + 1):
        available_utterances = all_utterances - used_utterances
        print("AVAILABLE UTTERANCES", len(available_utterances))
        shot_utterances = np.random.choice(
            list(available_utterances), N_SHOTS, replace=False
        )
        used_utterances = used_utterances | set(shot_utterances)
        print("USED UTTERANCES", len(used_utterances))
        val_utterances = np.random.choice(
            list(all_utterances - set(shot_utterances)), VAL_UTTERANCES, replace=False
        )

        for trial in range(1, n_trials + 1):
            ix += 1
            print("::::::::::::::", ix, sample_point, trial)

            rtl = RunTransferLearning(
                dest_dir=dest_dir,
                ix=ix,
                trial=trial,
                target_set=target_set,
                target=target,
                train_files=shot_utterances,
                val_files=val_utterances,
                unknown_sample=unknown_sample,
                unknown_files=unknown_files,
                num_epochs=sample_point.num_epochs,
                num_batches=sample_point.num_batches,
                batch_size=sample_point.batch_size,
                model_settings=model_settings,
                data_dir=data_dir,
                base_model_path="/home/mark/tinyspeech_harvard/train_100_augment/hundredword_efficientnet_1600_selu_specaug80.0146-0.8736",
                base_model_output="dense_2"
            )

            start = datetime.datetime.now()
            p = multiprocessing.Process(target=run_transfer_learning, args=(rtl,))
            p.start()
            p.join()
            #p.close()
            end = datetime.datetime.now()
            print(":::::::::::::: TIME", str(end - start)[:-7])

            trial_info[ix] = dict(
                ix=ix,
                trial=trial,
                sample_point=asdict(sample_point),
                target_set=target_set,
                train_files=shot_utterances,
                val_files=val_utterances,
                used_utterances=used_utterances,
            )
            with open(dest_dir + "trials/" + f"trial_info_{ix:03d}.pkl", "wb") as fh:
                pickle.dump(trial_info, fh)


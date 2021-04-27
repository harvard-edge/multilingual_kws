#%%
from dataclasses import dataclass, asdict
import os
import json
import glob
import shutil
import multiprocessing
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sox
import pydub
from pydub.playback import play

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
import input_data
import textgrid

import sklearn.model_selection

# fakedata = []
# for ci in [chr(i + 65) for i in range(10)]:
#     for cj in [chr(i + 65) for i in range(10)]:
#         fakedata.append(ci + cj)
# print(len(fakedata), fakedata[:15])

# print("\n\n\n\n\n")
# kf = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.3)
# for train_ix, test_ix in kf.split(fakedata):
#     print(test_ix, len(test_ix))
#     print("--")
# print("\n\n\n\n\n")

workdir = Path("/home/mark/tinyspeech_harvard/luganda")

# chooose random extractions from 1k alignments
# all_alignments = glob.glob(str(workdir / "1k_covid_alignments" / "*.wav"))
# selected_alignments = np.random.choice(all_alignments, 100, replace=False)
# with open(workdir / "hundred_alignments.txt", 'w') as fh:
#     fh.write("\n".join(selected_alignments))

with open(workdir / "hundred_alignments.txt", "r") as fh:
    hundred_alignments = fh.read().splitlines()
hundred_alignments = np.array(hundred_alignments)
print(len(hundred_alignments))


@dataclass(frozen=True)
class SweepData:
    train_files: List[os.PathLike]
    val_files: List[os.PathLike]
    n_batches: int
    n_epochs: int
    model_dest_dir: os.PathLike
    dest_pkl: os.PathLike
    dest_inf: os.PathLike
    primary_lr: float
    backprop_into_embedding: bool
    embedding_lr: float
    with_context: bool
    target: str = "covid"
    batch_size: int = 64
    streamwav: os.PathLike = workdir / "covid_stream.wav"
    gt: os.PathLike = workdir / "empty.txt"


def sweep_run(sd: SweepData, q):

    # load embedding model
    traindir = Path(f"/home/mark/tinyspeech_harvard/multilingual_embedding_wc")
    base_model_path = (
        traindir
        / "models"
        / "multilingual_context_resume20_.005-0.7236"
    )

    model_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis_ooe/")
    unknown_collection_path = model_dir / "unknown_collection.pkl"
    with open(unknown_collection_path, "rb") as fh:
        unknown_collection = pickle.load(fh)
    unknown_files = unknown_collection["unknown_files"]

    model_settings = input_data.standard_microspeech_model_settings(3)
    name, model, details = transfer_learning.transfer_learn(
        target=sd.target,
        train_files=sd.train_files,
        val_files=sd.val_files,
        unknown_files=unknown_files,
        num_epochs=sd.n_epochs,
        num_batches=sd.n_batches,
        batch_size=sd.batch_size,
        primary_lr=sd.primary_lr,
        backprop_into_embedding=sd.backprop_into_embedding,
        embedding_lr=sd.embedding_lr,
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
        csvlog_dest=sd.model_dest_dir / "log.csv",
    )
    print("saving", name)
    modelpath = sd.model_dest_dir / name
    # skip saving model for now, slow
    # model.save(modelpath)

    specs = [input_data.file2spec(model_settings, f) for f in sd.val_files]
    specs = np.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    # print(amx)
    val_accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    # this should maybe be thresholded also
    print("VAL ACCURACY", val_accuracy)

    streamtarget = sa.StreamTarget(
        "lu", "covid", modelpath, sd.streamwav, sd.gt, sd.dest_pkl, sd.dest_inf
    )
    start = datetime.datetime.now()
    sa.eval_stream_test(streamtarget, live_model=model)
    end = datetime.datetime.now()
    print("time elampsed (for all thresholds)", end - start)

    q.put(val_accuracy)


# %%
if __name__ == "__main__":
    q = multiprocessing.Queue()
    val_accuracies = []
    sweep_datas = []

    exp_dir = workdir / "hp_sweep" / "exp_10"
    os.makedirs(exp_dir, exist_ok=False)

    kf = sklearn.model_selection.ShuffleSplit(n_splits=1, test_size=0.95)
    for ix, (train_ixs, val_ixs) in enumerate(kf.split(hundred_alignments)):

        mdd = exp_dir / f"fold_{ix:02d}"
        dp = mdd / "result.pkl"
        di = mdd / "inferences.npy"
        print(mdd)
        os.makedirs(mdd)

        t = hundred_alignments[train_ixs]
        v = hundred_alignments[val_ixs]
        sd = SweepData(
            train_files=t,
            val_files=v,
            n_batches=2,
            n_epochs=4,
            model_dest_dir=mdd,
            dest_pkl=dp,
            dest_inf=di,
            primary_lr=0.001,
            backprop_into_embedding=False,
            embedding_lr=0.00001,
            with_context=True,
        )

        start = datetime.datetime.now()
        p = multiprocessing.Process(target=sweep_run, args=(sd, q))
        p.start()
        p.join()
        end = datetime.datetime.now()
        print("\n\n::::::: experiment run elapsed time", end-start, "\n\n")

        val_acc = q.get()
        val_accuracies.append(val_acc)

        sweep_datas.append(sd)

        # overwrite on every experiment
        with open(mdd / "sweep_data.pkl", "wb") as fh:
            d = dict(sweep_datas=sweep_datas, val_accuracies=val_accuracies)
            # fh.write(json.dumps(d))
            pickle.dump(d, fh)

# %%

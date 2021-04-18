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

#%%
fakedata = []
for ci in [chr(i + 65) for i in range(10)]:
    for cj in [chr(i + 65) for i in range(10)]:
        fakedata.append(ci + cj)
print(len(fakedata), fakedata[:15])

print("\n\n\n\n\n")
kf = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.3)
for train_ix, test_ix in kf.split(fakedata):
    print(test_ix, len(test_ix))
    print("--")
print("\n\n\n\n\n")

# %%
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
# %%
# chooose random extractions from 1k alignments
all_alignments = glob.glob(str(workdir / "1k_covid_alignments" / "*.wav"))
selected_alignments = np.random.choice(all_alignments, 100, replace=False)
# with open(workdir / "hundred_alignments.txt", 'w') as fh:
#     fh.write("\n".join(selected_alignments))
# %%
with open(workdir / "hundred_alignments.txt", "r") as fh:
    hundred_alignments = fh.read().splitlines()
hundred_alignments = np.array(hundred_alignments)
print(hundred_alignments)

# %%


@dataclass(frozen=True)
class SweepData:
    train_files: List[os.PathLike]
    val_files: List[os.PathLike]
    n_batches: int
    n_epochs: int
    model_dest_dir: os.PathLike
    target: str = "covid"
    batch_size: int = 64


def sweep_run(sd: SweepData, q):

    # load embedding model
    traindir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding")
    base_model_path = (
        traindir
        / "models"
        / "multilang_resume40_resume05_resume20_resume22.007-0.7981/"
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
        model_settings=model_settings,
        base_model_path=base_model_path,
        base_model_output="dense_2",
        csvlog_dest=sd.model_dest_dir / "log.csv",
    )
    print("saving", name)
    # model.save(sd.model_dest_dir / name)

    specs = [input_data.file2spec(model_settings, f) for f in sd.val_files]
    specs = np.expand_dims(specs, -1)
    preds = model.predict(specs)
    amx = np.argmax(preds, axis=1)
    # print(amx)
    val_accuracy = amx[amx == 2].shape[0] / preds.shape[0]
    print("VAL ACCURACY", val_accuracy)
    q.put(val_accuracy)


# %%

q = multiprocessing.Queue()
val_accuracies = []
sweep_datas = []

kf = sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=0.5)
for ix, (train_ixs, val_ixs) in enumerate(kf.split(hundred_alignments)):

    mdd = workdir / "hp_sweep" / f"fold_{ix:02d}"
    print(mdd)
    os.makedirs(mdd)

    t = hundred_alignments[train_ixs]
    v = hundred_alignments[val_ixs]
    sd = SweepData(
        train_files=t, val_files=v, n_batches=4, n_epochs=1, model_dest_dir=mdd
    )

    p = multiprocessing.Process(target=sweep_run, args=(sd, q))
    p.start()
    p.join()

    val_acc = q.get()
    val_accuracies.append(val_acc)

    sweep_datas.append(asdict(sd))

    with open(workdir / "sweep_results.pkl", "wb") as fh:
        d = dict(sweep_datas=sweep_datas, val_accuracies=val_accuracies)
        # fh.write(json.dumps(d))
        pickle.dump(d, fh)

# %%

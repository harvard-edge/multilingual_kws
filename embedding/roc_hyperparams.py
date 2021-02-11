#%%
import os
import logging
import re
from typing import Dict, List

import glob
import numpy as np
import pickle
import datetime

import sys

# sys.path.insert(0, "/home/mark/tinyspeech_harvard/tinyspeech/")
# import input_data

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set()
sns.set_palette("bright")

#%%
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
    # tprs = np.around(tprs, 2) 
    # fprs = np.around(fprs, 2)
    return tprs, fprs, threshs


#%%

# print(os.getcwd())
#%%
results_dir = os.listdir("./results/")
results_dir.sort()
latest = results_dir[-1]
print(latest)
with open(f"results/{latest}", "rb") as fh:
    results = pickle.load(fh)

# the sweep script crashed a few times
crashes = [
    "results/hpsweep_025.pkl",
    "results/hpsweep_050.pkl",
    "results/hpsweep_073.pkl",
    "results/hpsweep_096.pkl",
    "results/hpsweep_119.pkl",
    "results/hpsweep_144.pkl",
]
for crash in crashes:
    with open(crash, "rb") as fh:
        memcrash = pickle.load(fh)
        # python 3.9
        # results = results | memcrash1
        results = {**results, **memcrash}

print("number of results", len(results.keys()))

fig = go.Figure()
for model, rd in results.items():

    tprs, fprs, thresh_labels = roc_sc(rd["target_results"], rd["unknown_results"])
    legend = f"""ne: {rd['epochs']}, 
nb: {rd['n_batches']}, 
bs: {rd['details']['batch_size']}, 
trial: {rd['trial']},
va: {rd['details']['val accuracy']:0.2f}"""
    # if rd['details']['batch_size'] != 64:
    #     continue
    #fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=legend))
    legend_w_threshs = [legend + f"<br>thresh: {t}" for t in thresh_labels]
    fig.add_trace(go.Scatter(x=fprs, y=tprs, text=legend_w_threshs, name=legend))

fig.update_layout(
    xaxis_title="FPR",
    yaxis_title="TPR",
    title="target: two [speech commands classification accuracy] <br> [ne: #epochs, nb: #batches, bs: batch size, va: val accuracy]",
    # hoverlabel=dict(font_size=8), # https://plotly.com/python/hover-text-and-formatting/
)
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])
fig.write_html("hpsweep.html")
fig

# %%
a = datetime.datetime.now()
# %%
b = datetime.datetime.now()
d = b-a
print(str(d)[:-5])

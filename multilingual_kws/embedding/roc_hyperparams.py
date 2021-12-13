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


def nice(ne, nb, bs, trial):
    # if ne == 8 and nb == 1 and bs == 64:
    #     return True
    # if ne == 7 and nb == 2 and bs == 32:
    #     return True
    # if ne == 9 and nb == 2 and bs == 32:
    #     return True
    # if ne == 3 and nb == 3 and bs == 64:
    #     return True
    # if ne == 4 and nb == 2 and bs == 64:
    #     return True
    if ne in [8,9] and bs == 64:
        return True
    return False

#%%

results_dir = "/home/mark/tinyspeech_harvard/hyperparam_analysis/results/"
files = os.listdir(results_dir)
files.sort()
latest = files[-1]
print(latest)
with open(f"{results_dir}/{latest}", "rb") as fh:
    results = pickle.load(fh)

# the sweep script crashed a few times
crashes = [
    f"{results_dir}/hpsweep_025.pkl",
    f"{results_dir}/hpsweep_050.pkl",
    f"{results_dir}/hpsweep_073.pkl",
    f"{results_dir}/hpsweep_096.pkl",
    f"{results_dir}/hpsweep_119.pkl",
    f"{results_dir}/hpsweep_144.pkl",
]
for crash in crashes:
    with open(crash, "rb") as fh:
        memcrash = pickle.load(fh)
        # python 3.9
        # results = results | memcrash1
        results = {**results, **memcrash}

print("number of results", len(results.keys()))


#%%

#%%

#  fig = go.Figure()
#  # for model, rd in results.items():
#  for rd in results:
#  
#      tprs, fprs, thresh_labels = roc_sc(rd["target_results"], rd["unknown_results"])
#      ne = rd["num_epochs"]
#      nb = rd["num_batches"]
#      bs = rd["details"]["batch_size"]
#      trial = rd["trial"]
#      va = rd["details"]["val_accuracy"]
#      legend = f"""ne: {ne}, nb: {nb}, bs: {bs}, trial: {trial}, va: {va:0.2f}"""
#      # if bs != 64:
#      #     continue
#      # if not nice(ne, nb, bs, trial):
#      #     continue
#      # fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=legend))
#      legend_w_threshs = [legend + f"<br>thresh: {t}" for t in thresh_labels]
#      fig.add_trace(go.Scatter(x=fprs, y=tprs, text=legend_w_threshs, name=legend))
#  
#  fig.update_layout(
#      xaxis_title="FPR",
#      yaxis_title="TPR",
#      title="target: two [speech commands classification accuracy] <br> [ne: #epochs, nb: #batches, bs: batch size, va: val accuracy]",
#      # hoverlabel=dict(font_size=8), # https://plotly.com/python/hover-text-and-formatting/
#  )
#  fig.update_xaxes(range=[0,1])
#  fig.update_yaxes(range=[0,1])
#  # fig.update_xaxes(range=[0.075, 0.25])
#  # fig.update_yaxes(range=[0.8, 0.92])
#  #fig.write_html("hpsweep.html")
#  fig

# %%
test_dir="/home/mark/tinyspeech_harvard/utterance_sweep_2/"
rd = f"{test_dir}/results/"
rs = glob.glob(rd + "*.pkl")
# print(rs)

td = f"{test_dir}/trials/"
ts = glob.glob(rd + "*.pkl")
# print(ts)

results = []
for r,t in zip(rs,ts):
    with open(r, 'rb') as fh:
        result = pickle.load(fh)
    with open(t, 'rb') as fh:
        trial_info = pickle.load(fh)
    results.append((result,trial_info))
print("NUM RESULTS", len(results))

target_sets=set()
for rd, ti in results:
    target_sets.add(rd["rtl"]["target_set"])
print(target_sets)


fig = go.Figure()
# for model, rd in results.items():
for rd, ti in results:

    tprs, fprs, thresh_labels = roc_sc(rd["target_results"], rd["unknown_results"])
    ne = rd["rtl"]["num_epochs"]
    nb = rd["rtl"]["num_batches"]
    bs = rd["rtl"]["batch_size"]
    trial = rd["rtl"]["trial"]
    target_set = rd["rtl"]["target_set"]
    va = rd["details"]["val_accuracy"]
    legend = f"""ne: {ne}, nb: {nb}, bs: {bs}, tgset: {target_set} trial: {trial}, va: {va:0.2f}"""
    # if bs != 64:
    #     continue
    # if not nice(ne, nb, bs, trial):
    #     continue
    # fig.add_trace(go.Scatter(x=fprs, y=tprs, text=thresh_labels, name=legend))
    legend_w_threshs = [legend + f"<br>thresh: {t}" for t in thresh_labels]
    c = px.colors.qualitative.Dark24[target_set % 24]
    fig.add_trace(go.Scatter(x=fprs, y=tprs, text=legend_w_threshs, name=legend, marker_color=c))

fig.update_layout(
    xaxis_title="FPR",
    yaxis_title="TPR",
    title="target: two [speech commands classification accuracy] <br> [ne: #epochs, nb: #batches, bs: batch size, va: val accuracy]",
    # hoverlabel=dict(font_size=8), # https://plotly.com/python/hover-text-and-formatting/
)
fig.update_xaxes(range=[0,1])
fig.update_yaxes(range=[0,1])
# fig.update_xaxes(range=[0.075, 0.25])
# fig.update_yaxes(range=[0.8, 0.92])
#fig.write_html(f"{test_dir}/hpsweep.html")
fig


#%%
len(px.colors.qualitative.Dark24)
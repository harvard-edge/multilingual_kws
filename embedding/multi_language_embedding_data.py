# %%
import numpy as np
import os
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict
import functools
from collections import Counter, OrderedDict
import csv
import pathlib
import textgrid
import sox
from pathlib import Path
import pickle

import word_extraction

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1.6)

# %%

NUM_WAVS=1600

per_lang = {}

frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words/")
for lang in os.listdir(frequent_words):
    per_lang[lang] = []
    clips = frequent_words / lang / "clips"
    words = os.listdir(clips)
    for word in words:
        wavs = glob.glob(str(clips / word / "*.wav"))
        if len(wavs) > NUM_WAVS:
            per_lang[lang].append(clips/word)


# %%
plt.bar(per_lang.keys(), [len(v) for v in per_lang.values()])

# %%

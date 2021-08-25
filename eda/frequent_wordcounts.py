import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
import nltk
from typing import Set, List, Dict
import functools
from collections import Counter
import csv
import pathlib
from tqdm import tqdm
from pathlib import Path

fw_path1 = Path('/mnt/disks/std2/data/generated/common_voice/frequent_words')
fw_path1 = Path('/mnt/disks/std3/compressed_cleaned3/generated/common_voice/frequent_words')
# fw_path2 = Path('/mnt/disks/std3/data/generated/common_voice/frequent_words')
langs = os.listdir(fw_path1)
# langs += os.listdir(fw_path2)

# new_langs = ['gn','ha']

# langs = [i for i in langs if i in new
lang_to_c = {}

for l in tqdm(langs):
    c = Counter()
    words1_path = fw_path1 / l / 'clips'
    # words2_path = fw_path2 / l / 'clips'
    if words1_path.is_dir():
        words1 = os.listdir(words1_path)
    else:
        words1 = []
    # if words2_path.is_dir():
    #     words2 = os.listdir(words2_path)
    # else:
    #     words2 = []
    # words1 = os.listdir()
    # words2 = os.listdir(fw_path2 / l / 'clips')
    # words = os.listdir(f"/mnt/disks/std750/data/frequent_words/{l}/clips/")
    for w in tqdm(words1):
        wavs = os.listdir(fw_path1 / l / 'clips' / w)
        # wavs = glob.glob(f"/mnt/disks/std750/data/frequent_words/{l}/clips/{w}/*.wav")
        c[w] = len(wavs)

    # for w in tqdm(words2):
    #     wavs = os.listdir(fw_path2 / l / 'clips' / w)
    #     # wavs = glob.glob(f"/mnt/disks/std750/data/frequent_words/{l}/clips/{w}/*.wav")
    #     if w in c:
    #         c[w] += len(wavs)
    #     else:
    #         c[w] = len(wavs)
        # c[w] += len(wavs)
    
    lang_to_c[l] = c

import pickle
# prev = pickle.load(open('frequent_words_stats_std4.p','rb'))

# for k,v in lang_to_c.items():
#     prev[k] = v


pickle.dump(lang_to_c, open('frequent_words_stats_std_final.p','wb'))


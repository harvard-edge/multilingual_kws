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
import stopwordsiso as stopwords
import textgrid
import sox

from tqdm import tqdm

def wordcounts(csvpath, isocode, filter_stopwords=True, wordlength=5):
    all_frequencies = Counter()
    if filter_stopwords:
        frequencies_without_stopwords = Counter()
        stops = stopwords.stopwords(isocode)
        if not stops:
            stops = set()
            #raise ValueError(f"unknown isocode for {isocode}")
    else:
        frequencies_without_stopwords = None
        
    with open(csvpath, 'r') as fh:
        reader = csv.reader(fh)
        for ix,row in enumerate(reader):
            if ix == 0:
                continue # skips header
            words = row[2].split()
            for w in words:
                if len(w) >= wordlength:
                    all_frequencies[w] += 1
                    if filter_stopwords and not w in stops:
                        frequencies_without_stopwords[w] += 1
    
    return frequencies_without_stopwords

#freqs = wordcounts("/mnt/disks/std750/common-voice-forced-alignments/en/validated.csv", isocode='en')

langs = """common-voice-forced-alignments/ab/validated.csv
common-voice-forced-alignments/ar/validated.csv
common-voice-forced-alignments/as/validated.csv
common-voice-forced-alignments/br/validated.csv
common-voice-forced-alignments/ca/validated.csv
common-voice-forced-alignments/cnh/validated.csv
common-voice-forced-alignments/cs/validated.csv
common-voice-forced-alignments/cv/validated.csv
common-voice-forced-alignments/cy/validated.csv
common-voice-forced-alignments/de/validated.csv
common-voice-forced-alignments/dv/validated.csv
common-voice-forced-alignments/el/validated.csv
common-voice-forced-alignments/en/validated.csv
common-voice-forced-alignments/eo/validated.csv
common-voice-forced-alignments/es/validated.csv
common-voice-forced-alignments/et/validated.csv
common-voice-forced-alignments/eu/validated.csv
common-voice-forced-alignments/fa/validated.csv
common-voice-forced-alignments/fr/validated.csv
common-voice-forced-alignments/fy-NL/validated.csv
common-voice-forced-alignments/ga-IE/validated.csv
common-voice-forced-alignments/ia/validated.csv
common-voice-forced-alignments/id/validated.csv
common-voice-forced-alignments/it/validated.csv
common-voice-forced-alignments/ja/validated.csv
common-voice-forced-alignments/ka/validated.csv
common-voice-forced-alignments/ky/validated.csv
common-voice-forced-alignments/lv/validated.csv
common-voice-forced-alignments/mn/validated.csv
common-voice-forced-alignments/mt/validated.csv
common-voice-forced-alignments/nl/validated.csv
common-voice-forced-alignments/or/validated.csv
common-voice-forced-alignments/pa-IN/validated.csv
common-voice-forced-alignments/pl/validated.csv
common-voice-forced-alignments/pt/validated.csv
common-voice-forced-alignments/rm-sursilv/validated.csv
common-voice-forced-alignments/rm-vallader/validated.csv
common-voice-forced-alignments/ro/validated.csv
common-voice-forced-alignments/ru/validated.csv
common-voice-forced-alignments/rw/validated.csv
common-voice-forced-alignments/sah/validated.csv
common-voice-forced-alignments/sl/validated.csv
common-voice-forced-alignments/sv-SE/validated.csv
common-voice-forced-alignments/ta/validated.csv
common-voice-forced-alignments/tr/validated.csv
common-voice-forced-alignments/tt/validated.csv
common-voice-forced-alignments/uk/validated.csv
common-voice-forced-alignments/vi/validated.csv
common-voice-forced-alignments/zh-CN/validated.csv
common-voice-forced-alignments/zh-HK/validated.csv
common-voice-forced-alignments/zh-TW/validated.csv""".split('\n')

lang_to_c = {}
for l in tqdm(langs):
    words = wordcounts("/mnt/disks/std750/data/" + l, isocode=l.split('/')[1]) 
    lang_to_c[l.split('/')[1]] = words

import pickle
pickle.dump(lang_to_c, open('common_voice_stats.p','wb'))

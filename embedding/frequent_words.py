#%%
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
# import nltk
from typing import Set, List, Dict
import functools
from collections import Counter, OrderedDict
import csv
import pathlib
# import stopwordsiso as stopwords
import textgrid
import sox
from pathlib import Path
import pickle

import word_extraction

sns.set()
sns.set_palette("bright")
# sns.set(font_scale=1)

#%%
# how large is each language?
langs = {}
alignments = Path("/home/mark/tinyspeech_harvard/common-voice-forced-alignments")
for lang in os.listdir(alignments):
    if not os.path.isdir(alignments/lang):
        continue
    validated = alignments/lang/"validated.csv"
    if not os.path.isfile(validated):
        continue
    with open(validated, 'r') as fh:
        langs[lang] = len(fh.readlines())
langs = OrderedDict(sorted(langs.items(), key=lambda kv: kv[1], reverse=True))
# too many results in english
del langs["en"]

fig, ax = plt.subplots()
ax.bar(langs.keys(), langs.values())
ax.set_xticklabels(langs.keys(), rotation=90)
fig.set_size_inches(20,5)

# %%
#LANG_ISOCODE="de"
LANG_ISOCODE="el"

if not os.path.isdir(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}"):
    raise ValueError("need to create dir")
if not os.path.isdir(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/timings"):
    raise ValueError("need to create dir")
# %%
# generate most frequent words and their counts
alignments = Path("/home/mark/tinyspeech_harvard/common-voice-forced-alignments")
counts = word_extraction.wordcounts(alignments / LANG_ISOCODE / "validated.csv")

# %%
# look for stopwords that are too short
counts.most_common(20)

# %%
N_WORDS_TO_SAMPLE = 250
# get rid of words that are too short
SKIP_FIRST_N = 8
to_expunge = counts.most_common(8)
non_stopwords = counts.copy()
for k,_ in to_expunge:
    del non_stopwords[k]
longer_words = [kv for kv in non_stopwords.most_common() if len(kv[0]) > 2]

print("counts for last word", longer_words[N_WORDS_TO_SAMPLE - 1])

# %%
# visualize frequencies of top words
fig,ax = plt.subplots()
topn = longer_words[:N_WORDS_TO_SAMPLE]
ax.bar([c[0] for c in topn], [c[1] for c in topn]);
ax.set_xticklabels([c[0] for c in topn], rotation=70);
ax.set_ylim([0,3000])
fig.set_size_inches(40,10)

# %%
# fetch timings for all words of interest
words = set([w[0] for w in longer_words[:N_WORDS_TO_SAMPLE]])
tgs = word_extraction.generate_filemap(lang_isocode=LANG_ISOCODE, alignment_basedir=alignments)
print("extracting timings")
timings, notfound = word_extraction.generate_wordtimings(words_to_search_for=words, mp3_to_textgrid=tgs, lang_isocode=LANG_ISOCODE, alignment_basedir=alignments)
print("errors", len(notfound))

# %%
#with open(f"/home/mark/tinyspeech_harvard/frequent_words/{ISO_CODE}/all_timings.pkl", "wb") as fh:
#   pickle.dump(timings, fh)
#with open(f"/home/mark/tinyspeech_harvard/frequent_words/{ISO_CODE}/all_timings.pkl", "rb") as fh:
#    timings = pickle.load(fh)

# %%
# write timings to csvs per word
MAX_NUM_UTTERANCES_TO_SAMPLE = 2500
df_dest = pathlib.Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/timings")
for word, times in timings.items():
    df = pd.DataFrame(times, columns=["mp3_filename", "start_time_s", "end_time_s"])
    if df.shape[0] > MAX_NUM_UTTERANCES_TO_SAMPLE:
        print(word, "SUBSAMPLING")
        df = df.sample(n=MAX_NUM_UTTERANCES_TO_SAMPLE, replace=False)
    print(df_dest / (word + ".csv"))
    df.to_csv(df_dest / (word + ".csv"), quoting=csv.QUOTE_MINIMAL, index=False)
# %%

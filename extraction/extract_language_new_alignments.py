#%%

import csv
import functools
import glob
import multiprocessing
import os, sys
import pathlib
import pickle
import shutil
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path
# import nltk
from typing import Dict, List, Set

import matplotlib.pyplot as plt
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import sox
# import stopwordsiso as stopwords
import textgrid

sys.path.append("../multilingual_kws/embedding")
import word_extraction

# LANGUAGE = 'uk'

# local
# ['ar', 'cs', 'cy', 'et', 'eu', 'id', 'ky', 'pl', 'pt', 'ru', 'ta', 'tr', 'tt', 'uk']

# sns.set()
# sns.set_palette("bright")
# sns.set(font_scale=1.6)

#%%
# how large is each language?
langs = {}
# alignments = Path("/mnt/disks/std750/data/common-voice-forced-alignments")
new_alignments = Path("/mnt/disks/std3/alignments")

for lang in os.listdir(new_alignments):
    if not os.path.isdir(new_alignments / lang):
        continue
    folder = new_alignments / lang
    langs[lang] = len(os.listdir(folder)) - 1
langs = OrderedDict(sorted(langs.items(), key=lambda kv: kv[1], reverse=True))
# too many results in english
# del langs["en"]

# %%
iso2lang = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "de": "German", #
    "en": "English", #
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ky": "Kyrgyz",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "rw": "Kinyarwanda", #
    "ta": "Tamil",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukranian",
}

# %%
# paths
disk = Path("/mnt/disks/std3")
LANG_ISOCODE = "ha"
generated_common_voice_path = disk / "data/generated/common_voice"
frequent_words_dir = generated_common_voice_path / "frequent_words" / LANG_ISOCODE
timings_dir = frequent_words_dir / "timings"
errors_dir = frequent_words_dir / "errors"
clips_dir = frequent_words_dir / "clips"

frequent_words_dir.mkdir(parents=True, exist_ok=True)
timings_dir.mkdir(parents=True, exist_ok=True)
errors_dir.mkdir(parents=True, exist_ok=True)
clips_dir.mkdir(parents=True, exist_ok=True)
print("Generating", LANG_ISOCODE)#, iso2lang[LANG_ISOCODE])

# %%
# total
# ['ar', 'ca', 'cs', 'cy', 'de', 'en', 'es', 'et', 'eu', 'fa', 'fr', 'id', 'it', 'ky', 'nl', 'pl', 'pt', 'ru', 'rw', 'ta', 'tr', 'tt', 'uk']
# non-embedding classification:
# done: cs cy eu
# ['ar', 'cs', 'cy', 'et', 'eu', 'id', 'ky', 'pl', 'pt', 'ru', 'ta', 'tr', 'tt', 'uk']

# we are just using the speech commands dir directly, ignore this for now
# background_noise_dir=f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/_background_noise_"
# BASE_BACKGROUND_NOISE="/home/mark/tinyspeech_harvard/speech_commands/_background_noise_"
# shutil.copytree(BASE_BACKGROUND_NOISE, background_noise_dir)
# if not os.path.isdir(background_noise_dir):
#     raise ValueError("need to copy bg data", background_noise_dir)

# %%
# generate most frequent words and their counts
# alignments = Path("/home/mark/tinyspeech_harvard/common-voice-forced-alignments")
# csv = 

new_csv = pd.read_csv(f"/mnt/disks/std750/data/common-voice/cv-corpus-7.0-2021-07-21/{LANG_ISOCODE}/validated.tsv", delimiter='\t')[['path','sentence']].rename({'path':'wav_filename','sentence':'transcript'},axis=1)
# new_csv['transcript'] = open('/mnt/disks/std3/alignments/zh-CN/chinese_lines_segmented.txt','r',encoding='utf-8').read().split('\n')[:-1]
# new_csv['transcript'] = new_csv['transcript'].str.replace('.mp3','')
new_csv.to_csv(new_alignments / LANG_ISOCODE / "validated.csv", index=False)


# new_csv = pd.read_csv(f"/mnt/disks/std750/data/common-voice/cv-corpus-7.0-2021-07-21/{LANG_ISOCODE}/validated.tsv", delimiter='\t')[['path','sentence']].rename({'path':'wav_filename','sentence':'transcript'},axis=1)
# new_csv['transcript'] = open('/mnt/disks/std3/alignments/zh-CN/chinese_lines_segmented.txt','r',encoding='utf-8').read().split('\n')[:-1]
# new_csv.to_csv(new_alignments / LANG_ISOCODE / "validated.csv", index=False)

counts = word_extraction.wordcounts(new_alignments / LANG_ISOCODE / "validated.csv", True, 1)

# %%
MIN_COUNT = 5
MIN_CHAR_LEN = 2
SKIP_FIRST_N = 0

non_stopwords = counts.copy()
# get rid of words that are too short
to_expunge = counts.most_common(SKIP_FIRST_N)
for k, _ in to_expunge:
    del non_stopwords[k]

already_exists = os.listdir(clips_dir)
skipped = 0
for k in already_exists:
    if k in non_stopwords.keys():
        print("already exists", k)
        skipped += 1
        del non_stopwords[k]
print("skipped", skipped)
longer_words = [kv for kv in non_stopwords.most_common() if (kv[1] >= MIN_COUNT and len(kv[0]) >= MIN_CHAR_LEN)]
print("num words to be extracted", len(longer_words))

new_words_file = (
    disk / f"data/generated/common_voice/new_words/new_words_{LANG_ISOCODE}.txt"
)
new_words_file.parent.mkdir(parents=True, exist_ok=True)

with open(new_words_file, "w") as fh:
    fh.write(LANG_ISOCODE + "\n")
    fh.write(",".join([l[0] for l in longer_words]))
    fh.write("\n")


# %%
# fetch timings for all words of interest
timings_count = 0
dest_pkl = f"{frequent_words_dir}/all_timings_{timings_count}.pkl"
while os.path.isfile(dest_pkl):
    print("already exists", dest_pkl)
    timings_count += 1
    dest_pkl = f"{frequent_words_dir}/all_timings_{timings_count}.pkl"
words = set([w[0] for w in longer_words])
tgs = word_extraction.generate_new_filemap(
    lang_isocode=LANG_ISOCODE, alignment_basedir=new_alignments
)

# for k,v in tgs.items():


#%%
print("extracting timings")
timings, notfound = word_extraction.generate_wordtimings(
    words_to_search_for=words,
    mp3_to_textgrid=tgs,
    lang_isocode=LANG_ISOCODE,
    alignment_basedir=new_alignments,
)
print("errors", len(notfound))
print("saving timings to", dest_pkl)
with open(dest_pkl, "wb") as fh:
    pickle.dump(timings, fh)

# %%
# write timings to csvs per word
MAX_NUM_UTTERANCES_TO_SAMPLE = 300
df_dest = pathlib.Path(timings_dir)
from tqdm import tqdm
for word, times in tqdm(timings.items()):
    df = pd.DataFrame(times, columns=["mp3_filename", "start_time_s", "end_time_s"])
    # print(df_dest / (word + ".csv"))
    if "/" in word:
        print(f"Cant save {word}")
        continue
    df.to_csv(df_dest / (word + ".csv"), quoting=csv.QUOTE_MINIMAL, index=False)

# %%

################################################################
# now, run extract_frequent_words.py
# below, select splits between 165 commands and 85 other words
################################################################

# %%
# LANG_ISOCODE = "uk"
# disk = Path("/mnt/disks/std750")
disk750 = Path("/mnt/disks/std750")
common_voice_dir = disk / "data/generated/common_voice"
common_voice_data_dir = disk750 / "data/common-voice/cv-corpus-7.0-2021-07-21"
WORD_CSVS_DIR = timings_dir
# WORD_CSVS = Path(f"/mnt/disks/std750/data/generated/common_voice/timings/{LANG_ISOCODE}/*.csv")
# WORD_CSVS = f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/timings/*.csv"
# CV_CLIPS_DIR = Path(f"/media/mark/hyperion/common_voice/cv-corpus-5.1-2020-06-22/{LANG_ISOCODE}/clips/")
CV_CLIPS_DIR = common_voice_data_dir / LANG_ISOCODE / "clips"
# CV_CLIPS_DIR = Path(f"/media/mark/hyperion/common_voice/cv-corpus-6.1-2020-12-11/{LANG_ISOCODE}/clips/")
# SWTS_CLIPS_DIR = Path("/home/mark/tinyspeech_harvard/commonvoice_singleword/cv-corpus-5-singleword/en/clips")
OUT_DIR = clips_dir
# OUT_DIR = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/clips")
ERRORS_DIR = errors_dir
# ERRORS_DIR = Path(f"/home/mark/tinyspeech_harvard/frequent_words/{LANG_ISOCODE}/errors")
WRITE_PROGRESS = True

#%%
def extract_one_second(duration_s: float, start_s: float, end_s: float):
    """
    return one second around the midpoint between start_s and end_s
    """
    if duration_s < 1:
        return (0, duration_s)

    center_s = start_s + ((end_s - start_s) / 2.0)

    new_start_s = center_s - 0.5
    new_end_s = center_s + 0.5

    if new_end_s > duration_s:
        new_end_s = duration_s
        new_start_s = duration_s - 1.0

    if new_start_s < 0:
        new_start_s = 0
        new_end_s = np.minimum(duration_s, new_start_s + 1.0)

    #     print(
    #         "start",
    #         new_start_s,
    #         "end",
    #         new_end_s,
    #         "\nduration",
    #         new_end_s - new_start_s,
    #         "midpoint",
    #         new_start_s + ((new_end_s - new_start_s) / 2.0),
    #     )
    return (new_start_s, new_end_s)


def extract(csvpath):
    word = os.path.splitext(os.path.basename(csvpath))[0]
    # print(word)
    if os.path.isdir(OUT_DIR / word):
        raise ValueError("trying to extract to an existing dir", OUT_DIR / word)
    os.mkdir(OUT_DIR / word)

    with open(csvpath, "r") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header

        for ix, row in enumerate(reader):
            if ix % 1000 == 0:
                print(word, ix)
                if WRITE_PROGRESS:
                    with open("progress.txt", "a") as fh:
                        fh.write(f"{word} {ix}\n")
            mp3name_no_ext = row[0]
            if LANG_ISOCODE == "zh-CN":
                mp3name_no_ext = mp3name_no_ext.split('-')
                mp3name_no_ext = mp3name_no_ext[0] + '_' + mp3name_no_ext[1] + '_' + mp3name_no_ext[2] + '-' + mp3name_no_ext[3] + '_' + mp3name_no_ext[4]
            else:
                mp3name_no_ext = mp3name_no_ext.replace("-", "_")

            print(mp3name_no_ext)
            start_s = float(row[1])
            end_s = float(row[2])
            mp3path = CV_CLIPS_DIR / (mp3name_no_ext + ".mp3")
            # if not os.path.exists(mp3path): # must be in Mozilla SWTS
            # mp3path = SWTS_CLIPS_DIR / (mp3name_no_ext + ".mp3")
            if not os.path.exists(mp3path):
                # really don't know where this came from, skip it
                with open(ERRORS_DIR / mp3name_no_ext, "a") as fh:
                    pass
                continue

            duration = sox.file_info.duration(mp3path)
            if end_s - start_s < 1:
                pad_amt_s = (1.0 - (end_s - start_s)) / 2.0
            else:  # utterance is already longer than 1s, trim instead
                start_s, end_s = extract_one_second(duration, start_s, end_s)
                pad_amt_s = 0

            dest = OUT_DIR / word / (mp3name_no_ext + ".wav")
            # words can appear multiple times in a sentence: append w number
            count = 2
            while os.path.exists(dest):
                dest = OUT_DIR / word / (f"{mp3name_no_ext}__{count}.wav")
                count += 1

            transformer = sox.Transformer()
            transformer.convert(samplerate=48000)  # from 48K mp3s
            transformer.trim(start_s, end_s)
            # use smaller fadein/fadeout since we are capturing just the word
            # TODO(mmaz) is this appropriately sized?
            transformer.fade(fade_in_len=0.025, fade_out_len=0.025)
            transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
            transformer.build(str(mp3path), str(dest))
    return word

def main():
    if not os.path.isdir(OUT_DIR) or not os.path.isdir(ERRORS_DIR):
        raise ValueError("create outdir and errordir", OUT_DIR, ERRORS_DIR)
    if not os.path.isdir(CV_CLIPS_DIR):
        raise ValueError("data not found")
    # print(WORD_CSVS)
    all_csvs = [x for x in WORD_CSVS_DIR.glob("*.csv") if x.is_file()]
    print("num csvs found", len(all_csvs))
    if len(all_csvs) == 0:
        raise ValueError("no csvs")

    unextracted_csvs = []
    for csvpath in all_csvs:
        word = os.path.splitext(os.path.basename(csvpath))[0]
        wav_dir = OUT_DIR / word
        if os.path.exists(wav_dir):
            print("skipping", wav_dir)
            continue
        else:
            unextracted_csvs.append(csvpath)
    print("\n\n::::::::::::::::::::")
    print("unextracted csvs:", unextracted_csvs)
    print("\n\n--------------------")
    print("extracting", len(unextracted_csvs), "out of", len(all_csvs))

    pool = multiprocessing.Pool()
    for i, result in enumerate(pool.imap_unordered(extract, unextracted_csvs[:500]), start=1):
        print("counter: ", i, "word", result)
        if WRITE_PROGRESS:
            with open("finished.txt", "a") as fh:
                fh.write(f"counter {i} word {result}\n")
    pool.close()
    pool.join()

    if WRITE_PROGRESS:
        with open("finished.txt", "a") as fh:
            fh.write("complete\n")

main()

# %%

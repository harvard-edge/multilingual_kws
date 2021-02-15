#%%
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict
import functools
from collections import Counter
import csv
import pathlib
import textgrid
import sox
import pickle
from scipy.io import wavfile


#%%
def wordcounts(csvpath):
    all_frequencies = Counter()
    with open(csvpath, "r") as fh:
        reader = csv.reader(fh)
        for ix, row in enumerate(reader):
            if ix == 0:
                continue  # skips header
            words = row[2].split()
            for w in words:
                all_frequencies[w] += 1
    return all_frequencies


#%%

counts = wordcounts(
    "/home/mark/tinyspeech_harvard/common-voice-forced-alignments/en/validated.csv"
)
common = counts.most_common(600)
print(common[455])
# merchant: 2204

#%%

# generate a filepath map from mp3 filename to textgrid
def generate_filemap(
    lang_isocode="en",
    alignment_basedir="/home/mark/tinyspeech_harvard/common-voice-forced-alignments/",
):
    filemap = {}
    for root, dirs, files in os.walk(
        pathlib.Path(alignment_basedir) / lang_isocode / "alignments"
    ):
        if not files:
            continue  # this is the top level dir
        for textgrid in files:
            mp3name = os.path.splitext(textgrid)[0]
            if mp3name in filemap:
                raise ValueError(f"{mp3name} already present in filemap")
            filemap[mp3name] = os.path.join(root, textgrid)
    return filemap


def generate_wordtimings(
    words_to_search_for: Set[str],
    mp3_to_textgrid,
    lang_isocode="en",
    alignment_basedir="/home/mark/tinyspeech_harvard/common-voice-forced-alignments/",
):
    # word: pseudo-datframe of [(mp3_filename, start_time_s, end_time_s)]
    timings = {w: [] for w in words_to_search_for}
    notfound = []
    # common voice csv from DeepSpeech/import_cv2.py
    csvpath = pathlib.Path(alignment_basedir) / lang_isocode / "validated.csv"
    with open(csvpath, "r") as fh:
        reader = csv.reader(fh)
        for ix, row in enumerate(reader):
            if ix == 0:
                continue  # skips header
            if ix % 80_000 == 0:
                print(ix)
            # find words in common_words set from each row of csv
            mp3name_no_extension = os.path.splitext(row[0])[0]
            words = row[2].split()
            for word in words:
                if word not in words_to_search_for:
                    continue
                # get alignment timings for this word
                try:
                    tgf = mp3_to_textgrid[mp3name_no_extension]
                except KeyError as e:
                    notfound.append((mp3name_no_extension, word))
                    continue
                tg = textgrid.TextGrid.fromFile(tgf)
                for interval in tg[0]:
                    if interval.mark != word:
                        continue
                    start_s = interval.minTime
                    end_s = interval.maxTime
                    timings[word].append((mp3name_no_extension, start_s, end_s))
    return timings, notfound


# %%
mp3_to_textgrid = generate_filemap()
print(len(mp3_to_textgrid.keys()))
timings, notfound = generate_wordtimings({"merchant"}, mp3_to_textgrid)
print(len(notfound))
print(len(timings["merchant"]))
print(timings["merchant"][0])

# %%
# how many mp3s have two or more utterances of the target
target = "merchant"
mp3s = [t[0] for t in timings[target]]
n_samples = len(mp3s)
n_mp3s = len(set(mp3s))
print(n_samples, n_mp3s, n_mp3s - n_samples)

# %%

# %%
#####################################################
##    select 1k examples to string into a long wav
####################################################
NUM_SAMPLES = 1000
NUM_SHOTS = 5
# https://stackoverflow.com/questions/23445936/numpy-random-choice-of-tuples
ix_samples = np.random.choice(
    len(timings[target]), NUM_SAMPLES + NUM_SHOTS, replace=False
)
samples = np.array(timings[target])[ix_samples]
mp3s = set([t[0] for t in samples])
assert (
    len(mp3s) == NUM_SAMPLES + NUM_SHOTS
), "an mp3 was selected with multiple targets in the same sentence"

shots = samples[:NUM_SHOTS]
stream = samples[NUM_SHOTS:]

base_dir = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments")
dest_dir = base_dir / target
shot_targets = dest_dir / "n_shots"
streaming_test_data = dest_dir / "streaming_test_data.pkl"

print("generating", [dest_dir, shot_targets])
# os.makedirs(dest_dir, exist_ok=True)
# os.makedirs(shot_targets, exist_ok=True)
# with open(streaming_test_data, 'wb') as fh:
#     pickle.dump((mp3_to_textgrid, timings, shots, stream), fh)

# %%


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
    return (new_start_s, new_end_s)


def extract_shot_from_mp3(
    mp3name_no_ext,
    start_s,
    end_s,
    dest_dir,
    cv_clipsdir=pathlib.Path(
        "/home/mark/tinyspeech_harvard/common_voice/cv-corpus-6.1-2020-12-11/en/clips"
    ),
):
    mp3path = cv_clipsdir / (mp3name_no_ext + ".mp3")
    if not os.path.exists(mp3path):
        raise ValueError("could not find", mp3path)

    duration = sox.file_info.duration(mp3path)
    if end_s - start_s < 1:
        pad_amt_s = (1.0 - (end_s - start_s)) / 2.0
    else:  # utterance is already longer than 1s, trim instead
        start_s, end_s = extract_one_second(duration, start_s, end_s)
        pad_amt_s = 0

    dest = dest_dir / (mp3name_no_ext + ".wav")
    # words can appear multiple times in a sentence: above should have filtered these
    if os.path.exists(dest):
        raise ValueError("already exists:", dest)

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(start_s, end_s)
    # use smaller fadein/fadeout since we are capturing just the word
    # TODO(mmaz) is this appropriately sized?
    transformer.fade(fade_in_len=0.025, fade_out_len=0.025)
    transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
    transformer.build(str(mp3path), str(dest))


# %%
# GENERATE SHOTS
for mp3name_no_ext, start_s, end_s in shots:
    print(mp3name_no_ext, start_s, end_s)
    # extract_shot_from_mp3(mp3name_no_ext, float(start_s), float(end_s), dest_dir=shot_targets)

# %%
###############################################
##               LISTEN
###############################################
import pydub
from pydub.playback import play
import time

audiofiles = glob.glob(str(shot_targets / "*.wav"))
for ix, f in enumerate(audiofiles):
    print(ix, f)

for ix, f in enumerate(audiofiles):
    print(f)
    play(pydub.AudioSegment.from_wav(f))
    time.sleep(0.5)

#%%
###############################################
## GENERATE LONG WAVFILE AND LABELS
###############################################

# step 1: convert to wavs (to avoid slop in mp3 timings)
# food for thought: sox mp3s may have a gap when concatenating
# https://stackoverflow.com/questions/25280958/sox-concatenate-multiple-audio-files-without-a-gap-in-between

cv_clipsdir = pathlib.Path(
    "/home/mark/tinyspeech_harvard/common_voice/cv-corpus-6.1-2020-12-11/en/clips"
)
wavs_dest = dest_dir / "wavs"

wavs = []
total_duration_mp3s_s = 0
for ix, (mp3name_no_ext, start_s, end_s) in enumerate(stream):
    if ix % 250 == 0:
        print(ix)
    mp3path = cv_clipsdir / (mp3name_no_ext + ".mp3")
    if not os.path.exists(mp3path):
        raise ValueError("could not find", mp3path)

    duration_s = sox.file_info.duration(mp3path)
    total_duration_mp3s_s += duration_s

    wav = str(wavs_dest / (mp3name_no_ext + ".wav"))
    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s

    transformer.build(str(mp3path), wav)
    wavs.append(wav)

print(total_duration_mp3s_s, "sec = ", total_duration_mp3s_s / 60, "min")
print(len(wavs))

# step 2: how long is the sum of each wav according to sox?
total_duration_wavs_s = 0
for w in wavs:
    duration_s = sox.file_info.duration(w)
    total_duration_wavs_s += duration_s
print(total_duration_wavs_s, "sec = ", total_duration_wavs_s / 60, "min")

#%%

# step 3: combine the wavs. godspeed.
combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(wavs, str(dest_dir / "stream.wav"), "concatenate")

#%%
# step 4: how long is the total wavfile? should be the sum of the individual wavs
duration_s = sox.file_info.duration(dest_dir / "stream.wav")
print(duration_s, "sec = ", duration_s / 60, "min")

#%%
# step 5: generate labels using the wav file durations, not the sloppy mp3 file durations

target_times_s = []
current_sentence_start_s = 0
for ix, (mp3name_no_ext, start_s, end_s) in enumerate(stream):
    wavpath = dest_dir / "wavs" / (mp3name_no_ext + ".wav")
    sentence_duration_s = sox.file_info.duration(wavpath)
    target_utterance_start_s = current_sentence_start_s + float(start_s)
    target_utterance_end_s = current_sentence_start_s + float(end_s)
    target_times_s.append((target_utterance_start_s, target_utterance_end_s))
    current_sentence_start_s += sentence_duration_s

#%%
# step 6: write labels out
# the label timings should indicate the start of each target utterance in ms
label_file = dest_dir / "labels.txt"
with open(label_file, "w") as fh:
    for start_s, _ in target_times_s:
        start_ms = start_s * 1000
        fh.write(f"{target}, {start_ms}\n")

#%%
# how good are the timings? listen to random extractions from the full stream

sr, data = wavfile.read(dest_dir / "stream.wav")
print(data.shape[0] / sr, "sec")

rand_ix = np.random.randint(len(target_times_s))
print(rand_ix, "/", len(target_times_s))

start_s, end_s = target_times_s[rand_ix]
start_samples, end_samples = int(start_s * sr), int(end_s * sr)
utterance = data[start_samples:end_samples]
print(start_s, end_s)

#https://github.com/jiaaro/pydub/blob/master/API.markdown
play(pydub.AudioSegment(data=utterance, sample_width=2, frame_rate=sr, channels=1))


#%%
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
from typing import Set, List, Dict, Set
import functools
from collections import Counter
import csv
import pathlib
import textgrid
import sox
import pickle
from scipy.io import wavfile
from pathlib import Path

import word_extraction


#%%

target_lang = "cy"
counts = word_extraction.wordcounts(
    f"/home/mark/tinyspeech_harvard/common-voice-forced-alignments/{target_lang}/validated.csv"
)
#%%
target = "iaith"  # 243

common = counts.most_common(600)
# print(common[455])
for ix, (w, c) in enumerate(common):
    if w == target:
        print(target, ix, c)
# saint (415)
# dream (425): 2364
# merchant (455): 2204


# %%

mp3_to_textgrid = word_extraction.generate_filemap(lang_isocode=target_lang)
print(len(mp3_to_textgrid.keys()))
timings_w_dups, notfound = word_extraction.generate_wordtimings(
    {target}, mp3_to_textgrid, lang_isocode=target_lang
)
print("notfound", len(notfound))
print("num timings", len(timings_w_dups[target]))
print(timings_w_dups[target][0])

# %%
# how many mp3s have two or more utterances of the target
mp3s = [t[0] for t in timings_w_dups[target]]
n_samples = len(mp3s)
n_mp3s = len(set(mp3s))
print(n_samples, n_mp3s, "2 or more utterances of the target:", n_samples - n_mp3s)

# remove duplicates (removes any mp3 that shows up more than once, not just subsequent occurences)
seen_mp3s = set()
seen_multiple = set()
for t in timings_w_dups[target]:
    mp3_filename = t[0]
    if mp3_filename in seen_mp3s:
        seen_multiple.add(mp3_filename)
    seen_mp3s.add(mp3_filename)

timings = {target: []}
for t in timings_w_dups[target]:
    mp3_filename = t[0]
    if mp3_filename in seen_multiple:
        continue
    timings[target].append(t)

print("after removing duplicates", len(timings[target]), ", dropped mp3s:", n_mp3s - len(timings[target]))

# %%
#####################################################
##    select examples to string into a long wav
####################################################
NUM_SAMPLES_FOR_STREAMING_WAV = 100
NUM_SHOTS = 5
NUM_VAL = 30
# https://stackoverflow.com/questions/23445936/numpy-random-choice-of-tuples
ix_samples = np.random.choice(
    len(timings[target]),
    NUM_SAMPLES_FOR_STREAMING_WAV + NUM_SHOTS + NUM_VAL,
    replace=False,
)
samples = np.array(timings[target])[ix_samples]
mp3s = set([t[0] for t in samples])
assert (
    len(mp3s) == NUM_SAMPLES_FOR_STREAMING_WAV + NUM_SHOTS + NUM_VAL
), "an mp3 was selected with multiple targets in the same sentence"

shots = samples[:NUM_SHOTS]
val = samples[NUM_SHOTS : NUM_SHOTS + NUM_VAL]
target_stream = samples[NUM_SHOTS + NUM_VAL :]
print(len(shots), len(val), len(target_stream))

# select non-target sentences to intersperse
non_targets = word_extraction.random_non_target_sentences(
    num_sentences=len(target_stream),
    words_to_exclude={target},
    lang_isocode=target_lang,
)

wav_data = []
for (target_sample, non_target_sample) in zip(target_stream, non_targets):
    target_data = dict(
        is_target=True,
        mp3name_no_ext=target_sample[0],
        start_s=target_sample[1],
        end_s=target_sample[2],
    )
    non_target_data = dict(is_target=False, mp3name_no_ext=non_target_sample)
    wav_data.append(target_data)
    wav_data.append(non_target_data)

assert len(wav_data) == len(target_stream) * 2, "missing a pair for the targets"

# %%
# write out data
base_dir = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments")
dest_dir = base_dir / target
shot_targets = dest_dir / "n_shots"
val_targets = dest_dir / "val"
wav_intermediates = dest_dir / "wavs"
streaming_test_data = dest_dir / "streaming_test_data.pkl"

print("generating", [dest_dir, shot_targets])
os.makedirs(dest_dir, exist_ok=False)
os.makedirs(shot_targets, exist_ok=False)
os.makedirs(val_targets, exist_ok=False)
os.makedirs(wav_intermediates, exist_ok=False)
with open(streaming_test_data, "wb") as fh:
    stream_data = dict(
        target_word=target,
        target_lang=target_lang,
        mp3_to_textgrid=mp3_to_textgrid,
        timings=timings,
        shots=shots,
        val=val,
        target_stream=target_stream,
        non_targets=non_targets,
        wav_data=wav_data,
    )
    pickle.dump(stream_data, fh)


# %%
# GENERATE SHOTS

# fmt: off
cv_clipsdir = Path("/media/mark/hyperion/common_voice/cv-corpus-6.1-2020-12-11/") / target_lang / "clips"
# fmt: on

print("generating shots")
for mp3name_no_ext, start_s, end_s in shots:
    print(mp3name_no_ext, start_s, end_s)
    word_extraction.extract_shot_from_mp3(
        mp3name_no_ext,
        float(start_s),
        float(end_s),
        dest_dir=shot_targets,
        cv_clipsdir=cv_clipsdir,
    )

print("generating val")
for mp3name_no_ext, start_s, end_s in val:
    print(mp3name_no_ext, start_s, end_s)
    word_extraction.extract_shot_from_mp3(
        mp3name_no_ext,
        float(start_s),
        float(end_s),
        dest_dir=val_targets,
        cv_clipsdir=cv_clipsdir,
    )

# %%
###############################################
##               LISTEN
###############################################
import pydub
from pydub.playback import play
import time

audiofiles = glob.glob(str(shot_targets / "*.wav"))
# audiofiles = glob.glob("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/dream/n_shots/*.wav")
for ix, f in enumerate(audiofiles):
    print(ix, f)

print("listening:")
for ix, f in enumerate(audiofiles):
    print(ix, f)
    play(pydub.AudioSegment.from_wav(f))
    time.sleep(0.5)
    if ix > 5:
        break

#%%
###############################################
## GENERATE LONG WAVFILE AND LABELS
###############################################

# step 1: convert to wavs (to avoid slop in mp3 timings)
# food for thought: sox mp3s may have a gap when concatenating
# https://stackoverflow.com/questions/25280958/sox-concatenate-multiple-audio-files-without-a-gap-in-between

# fmt: off
#cv_clipsdir = Path("/home/mark/tinyspeech_harvard/common_voice/cv-corpus-6.1-2020-12-11/en/clips")
cv_clipsdir = Path("/media/mark/hyperion/common_voice/cv-corpus-6.1-2020-12-11/") / target_lang / "clips"
# fmt: on

assert os.path.isdir(wav_intermediates), "no destination dir available"

wavs = []
total_duration_mp3s_s = 0
for ix, stream_component in enumerate(wav_data):
    mp3name_no_ext = stream_component["mp3name_no_ext"]

    if ix % 250 == 0:
        print(ix)
    mp3path = cv_clipsdir / (mp3name_no_ext + ".mp3")
    if not os.path.exists(mp3path):
        raise ValueError("could not find", mp3path)

    duration_s = sox.file_info.duration(mp3path)
    total_duration_mp3s_s += duration_s

    wav = str(wav_intermediates / (mp3name_no_ext + ".wav"))
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
print(
    "individual wavs:",
    total_duration_wavs_s,
    "sec = ",
    total_duration_wavs_s / 60,
    "min",
)

# step 3: combine the wavs. godspeed.
combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(wavs, str(dest_dir / "stream.wav"), "concatenate")

# step 4: how long is the total wavfile? should be the sum of the individual wavs
duration_s = sox.file_info.duration(dest_dir / "stream.wav")
print("concatenated wav:", duration_s, "sec = ", duration_s / 60, "min")

# step 5: generate labels using the wav file durations, not the sloppy mp3 file durations

target_times_s = []
current_sentence_start_s = 0
for ix, stream_component in enumerate(wav_data):
    mp3name_no_ext = stream_component["mp3name_no_ext"]
    wavpath = dest_dir / "wavs" / (mp3name_no_ext + ".wav")
    sentence_duration_s = sox.file_info.duration(wavpath)
    if not stream_component["is_target"]:
        # add full duration of non-target sentence to current offset
        current_sentence_start_s += sentence_duration_s
        continue
    start_s = stream_component["start_s"]
    end_s = stream_component["end_s"]
    target_utterance_start_s = current_sentence_start_s + float(start_s)
    target_utterance_end_s = current_sentence_start_s + float(end_s)
    target_times_s.append((target_utterance_start_s, target_utterance_end_s))
    current_sentence_start_s += sentence_duration_s

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
print((end_samples - start_samples) / sr)

# https://github.com/jiaaro/pydub/blob/master/API.markdown
play(pydub.AudioSegment(data=utterance, sample_width=2, frame_rate=sr, channels=1))


#%%
# Count the number of words in the stream.wav file to estimate word rate FPR
# NOTE: skip the target words or else the FPR will be inaccurate!
sse = pathlib.Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
stream_data_pkl = sse / "iaith" / "streaming_test_data.pkl"

with open(stream_data_pkl, "rb") as fh:
    stream_data = pickle.load(fh)

#%%
def count_number_of_non_target_words_in_stream(
    target_word: str,
    mp3_files_noext_used_in_stream: Set[str],
    lang_isocode="en",
    alignment_basedir="/home/mark/tinyspeech_harvard/common-voice-forced-alignments/",
):
    word_count_without_targets = 0
    target_word_counts = 0
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
            if mp3name_no_extension not in mp3_files_noext_used_in_stream:
                continue
            words = row[2].split()
            for word in words:
                if word == target_word:
                    # print(words, mp3name_no_extension)
                    target_word_counts += 1
                else:
                    word_count_without_targets += 1
    return word_count_without_targets, target_word_counts


files_no_ext = set([d["mp3name_no_ext"] for d in stream_data["wav_data"]])
target_word = stream_data["target_word"]
target_lang = stream_data["target_lang"]
wcwt = count_number_of_non_target_words_in_stream(
    target_word, files_no_ext, lang_isocode=target_lang
)
print("# WORDS IN STREAM:", wcwt)

#%%


#%%
######## GENERATE FULL TRANSCRIPTION FOR VIDEO ###

sse = Path("/home/mark/tinyspeech_harvard/streaming_sentence_experiments/")
stream_data_pkl = sse / target / "streaming_test_data.pkl"

destination_pkl = sse / target / "full_transcription.pkl"
assert not os.path.isfile(destination_pkl), f"{destination_pkl} already exists"

with open(stream_data_pkl, "rb") as fh:
    stream_data = pickle.load(fh)

full_transcription = []
prev_wav_start = 0
for stream_component in stream_data["wav_data"]:
    mp3name_no_ext = stream_component["mp3name_no_ext"]
    tg_path = mp3_to_textgrid[mp3name_no_ext]
    transcription = word_extraction.full_transcription_timings(tg_path)
    # add offset for previous wavs
    transcription = [
        (w, start + prev_wav_start, end + prev_wav_start)
        for (w, start, end) in transcription
    ]
    full_transcription.extend(transcription)

    # "wavs" might now be "intermediate_wavs"
    wav = sse / target / "wavs" / (mp3name_no_ext + ".wav")
    duration_s = sox.file_info.duration(wav)
    prev_wav_start += duration_s

with open(destination_pkl, "wb") as fh:
    pickle.dump(full_transcription, fh)

# %%

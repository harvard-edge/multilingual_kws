#%%
import numpy as np
import os
import glob
import datetime
import pandas as pd
import shutil
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


# %%


def timings_for_target(target_word, target_lang):
    """extract all timings for a given target word and language"""
    mp3_to_textgrid = word_extraction.generate_filemap(lang_isocode=target_lang)
    print("total num mp3s in language", len(mp3_to_textgrid.keys()))
    timings_w_dups, notfound = word_extraction.generate_wordtimings(
        {target_word}, mp3_to_textgrid, lang_isocode=target_lang
    )
    # debugging
    # how many mp3s have two or more utterances of the target
    print("notfound", len(notfound))
    print("num timings", len(timings_w_dups[target_word]))
    print(timings_w_dups[target_word][0])
    mp3s = [t[0] for t in timings_w_dups[target_word]]
    n_samples = len(mp3s)
    n_mp3s = len(set(mp3s))
    # fmt: off
    print("num samples", n_samples, "num mp3s for samples", n_mp3s, " -- have 2 or more utterances of the target:", n_samples - n_mp3s)
    # fmt: on

    # remove duplicates (removes any mp3 that shows up more than once, not just subsequent occurences)
    seen_mp3s = set()
    seen_multiple = set()
    for t in timings_w_dups[target_word]:
        mp3_filename = t[0]
        if mp3_filename in seen_mp3s:
            seen_multiple.add(mp3_filename)
        seen_mp3s.add(mp3_filename)

    timings = {target_word: []}
    for t in timings_w_dups[target_word]:
        mp3_filename = t[0]
        if mp3_filename in seen_multiple:
            continue
        timings[target_word].append(t)

    # fmt: off
    print("after removing duplicates", len(timings[target_word]), ", dropped mp3s:", n_mp3s - len(timings[target_word]))
    # fmt: on
    return mp3_to_textgrid, timings


def select_samples(
    target_word, timings, NUM_SAMPLES_FOR_STREAMING_WAV=100, NUM_SHOTS=5, NUM_VAL=30
):
    #####################################################
    ##    select examples to string into a long wav
    ####################################################
    # https://stackoverflow.com/questions/23445936/numpy-random-choice-of-tuples
    ix_samples = np.random.choice(
        len(timings[target_word]),
        NUM_SAMPLES_FOR_STREAMING_WAV + NUM_SHOTS + NUM_VAL,
        replace=False,
    )
    samples = np.array(timings[target_word])[ix_samples]
    mp3s = set([t[0] for t in samples])
    assert (
        len(mp3s) == NUM_SAMPLES_FOR_STREAMING_WAV + NUM_SHOTS + NUM_VAL
    ), "an mp3 was selected with multiple targets in the same sentence"

    shots = samples[:NUM_SHOTS]
    val = samples[NUM_SHOTS : NUM_SHOTS + NUM_VAL]
    target_stream = samples[NUM_SHOTS + NUM_VAL :]

    # fmt: off
    print("numshots", len(shots), "numval", len(val), "num stream components", len(target_stream))
    # fmt: on

    # select non-target sentences to intersperse
    non_targets = word_extraction.random_non_target_sentences(
        num_sentences=len(target_stream),
        words_to_exclude={target_word},
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
    return dict(
        shot_targets=shots,
        val_targets=val,
        wav_data=wav_data,
        target_stream=target_stream,
        non_targets=non_targets,
    )


def generate_extractions(
    timings_for_split,
    dest_dir,
    target_lang,
    cv_clipsdir,
):
    assert os.path.isdir(cv_clipsdir), "cv data not found"
    assert os.path.isdir(dest_dir), "no dir exists"
    assert os.listdir(dest_dir) == [], "data is present already"
    for mp3name_no_ext, start_s, end_s in timings_for_split:
        print(mp3name_no_ext, start_s, end_s)
        word_extraction.extract_shot_from_mp3(
            mp3name_no_ext,
            float(start_s),
            float(end_s),
            dest_dir=dest_dir,
            cv_clipsdir=cv_clipsdir,
        )


def generate_stream_and_labels(
    dest_dir,
    wav_intermediates,
    wav_data,
    target_word,
    target_lang,
    cv_clipsdir,
):
    ###############################################
    ## GENERATE LONG WAVFILE AND LABELS
    ###############################################
    # step 1: convert to wavs (to avoid slop in mp3 timings)
    # food for thought: sox mp3s may have a gap when concatenating
    # https://stackoverflow.com/questions/25280958/sox-concatenate-multiple-audio-files-without-a-gap-in-between

    assert os.path.isdir(cv_clipsdir), "cv data not found"

    assert os.path.isdir(dest_dir), "no dest dir available"
    assert os.path.isdir(
        wav_intermediates
    ), "no destination intermediate wav dir available"
    assert os.listdir(wav_intermediates) == [], "intermediate wav dir not empty"

    label_file = dest_dir / "streaming_labels.txt"
    wav_stream_file = str(dest_dir / "streaming_test.wav")

    assert not os.path.isfile(label_file), "label file exists already"
    assert not os.path.isfile(wav_stream_file), "wav stream exists already"

    wavs = []
    total_duration_mp3s_s = 0
    for ix, stream_component in enumerate(wav_data):
        mp3name_no_ext = stream_component["mp3name_no_ext"]

        if ix % 250 == 0:
            print("mp3 to wav", ix)
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
    combiner.build(wavs, wav_stream_file, "concatenate")

    # step 4: how long is the total wavfile? should be the sum of the individual wavs
    duration_s = sox.file_info.duration(wav_stream_file)
    print("concatenated wav:", duration_s, "sec = ", duration_s / 60, "min")

    # step 5: generate labels using the wav file durations, not the sloppy mp3 file durations

    target_times_s = []
    current_sentence_start_s = 0
    for ix, stream_component in enumerate(wav_data):
        mp3name_no_ext = stream_component["mp3name_no_ext"]
        wavpath = wav_intermediates / (mp3name_no_ext + ".wav")
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
    with open(label_file, "w") as fh:
        for start_s, _ in target_times_s:
            start_ms = start_s * 1000
            fh.write(f"{target_word}, {start_ms}\n")
    return target_times_s


def count_number_of_non_target_words_in_stream(
    target_word: str,
    target_lang: str,
    wav_data: List,
    alignment_basedir="/home/mark/tinyspeech_harvard/common-voice-forced-alignments/",
):
    # Count the number of words in streaming_test.wav file to estimate word rate FPR
    # NOTE: this skips the target word (or else the FPR will be inaccurate!)
    mp3_files_noext_used_in_stream = set([d["mp3name_no_ext"] for d in wav_data])

    non_target_word_count = 0
    target_word_count = 0
    # common voice csv from DeepSpeech/import_cv2.py
    csvpath = pathlib.Path(alignment_basedir) / target_lang / "validated.csv"
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
                    target_word_count += 1
                else:
                    non_target_word_count += 1
    return target_word_count, non_target_word_count


def write_full_transcription(
    transcript_destination_pkl_file, wav_intermediates, mp3_to_textgrid, wav_data
):
    ## GENERATE FULL TRANSCRIPTION FOR VIDEO ###
    assert not os.path.isfile(
        transcript_destination_pkl_file
    ), f"{transcript_destination_pkl_file} exists"
    assert os.path.isdir(wav_intermediates), "no intermediate wavs found"

    full_transcription = []
    prev_wav_start = 0
    for stream_component in wav_data:
        mp3name_no_ext = stream_component["mp3name_no_ext"]
        tg_path = mp3_to_textgrid[mp3name_no_ext]
        transcription = word_extraction.full_transcription_timings(tg_path)
        # add offset for previous wavs
        transcription = [
            (w, start + prev_wav_start, end + prev_wav_start)
            for (w, start, end) in transcription
        ]
        full_transcription.extend(transcription)

        wav = wav_intermediates / (mp3name_no_ext + ".wav")
        duration_s = sox.file_info.duration(wav)
        prev_wav_start += duration_s

    with open(transcript_destination_pkl_file, "wb") as fh:
        pickle.dump(full_transcription, fh)


#%%
def find_target_counts(target_word, common_counts):
    for ix, (w, c) in enumerate(common_counts):
        if w == target_word:
            print(target_word, ix, c)


#%%

# TODO(mmaz): choose words + models already trained in paper_data -- exclude mp3file_no_ext in N_SHOTS from timings

paper_data = Path("/home/mark/tinyspeech_harvard/paper_data")
# in_embedding_mlc_pkl = paper_data / "multilang_classification_in_embedding_all_lang_targets.pkl"
# with open(in_embedding_mlc_pkl, 'rb') as fh:
#     in_embedding_mlc = pickle.load(fh)
# for target_data in in_embedding_mlc:
#     print(target_data.keys())
#     break
data_dir = Path("/home/mark/tinyspeech_harvard/frequent_words")
target_data = []
target_word_counts = {}
#multilang_results_dir = paper_data / "multilang_classification"
multilang_results_dir = paper_data / "ooe_multilang_classification"
for multiclass_lang in os.listdir(multilang_results_dir):
    lang_isocode = multiclass_lang.split("_")[-1]
    print("lang_isocode", lang_isocode)
    for result_file in os.listdir(multilang_results_dir / multiclass_lang / "results"):
        target_word = os.path.splitext(result_file.split("_")[-1])[0]
        # find model path for this target
        model_file = None
        for m in os.listdir(multilang_results_dir / multiclass_lang / "models"):
            m_target = m.split("_")[-1]
            if m_target == target_word:
                model_file = multilang_results_dir / multiclass_lang / "models" / m
        if not model_file:
            raise ValueError
        print(lang_isocode, target_word)
        wav_dir = data_dir / lang_isocode / "clips" / target_word
        num_wavs = len(glob.glob(str(wav_dir / "*.wav")))
        target_word_counts[f"{lang_isocode}_{target_word}"] = num_wavs
        d = (lang_isocode, target_word, multilang_results_dir / multiclass_lang, model_file)
        target_data.append(d)
print(len(target_word_counts.keys()))

#%%

#%%
# number of target wavs per keyword
fig,ax = plt.subplots()
ax.bar(target_word_counts.keys(), target_word_counts.values())
ax.set_xticklabels(target_word_counts.keys(), rotation=90)
#  # ax.set_xlabel("language")
#  # ax.set_ylabel("")
ax.set_ylim(0,800)
fig.set_size_inches(30,10)

#%%
# use existing models and keywords

#base_dir = Path("/home/mark/tinyspeech_harvard/paper_data/streaming_batch_sentences")
base_dir = Path("/home/mark/tinyspeech_harvard/paper_data/ooe_streaming_batch_sentences")
frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")
n_targets = len(target_data)
for ix, (target_lang, target_word, multilang_class_dir, model_file) in enumerate(target_data):
    if target_lang == "fr" or target_lang == "rw":
        continue
    print(f"\n\n\n:::::::::::{ix} / {n_targets} ::::::::::: TARGET LANG: {target_lang}")
    start_gen = datetime.datetime.now()

    counts = word_extraction.wordcounts(f"/home/mark/tinyspeech_harvard/common-voice-forced-alignments/{target_lang}/validated.csv")

    dest_dir = base_dir / f"streaming_{target_lang}" / f"streaming_{target_word}"
    if os.path.isdir(dest_dir):
        print("already generated data for this word", dest_dir)
        continue

    mp3_to_textgrid, timings = timings_for_target(target_word, target_lang)

    if len(timings[target_word]) < 45: # even though we already have a trained model 
        print("ERROR: not enough data in timings")
        continue
    # fmt:on
    sample_data = select_samples(target_word, timings, NUM_SAMPLES_FOR_STREAMING_WAV=40, NUM_SHOTS=5, NUM_VAL=0)
    # fmt:off

    n_target_in_stream, n_nontarget_in_stream = count_number_of_non_target_words_in_stream(
        target_word, target_lang, sample_data["wav_data"]
    )
    # find cv clips dir
    cv_clipsdir=None
    for cv_datadir in ["cv-corpus-6.1-2020-12-11/", "cv-corpus-5.1-2020-06-22"]:
        #fmt: off
        cv_clipsdir = Path("/media/mark/hyperion/common_voice") / cv_datadir / target_lang / "clips"
        # fmt : on
        if os.path.isdir(cv_clipsdir):
            break
    if cv_clipsdir is None:
        print("cant find cv data", target_lang)
        continue

    shot_targets = dest_dir / "n_shots"
    val_targets = dest_dir / "val"
    wav_intermediates = dest_dir / "wav_intermediates"
    model_dir = dest_dir / "model"
    streaming_test_data = dest_dir / "streaming_test_data.pkl"
    transcript_destination_pkl_file = dest_dir / "full_transcript.pkl"

    print("CREATING DIR STRUCTURE")
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(shot_targets, exist_ok=False)
    os.makedirs(val_targets, exist_ok=False)
    os.makedirs(wav_intermediates, exist_ok=False)
    os.makedirs(model_dir, exist_ok=False)
    with open(streaming_test_data, "wb") as fh:
        stream_data = dict(
            target_word=target_word,
            target_lang=target_lang,
            mp3_to_textgrid=mp3_to_textgrid,
            timings=timings,
            sample_data=sample_data,
            num_target_words_in_stream=n_target_in_stream,
            num_non_target_words_in_stream=n_nontarget_in_stream,
        )
        pickle.dump(stream_data, fh)

    model_name = os.path.split(model_file)[-1]
    shutil.copytree(model_file, model_dir / model_name)

    generate_extractions(
        timings_for_split=sample_data["shot_targets"],
        dest_dir=shot_targets,
        target_lang=target_lang,
        cv_clipsdir=cv_clipsdir,
    )
    generate_extractions(
        timings_for_split=sample_data["val_targets"],
        dest_dir=val_targets,
        target_lang=target_lang,
        cv_clipsdir=cv_clipsdir,
    )

    target_times_s = generate_stream_and_labels(
        dest_dir, wav_intermediates, sample_data["wav_data"], target_word, target_lang, cv_clipsdir=cv_clipsdir,
    )

    print("saving transcription to", transcript_destination_pkl_file)
    write_full_transcription(
        transcript_destination_pkl_file,
        wav_intermediates,
        mp3_to_textgrid,
        sample_data["wav_data"],
    )
    end_gen = datetime.datetime.now()
    print("elapsed time", end_gen - start_gen)

#%%
###############################
############
#####
###
#
# generate new streaming data
###
#####
#########

base_dir = Path("/home/mark/tinyspeech_harvard/streaming_batch_sentences")
frequent_words = Path("/home/mark/tinyspeech_harvard/frequent_words")
for ix in range(3):
    langs = os.listdir(frequent_words)
    langs = ["cy", "eu", "cs", "it", "nl", "fr"]
    target_lang = np.random.choice(langs)
    if target_lang == "fa":
        print("skipping persian due to unicode issues")
        continue
    print(f":::::::::::{ix}::::::::::: TARGET LANG: {target_lang}")

    counts = word_extraction.wordcounts(
        f"/home/mark/tinyspeech_harvard/common-voice-forced-alignments/{target_lang}/validated.csv"
    )
    START_OFFSET = 180
    NUM_COMMON = 300
    MIN_NUM_WORDS = 200
    common_words = counts.most_common(NUM_COMMON)

    N_ATTEMPTS = 30
    attempts = 0
    while attempts < N_ATTEMPTS:
        random_common_ix = np.random.randint(low=START_OFFSET, high=NUM_COMMON)
        if common_words[random_common_ix][1] > MIN_NUM_WORDS:
            break
        attempts += 1
    if attempts > N_ATTEMPTS:
        print("not enough data", target_lang)
        continue
    print(common_words[random_common_ix])

    target_word = common_words[random_common_ix][0]

    dest_dir = base_dir / target_word
    if os.path.isdir(dest_dir):
        print("already generated data for this word", dest_dir)
        continue

    mp3_to_textgrid, timings = timings_for_target(target_word, target_lang)
    if len(timings[target_word]) < 100 + 5 + 30:
        print("ERROR: not enough data in timings")
        continue
    sample_data = select_samples(target_word, timings)

    n_target_in_stream, n_nontarget_in_stream = count_number_of_non_target_words_in_stream(
        target_word, target_lang, sample_data["wav_data"]
    )
    # find cv clips dir
    cv_clipsdir=None
    for cv_datadir in ["cv-corpus-6.1-2020-12-11/", "cv-corpus-5.1-2020-06-22"]:
        #fmt: off
        cv_clipsdir = Path("/media/mark/hyperion/common_voice") / cv_datadir / target_lang / "clips"
        # fmt : on
        if os.path.isdir(cv_clipsdir):
            break
    if cv_clipsdir is None:
        print("cant find cv data", target_lang)
        continue


    shot_targets = dest_dir / "n_shots"
    val_targets = dest_dir / "val"
    wav_intermediates = dest_dir / "wav_intermediates"
    streaming_test_data = dest_dir / "streaming_test_data.pkl"
    transcript_destination_pkl_file = dest_dir / "full_transcript.pkl"

    print("CREATING DIR STRUCTURE")
    os.makedirs(dest_dir, exist_ok=False)
    os.makedirs(shot_targets, exist_ok=False)
    os.makedirs(val_targets, exist_ok=False)
    os.makedirs(wav_intermediates, exist_ok=False)
    with open(streaming_test_data, "wb") as fh:
        stream_data = dict(
            target_word=target_word,
            target_lang=target_lang,
            mp3_to_textgrid=mp3_to_textgrid,
            timings=timings,
            sample_data=sample_data,
            num_target_words_in_stream=n_target_in_stream,
            num_non_target_words_in_stream=n_nontarget_in_stream,
        )
        pickle.dump(stream_data, fh)

    generate_extractions(
        timings_for_split=sample_data["shot_targets"],
        dest_dir=shot_targets,
        target_lang=target_lang,
        cv_clipsdir=cv_clipsdir,
    )
    generate_extractions(
        timings_for_split=sample_data["val_targets"],
        dest_dir=val_targets,
        target_lang=target_lang,
        cv_clipsdir=cv_clipsdir,
    )

    target_times_s = generate_stream_and_labels(
        dest_dir, wav_intermediates, sample_data["wav_data"], target_word, target_lang, cv_clipsdir=cv_clipsdir,
    )

    print("saving transcription to", transcript_destination_pkl_file)
    write_full_transcription(
        transcript_destination_pkl_file,
        wav_intermediates,
        mp3_to_textgrid,
        sample_data["wav_data"],
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
######################
# how good are the timings? listen to random extractions from the full stream
#####################
sr, data = wavfile.read(dest_dir / "streaming_test.wav")
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


# %%

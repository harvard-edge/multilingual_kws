#%%
from dataclasses import asdict
import os
import glob
import shutil
import json
import subprocess
import time
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sox
import pydub
import pydub.playback
import textgrid
import seaborn as sns

sns.set()
sns.set_style("darkgrid")
sns.set_palette("bright")

# %%

import sys

sys.path.insert(0, str(Path.cwd().parents[0]))
from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
from embedding import input_data
from luganda_info import WavTranscript

# %%
l_csv = Path("/media/mark/hyperion/makerere/uliza-clips/transcripts.csv")
counts = word_extraction.wordcounts(l_csv, skip_header=True, transcript_column=2)

# %%
words = os.listdir(Path.home() / "tinyspeech_harvard/luganda/covid_keyword_utterances")
words.append('covid')
cs = []
for w in words:
    cs.append((w, counts[w]))
cs.sort(key=lambda c: c[1], reverse=True)
cs[:20]
#%%


# %%
# Luganda

l_data = Path("/media/mark/hyperion/makerere/luganda/luganda/")
l_csv = l_data / "data.csv"
counts = word_extraction.wordcounts(l_csv, skip_header=False, transcript_column=1)
workdir = Path("/home/mark/tinyspeech_harvard/luganda")

# %%
# find keywords
N_WORDS_TO_SAMPLE = 10
MIN_CHAR_LEN = 4
SKIP_FIRST_N = 5

counts.most_common(SKIP_FIRST_N)
# %%
non_stopwords = counts.copy()
# get rid of words that are too short
to_expunge = counts.most_common(SKIP_FIRST_N)
for k, _ in to_expunge:
    del non_stopwords[k]

longer_words = [kv for kv in non_stopwords.most_common() if len(kv[0]) >= MIN_CHAR_LEN]

print("num words to be extracted", len(longer_words[:N_WORDS_TO_SAMPLE]))
print("counts for last word", longer_words[N_WORDS_TO_SAMPLE - 1])
print("words:\n", " ".join([l[0] for l in longer_words[:N_WORDS_TO_SAMPLE]]))


# %%
# visualize frequencies of top words
fig, ax = plt.subplots()
topn = longer_words[:N_WORDS_TO_SAMPLE]
ax.bar([c[0] for c in topn], [c[1] for c in topn])
ax.set_xticklabels([c[0] for c in topn], rotation=70)
# ax.set_ylim([0, 3000])
# fig.set_size_inches(40, 10)

# %%
keyword = "covid"
wav_transcript = []
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    for ix, row in enumerate(reader):
        words = row[1].split()
        for w in words:
            if w == keyword:
                wav_transcript.append((row[0], row[1]))
                break  # may contain more than one instance
print(len(wav_transcript))

# %%
# listen to random samples
ix = np.random.randint(len(wav_transcript))
w = l_data / "clips" / wav_transcript[ix][0]
print(ix, w, wav_transcript[ix][1])
pydub.playback.play(pydub.AudioSegment.from_file(w))

# %%
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
silence_padded = workdir / "silence_padded"

# %%
# extract covid from first 5 wavs using audacity
fiveshot_dest = workdir / "originals"
os.makedirs(fiveshot_dest)
for ix in range(5):
    w = l_data / "clips" / wav_transcript[ix][0]
    shutil.copy2(w, fiveshot_dest)

# %%
# pad with silence out to 1 second

unpadded = workdir / "unpadded"
# os.makedirs(silence_padded)

for f in os.listdir(unpadded):
    src = str(unpadded / f)
    print(src)
    duration_s = sox.file_info.duration(src)
    if duration_s < 1:
        pad_amt_s = (1.0 - duration_s) / 2.0
    else:
        raise ValueError("utterance longer than 1s", src)

    dest = silence_padded / f
    # words can appear multiple times in a sentence: above should have filtered these
    if os.path.exists(dest):
        raise ValueError("already exists:", dest)

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
    transformer.build(src, str(dest))


# %%
# select random wavs without the keyword to intersperse stream with
non_targets = []
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    for ix, row in enumerate(reader):
        words = row[1].split()
        has_keyword = False
        for w in words:
            if w == keyword:
                has_keyword = True
        if not has_keyword:
            non_targets.append((row[0], row[1]))
# %%
stream_data = wav_transcript[5:]
n_stream_wavs = len(stream_data)
print(n_stream_wavs)
selected_nontargets = np.random.choice(
    range(len(non_targets)), n_stream_wavs, replace=False
)

# %%
stream_info_file = workdir / "stream_info.pkl"
assert not os.path.isfile(stream_info_file), "already exists"
# make streaming wav
intermediate_wavdir = workdir / "intermediate_wavs"
os.makedirs(intermediate_wavdir)

stream_info = []
stream_wavs = []
for ix, ((target_wav, target_transcript), nontarget_ix) in enumerate(
    zip(stream_data, selected_nontargets)
):
    tw = l_data / "clips" / target_wav
    nw = l_data / "clips" / non_targets[nontarget_ix][0]

    durations_s = []
    # convert all to same samplerate
    for w in [tw, nw]:
        dest = str(intermediate_wavdir / w.name)
        transformer = sox.Transformer()
        transformer.convert(samplerate=16000)  # from 48K mp3s
        transformer.build(str(w), dest)
        stream_wavs.append(dest)
        durations_s.append(sox.file_info.duration(dest))

    # record transcript info
    tw_info = dict(
        ix=2 * ix,
        wav=target_wav,
        transcript=target_transcript,
        duration_s=durations_s[0],
    )
    nw_info = dict(
        ix=2 * ix + 1,
        wav=non_targets[nontarget_ix][0],
        transcript=non_targets[nontarget_ix][1],
        duration_s=durations_s[1],
    )
    stream_info.extend([tw_info, nw_info])

assert len(stream_wavs) == n_stream_wavs * 2, "not enough stream data"
stream_wavfile = str(workdir / "covid_stream.wav")

combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(stream_wavs, stream_wavfile, "concatenate")

dur_info = sum([d["duration_s"] for d in stream_info])
print(sox.file_info.duration(stream_wavfile), "seconds in length", dur_info)

with open(stream_info_file, "wb") as fh:
    pickle.dump(stream_info, fh)

# %%

# load embedding model
traindir = Path(f"/home/mark/tinyspeech_harvard/multilang_embedding")

# SELECT MODEL
base_model_path = (
    traindir / "models" / "multilang_resume40_resume05_resume20_resume22.007-0.7981/"
)

model_dir = Path(f"/home/mark/tinyspeech_harvard/multilang_analysis_ooe/")
unknown_collection_path = model_dir / "unknown_collection.pkl"
with open(unknown_collection_path, "rb") as fh:
    unknown_collection = pickle.load(fh)
# unknown_lang_words = unknown_collection["unknown_lang_words"]
unknown_files = unknown_collection["unknown_files"]
# oov_lang_words = unknown_collection["oov_lang_words"]
# commands = unknown_collection["commands"]
# unknown_words = set([lw[1] for lw in unknown_lang_words])

# %%

target_n_shots = os.listdir(silence_padded)

train_files = [str(silence_padded / w) for w in target_n_shots]
# reuse train for val
val_files = [str(silence_padded / w) for w in target_n_shots]
print(train_files)

# %%

model_dest_dir = workdir / "model"
model_settings = input_data.standard_microspeech_model_settings(3)
name, model, details = transfer_learning.transfer_learn(
    target=keyword,
    train_files=train_files,
    val_files=val_files,
    unknown_files=unknown_files,
    num_epochs=4,  # 9
    num_batches=1,  # 3
    batch_size=64,
    model_settings=model_settings,
    base_model_path=base_model_path,
    base_model_output="dense_2",
    csvlog_dest=model_dest_dir / "log.csv",
)
print("saving", name)
model.save(model_dest_dir / name)

# %%
# sanity check model outputs
specs = [input_data.file2spec(model_settings, f) for f in val_files]
specs = np.expand_dims(specs, -1)
print(specs.shape)
preds = model.predict(specs)
amx = np.argmax(preds, axis=1)
print(amx)
print("VAL ACCURACY", amx[amx == 2].shape[0] / preds.shape[0])
print("--")

with np.printoptions(precision=3, suppress=True):
    print(preds)

# %%

# use model above
# modelpath = model_dest_dir / name
# or load existing model
modelpath = workdir / "model" / "xfer_epochs_4_bs_64_nbs_1_val_acc_1.00_target_covid"

# %%
# run inference
os.makedirs(workdir / "results")
streamwav = workdir / "covid_stream.wav"
# create empty label file
with open(workdir / "empty.txt", "w"):
    pass
empty_gt = workdir / "empty.txt"
dest_pkl = workdir / "results" / "streaming_results.pkl"
dest_inf = workdir / "results" / "inferences.npy"
streamtarget = sa.StreamTarget(
    "lu", "covid", modelpath, streamwav, empty_gt, dest_pkl, dest_inf
)

# %%
start = datetime.datetime.now()
sa.eval_stream_test(streamtarget)
end = datetime.datetime.now()
print("time elampsed (for all thresholds)", end - start)

# %%
# DONE 0.05
# No ground truth yet, 151false positives
# DONE 0.1
# No ground truth yet, 374false positives
# DONE 0.15
# No ground truth yet, 478false positives
# DONE 0.2
# No ground truth yet, 535false positives
# DONE 0.25
# No ground truth yet, 505false positives
# DONE 0.3
# No ground truth yet, 440false positives
# DONE 0.35
# No ground truth yet, 365false positives
# DONE 0.39999999999999997
# No ground truth yet, 295false positives
# DONE 0.44999999999999996
# No ground truth yet, 231false positives
# DONE 0.49999999999999994
# No ground truth yet, 171false positives
# DONE 0.5499999999999999
# No ground truth yet, 130false positives
# DONE 0.6
# No ground truth yet, 98false positives
# DONE 0.65
# No ground truth yet, 72false positives
# DONE 0.7
# No ground truth yet, 46false positives
# DONE 0.75
# No ground truth yet, 26false positives
# DONE 0.7999999999999999
# No ground truth yet, 15false positives
# DONE 0.85
# No ground truth yet, 6false positives
# DONE 0.9
# No ground truth yet, 1false positives
# DONE 0.95
# No ground truth yet, 0false positives
# DONE 1.0
# No ground truth yet, 0false positives

# %%
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
with open(workdir / "stream_info.pkl", "rb") as fh:
    stream_info = pickle.load(fh)
# with open(streamtarget.destination_result_pkl, "rb") as fh:
with open(workdir / "results" / "streaming_results.pkl", "rb") as fh:
    results = pickle.load(fh)
# %%
keyword = "covid"
operating_point = 0.7
for thresh, (_, found_words, all_found_w_confidences) in results[keyword].items():
    if np.isclose(thresh, operating_point):
        break
print(len(found_words), "targets found")

# %%
stream = pydub.AudioSegment.from_file(workdir / "covid_stream.wav")

# %%
ix = 0
# %%
# listen one-by-one with transcript, or randomly
select_random = False
if select_random:
    ix = np.random.randint(len(found_words))
time_ms = found_words[ix][1]
time_s = time_ms / 1000
print(ix, time_s)
context_ms = 1000

current_duration_s = 0
for si in stream_info:
    current_duration_s += si["duration_s"]
    if time_s < current_duration_s:
        print(ix, si["transcript"])
        break

play(stream[time_ms - context_ms : time_ms + context_ms])

# %%
ix += 1

# %%
with open(workdir / "tabulate_stream_info.csv", "w") as fh:
    start_s = 0
    for si in stream_info:
        end_s = start_s + si["duration_s"]
        # fh.write(",", si["transcript"], "\n")
        result = f"False,0,{start_s:02f},{end_s:02f},{si['transcript']}\n"
        fh.write(result)
        start_s = end_s


# %%
# save detections from stream
extractions = workdir / "extractions"
os.makedirs(extractions)
for ix, (_, time_ms) in enumerate(found_words):
    print(time_ms)

    dest_wav = str(
        extractions
        / f"{ix:03d}_{keyword}_detection_thresh_{operating_point}_{time_ms}ms.wav"
    )
    print(dest_wav)
    time_s = time_ms / 1000.0

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(time_s - 1, time_s + 1)
    transformer.build(str(streamtarget.stream_wav), dest_wav)


# %%
# listen to random alignments for keyword
covid_alignments = workdir / "covid_alignments"
alignment_speakers = [
    d for d in os.listdir(covid_alignments) if os.path.isdir(covid_alignments / d)
]

speaker = np.random.choice(alignment_speakers)
tgfiles = os.listdir(covid_alignments / speaker)
tgfile = covid_alignments / speaker / tgfiles[0]

radio_data = Path("/media/mark/hyperion/makerere/uliza-clips")
wavpath = radio_data / (tgfile.stem + ".wav")

transcript = (
    Path("/media/mark/hyperion/makerere/alignment/covid_clips")
    / tgfile.stem
    / (tgfile.stem + ".lab")
)
with open(transcript, "r") as fh:
    print(fh.read())

print(tgfile, wav)

tg = textgrid.TextGrid.fromFile(tgfile)
for interval in tg[0]:
    if interval.mark != "covid":
        continue
    start_s = interval.minTime
    end_s = interval.maxTime

wav = pydub.AudioSegment.from_file(wavpath)
play(wav[start_s * 1000 : end_s * 1000])

# %%
# save out covid alignments
covid_alignments = workdir / "covid_alignments"
alignment_speakers = os.listdir(covid_alignments)
for ix, speaker in enumerate(alignment_speakers):
    if not os.path.isdir(covid_alignments / speaker):
        continue  # skip unaligned.txt
    if ix % 100 == 0:
        print(ix)
    tgfiles = os.listdir(covid_alignments / speaker)
    tgfile = covid_alignments / speaker / tgfiles[0]

    radio_data = Path("/media/mark/hyperion/makerere/uliza-clips")
    wavpath = radio_data / (tgfile.stem + ".wav")

    tg = textgrid.TextGrid.fromFile(tgfile)
    for interval in tg[0]:
        if interval.mark != "covid":
            continue
        start_s = interval.minTime
        end_s = interval.maxTime

    wav = pydub.AudioSegment.from_file(wavpath)

    dest = workdir / "1k_covid_alignments" / (tgfile.stem + ".wav")

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(start_s, end_s)
    if end_s - start_s < 1:
        pad_amt_s = (1.0 - (end_s - start_s)) / 2.0
        transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
    else:
        raise ValueError("generated extraction longer than 1s")
    transformer.build(str(wavpath), str(dest))

# %%
# combine
dest = str(workdir / "all_covid_alignments.mp3")
mp3s = glob.glob(str(workdir / "1k_covid_alignments" / "*.mp3"))
combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(mp3s, dest, "concatenate")

# %%
###########################################
#########  generate streaming data ########
###########################################

l_data = Path("/media/mark/hyperion/makerere/uliza-clips")
l_csv = l_data / "transcripts.csv"

# l_test = Path("/media/mark/hyperion/makerere/alignment/akawuka/")
# l_clips = l_test / "akawuka_clips"

l_test = Path("/media/mark/hyperion/makerere/alignment/cs288/")
l_clips = l_test / "cs288_clips"

# l_test = Path("/media/mark/hyperion/makerere/alignment/covid/")
# l_clips = l_test / "covid_clips"

l_alignments = l_test / "alignments"

# corona  covid  mask  okugema  ssennyiga
# alt spelling: senyiga

keyword = "mask"

# %%

# select random wavs without the keyword to intersperse stream with
non_targets = []
keywords_to_exclude = set(
    # TODO(mmaz): find a better method to exclude variants in spelling/plurals/etc
    [keyword, "mask", "masiki", "masks"]
    # [keyword, "corona", "korona", "kolona", "coronavirus"]
)
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header [wav_filename,wav_filesize,transcript]
    for ix, row in enumerate(reader):
        wav_filename = l_data / row[0]
        assert os.path.isfile(wav_filename), "wav not found"
        transcript = row[2]
        has_kw = any([w in keywords_to_exclude for w in transcript.split()])
        if not has_kw:
            non_targets.append(WavTranscript(wav=wav_filename, transcript=transcript))

# assemble wav transcripts with timings from alignments
# supports multiple keywords in a single wav
keyword_wav_transcripts = []
for a in os.listdir(l_alignments):
    if not os.path.isdir(l_alignments / a):
        continue  # unaligned.txt

    wav = l_clips / a / f"{a}.wav"

    lab = l_clips / a / f"{a}.lab"
    with open(lab, "r") as fh:
        transcript = fh.read()

    has_kw = any([keyword == w for w in transcript.split()])
    if not has_kw:
        continue

    tgfile = l_alignments / a / f"{a}.TextGrid"
    tg = textgrid.TextGrid.fromFile(tgfile)

    occurences_s = []

    for interval in tg[0]:
        if interval.mark != keyword:
            continue
        start_s = interval.minTime
        end_s = interval.maxTime
        occurences_s.append((start_s, end_s))

    if occurences_s == []:
        raise ValueError("why did we get here")

    keyword_wav_transcripts.append(
        WavTranscript(
            wav=wav,
            transcript=transcript,
            keyword=keyword,
            occurences_s=occurences_s,
            tgfile=tgfile,
        )
    )

print("data", len(keyword_wav_transcripts), len(non_targets))

# %%
# listen to random samples
ix = np.random.randint(len(keyword_wav_transcripts))
w = keyword_wav_transcripts[ix]
print(ix, w.wav)


def decorate(word, keyword):
    if word == keyword:
        return f"[::{word}::]"
    return word


decorated = [decorate(word, keyword=keyword) for word in w.transcript.split()]
print(" ".join(decorated))
start_s = w.occurences_s[0][0]
end_s = w.occurences_s[0][1]
audio = pydub.AudioSegment.from_file(w.wav)
pydub.playback.play(audio[start_s * 1000 - 500 : end_s * 1000 + 500])

# %%
# generate stream and groundtruth data
NUM_TARGETS = 80

workdir = Path("/home/mark/tinyspeech_harvard/luganda/demo_eval") / keyword
os.makedirs(workdir, exist_ok=True)
dest_wavfile = str(workdir / f"{keyword}_stream.wav")
dest_mp3 = str(workdir / f"{keyword}_stream.mp3")
groundtruth_data = workdir / f"{keyword}_groundtruth.pkl"
full_transcript_file = workdir / f"{keyword}_full_transcript.json"
groundtruth_txt = workdir / f"{keyword}_groundtruth_labels.txt"
# os.makedirs(workdir / "cs288_test" / keyword, exist_ok=True)
# dest_wavfile = str(workdir / "cs288_test" / keyword / f"{keyword}_stream.wav")
# groundtruth_data = workdir / "cs288_test" / keyword / f"{keyword}_groundtruth.pkl"
assert not os.path.isfile(dest_wavfile), "already exists"
assert not os.path.isfile(dest_mp3), "already exists"
assert not os.path.isfile(groundtruth_data), "already exists"
assert not os.path.isfile(full_transcript_file), "already exists"
assert not os.path.isfile(groundtruth_txt), "already exists"

ntlog = set()
ixs = np.random.choice(range(len(keyword_wav_transcripts)), NUM_TARGETS, replace=False)

total_wav_duration_s = 0.0
to_combine = []
groundtruth_target_times_ms = []

stream_data = []
transcript = []  # TODO(mmaz): need TGFiles for non-targets also

for ix in ixs:
    target = keyword_wav_transcripts[ix]
    non_target_ix = np.random.randint(len(non_targets))
    ntlog.add(non_target_ix)
    non_target = non_targets[non_target_ix]

    target_duration_s = sox.file_info.duration(target.wav)
    nontarget_duration_s = sox.file_info.duration(non_target.wav)

    for start_s, _ in target.occurences_s:
        t_ms = (total_wav_duration_s + start_s) * 1000
        groundtruth_target_times_ms.append(t_ms)

    ## full transcript
    target_transcription = word_extraction.full_transcription_timings(target.tgfile)
    target_transcription = [
        dict(word=w, start=start + total_wav_duration_s, end=end + total_wav_duration_s)
        for (w, start, end) in target_transcription
    ]
    transcript.append(
        dict(
            transcript_type="target",
            transcript=target.transcript,
            start=total_wav_duration_s,
            end=total_wav_duration_s + target_duration_s,
            transcript_perword=target_transcription,
        )
    )
    transcript.append(
        dict(
            trancript_type="nontarget",
            transcript=non_target.transcript,
            start=(total_wav_duration_s + target_duration_s),
            end=(total_wav_duration_s + target_duration_s + nontarget_duration_s),
        )
    )

    total_wav_duration_s += target_duration_s + nontarget_duration_s

    to_combine.extend([str(target.wav), str(non_target.wav)])

    td = asdict(target)
    td["duration_s"] = target_duration_s
    nd = asdict(non_target)
    nd["duration_s"] = nontarget_duration_s
    stream_data.extend([td, nd])

groundtruth = dict(
    groundtruth_target_times_ms=groundtruth_target_times_ms,
    stream_data=stream_data,
    keyword_wav_transcripts=keyword_wav_transcripts,
    non_targets=non_targets,
    ixs=ixs,
)

if len(ntlog) < NUM_TARGETS:
    print("Warning: used duplicate nontarget wavs", NUM_TARGETS - len(ntlog))

print("duration in minutes", total_wav_duration_s / 60)

combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
combiner.build(to_combine, dest_wavfile, "concatenate")

combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
combiner.build(to_combine, dest_mp3, "concatenate")

with open(groundtruth_data, "wb") as fh:
    pickle.dump(groundtruth, fh)
with open(full_transcript_file, "w") as fh:
    json.dump(transcript, fh)
with open(groundtruth_txt, "w") as fh:
    for t in groundtruth_target_times_ms:
        fh.write(f"{keyword},{t}\n")


# %% validate data
t_ms = np.random.choice(groundtruth_target_times_ms)
audio = pydub.AudioSegment.from_file(dest_wavfile)
pydub.playback.play(audio[t_ms : t_ms + 1000])

# %%
# listen to detections
keyword = "akawuka"
workdir = Path("/home/mark/tinyspeech_harvard/luganda")
pkl = workdir / "hp_sweep" / "exp_01" / "fold_00" / "result.pkl"
with open(pkl, "rb") as fh:
    result = pickle.load(fh)

op_point = 0.95

for post_processing_settings, results_per_thresh in result[keyword]:
    for thresh, (found_words, _) in results_per_thresh.items():
        if np.isclose(thresh, op_point):
            break

audio = pydub.AudioSegment.from_file(workdir / "akawuka_stream.wav")
with open(workdir / "akawuka_groundtruth.pkl", "rb") as fh:
    gt = pickle.load(fh)

print("detections", len(found_words))
print("num gt", len(gt["groundtruth_target_times_ms"]))

# %%
# listen to random detections


def decorate(word, keyword):
    if word == keyword:
        return f"[::{word}::]"
    return word


def transcript_by_offset(time_ms, groundtruth):
    offset_ms = 0
    for w in groundtruth["stream_data"]:
        duration_ms = w["duration_s"] * 1000
        if time_ms < offset_ms + duration_ms:
            offset_in_clip = time_ms - offset_ms
            pct_in_clip = offset_in_clip / duration_ms

            decorated = [
                decorate(word, keyword=keyword) for word in w["transcript"].split()
            ]
            print(" ".join(decorated))
            print(f"detection pct {pct_in_clip:0.2f}")
            if len(w["occurences_s"]) == 0:
                print("---- CERTAIN FALSE POSITIVE")
            break
        offset_ms += duration_ms


ix = np.random.randint(len(found_words))
rand_ms = found_words[ix][1]
transcript_by_offset(rand_ms, gt)
play(audio[rand_ms - 750 : rand_ms + 1000])

# %%
ix = 0
# %%
print(ix)
detection_ms = found_words[ix][1]
transcript_by_offset(detection_ms, gt)
play(audio[detection_ms - 750 : detection_ms + 1000])
false = "ffffffffffffffff"
print(len(false))

# %%
ix += 1

# %%
# listen to all detections
for ix, detection_ms in enumerate([d[1] for d in found_words]):
    print(ix)
    transcript_by_offset(detection_ms, gt)
    play(audio[detection_ms - 750 : detection_ms + 1000])
    time.sleep(1)

# %%
# create CSV of detection information
offset_ms = 0
rows = []
time_tolerance_ms = 750
for w in gt["stream_data"]:
    duration_ms = w["duration_s"] * 1000

    num_ds_for_segment = 0
    detections = []
    for detection_ms in [d[1] for d in found_words]:
        if detection_ms > offset_ms and detection_ms < (offset_ms + duration_ms):
            num_ds_for_segment += 1
            detections.append(detection_ms)

    wav = Path(w["wav"]).name
    transcript = w["transcript"]

    is_tp = False
    for d in detections:
        latest_time = d + time_tolerance_ms
        earliest_time = d - time_tolerance_ms
        for gt_time in gt["groundtruth_target_times_ms"]:
            if gt_time > latest_time:
                break
            if gt_time < earliest_time:
                continue
            is_tp = True

    has_kw = any([word == keyword for word in transcript.split()])
    detections_s = "/".join([str((d - offset_ms) / 1000) for d in detections])
    is_fn = has_kw and not is_tp
    is_fp = len(detections) > 0 and not is_tp

    row = (
        detections_s,
        num_ds_for_segment,
        has_kw,
        is_fn,
        is_fp,
        is_tp,
        w["transcript"],
        wav,
    )
    rows.append(row)
    offset_ms += duration_ms


# %%
fieldnames = [
    "detections (sec)",
    "num detections",
    "has keyword",
    "false negative",
    "false positive",
    "true positive",
    "transcript",
    "wav",
]
with open(workdir / "akawuka_analysis.csv", "w") as fh:
    writer = csv.writer(fh)
    writer.writerow(fieldnames)
    writer.writerows(rows)

# %%
# save extractions
detections_dest = workdir / "akawuka_detections"
source = workdir / "akawuka_stream.wav"
for ix, detection_ms in enumerate([d[1] for d in found_words]):

    dest = detections_dest / f"detection_{ix:03d}_{detection_ms}_ms.wav"
    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    start_s = (detection_ms - 750) / 1000
    end_s = (detection_ms + 1000) / 1000
    transformer.trim(start_s, end_s)
    transformer.build(str(source), str(dest))


# %%
keyword = "senyiga"
data = Path("/media/mark/hyperion/makerere/alignment/cs288")
alignments = data / "alignments"
clips = data / "cs288_clips"

tgs_for_kw = []
alignment_speakers = [
    d for d in os.listdir(alignments) if os.path.isdir(alignments / d)
]
for speaker in alignment_speakers:
    lab = clips / speaker / f"{speaker}.lab"
    wav = clips / speaker / f"{speaker}.wav"
    tgfile = alignments / speaker / f"{speaker}.TextGrid"
    with open(lab, "r") as fh:
        transcript = fh.read()
    if keyword in transcript:
        tgs_for_kw.append((tgfile, transcript, wav))
print(len(tgs_for_kw))

# %%
heard_transcripts = []
# %%
# listen to random alignment clips for keyword
rand_ix = np.random.randint(len(tgs_for_kw))
(tgfile, transcript, wavpath) = tgs_for_kw[rand_ix]

radio_data = Path("/media/mark/hyperion/makerere/uliza-clips")

print("TRANSCRIPT", rand_ix, transcript)
for t in heard_transcripts:
    d = levenshteinDistance(transcript, t)
    if d < 50:
        print("DUPLICATE", d)
        print(f"s1 {t}\ns2 {transcript}")
heard_transcripts.append(transcript)

tg = textgrid.TextGrid.fromFile(tgfile)
for interval in tg[0]:
    if interval.mark != keyword:
        continue
    start_s = interval.minTime
    end_s = interval.maxTime

wav = pydub.AudioSegment.from_file(wavpath)
play(wav[start_s * 1000 - 500 : end_s * 1000 + 500])

# %%

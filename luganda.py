#%%
import os
import glob
import shutil
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint

import matplotlib.pyplot as plt
import numpy as np
import sox
import pydub
from pydub.playback import play

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
import input_data
import textgrid


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
play(pydub.AudioSegment.from_file(w))

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
timing_csv = workdir / "covid_19_timing.csv"
covid_timings = {}
keyword = "covid"
with open(timing_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header
    for ix, row in enumerate(reader):
        wav = row[0]
        transcript = row[1]
        start_time_s = row[3]
        contains_keyword = any([w == keyword for w in transcript.split()])
        if contains_keyword:
            covid_timings[wav] = (start_time_s, transcript)
print(covid_timings)

# %%
# generate groundtruth timings
gt_target_times_ms = []
keyword = "covid"
cur_time_s = 0.0
for segment in stream_info:
    transcript = segment["transcript"]
    contains_keyword = any([w == keyword for w in transcript.split()])
    if contains_keyword:
        wav = segment["wav"]
        offset_s = float(covid_timings[wav][0])
        keyword_start_s = cur_time_s + offset_s
        gt_target_times_ms.append(keyword_start_s * 1000)
    cur_time_s += segment["duration_s"]

# %%

time_tolerance_ms = 1000
found_target_times = [t for f, t in found_words if f == keyword]

# find false negatives
false_negatives = 0
for time_ms in gt_target_times_ms:
    latest_time = time_ms + time_tolerance_ms
    earliest_time = time_ms - time_tolerance_ms
    potential_match = False
    for found_time in found_target_times:
        if found_time > latest_time:
            break
        if found_time < earliest_time:
            continue
        potential_match = True
    if not potential_match:
        false_negatives += 1

# find true/false positives
false_positives = 0  # no groundtruth match for model-found word
true_positives = 0
for word, time in found_words:
    if word == keyword:
        # highlight spurious words
        latest_time = time + time_tolerance_ms
        earliest_time = time - time_tolerance_ms
        potential_match = False
        for gt_time in gt_target_times_ms:
            if gt_time > latest_time:
                break
            if gt_time < earliest_time:
                continue
            potential_match = True
        if not potential_match:
            false_positives += 1
        else:
            true_positives += 1
if true_positives > len(gt_target_times_ms):
    print("WARNING: weird timing issue")
    true_positives = len(gt_target_times_ms)  # if thresh is low -- what causes this?
#     continue
tpr = true_positives / len(gt_target_times_ms)
false_rejections_per_instance = false_negatives / len(gt_target_times_ms)
false_positives = len(found_target_times) - true_positives
pp = pprint.PrettyPrinter()
pp.pprint(
    dict(
        tpr=tpr,
        false_positives=false_positives,
        false_negatives=false_negatives,
        false_rejections_per_instance=false_rejections_per_instance,
    )
)
# print("thresh", thresh, false_rejections_per_instance)
# print("thresh", thresh, "true positives ", true_positives, "TPR:", tpr)
# TODO(MMAZ) is there a beter way to calculate false positive rate?
# fpr = false_positives / (false_positives + true_negatives)
# fpr = false_positives / negatives
# print(
#     "false positives (model detection when no groundtruth target is present)",
#     false_positives,
# )
# fpr = false_positives / num_nontarget_words
# false_accepts_per_seconds = false_positives / (duration_s / (3600))

# %%
# listen to random alignments for covid
covid_alignments = workdir / "covid_alignments"
alignment_speakers = os.listdir(covid_alignments)

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
with open(transcript, 'r') as fh:
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
        continue #skip unaligned.txt
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

    dest = workdir / "1k_covid_alignments" / (tgfile.stem + ".mp3")

    transformer = sox.Transformer()
    transformer.convert(samplerate=16000)  # from 48K mp3s
    transformer.trim(start_s, end_s)
    if end_s - start_s < 1:
        pad_amt_s = (1. - (end_s - start_s))/2.
        transformer.pad(start_duration=pad_amt_s, end_duration=pad_amt_s)
    transformer.build(str(wavpath), str(dest))

# %%
# combine
dest = str(workdir / "all_covid_alignments.mp3")
mp3s = glob.glob(str(workdir / "1k_covid_alignments" / "*.mp3" ))
combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
# https://github.com/rabitt/pysox/blob/master/sox/combine.py#L46
combiner.build(mp3s, dest, "concatenate")

# %%

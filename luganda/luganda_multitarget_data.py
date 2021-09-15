#%%
from dataclasses import asdict, field, dataclass
import os
import glob
import shutil
import json
import subprocess
import time
import csv
import pickle
import datetime
from pathlib import Path
import pprint
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import sox
import pydub
import pydub.playback
import textgrid

# %%
l_data = Path("/media/mark/hyperion/makerere/uliza-clips")
l_csv = l_data / "transcripts.csv"

l_test = Path("/media/mark/hyperion/makerere/alignment/cs288/")
l_clips = l_test / "cs288_clips"
l_alignments = l_test / "alignments"

# TODO(mmaz): find a better method to exclude variants in spelling/plurals/etc
keyword_set = {
    "mask",
    "masiki",
    "masks",
    "korona",
    "corona",
}
# {"corona", "korona", "kolona", "coronavirus"}
# corona  covid  mask  okugema  ssennyiga
# alt spelling: senyiga

# %%
kwcount = {w: 0 for w in keyword_set}
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header [wav_filename,wav_filesize,transcript]
    for ix, row in enumerate(reader):
        transcript = row[2]
        for w in transcript.split():
            if w in keyword_set:
                kwcount[w] += 1
pprint.pprint(kwcount)
print("num lines", ix)

# %%
@dataclass
class MultiTargetWavTranscript:
    wav: str
    transcript: str
    keywords: Optional[List[str]] = None
    occurences_s: List[Dict[str, float]] = field(default_factory=list)
    tgfile: Optional[str] = None


# select random wavs without the keyword to intersperse stream with
non_targets = []
with open(l_csv, "r") as fh:
    reader = csv.reader(fh)
    next(reader)  # skip header [wav_filename,wav_filesize,transcript]
    for ix, row in enumerate(reader):
        wav_filename = l_data / row[0]
        assert os.path.isfile(wav_filename), "wav not found"
        transcript = row[2]
        has_kw = any([w in keyword_set for w in transcript.split()])
        if not has_kw:
            non_targets.append(
                MultiTargetWavTranscript(wav=str(wav_filename), transcript=transcript)
            )

# assemble wav transcripts with timings from alignments
# supports multiple keywords in a single wav
keyword_wav_transcripts = []
for a in os.listdir(l_alignments):
    if not os.path.isdir(l_alignments / a):
        continue  # unaligned.txt, alignment failed

    wav = l_clips / a / f"{a}.wav"

    lab = l_clips / a / f"{a}.lab"
    with open(lab, "r") as fh:
        transcript = fh.read()

    has_kw = any([w in keyword_set for w in transcript.split()])
    if not has_kw:
        continue

    tgfile = l_alignments / a / f"{a}.TextGrid"
    tg = textgrid.TextGrid.fromFile(tgfile)

    occurences_s = []
    keywords_present = set()

    for interval in tg[0]:
        if interval.mark not in keyword_set:
            continue
        keywords_present.add(interval.mark)
        start_s = interval.minTime
        end_s = interval.maxTime
        occurences_s.append(dict(keyword=interval.mark, start_s=start_s, end_s=end_s))

    if occurences_s == []:
        raise ValueError("why did we get here")

    keyword_wav_transcripts.append(
        MultiTargetWavTranscript(
            wav=str(wav),
            transcript=transcript,
            keywords=list(keywords_present),
            occurences_s=occurences_s,
            tgfile=str(tgfile),
        )
    )

print("data", len(keyword_wav_transcripts), len(non_targets))

mt, st = 0, 0
kwcount = {w: 0 for w in keyword_set}
for k in keyword_wav_transcripts:
    if len(k.keywords) > 1:
        mt += 1
    else:
        st += 1
        kwcount[k.keywords[0]] += 1
print("multi target", mt, "single target", st)
pprint.pprint(kwcount)

# %%
# listen to random samples
ix = np.random.randint(len(keyword_wav_transcripts))
w = keyword_wav_transcripts[ix]
print(ix, w.wav)


def decorate(word, keyword_set):
    if word in keyword_set:
        return f"[::{word}::]"
    return word


decorated = [decorate(word, keyword_set) for word in w.transcript.split()]
print(" ".join(decorated))
start_s = w.occurences_s[0]["start_s"]
end_s = w.occurences_s[0]["end_s"]
audio = pydub.AudioSegment.from_file(w.wav)
pydub.playback.play(audio[start_s * 1000 - 500 : end_s * 1000 + 500])

# %%
NUM_TARGETS = 80
ixs = np.random.choice(range(len(keyword_wav_transcripts)), NUM_TARGETS, replace=False)

kwcount = {w: 0 for w in keyword_set}
for ix in ixs:
    for kw in keyword_wav_transcripts[ix].keywords:
        kwcount[kw] += 1
pprint.pprint(kwcount)

# %%
# generate stream and groundtruth data
print(len(ixs))
workdir = Path.home() / "tinyspeech_harvard/luganda" / "mt_eval"
os.makedirs(workdir, exist_ok=False)  # ensure we dont overwrite files
dest_wavfile = str(workdir / f"stream.wav")
dest_mp3 = str(workdir / f"stream.mp3")
groundtruth_data = workdir / f"groundtruth.json"
full_transcript_file = workdir / f"full_transcript.json"
groundtruth_txt = workdir / f"groundtruth_labels.txt"
kwlist = workdir / "keyword_list.txt"

with open(kwlist, "w") as fh:
    fh.write("\n".join(keyword_set))

ntlog = set()

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

    for o in target.occurences_s:
        start_s = o["start_s"]
        kw = o["keyword"]
        t_ms = (total_wav_duration_s + start_s) * 1000
        groundtruth_target_times_ms.append(dict(keyword=kw, time_ms=t_ms))

    transcript.append(
        dict(
            transcript_type="target",
            transcript=target.transcript,
            start=total_wav_duration_s,
            end=total_wav_duration_s + target_duration_s,
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

    to_combine.extend([target.wav, non_target.wav])

    td = asdict(target)
    td["duration_s"] = target_duration_s
    nd = asdict(non_target)
    nd["duration_s"] = nontarget_duration_s
    stream_data.extend([td, nd])

groundtruth = dict(
    groundtruth_target_times_ms=groundtruth_target_times_ms,
    stream_data=stream_data,
    keyword_wav_transcripts=[asdict(t) for t in keyword_wav_transcripts],
    non_targets=[asdict(n) for n in non_targets],
    ixs=ixs.tolist(),
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

with open(groundtruth_data, "w") as fh:
    json.dump(groundtruth, fh)
with open(full_transcript_file, "w") as fh:
    json.dump(transcript, fh)
with open(groundtruth_txt, "w") as fh:
    for gt in groundtruth_target_times_ms:
        kw = gt["keyword"]
        t_ms = gt["time_ms"]
        fh.write(f"{kw},{t_ms}\n")

# %% validate data
t_ix = np.random.choice(len(groundtruth_target_times_ms))
gt = groundtruth_target_times_ms[t_ix]
kw = gt["keyword"]
t_ms = gt["time_ms"]
print(kw)
audio = pydub.AudioSegment.from_file(dest_wavfile)
pydub.playback.play(audio[t_ms : t_ms + 1000])

# %%

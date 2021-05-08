# %%
from dataclasses import asdict
import os
import glob
import shutil
import subprocess
import time
from collections import Counter
import csv
import pickle
import datetime
from pathlib import Path
import pprint
from typing import List, Optional, Tuple

import numpy as np
import sox
import pydub
from pydub.playback import play
import textgrid

from embedding import word_extraction, transfer_learning
from embedding import batch_streaming_analysis as sa
import input_data
from luganda_info import WavTranscript

# %%
# generate nontarget data
l_data = Path("/media/mark/hyperion/makerere/uliza-clips")
l_csv = l_data / "transcripts.csv"

l_test = Path("/media/mark/hyperion/makerere/alignment/cs288/")
l_clips = l_test / "cs288_clips"

workdir = Path("/home/mark/tinyspeech_harvard/luganda")
os.makedirs(workdir / "cs288_eval" / "nontarget", exist_ok=True)
dest_wavfile = str(workdir / "cs288_eval" / "nontarget" / "nontarget_stream.wav")
groundtruth_data = workdir / "cs288_eval" / "nontarget" / "nontarget_groundtruth.pkl"
assert not os.path.isfile(dest_wavfile), "already exists"
assert not os.path.isfile(groundtruth_data), "already exists"

NUM_WAVS = 160

# select random wavs without keyword to intersperse stream with
keywords_to_exclude = set(["corona", "korona", "kolona", "coronavirus", "mask", "masiki", "masks" ])
non_targets = []
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
ntlog = set()

total_wav_duration_s = 0.0
to_combine = []

stream_data = []

for ix in range(NUM_WAVS):
    non_target_ix = np.random.randint(len(non_targets))
    ntlog.add(non_target_ix)
    non_target = non_targets[non_target_ix]

    nontarget_duration_s = sox.file_info.duration(non_target.wav)

    total_wav_duration_s += nontarget_duration_s

    to_combine.append(str(non_target.wav))

    nd = asdict(non_target)
    nd["duration_s"] = nontarget_duration_s
    stream_data.append(nd)

groundtruth = dict(
    groundtruth_target_times_ms=[],
    stream_data=stream_data,
    keyword_wav_transcripts=[],
    non_targets=non_targets,
    ixs=[],
)

print("duration in minutes", total_wav_duration_s / 60)

combiner = sox.Combiner()
combiner.convert(samplerate=16000, n_channels=1)
combiner.build(to_combine, dest_wavfile, "concatenate")

with open(groundtruth_data, "wb") as fh:
    pickle.dump(groundtruth, fh)

# %%
groundtruth_data = workdir / "cs288_eval" / "nontarget" / "nontarget_groundtruth.pkl"
with open(groundtruth_data, 'rb') as fh:
    groundtruth = pickle.load(fh)

for stream_segment in groundtruth["stream_data"]:
    transcript = stream_segment["transcript"]
    for w in transcript.split():
        if w in keywords_to_exclude:
            raise ValueError("keyword present", transcript)

# %%
# corona:
# results for 0.80
# No ground truth yet, 52false positives
# results for 0.85
# No ground truth yet, 33false positives
# results for 0.90
# No ground truth yet, 23false positives
# results for 0.95
# No ground truth yet, 7false positives
#
# mask:
# results for 0.75
# No ground truth yet, 30false positives
# results for 0.80
# No ground truth yet, 19false positives
# results for 0.85
# No ground truth yet, 11false positives
# results for 0.90
# No ground truth yet, 5false positives
# results for 0.95
# No ground truth yet, 3false positives
# results for 1.00
# No ground truth yet, 0false positives
